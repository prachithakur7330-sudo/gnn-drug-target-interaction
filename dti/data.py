"""Load the DAVIS kinase binding dataset and turn each drug-target pair into a graph.

DAVIS (Davis et al. 2011) measured Kd for 68 drugs against 442 kinase targets, a
dense 68x442 grid. Following DeepDTA we model affinity as pKd = -log10(Kd / 1e9); the
10,000 nM ceiling used for non-binders maps to pKd = 5.

Each drug SMILES becomes a molecular graph (atoms = nodes). Each protein is encoded as
its amino-acid sequence (handled by the model's CNN), so a sample is a drug graph plus
a target index into the shared sequence tensor.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

# ---- protein sequence encoding --------------------------------------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i + 1 for i, a in enumerate(AMINO_ACIDS)}  # 0 is padding / unknown
MAX_PROTEIN_LEN = 1000


def encode_protein(seq, max_len=MAX_PROTEIN_LEN):
    idx = np.zeros(max_len, dtype=np.int64)
    for i, aa in enumerate(seq[:max_len]):
        idx[i] = AA_TO_IDX.get(aa, 0)
    return idx


# ---- atom featurisation ----------------------------------------------------
ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "other"]
HYBRIDISATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def _one_hot(value, choices):
    vec = [int(value == c) for c in choices]
    if not any(vec):  # unseen category falls into the last "other" slot
        vec[-1] = 1
    return vec


def atom_features(atom):
    return (
        _one_hot(atom.GetSymbol(), ATOM_TYPES)
        + _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + _one_hot(atom.GetHybridization(), HYBRIDISATIONS)
        + [
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ]
    )


ATOM_FEATURE_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))


def smiles_to_graph(smiles):
    """SMILES -> (node_features, edge_index) as tensors, or None if RDKit can't parse it.

    Bonds are added in both directions so message passing is symmetric. A self-loop
    is added so single-atom molecules still have an edge for the GNN layers.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edges = [[], []]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges[0] += [i, j]
        edges[1] += [j, i]
    if not edges[0]:
        edges = [list(range(mol.GetNumAtoms())), list(range(mol.GetNumAtoms()))]
    edge_index = torch.tensor(edges, dtype=torch.long)
    return x, edge_index


def affinity_to_pkd(kd_nm):
    return -np.log10(np.asarray(kd_nm, dtype=np.float64) / 1e9)


class DavisDataset(InMemoryDataset):
    """DAVIS as PyG graphs. split is 'train' or 'test' using the DeepDTA fold files."""

    def __init__(self, root, split="train", raw_dir="data/davis"):
        self.split = split
        self._raw_dir = Path(raw_dir)
        super().__init__(root)
        path = self.processed_paths[0 if split == "train" else 1]
        self.data, self.slices, self.proteins = torch.load(path, weights_only=False)

    @property
    def processed_file_names(self):
        return ["davis_train.pt", "davis_test.pt"]

    def process(self):
        ligands = json.load(open(self._raw_dir / "ligands_can.txt"))
        proteins = json.load(open(self._raw_dir / "proteins.txt"))
        Y = pickle.load(open(self._raw_dir / "Y", "rb"), encoding="latin1")
        Y = np.asarray(Y, dtype=np.float64)

        drug_smiles = list(ligands.values())
        target_seqs = list(proteins.values())
        drug_graphs = [smiles_to_graph(s) for s in drug_smiles]
        protein_tensor = torch.tensor(
            np.stack([encode_protein(s) for s in target_seqs]), dtype=torch.long
        )

        train_folds = json.load(open(self._raw_dir / "train_fold_setting1.txt"))
        train_idx = sorted(i for fold in train_folds for i in fold)
        test_idx = sorted(json.load(open(self._raw_dir / "test_fold_setting1.txt")))

        rows, cols = np.unravel_index(np.arange(Y.size), Y.shape)
        pkd = affinity_to_pkd(Y).reshape(-1)

        for split, indices, out_path in [
            ("train", train_idx, self.processed_paths[0]),
            ("test", test_idx, self.processed_paths[1]),
        ]:
            samples = []
            for flat in indices:
                drug_i, target_j = int(rows[flat]), int(cols[flat])
                graph = drug_graphs[drug_i]
                if graph is None:
                    continue
                x, edge_index = graph
                samples.append(Data(
                    x=x,
                    edge_index=edge_index,
                    target_idx=torch.tensor([target_j], dtype=torch.long),
                    y=torch.tensor([pkd[flat]], dtype=torch.float),
                    drug_idx=torch.tensor([drug_i], dtype=torch.long),
                ))
            data, slices = self.collate(samples)
            torch.save((data, slices, protein_tensor), out_path)
