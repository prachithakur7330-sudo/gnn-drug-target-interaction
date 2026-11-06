import numpy as np
import torch

from dti.data import (
    ATOM_FEATURE_DIM,
    MAX_PROTEIN_LEN,
    affinity_to_pkd,
    encode_protein,
    smiles_to_graph,
)


def test_encode_protein_length_and_padding():
    idx = encode_protein("ACDEF")
    assert idx.shape == (MAX_PROTEIN_LEN,)
    assert idx[0] == 1 and idx[4] == 5          # A..F map to 1..5
    assert idx[5] == 0                          # padded after the sequence


def test_encode_protein_truncates_and_handles_unknown():
    long_seq = "A" * (MAX_PROTEIN_LEN + 50)
    assert encode_protein(long_seq).shape == (MAX_PROTEIN_LEN,)
    assert encode_protein("XBZ")[0] == 0        # non-standard letters -> 0


def test_smiles_to_graph_benzene():
    x, edge_index = smiles_to_graph("c1ccccc1")
    assert x.shape == (6, ATOM_FEATURE_DIM)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == 12            # 6 aromatic bonds, both directions
    assert x[:, -2].sum() == 6                  # every atom flagged aromatic


def test_smiles_to_graph_single_atom_gets_self_loop():
    x, edge_index = smiles_to_graph("[Na+]")
    assert x.shape[0] == 1
    assert edge_index.shape[1] >= 1             # self-loop keeps message passing valid


def test_smiles_to_graph_rejects_garbage():
    assert smiles_to_graph("not a molecule") is None


def test_affinity_to_pkd_reference_points():
    # 1 nM -> pKd 9, 10000 nM ceiling -> pKd 5
    assert np.isclose(affinity_to_pkd(1.0), 9.0)
    assert np.isclose(affinity_to_pkd(1e4), 5.0)


def test_graph_edges_are_symmetric():
    x, edge_index = smiles_to_graph("CCO")       # ethanol
    pairs = {(int(a), int(b)) for a, b in edge_index.t()}
    for a, b in list(pairs):
        assert (b, a) in pairs
