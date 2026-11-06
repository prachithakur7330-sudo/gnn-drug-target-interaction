"""GraphDTA-style model: a GNN over the drug graph plus a 1D CNN over the protein sequence.

The two embeddings are concatenated and passed through an MLP that predicts pKd. This
follows the design of Nguyen et al. 2021 (GraphDTA) but is written from scratch here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

from .data import AMINO_ACIDS, ATOM_FEATURE_DIM


class DrugEncoder(nn.Module):
    """Three GIN layers, then mean-pool the atom embeddings into one drug vector."""

    def __init__(self, in_dim=ATOM_FEATURE_DIM, hidden=128, out_dim=128, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_dim, hidden, hidden, hidden]
        for a, b in zip(dims[:-1], dims[1:]):
            mlp = nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, b))
            self.convs.append(GINConv(mlp))
        self.bns = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in self.convs)
        self.head = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.relu(self.head(global_mean_pool(x, batch)))


class ProteinEncoder(nn.Module):
    """Embed amino acids, then a stack of 1D convolutions and global max-pool."""

    def __init__(self, vocab=len(AMINO_ACIDS) + 1, embed_dim=128, out_dim=128, n_filters=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=0)
        # MaxPool after the first two blocks: a kinase motif is short relative to a
        # 1000-residue sequence, so downsampling keeps the signal and makes the later
        # (wider) convolutions roughly 9x cheaper.
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, n_filters, kernel_size=8), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(n_filters, n_filters * 2, kernel_size=8), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(n_filters * 2, n_filters * 3, kernel_size=8), nn.ReLU(),
        )
        self.head = nn.Linear(n_filters * 3, out_dim)

    def forward(self, seq):
        h = self.embed(seq).transpose(1, 2)        # (B, embed, L)
        h = self.conv(h)
        h = torch.max(h, dim=2).values             # global max over sequence length
        return F.relu(self.head(h))


class GraphDTA(nn.Module):
    def __init__(self, drug_dim=128, prot_dim=128, hidden=512, dropout=0.2):
        super().__init__()
        self.drug = DrugEncoder(out_dim=drug_dim, dropout=dropout)
        self.protein = ProteinEncoder(out_dim=prot_dim)
        self.regressor = nn.Sequential(
            nn.Linear(drug_dim + prot_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch, protein_tensor):
        drug_vec = self.drug(batch.x, batch.edge_index, batch.batch)
        seqs = protein_tensor[batch.target_idx.view(-1)]
        prot_vec = self.protein(seqs)
        joined = torch.cat([drug_vec, prot_vec], dim=1)
        return self.regressor(joined).view(-1)
