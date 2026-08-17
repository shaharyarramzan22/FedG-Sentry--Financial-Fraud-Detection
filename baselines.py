"""
Baseline models corresponding to Table IV of the manuscript:
FedAvg-GCN, GraphSAGE-FL, GAT-FL, SGLD-FL.

("GAGA" in the manuscript refers to a specific published architecture
[group aggregation for graph anomaly detection]; a faithful reproduction
requires implementing that paper directly rather than approximating it
here, so it is intentionally omitted -- don't silently substitute
something else and call it GAGA in your table.)

Each baseline plugs into the same `run_federated_training` loop in
federated.py (FedAvg-style weighted averaging, no DP/LAC by default,
matching how these methods are actually specified in the literature)
by being swapped in as the `FedGSentry`-equivalent model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.sparse.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj, self.lin(x))


class FedAvgGCN(nn.Module):
    """Vanilla 2-layer GCN trained with FedAvg (homophily-assuming baseline)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.gc2(h, adj))
        return self.head(h)


class GraphSAGELayer(nn.Module):
    """Mean-aggregator SAGE layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        neigh = torch.sparse.mm(adj, x)
        return self.lin_self(x) + self.lin_neigh(neigh)


class GraphSAGEFL(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.sage1 = GraphSAGELayer(in_dim, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, adj):
        h = F.relu(self.sage1(x, adj))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.sage2(h, adj))
        return self.head(h)


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim * n_heads)
        self.attn_src = nn.Parameter(torch.randn(n_heads, out_dim) * 0.1)
        self.attn_dst = nn.Parameter(torch.randn(n_heads, out_dim) * 0.1)
        self.dropout = dropout

    def forward(self, x, adj):
        n = x.shape[0]
        h = self.lin(x).view(n, self.n_heads, self.out_dim)
        mask = adj.to_dense() != 0

        src_score = torch.einsum("nhd,hd->nh", h, self.attn_src)
        dst_score = torch.einsum("nhd,hd->nh", h, self.attn_dst)
        scores = src_score.unsqueeze(1) + dst_score.unsqueeze(0)  # [n, n, H]
        scores = F.leaky_relu(scores, 0.2).permute(2, 0, 1)       # [H, n, n]
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.einsum("hij,jhd->ihd", attn, h)
        return out.reshape(n, -1)


class GATFL(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        head_dim = hidden_dim // n_heads
        self.gat1 = GATLayer(in_dim, head_dim, n_heads, dropout)
        self.gat2 = GATLayer(hidden_dim, head_dim, n_heads, dropout)
        self.head = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, adj):
        h = F.elu(self.gat1(x, adj))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.gat2(h, adj))
        return self.head(h)


class SGLDFL(nn.Module):
    """Same backbone as FedAvgGCN; the distinguishing factor for 'SGLD-FL'
    is the *optimizer* used during local training (SGLD instead of Adam/SGD)
    -- pass optim.SGLD when training this model, matching how it's trained
    in train.py's baseline loop."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.gc2(h, adj))
        return self.head(h)


BASELINE_REGISTRY = {
    "fedavg-gcn": FedAvgGCN,
    "graphsage-fl": GraphSAGEFL,
    "gat-fl": GATFL,
    "sgld-fl": SGLDFL,
}
