"""
FedG-Sentry model components:
  - HeterophilyAwareSpectralFilter  (Eq. 2 in the paper)
  - RelationalTransformerEncoder    (Eq. 3)
  - FedGSentry (full model = spectral filter -> transformer -> MLP head)

Implemented with plain PyTorch (dense small-graph eigendecomposition for
the spectral filter, sparse adjacency for message passing) so the whole
package has no PyTorch-Geometric dependency.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeterophilyAwareSpectralFilter(nn.Module):
    """Learnable spectral filter g_phi(Lambda) applied in the graph Fourier domain.

    H_spec = U * g_phi(Lambda) * U^T * X

    g_phi is parameterized as a small MLP over eigenvalues so it can learn
    an arbitrary (not just low-pass) frequency response -- initialized to
    retain mid/high frequencies rather than smooth them away, which is
    what distinguishes this from a standard GCN filter.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, max_nodes_for_exact_eig: int = 4000):
        super().__init__()
        self.freq_response = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Bias the initial response to emphasize high frequencies (bandpass/high-pass
        # rather than the implicit low-pass of vanilla GCN aggregation).
        with torch.no_grad():
            self.freq_response[-1].bias.fill_(0.5)
        self.max_nodes_for_exact_eig = max_nodes_for_exact_eig

    def forward(self, x: torch.Tensor, adj: torch.sparse.Tensor) -> torch.Tensor:
        n = x.shape[0]
        dense_adj = adj.to_dense()
        # Laplacian L = I - D^{-1/2} A D^{-1/2}; `adj` passed in is already
        # symmetrically normalized (see data._normalize_adj), so L = I - adj.
        L = torch.eye(n, device=x.device) - dense_adj

        if n <= self.max_nodes_for_exact_eig:
            eigvals, eigvecs = torch.linalg.eigh(L)  # symmetric -> real eigendecomposition
            response = self.freq_response(eigvals.unsqueeze(-1)).squeeze(-1)  # [n]
            response = torch.sigmoid(response) * 2.0  # keep response in a stable, bounded range
            filtered = eigvecs @ torch.diag(response) @ eigvecs.T @ x
        else:
            # For large client graphs, exact eigendecomposition is O(n^3) and
            # infeasible. Fall back to a K-term Chebyshev-style polynomial
            # approximation of the same learnable filter, which avoids
            # materializing U, Lambda explicitly while still allowing an
            # arbitrary (non low-pass-only) frequency response.
            filtered = self._chebyshev_approx(x, L)
        return filtered

    def _chebyshev_approx(self, x: torch.Tensor, L: torch.Tensor, K: int = 6) -> torch.Tensor:
        # Rescale L to [-1, 1] assuming normalized-Laplacian eigenvalues in [0, 2].
        L_hat = L - torch.eye(L.shape[0], device=L.device)
        Tx_0 = x
        Tx_1 = L_hat @ x
        out = 0.5 * Tx_0 + 0.5 * Tx_1
        Tx_prev, Tx_cur = Tx_0, Tx_1
        for k in range(2, K):
            Tx_next = 2 * (L_hat @ Tx_cur) - Tx_prev
            weight = torch.sigmoid(torch.tensor(float(k) / K)) * 2.0 / K
            out = out + weight * Tx_next
            Tx_prev, Tx_cur = Tx_cur, Tx_next
        return out


class RelationalTransformerEncoder(nn.Module):
    """Multi-head, graph-masked self-attention (Eq. 3).

    A single relation type is assumed unless `n_relations` > 1, in which
    case per-relation key/value projections are learned and summed
    (mirrors Eq. 3's sum over r).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_heads: int = 8,
                 n_relations: int = 1, dropout: float = 0.3):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads
        self.n_relations = n_relations

        self.q_proj = nn.Linear(in_dim, hidden_dim)
        self.k_proj = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(n_relations)])
        self.v_proj = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(n_relations)])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.residual_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor, adj: torch.sparse.Tensor) -> torch.Tensor:
        # Sparse, edge-restricted attention: O(|E| * d * H) as claimed in the
        # manuscript's complexity analysis (Section III-I) -- a dense [n,n]
        # score matrix (the naive way to implement Eq. 3) is O(n^2) and runs
        # out of memory well before real dataset scale (Elliptic ~200K nodes,
        # IEEE-CIS ~590K rows), so it does not actually match the paper's own
        # claim. This version only ever scores existing edges (+self-loops).
        n = x.shape[0]
        device = x.device
        adj = adj.coalesce()
        src, dst = adj.indices()[0], adj.indices()[1]
        self_idx = torch.arange(n, device=device)
        src = torch.cat([src, self_idx])
        dst = torch.cat([dst, self_idx])

        q = self.q_proj(x).view(n, self.n_heads, self.d_k)

        head_outs = []
        for r in range(self.n_relations):
            k = self.k_proj[r](x).view(n, self.n_heads, self.d_k)
            v = self.v_proj[r](x).view(n, self.n_heads, self.d_k)

            q_e = q[dst]   # query at the receiving node
            k_e = k[src]   # key at the sending (neighbor) node
            v_e = v[src]

            scores = (q_e * k_e).sum(-1) / (self.d_k ** 0.5)  # [E, H]
            attn = self._segment_softmax(scores, dst, n)       # softmax over each node's incoming edges
            attn = self.dropout(attn)

            weighted = attn.unsqueeze(-1) * v_e                 # [E, H, d_k]
            out_r = torch.zeros(n, self.n_heads, self.d_k, device=device)
            out_r.index_add_(0, dst, weighted)
            head_outs.append(out_r)

        combined = torch.stack(head_outs, dim=0).sum(0).reshape(n, -1)  # sum over relations
        combined = self.out_proj(combined)
        return self.norm(combined + self.residual_proj(x))

    @staticmethod
    def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
        """Softmax of `scores` [E, H] grouped by `index` [E] (each node's incoming edges)."""
        H = scores.shape[1]
        seg_max = torch.full((n, H), float("-inf"), device=scores.device)
        seg_max.scatter_reduce_(0, index.unsqueeze(-1).expand(-1, H), scores, reduce="amax", include_self=True)
        scores = scores - seg_max[index]
        exp_scores = scores.exp()
        seg_sum = torch.zeros(n, H, device=scores.device)
        seg_sum.index_add_(0, index, exp_scores)
        return exp_scores / (seg_sum[index] + 1e-12)


class FedGSentry(nn.Module):
    """Full model: spectral filter -> relational transformer -> classifier head."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_heads: int = 8,
                 dropout: float = 0.3, use_spectral_filter: bool = True,
                 use_transformer: bool = True):
        super().__init__()
        self.use_spectral_filter = use_spectral_filter
        self.use_transformer = use_transformer

        self.spectral_filter = HeterophilyAwareSpectralFilter(in_dim) if use_spectral_filter else None
        encoder_in_dim = in_dim
        self.transformer = (
            RelationalTransformerEncoder(encoder_in_dim, hidden_dim, n_heads, dropout=dropout)
            if use_transformer else nn.Linear(encoder_in_dim, hidden_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor, adj: torch.sparse.Tensor) -> torch.Tensor:
        h = self.spectral_filter(x, adj) if self.use_spectral_filter else x
        h = self.transformer(h, adj) if self.use_transformer else self.transformer(h)
        return self.head(h)  # logits, [n_nodes, 2]
