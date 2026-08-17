"""
Differential privacy + secure aggregation (Eqs. 5-6) and a real privacy
accountant.

This module fills in the "privacy composition was not audited" gap noted
in the manuscript's Limitations section -- with an actual accountant
(Renyi Differential Privacy / Mironov 2017), not invented numbers. Given
your real clip norm C, noise scale sigma, and number of rounds T, it
computes the true cumulative (epsilon, delta) for the Gaussian mechanism
used in Eq. 5. Run `python -m fedgsentry.privacy` to see it applied to
the hyperparameters in Table III of the manuscript (C=1.0, sigma=0.5,
T=100) as a worked example -- you should re-run it with whatever values
you actually train with, and report the output directly rather than
retyping these example numbers.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------
# Eq. 5: per-client clip + Gaussian noise
# --------------------------------------------------------------------------

def clip_and_noise(delta_theta: torch.Tensor, clip_norm: float, noise_scale: float) -> torch.Tensor:
    """delta_theta_tilde = delta_theta / max(1, ||delta_theta||_2 / C) + N(0, sigma^2 C^2 I)."""
    norm = torch.norm(delta_theta, p=2)
    clipped = delta_theta / max(1.0, (norm / clip_norm).item())
    noise = torch.randn_like(delta_theta) * (noise_scale * clip_norm)
    return clipped + noise


# --------------------------------------------------------------------------
# Latent Anchor Consistency screening + Eq. 6 weighted aggregation
# --------------------------------------------------------------------------

def latent_anchor_consistency(
    updates: List[torch.Tensor],
    anchor_basis: torch.Tensor,
    mad_threshold: float = 3.5,
) -> List[int]:
    """Screen client updates for poisoning.

    Projects each flattened update onto a low-rank anchor subspace
    (estimated upstream from previously accepted updates -- see
    `estimate_anchor_basis`) and flags updates whose residual norm is a
    robust outlier (modified Z-score via median absolute deviation).

    Returns indices of ACCEPTED clients.
    """
    residuals = []
    for u in updates:
        proj = anchor_basis @ (anchor_basis.T @ u)
        residuals.append(torch.norm(u - proj, p=2).item())
    residuals = np.array(residuals)

    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median)) + 1e-8
    modified_z = 0.6745 * (residuals - median) / mad

    accepted = [i for i, z in enumerate(modified_z) if z <= mad_threshold]
    if not accepted:  # never fully empty a round
        accepted = [int(np.argmin(residuals))]
    return accepted


def estimate_anchor_basis(prev_updates: List[torch.Tensor], rank: int = 8) -> torch.Tensor:
    """Low-rank anchor subspace from last round's accepted updates via SVD."""
    stacked = torch.stack(prev_updates, dim=0)  # [k, d]
    stacked = stacked - stacked.mean(dim=0, keepdim=True)
    u, s, vh = torch.linalg.svd(stacked, full_matrices=False)
    rank = min(rank, vh.shape[0])
    return vh[:rank].T  # [d, rank]


def weighted_average(updates: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    """Eq. 6: theta^(t+1) contribution = sum_k (n_k / sum n_j) * delta_theta_tilde_k."""
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    stacked = torch.stack(updates, dim=0)
    return (w.view(-1, 1) * stacked).sum(dim=0)


# --------------------------------------------------------------------------
# Real RDP accountant for the per-round Gaussian mechanism, composed over
# T rounds. Standard result (Mironov, 2017): for the Gaussian mechanism
# with noise multiplier z = sigma (since sensitivity is normalized to 1
# by clipping to C and the noise below is expressed in units of C),
# RDP(alpha) = alpha / (2 * z^2) per round. Composition over T
# *statistically independent* rounds simply sums RDP. Convert to
# (epsilon, delta)-DP by minimizing over the Renyi order alpha.
# --------------------------------------------------------------------------

def rdp_gaussian(alpha: float, noise_multiplier: float) -> float:
    return alpha / (2.0 * noise_multiplier ** 2)


def compose_epsilon(
    noise_multiplier: float,
    num_rounds: int,
    delta: float,
    alphas: Tuple[float, ...] = tuple(
        list(np.arange(1.5, 64, 0.5))
    ),
) -> Tuple[float, float]:
    """Return (epsilon, best_alpha) for the composed mechanism over `num_rounds`.

    NOTE on the trust model: this treats each client's per-round release as
    a fresh application of the Gaussian mechanism (no privacy amplification
    by subsampling is assumed, since in cross-institution FL every client
    typically participates every round). If your deployment subsamples
    clients per round, amplification would tighten this bound further --
    that extension is not included here and should be added before
    reporting composition-by-subsampling claims.
    """
    best_eps, best_alpha = float("inf"), None
    for alpha in alphas:
        if alpha <= 1:
            continue
        rdp_total = num_rounds * rdp_gaussian(alpha, noise_multiplier)
        eps = rdp_total + np.log(1.0 / delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps, best_alpha = eps, alpha
    return best_eps, best_alpha


if __name__ == "__main__":
    # Worked example using the manuscript's own Table III values
    # (C=1.0, sigma=0.5 -> noise_multiplier z = sigma/C = 0.5; T=100 rounds).
    # This is a real computation, not a placeholder -- swap in your own
    # values from an actual training run.
    for delta in (1e-5, 1e-4):
        eps, alpha = compose_epsilon(noise_multiplier=0.5, num_rounds=100, delta=delta)
        print(f"delta={delta:.0e}  ->  cumulative epsilon ~= {eps:.2f}  (best Renyi order alpha={alpha})")
    print(
        "\nInterpretation: at sigma/C = 0.5 composed naively (independent Gaussian\n"
        "mechanism per round, no subsampling amplification) over 100 rounds, the\n"
        "cumulative epsilon is large. If you want a tighter, deployment-realistic\n"
        "epsilon, either (a) increase noise_multiplier, (b) reduce communication\n"
        "rounds, or (c) add per-round client subsampling and use its amplification\n"
        "bound. Report whichever (C, sigma, T, delta) you actually deploy with."
    )
