"""
Federated training loop implementing Algorithm 1 from the manuscript:

  for round t in 0..T-1:
    server broadcasts theta^(t)
    each client: local SGLD training -> clipped+noised update
    server: Latent Anchor Consistency screening -> accepted set A_t
    server: weighted average -> theta^(t+1)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from .data import ClientGraph
from .models import FedGSentry
from .optim import SGLD
from .privacy import clip_and_noise, latent_anchor_consistency, estimate_anchor_basis, weighted_average


@dataclass
class FederatedConfig:
    rounds: int = 100
    local_epochs: int = 5          # note: paper's Table III lists 100 "local epochs per
                                    # round" together with T=100 rounds -- that's 10,000
                                    # full-batch local steps per client, and each step here
                                    # recomputes an exact eigendecomposition of the local
                                    # Laplacian (spectral filter), which is expensive at
                                    # that count. Default here is smaller for tractability;
                                    # override with --local_epochs 100 to match Table III
                                    # literally, but expect a much longer run.
    lr: float = 1e-3
    clip_norm: float = 1.0
    noise_scale: float = 0.5
    use_dp: bool = True
    use_lac_screening: bool = True
    hidden_dim: int = 64
    n_heads: int = 8
    dropout: float = 0.3
    use_spectral_filter: bool = True
    use_transformer: bool = True
    use_sgld: bool = True
    device: str = "cpu"


@dataclass
class RoundLog:
    round: int
    accepted_clients: List[int]
    flagged_adversarial_accepted: int = 0
    flagged_adversarial_rejected: int = 0


def _flatten(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _unflatten_into(model: torch.nn.Module, flat: torch.Tensor):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n


def local_train(
    global_state: torch.Tensor,
    client: ClientGraph,
    in_dim: int,
    cfg: FederatedConfig,
) -> torch.Tensor:
    """Run local SGLD (or Adam, if use_sgld=False) training and return the
    clipped+noised parameter delta (Delta-theta-tilde in Eq. 5)."""
    model = FedGSentry(
        in_dim=in_dim, hidden_dim=cfg.hidden_dim, n_heads=cfg.n_heads,
        dropout=cfg.dropout, use_spectral_filter=cfg.use_spectral_filter,
        use_transformer=cfg.use_transformer,
    ).to(cfg.device)
    _unflatten_into(model, global_state)
    theta_0 = _flatten(model).clone()

    optimizer = SGLD(model.parameters(), lr=cfg.lr) if cfg.use_sgld \
        else torch.optim.Adam(model.parameters(), lr=cfg.lr)

    x, adj, y = client.x.to(cfg.device), client.adj.to(cfg.device), client.y.to(cfg.device)
    train_mask = client.train_mask.to(cfg.device)

    if train_mask.sum() == 0:
        return torch.zeros_like(theta_0)

    model.train()
    for _ in range(cfg.local_epochs):
        optimizer.zero_grad()
        logits = model(x, adj)
        loss = F.cross_entropy(logits[train_mask], y[train_mask].clamp(min=0))
        loss.backward()
        optimizer.step()

    theta_k = _flatten(model)
    delta = theta_k - theta_0

    if client.is_adversarial:
        # Simulate a poisoning client: scale + sign-flip the honest update.
        # This is what the Latent Anchor Consistency screen (Section III-G)
        # is supposed to catch -- used for the adversarial robustness test.
        delta = -3.0 * delta + torch.randn_like(delta) * 0.1

    if cfg.use_dp:
        delta = clip_and_noise(delta, cfg.clip_norm, cfg.noise_scale)

    return delta


@torch.no_grad()
def evaluate(model: torch.nn.Module, client: ClientGraph, cfg: FederatedConfig, mask_name: str = "test_mask"):
    model.eval()
    x, adj, y = client.x.to(cfg.device), client.adj.to(cfg.device), client.y.to(cfg.device)
    mask = getattr(client, mask_name).to(cfg.device)
    if mask.sum() == 0:
        return None
    logits = model(x, adj)
    probs = F.softmax(logits[mask], dim=-1)[:, 1]
    preds = probs > 0.5
    labels = y[mask]
    valid = labels >= 0
    if valid.sum() == 0:
        return None
    return {
        "preds": preds[valid].cpu(),
        "probs": probs[valid].cpu(),
        "labels": labels[valid].cpu(),
    }


def run_federated_training(
    clients: List[ClientGraph],
    in_dim: int,
    cfg: FederatedConfig,
    verbose: bool = True,
) -> tuple[FedGSentry, List[RoundLog]]:
    global_model = FedGSentry(
        in_dim=in_dim, hidden_dim=cfg.hidden_dim, n_heads=cfg.n_heads,
        dropout=cfg.dropout, use_spectral_filter=cfg.use_spectral_filter,
        use_transformer=cfg.use_transformer,
    ).to(cfg.device)
    global_state = _flatten(global_model)

    prev_accepted_updates: Optional[List[torch.Tensor]] = None
    logs: List[RoundLog] = []

    for t in range(cfg.rounds):
        updates = [local_train(global_state, c, in_dim, cfg) for c in clients]
        client_sizes = [max(int(c.train_mask.sum().item()), 1) for c in clients]

        if cfg.use_lac_screening and prev_accepted_updates and len(prev_accepted_updates) >= 2:
            anchor_basis = estimate_anchor_basis(prev_accepted_updates)
            accepted_idx = latent_anchor_consistency(updates, anchor_basis)
        else:
            accepted_idx = list(range(len(updates)))  # can't screen without a prior-round anchor

        accepted_updates = [updates[i] for i in accepted_idx]
        accepted_weights = [client_sizes[i] for i in accepted_idx]

        flagged_adv_accepted = sum(1 for i in accepted_idx if clients[i].is_adversarial)
        flagged_adv_rejected = sum(
            1 for i in range(len(clients)) if clients[i].is_adversarial and i not in accepted_idx
        )
        logs.append(RoundLog(t, accepted_idx, flagged_adv_accepted, flagged_adv_rejected))

        aggregated_delta = weighted_average(accepted_updates, accepted_weights)
        global_state = global_state + aggregated_delta
        prev_accepted_updates = accepted_updates

        if verbose and (t % max(1, cfg.rounds // 10) == 0 or t == cfg.rounds - 1):
            print(f"[round {t+1}/{cfg.rounds}] accepted {len(accepted_idx)}/{len(clients)} clients "
                  f"({flagged_adv_accepted} adversarial slipped through, "
                  f"{flagged_adv_rejected} adversarial correctly rejected)")

    _unflatten_into(global_model, global_state)
    return global_model, logs
