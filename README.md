# FedG-Sentry--Financial-Fraud-Detection
Heterophily-Aware Federated Graph Learning for Cross-Border Financial Fraud Detection
# FedG-Sentry — reference implementation

A real, runnable implementation of the architecture described in the
manuscript: heterophily-aware spectral filtering, a relational
Transformer encoder, SGLD-based Bayesian federated optimization, and
differentially-private secure aggregation with anomaly screening
(Latent Anchor Consistency).

**This is not a results generator.** It doesn't ship or hardcode the
numbers already in the manuscript (96.4% accuracy, etc.). Whatever you
get from running it on real data is what you actually have. Read the
caveats below before you put any number from this code into a paper.

## Install

```bash
pip install -r requirements.txt
```

## Quick smoke test (no data download needed)

```bash
python train.py --dataset synthetic --rounds 20 --n_clients 6 --seeds 1
```

This uses a synthetic, controlled "FedG-Sim"-style generator
(`fedgsentry/data.py::_load_synthetic_fedgsim`) so you can confirm the
whole pipeline runs before pointing it at real data. Don't report
synthetic-data numbers as if they came from Elliptic / IEEE-CIS.

## Real datasets

Both require you to accept dataset terms on Kaggle yourself — this code
cannot and does not download them for you.

- **Elliptic Bitcoin**: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
  → place `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`,
  `elliptic_txs_edgelist.csv` in `--data_dir`.
- **IEEE-CIS Fraud Detection**: https://www.kaggle.com/competitions/ieee-fraud-detection
  → place `train_transaction.csv` in `--data_dir`. (IEEE-CIS has no native
  graph; a k-NN similarity graph is constructed over numeric transaction
  fields as a relational proxy — this is a modeling choice you should
  state explicitly in the paper, not present as an inherent graph.)

```bash
python train.py --dataset elliptic --data_dir ./data --rounds 100 \
    --n_clients 8 --seeds 5 --run_baselines --run_ablation
```

## Adversarial-ratio sweep

The manuscript's Limitations section notes only one adversarial-client
ratio (10%) was tested. This code can actually run a sweep instead of
just flagging the gap:

```bash
python train.py --dataset elliptic --data_dir ./data \
    --adversarial_sweep 0.0 0.1 0.2 0.3 0.4
```

## Privacy composition

```bash
python -m fedgsentry.privacy
```

Computes the real cumulative (ε, δ) for the Gaussian mechanism in Eq. 5,
via Renyi-DP composition (Mironov 2017), using Table III's own
hyperparameters (C=1.0, σ=0.5, T=100) as a worked example. **This is a
genuine finding, not a formality**: at those values, composed epsilon is
roughly 320 for δ=1e-5 — a very weak guarantee despite the per-round
noise looking reasonable in isolation. Before claiming a meaningful
formal privacy guarantee in the paper, either increase the noise scale,
reduce the number of rounds, add per-round client subsampling (with its
amplification bound — not implemented here), or explicitly scope the
privacy claim as "per-round" rather than "end-to-end."

## Known gaps / honest limitations of this code

- **GAGA baseline is intentionally not implemented.** It refers to a
  specific published architecture; approximating it and labeling the
  approximation "GAGA" in a comparison table would misrepresent that
  baseline. Implement it from its source paper if you need a faithful
  comparison point.
- **Table III's "100 local epochs × 100 rounds"** is 10,000 full-batch
  local steps per client, each recomputing an exact eigendecomposition
  of the local Laplacian in the spectral filter. That's expensive at
  real dataset scale — `federated.py` defaults to fewer local epochs
  and documents the tradeoff; override `--local_epochs 100` if you want
  to match Table III literally, but budget real compute time for it.
- **Exact eigendecomposition** (used for the spectral filter on graphs
  ≤4000 nodes) is O(n³) — client graphs larger than that automatically
  fall back to a Chebyshev polynomial approximation of the same learned
  filter (`models.py::HeterophilyAwareSpectralFilter._chebyshev_approx`).
  This is a legitimate standard technique, but it is an approximation,
  and you should say so in the methodology section if you use it at
  scale.
- **Attention is edge-restricted (sparse), not the naive dense n×n
  formulation** — a literal dense implementation of Eq. 3 is O(n²)
  memory and runs out of memory well before Elliptic/IEEE-CIS scale,
  which would actually contradict the O(|E|·d·H) complexity the
  manuscript claims in Section III-I. This implementation only scores
  existing edges plus self-loops, which is what makes that complexity
  claim true.
- **CPU-only by default.** Pass `--device cuda` if you have a GPU;
  nothing in the code assumes CPU, but nothing was tuned for GPU either.

## Files

- `fedgsentry/data.py` — dataset loaders + non-IID federated partitioning (Dirichlet)
- `fedgsentry/models.py` — spectral filter, relational transformer, full model
- `fedgsentry/optim.py` — SGLD optimizer (Eq. 4)
- `fedgsentry/privacy.py` — DP clip+noise (Eq. 5), Latent Anchor Consistency (Eq. 6), RDP accountant
- `fedgsentry/federated.py` — Algorithm 1 training loop
- `fedgsentry/baselines.py` — FedAvg-GCN, GraphSAGE-FL, GAT-FL, SGLD-FL
- `train.py` — experiment driver (metrics, ablation, adversarial sweep, comm cost)
