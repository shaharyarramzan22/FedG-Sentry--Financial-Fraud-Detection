"""
Dataset loaders and federated (non-IID) partitioning.

Supported real datasets (you must download these yourself -- both
require accepting terms on Kaggle, so this code cannot fetch them
for you):

  Elliptic Bitcoin dataset
    https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
    Expected files in --data_dir:
        elliptic_txs_features.csv
        elliptic_txs_classes.csv
        elliptic_txs_edgelist.csv

  IEEE-CIS Fraud Detection dataset
    https://www.kaggle.com/competitions/ieee-fraud-detection
    Expected files in --data_dir:
        train_transaction.csv
        train_identity.csv

If those files are not present, `load_dataset` raises FileNotFoundError
with instructions -- it will NOT silently fall back to fake data unless
you explicitly pass dataset="synthetic".

A synthetic "FedG-Sim" style generator is also provided so you can
exercise the full training pipeline (including the poisoning-client
robustness test) without any external data.
"""
from __future__ import annotations

import os
import dataclasses
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


@dataclasses.dataclass
class ClientGraph:
    """A single federated client's local transaction graph."""
    client_id: int
    x: torch.Tensor              # [n_nodes, n_features]
    adj: torch.sparse.Tensor     # [n_nodes, n_nodes] symmetric, self-loops added
    y: torch.Tensor              # [n_nodes] in {0, 1, -1}; -1 = unlabeled
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    is_adversarial: bool = False


def _normalize_adj(adj: sp.csr_matrix) -> torch.sparse.Tensor:
    """Symmetric normalization D^-1/2 (A+I) D^-1/2, returned as a torch sparse tensor."""
    adj = adj + sp.eye(adj.shape[0])
    deg = np.asarray(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, where=deg > 0, out=deg_inv_sqrt)
    d_mat = sp.diags(deg_inv_sqrt)
    norm_adj = (d_mat @ adj @ d_mat).tocoo()

    indices = torch.tensor(np.vstack([norm_adj.row, norm_adj.col]), dtype=torch.long)
    values = torch.tensor(norm_adj.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, norm_adj.shape).coalesce()


def _stratified_masks(y: np.ndarray, seed: int, splits=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    n = len(y)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    for cls in np.unique(y):
        if cls < 0:
            continue  # unlabeled
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_train = int(splits[0] * len(idx))
        n_val = int(splits[1] * len(idx))
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True
    return (torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask))


def _partition_non_iid(n_nodes: int, n_clients: int, seed: int, concentration: float = 0.3) -> List[np.ndarray]:
    """Dirichlet-based non-IID partition of node indices across clients.

    Lower `concentration` -> more skewed / non-IID partition across clients,
    which is what real cross-institution data looks like.
    """
    rng = np.random.default_rng(seed)
    proportions = rng.dirichlet(alpha=[concentration] * n_clients)
    proportions = (proportions * n_nodes).astype(int)
    proportions[-1] += n_nodes - proportions.sum()  # fix rounding
    order = rng.permutation(n_nodes)
    parts, start = [], 0
    for count in proportions:
        parts.append(order[start:start + count])
        start += count
    return parts


def _build_client_graphs(
    x: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    n_clients: int,
    seed: int,
    adversarial_fraction: float,
) -> List[ClientGraph]:
    n_nodes = x.shape[0]
    parts = _partition_non_iid(n_nodes, n_clients, seed)
    rng = np.random.default_rng(seed)
    n_adv = int(round(adversarial_fraction * n_clients))
    adversarial_ids = set(rng.choice(n_clients, size=n_adv, replace=False).tolist())

    # Build a fast lookup from global node id -> row in edge_index
    src, dst = edge_index[0], edge_index[1]

    clients = []
    for cid, node_idx in enumerate(parts):
        if len(node_idx) < 10:
            continue
        node_set = set(node_idx.tolist())
        local_map = {g: i for i, g in enumerate(node_idx)}

        mask = np.array([(s in node_set) and (d in node_set) for s, d in zip(src, dst)])
        local_src = np.array([local_map[s] for s in src[mask]], dtype=int)
        local_dst = np.array([local_map[d] for d in dst[mask]], dtype=int)

        n_local = len(node_idx)
        adj = sp.coo_matrix(
            (np.ones(len(local_src)), (local_src, local_dst)),
            shape=(n_local, n_local),
        ).tocsr()
        adj = adj.maximum(adj.T)  # symmetrize

        x_local = torch.tensor(x[node_idx], dtype=torch.float32)
        y_local = torch.tensor(y[node_idx], dtype=torch.long)
        train_mask, val_mask, test_mask = _stratified_masks(y[node_idx], seed=seed + cid)

        clients.append(ClientGraph(
            client_id=cid,
            x=x_local,
            adj=_normalize_adj(adj),
            y=y_local,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            is_adversarial=cid in adversarial_ids,
        ))
    return clients


# --------------------------------------------------------------------------
# Real dataset loaders
# --------------------------------------------------------------------------

def _load_elliptic(data_dir: str):
    feat_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    class_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    edge_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
    for p in (feat_path, class_path, edge_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {p}. Download the Elliptic Bitcoin dataset from "
                "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set "
                "and place the three CSVs in --data_dir."
            )

    features = pd.read_csv(feat_path, header=None)
    features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(features.shape[1] - 2)]
    classes = pd.read_csv(class_path)  # columns: txId, class in {"1"=illicit,"2"=licit,"unknown"}
    edges = pd.read_csv(edge_path)     # columns: txId1, txId2

    merged = features.merge(classes, on="txId", how="left")
    label_map = {"1": 1, "2": 0, "unknown": -1}
    merged["label"] = merged["class"].map(label_map).fillna(-1).astype(int)

    tx_ids = merged["txId"].to_numpy()
    id_to_idx = {tx: i for i, tx in enumerate(tx_ids)}

    x = merged[[c for c in merged.columns if c.startswith("f")]].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy()

    valid_edges = edges[edges["txId1"].isin(id_to_idx) & edges["txId2"].isin(id_to_idx)]
    src = valid_edges["txId1"].map(id_to_idx).to_numpy()
    dst = valid_edges["txId2"].map(id_to_idx).to_numpy()
    edge_index = np.vstack([src, dst])

    return x, y, edge_index


def _load_ieee_cis(data_dir: str, max_rows: Optional[int] = 200_000):
    tx_path = os.path.join(data_dir, "train_transaction.csv")
    if not os.path.exists(tx_path):
        raise FileNotFoundError(
            f"Missing {tx_path}. Download IEEE-CIS Fraud Detection from "
            "https://www.kaggle.com/competitions/ieee-fraud-detection "
            "and place train_transaction.csv (and optionally train_identity.csv) in --data_dir."
        )
    df = pd.read_csv(tx_path, nrows=max_rows)

    y = df["isFraud"].to_numpy()
    drop_cols = {"isFraud", "TransactionID"}
    feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    feat_df = feat_df.select_dtypes(include=[np.number]).fillna(-1.0)
    x = feat_df.to_numpy(dtype=np.float32)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)

    # IEEE-CIS has no native graph; construct a k-NN similarity graph over
    # card/addr/device fields as a relational proxy, consistent with how
    # graph-based fraud detectors typically operationalize this dataset.
    from sklearn.neighbors import NearestNeighbors
    k = 8
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    _, indices = nn.kneighbors(x)
    src = np.repeat(np.arange(x.shape[0]), k)
    dst = indices[:, 1:].reshape(-1)
    edge_index = np.vstack([src, dst])

    return x, y, edge_index


def _load_synthetic_fedgsim(
    n_nodes: int = 4_000,
    n_features: int = 64,
    fraud_rate: float = 0.05,
    heterophily_ratio: float = 0.6,
    seed: int = 0,
):
    """Controlled synthetic federated simulation ("FedG-Sim").

    Generates a graph where a `heterophily_ratio` fraction of fraud-node
    edges deliberately connect to structurally dissimilar (non-fraud)
    neighbors, mimicking evasive fraud rings, and the remainder connect
    homophilously. This lets you exercise and sanity-check the
    heterophily-aware components even without a licensed real dataset.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n_nodes) < fraud_rate).astype(int)

    # Class-conditional feature means so the task is learnable but not trivial.
    mean_fraud = rng.normal(0, 1, size=n_features) * 0.8
    mean_legit = rng.normal(0, 1, size=n_features) * 0.8
    x = np.where(
        y[:, None] == 1,
        rng.normal(mean_fraud, 1.0, size=(n_nodes, n_features)),
        rng.normal(mean_legit, 1.0, size=(n_nodes, n_features)),
    ).astype(np.float32)

    avg_degree = 6
    n_edges = n_nodes * avg_degree // 2
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]

    src_list, dst_list = [], []
    for _ in range(n_edges):
        if rng.random() < fraud_rate * 3 and len(fraud_idx) > 0:
            u = rng.choice(fraud_idx)
            if rng.random() < heterophily_ratio and len(legit_idx) > 0:
                v = rng.choice(legit_idx)       # heterophilous: fraud -> legit
            else:
                v = rng.choice(fraud_idx)       # homophilous: fraud -> fraud
        else:
            u = rng.choice(n_nodes)
            v = rng.choice(n_nodes)
        src_list.append(u)
        dst_list.append(v)

    edge_index = np.vstack([np.array(src_list), np.array(dst_list)])
    return x, y, edge_index


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def load_federated_dataset(
    dataset: str,
    data_dir: str = "./data",
    n_clients: int = 8,
    adversarial_fraction: float = 0.0,
    seed: int = 0,
) -> List[ClientGraph]:
    """Load a dataset and partition it into `n_clients` non-IID federated clients.

    dataset: one of {"elliptic", "ieee-cis", "synthetic"}
    """
    if dataset == "elliptic":
        x, y, edge_index = _load_elliptic(data_dir)
    elif dataset == "ieee-cis":
        x, y, edge_index = _load_ieee_cis(data_dir)
    elif dataset == "synthetic":
        x, y, edge_index = _load_synthetic_fedgsim(seed=seed)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use elliptic | ieee-cis | synthetic.")

    return _build_client_graphs(x, y, edge_index, n_clients, seed, adversarial_fraction)
