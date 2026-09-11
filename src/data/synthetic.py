"""Tiny synthetic dataset in LightGCN format, for tests and for building the
VAE against something that runs in seconds (October track).

    python -m src.data.synthetic --out data/synthetic --users 300 --items 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_synthetic(out: str | Path, n_users: int = 300, n_items: int = 200,
                   n_clusters: int = 5, seed: int = 0, test_frac: float = 0.2) -> Path:
    """Clustered preferences so that the graph has real structure (k-NN and
    Jaccard neighbourhoods are meaningful), unlike uniform random noise."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    u_c = rng.integers(0, n_clusters, n_users)
    i_c = rng.integers(0, n_clusters, n_items)
    train, test = {}, {}
    for u in range(n_users):
        same = np.flatnonzero(i_c == u_c[u])
        other = np.flatnonzero(i_c != u_c[u])
        n_pos = rng.integers(8, 25)
        n_same = int(0.8 * n_pos)
        items = np.concatenate([rng.choice(same, min(n_same, len(same)), replace=False),
                                rng.choice(other, n_pos - min(n_same, len(same)), replace=False)])
        items = np.unique(items)
        rng.shuffle(items)
        n_te = max(1, int(test_frac * len(items)))
        test[u] = sorted(items[:n_te].tolist())
        train[u] = sorted(items[n_te:].tolist())
    with open(out / "train.txt", "w") as f:
        for u in range(n_users):
            f.write(f"{u} " + " ".join(map(str, train[u])) + "\n")
    with open(out / "test.txt", "w") as f:
        for u in range(n_users):
            f.write(f"{u} " + " ".join(map(str, test[u])) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_synthetic(a.out, a.users, a.items, seed=a.seed))
