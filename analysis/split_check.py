"""Does a benchmark's test file look like a random per-user holdout?

Written 14 Oct 2026 while diagnosing the VAE-CF baseline (MIGRATION.md,
"VAE-CF on Gowalla: diagnosing the gap"). Our validation set is a uniformly
random 10% of each user's training items (src/data/splits.py), the procedure
NGCF describes for these benchmarks. Validation is only a stand-in for test if
the test file was drawn the same way. This script checks that from the files
alone (they carry no timestamps).

Statistic: item dispersion index. Let m_i be item i's interactions in
train + test and b_i those in test, p = (test size) / (train + test size).
Under a uniformly random per-user split, b_i ~ approximately Binomial(m_i, p),
so

    D = mean_i (b_i - p m_i)^2 / (m_i p (1 - p))

is close to 1. The reference value is not assumed: the script redraws each
user's test items uniformly from that user's own train + test items (same
per-user test sizes) and reports D for those simulated splits too. D well
above the simulated value means some items sit in the test file far more (or
less) often than chance -- the split is not exchangeable with a random
holdout, whatever procedure produced it.

Also reported: training popularity of test items vs a random 10% of training
items (what our validation set is).

    python -m analysis.split_check                          # data/{gowalla,yelp2018,amazon-book}
    python -m analysis.split_check --root data gowalla
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.dataset import _read_lightgcn_file


def dispersion_index(b: np.ndarray, m: np.ndarray, p: float) -> float:
    mask = m > 0
    return float(((b[mask] - p * m[mask]) ** 2 / (m[mask] * p * (1 - p))).mean())


def _by_user(users: np.ndarray, items: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(users, kind="stable")
    u, i = users[order], items[order]
    uniq, start = np.unique(u, return_index=True)
    end = np.append(start[1:], len(u))
    return {int(x): i[s:e] for x, s, e in zip(uniq, start, end)}


def check(tr_u, tr_i, te_u, te_i, n_sims: int = 3, seed: int = 0) -> dict:
    n_items = int(max(tr_i.max(), te_i.max())) + 1
    a = np.bincount(tr_i, minlength=n_items).astype(float)
    b = np.bincount(te_i, minlength=n_items).astype(float)
    m = a + b
    p = b.sum() / m.sum()
    real = dispersion_index(b, m, p)

    tr, te = _by_user(tr_u, tr_i), _by_user(te_u, te_i)
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_sims):
        bs = np.zeros(n_items)
        for u, t in te.items():
            pool = np.concatenate([tr.get(u, np.empty(0, np.int64)), t])
            np.add.at(bs, rng.choice(pool, size=len(t), replace=False), 1)
        sims.append(dispersion_index(bs, m, p))

    rnd = np.concatenate([rng.choice(v, size=max(1, int(round(0.1 * len(v)))), replace=False)
                          for v in tr.values() if len(v) >= 2])
    return {"test_frac": float(p), "dispersion": real, "dispersion_random_split": sims,
            "train_pop_test_items_mean": float(a[te_i].mean()),
            "train_pop_test_items_median": float(np.median(a[te_i])),
            "train_pop_random10_mean": float(a[rnd].mean()),
            "train_pop_random10_median": float(np.median(a[rnd]))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["gowalla", "yelp2018", "amazon-book"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--sims", type=int, default=3)
    args = ap.parse_args(argv)
    print(f"{'dataset':12s} {'test frac':>9s} {'D (file)':>9s} {'D (random split, sims)':>24s} "
          f"{'train pop of test items':>24s} {'of random 10% of train':>24s}")
    for name in args.datasets:
        d = Path(args.root) / name
        tr_u, tr_i = _read_lightgcn_file(d / "train.txt")
        te_u, te_i = _read_lightgcn_file(d / "test.txt")
        r = check(tr_u, tr_i, te_u, te_i, n_sims=args.sims)
        sims = ", ".join(f"{x:.2f}" for x in r["dispersion_random_split"])
        print(f"{name:12s} {r['test_frac']:9.3f} {r['dispersion']:9.2f} {sims:>24s} "
              f"{r['train_pop_test_items_mean']:13.1f} (med {r['train_pop_test_items_median']:4.0f}) "
              f"{r['train_pop_random10_mean']:13.1f} (med {r['train_pop_random10_median']:4.0f})")


if __name__ == "__main__":
    main()
