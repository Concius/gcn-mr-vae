"""Baseline table: LightGCN, VAE-CF (tuned) and MF-BPR on the three datasets.

Declared 19 Oct 2026, before the Amazon VAE-CF seeds were read (MIGRATION.md,
"Baseline table"). Descriptive comparison of the committed baselines under the
corrected protocol; no hypothesis of the dissertation rides on it.

Arms (folders under runs/<dataset>/):
    LightGCN  mr_off_w100_K3_d256   (the MR-off control, K = 3)
    VAE-CF    vae_cf_tuned          (Mult-VAE^PR, dropout 0.5, constant beta tuned per dataset)
    MF-BPR    mr_off_w100_K0_d256   (the same trainer with K = 0)

Metrics (test split, at the validation-selected checkpoint): recall@20,
ndcg@20, gini@20 (lower = more even exposure), tail_catalog_coverage@20.

Tests: VAE-CF - LightGCN and VAE-CF - MF-BPR, paired by seed (the seeds both
arms have), two-sided exact Wilcoxon. Holm within each metric over the six
tests (2 comparisons x 3 datasets). With 10 seeds the smallest p is 2/1024, so
a 10/10 result survives Holm in a family of six (6 x 0.00195 = 0.012).

Nothing is tested unless every run is healthy (all metrics and losses finite).

    python -m analysis.baselines                    # runs/ -> runs/_baselines/
    python -m analysis.baselines --runs runs --out runs/_baselines
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.lambda_sweep import health_gate, holm, load, wtest

DATASETS = ("gowalla", "yelp2018", "amazon-book")
ARMS = {"LightGCN": "mr_off_w100_K3_d256", "VAE-CF": "vae_cf_tuned", "MF-BPR": "mr_off_w100_K0_d256"}
METRICS = ("recall@20", "ndcg@20", "gini@20", "tail_catalog_coverage@20")
COMPARISONS = (("VAE-CF", "LightGCN"), ("VAE-CF", "MF-BPR"))


def gather(runs_root, datasets=DATASETS, arms=ARMS):
    return {(ds, lab): load(runs_root, ds, tag) for ds in datasets for lab, tag in arms.items()}


def describe(data, datasets=DATASETS, arms=ARMS, metrics=METRICS) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        for lab in arms:
            rs = data[(ds, lab)]
            row = {"dataset": ds, "arm": lab, "n": len(rs)}
            for m in metrics:
                x = np.array([r["test"][m] for r in rs.values()], float)
                row[f"{m}_mean"] = x.mean() if len(x) else np.nan
                row[f"{m}_sd"] = x.std(ddof=1) if len(x) > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def compare(data, datasets=DATASETS, metrics=METRICS, comparisons=COMPARISONS) -> pd.DataFrame:
    rows = []
    for m in metrics:
        block = []
        for a, b in comparisons:
            for ds in datasets:
                ra, rb = data[(ds, a)], data[(ds, b)]
                seeds = sorted(set(ra) & set(rb))
                if not seeds:
                    continue
                d = np.array([ra[s]["test"][m] - rb[s]["test"][m] for s in seeds], float)
                mb = np.mean([rb[s]["test"][m] for s in seeds])
                block.append({"metric": m, "comparison": f"{a} - {b}", "dataset": ds, "n": len(seeds),
                              "mean_diff": d.mean(), "rel_pct": 100 * d.mean() / mb if mb else np.nan,
                              "a_higher": int((d > 0).sum()), "p": wtest(d)})
        if block:
            adj = holm([r["p"] for r in block])
            for r, ph in zip(block, adj):
                r["p_holm"] = float(ph)
        rows.extend(block)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/_baselines")
    args = ap.parse_args(argv)
    data = gather(args.runs)
    missing = [f"{ds}/{ARMS[lab]}" for (ds, lab), rs in data.items() if not rs]
    if missing:
        print("missing (no results.json): " + ", ".join(missing))
    counts = {f"{ds}/{lab}": len(rs) for (ds, lab), rs in data.items()}
    print("seeds per arm: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if not health_gate({f"{ds}/{lab}": rs for (ds, lab), rs in data.items()}):
        return 1
    desc, comp = describe(data), compare(data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    desc.to_csv(out / "baselines_describe.csv", index=False)
    comp.to_csv(out / "baselines_tests.csv", index=False)

    print("\nTest metrics at the validation-selected checkpoint (mean +- sd over seeds)")
    for ds in DATASETS:
        print(f"\n  {ds}")
        print(f"    {'arm':9s} {'n':>3s}  {'R@20':>17s}  {'N@20':>17s}  {'Gini@20':>7s}  {'tail cov':>8s}")
        for _, r in desc[desc.dataset == ds].iterrows():
            print(f"    {r.arm:9s} {r.n:3d}  {r['recall@20_mean']:.4f} +- {r['recall@20_sd']:.4f}  "
                  f"{r['ndcg@20_mean']:.4f} +- {r['ndcg@20_sd']:.4f}  {r['gini@20_mean']:7.4f}  "
                  f"{r['tail_catalog_coverage@20_mean']:8.4f}")
    print("\nPaired two-sided exact Wilcoxon by seed; Holm within each metric over 6 tests")
    for m in METRICS:
        print(f"\n  {m}")
        for _, r in comp[comp.metric == m].iterrows():
            print(f"    {r.comparison:18s} {r.dataset:12s} diff {r.mean_diff:+.5f} ({r.rel_pct:+6.1f}%)  "
                  f"VAE-CF higher {r.a_higher}/{r.n}  p {r.p:.4f}  Holm {r.p_holm:.4f}")
    print(f"\nwritten: {out}/baselines_describe.csv, {out}/baselines_tests.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
