"""Collect ``results.json`` files and run the paired-by-seed comparison.

Replaces ``inventory_drive`` / ``wilcoxon_test_block_c`` / ``build_cap5_table``
(Cell 16) and the whole Section 3 recovery subsystem. There is nothing to
recover: every run writes its own ``results.json`` and ``history.json``.

    python -m analysis.summarize runs                      # table + Wilcoxon emb_mr vs mr_off
    python -m analysis.summarize runs --a emb_mr --b baseline --metric ndcg@20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

GEOM_KEYS = ["er_table", "er_prop", "er_table_at_ref", "er_prop_at_ref",
             "delta_er_table", "delta_er_prop", "np_ref@10", "np_ref@20",
             "np_vs_ref@10", "np_vs_ref@20"]
TEST_KEYS = ["recall@20", "ndcg@20", "precision@10", "recall@10", "ndcg@10",
             "ndcg_short@10", "ndcg_long@10", "gini@20", "tail_share_user@20",
             "tail_catalog_coverage@20"]


def collect_results(runs_root: str | Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(runs_root).rglob("results.json")):
        r = json.load(open(p))
        row = {"dataset": r["dataset"], "arm": r["arm"], "arm_tag": r.get("arm_tag", r["arm"]), "seed": r["seed"],
               "best_epoch": r["best_epoch"], "selected_on": r["selected_on"],
               "epochs": r["epochs_budget"], "warmup": r["warmup_epochs"],
               "lambda": r["lambda_manifold"], "path": str(p.parent)}
        row.update({f"test_{k}": r["test"].get(k) for k in TEST_KEYS})
        row.update({f"val_{k}": r["val"].get(k) for k in ["recall@20", "ndcg@20"]})
        row.update({k: r["geometry"].get(k) for k in GEOM_KEYS})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_wilcoxon(df: pd.DataFrame, arm_a: str, arm_b: str, metric: str = "test_recall@20",
                    alternative: str = "greater", alpha: float = 0.05) -> pd.DataFrame:
    """H1: metric(arm_a) > metric(arm_b), paired by seed, one test per dataset.

    With n=5 seeds the smallest attainable one-sided p is 1/32 = 0.031; with
    n=10 it is 1/1024. Report n alongside p. Holm-Bonferroni across datasets
    is applied in the ``p_holm`` column.
    """
    out = []
    for ds, g in df.groupby("dataset"):
        # match by arm_tag first (exact hyperparameters), fall back to arm name
        sel = lambda x: g[g.arm_tag == x] if (g.arm_tag == x).any() else g[g.arm == x]
        a = sel(arm_a).set_index("seed")[metric]
        b = sel(arm_b).set_index("seed")[metric]
        seeds = sorted(set(a.index) & set(b.index))
        if len(seeds) < 3:
            out.append({"dataset": ds, "n": len(seeds), "note": "need >= 3 paired seeds"})
            continue
        x, y = a.loc[seeds].values, b.loc[seeds].values
        try:
            stat, p = wilcoxon(x, y, alternative=alternative)
        except ValueError as e:      # all differences zero
            stat, p = np.nan, 1.0
        out.append({"dataset": ds, "n": len(seeds),
                    f"mean_{arm_a}": x.mean(), f"mean_{arm_b}": y.mean(),
                    "mean_delta_pct": float(((x - y) / y * 100).mean()),
                    "wins": int((x > y).sum()), "stat": stat, "p": p})
    res = pd.DataFrame(out)
    if "p" in res:
        ps = res["p"].fillna(1.0).values
        order = np.argsort(ps)
        holm = np.empty_like(ps)
        m = len(ps)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * ps[idx])
            holm[idx] = min(1.0, running)
        res["p_holm"] = holm
        res["significant_holm"] = res["p_holm"] < alpha
    return res


def summary_table(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    metrics = metrics or ["test_recall@20", "test_ndcg@20", "er_table", "er_prop", "np_ref@20",
                          "test_gini@20", "test_tail_catalog_coverage@20"]
    grp = df.groupby(["dataset", "arm_tag"])
    g = grp[metrics]
    out = pd.concat({"mean": g.mean(), "std": g.std()}, axis=1)
    out[("n", "seeds")] = grp.size()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--a", default="emb_mr")
    ap.add_argument("--b", default="mr_off")
    ap.add_argument("--metric", default="recall@20")
    a = ap.parse_args(argv)
    df = collect_results(a.runs)
    if df.empty:
        print(f"no results.json under {a.runs}")
        return
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(summary_table(df).round(4))
    print()
    print(paired_wilcoxon(df, a.a, a.b, f"test_{a.metric}").round(5))


if __name__ == "__main__":
    main()
