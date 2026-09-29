"""Differentiable-kernel experiment (post-gate plan, step 2), Gowalla.

Pre-registered before running:
  PRIMARY    emb_mr_dw vs emb_mr_fresh on er_table -- isolates the gradient
             through the kernel. One test, reported uncorrected.
  SECONDARY  everything else, Holm-corrected together:
             emb_mr_dw    vs emb_mr_fresh  (other metrics)
             emb_mr_fresh vs emb_mr        (current vs build-time weights)
             emb_mr_dw    vs mr_off        (does MR still lower ER?)
Two-sided exact Wilcoxon, paired by seed.

    python -m analysis.compare_kernel runs
"""
import json
import pathlib
import sys

import numpy as np
from scipy.stats import wilcoxon

DS = "gowalla"
TAG = {"dw": "emb_mr_dw_w100_lam1e-05_k20_r50_K3_d256",
       "fresh": "emb_mr_fresh_w100_lam1e-05_k20_r50_K3_d256",
       "emb_mr": "emb_mr_w100_lam1e-05_k20_r50_K3_d256",
       "mr_off": "mr_off_w100_K3_d256"}
METRICS = [("test", "recall@20"), ("geometry", "er_table"), ("geometry", "np_ref@20"),
           ("test", "gini@20"), ("test", "tail_catalog_coverage@20")]
PAIRS = [("dw", "fresh"), ("fresh", "emb_mr"), ("dw", "mr_off")]


def load(root, tag):
    out = {}
    for p in pathlib.Path(root, DS, tag).glob("seed*/results.json"):
        r = json.load(open(p)); out[r["seed"]] = r
    return out


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(ps) - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


def crossover(root, tag):
    fr = []
    for p in pathlib.Path(root, DS, tag).glob("seed*/history.json"):
        fr += [h["frac_beyond_crossover"] for h in json.load(open(p))
               if h.get("event") == "laplacian_built" and "frac_beyond_crossover" in h]
    return fr


def main(root="runs"):
    data = {k: load(root, t) for k, t in TAG.items()}
    for k, v in data.items():
        print(f"  {k:7s} {TAG[k]:44s} seeds: {len(v)}")
    rows = []
    for a, b in PAIRS:
        seeds = sorted(set(data[a]) & set(data[b]))
        if len(seeds) < 3:
            print(f"  skipping {a} vs {b}: only {len(seeds)} paired seeds"); continue
        for sec, m in METRICS:
            d = np.array([data[a][s][sec][m] - data[b][s][sec][m] for s in seeds])
            rows.append(dict(cmp=f"{a} - {b}", m=m, n=len(seeds), mean=d.mean(),
                             lo=int((d < 0).sum()), hi=int((d > 0).sum()),
                             p=wilcoxon(d, alternative="two-sided").pvalue,
                             primary=(a, b, m) == ("dw", "fresh", "er_table")))
    sec_idx = [i for i, r in enumerate(rows) if not r["primary"]]
    for i, h in zip(sec_idx, holm([rows[i]["p"] for i in sec_idx])):
        rows[i]["holm"] = h
    print("\nPRIMARY (pre-registered, uncorrected)")
    for r in rows:
        if r["primary"]:
            print(f"  {r['cmp']:15s} {r['m']:26s} n={r['n']:2d} mean {r['mean']:+.5f} "
                  f"{r['lo']}/{r['hi']} lower/higher  p={r['p']:.4f}{'  *' if r['p'] < 0.05 else ''}")
    print(f"\nSECONDARY (Holm across all {len(sec_idx)} tests)")
    last = None
    for r in rows:
        if r["primary"]: continue
        if r["cmp"] != last: print(f"  {r['cmp']}"); last = r["cmp"]
        print(f"    {r['m']:26s} n={r['n']:2d} mean {r['mean']:+.5f} {r['lo']:2d}/{r['hi']:<2d} "
              f"p={r['p']:.4f} Holm={r['holm']:.4f}{'  *' if r['holm'] < 0.05 else ''}")
    fr = crossover(root, TAG["dw"])
    if fr:
        print(f"\nemb_mr_dw neighbour pairs beyond the repulsion crossover, over {len(fr)} rebuilds: "
              f"median {100*np.median(fr):.2f}%, max {100*max(fr):.2f}%")


if __name__ == "__main__":
    main(*sys.argv[1:])
