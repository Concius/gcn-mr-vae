"""Pair-split of the kernel gradient: pre-registered analysis (written
4 Oct 2026, before any emb_mr_dw_weaken / emb_mr_dw_repel run existed).

Design: a 2x2 factorial on Gowalla, 10 seeds, paired by seed. Factor A is
"kernel gradient on pairs that stay net-attractive" (weaken pairs), factor B
is "kernel gradient on net-repulsive pairs" (repel pairs), with net-repulsive
defined by the exact rule d2 > 2 sigma^2 |z_i||z_j| re-evaluated every batch:

                      B off            B on
    A off       emb_mr_fresh     emb_mr_dw_repel
    A on    emb_mr_dw_weaken          emb_mr_dw

emb_mr_fresh and emb_mr_dw are the existing runs: their code paths were
verified bit-identical (value and gradient, float32 and float64) after the
split modes were added.

PRIMARY (one test, uncorrected): weaken - repel on er_table, two-sided exact
Wilcoxon. Positive = the pairs whose attraction is merely weakened carry more
of emb_mr_dw's effective-rank gain than the net-repulsive pairs do.

SECONDARY (Holm-corrected together, 11 tests):
    weaken - fresh, repel - fresh, interaction    x  er_table, np_ref@20, recall@20
    weaken - repel                                x  np_ref@20, recall@20
with interaction = dw - weaken - repel + fresh per seed (0 if the two
classes' effects simply add).

DESCRIPTIVE (no tests): per-seed shares of dw's ER gain, (weaken - fresh) /
(dw - fresh) and (repel - fresh) / (dw - fresh); and, from history.json, the
net-repulsive share (exact and unit-sphere rule) and the kernel-budget share of
net-repulsive pairs in the early / mid / late MR windows of each split arm.

Caveat fixed in advance: the weaken class is the larger one, so a positive
primary result can partly reflect how many pairs each class holds. The
kernel-budget share is reported so the two can be read together; no
size-normalised test is run.

    python -m analysis.compare_split runs              # gowalla
    python -m analysis.compare_split runs <dataset>
"""
import json
import pathlib
import sys

import numpy as np
from scipy.stats import wilcoxon

SUFFIX = "_w100_lam1e-05_k20_r50_K3_d256"
ARMS = {"fresh": "emb_mr_fresh", "weaken": "emb_mr_dw_weaken",
        "repel": "emb_mr_dw_repel", "dw": "emb_mr_dw"}
METRICS = {"er_table": ("geometry", "er_table"), "np_ref@20": ("geometry", "np_ref@20"),
           "recall@20": ("test", "recall@20")}
PRIMARY = ("weaken - repel", "er_table")
SECONDARY = ([(c, m) for c in ("weaken - fresh", "repel - fresh", "interaction")
              for m in ("er_table", "np_ref@20", "recall@20")]
             + [("weaken - repel", m) for m in ("np_ref@20", "recall@20")])
WINDOW_FRACTIONS = (0.0, 7 / 18, 13 / 18, 1.0)       # W=100, E=1000 -> 100, 450, 750, 1000
WINDOW_NAMES = ("early", "mid", "late")


def load(root, ds, tag):
    out = {}
    for p in pathlib.Path(root, ds, tag).glob("seed*/results.json"):
        r = json.load(open(p))
        out[r["seed"]] = (r, p.parent)
    return out


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(ps) - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


def contrast(v, name):
    """Per-seed contrast from {arm: value}."""
    if name == "interaction":
        return v["dw"] - v["weaken"] - v["repel"] + v["fresh"]
    a, b = name.split(" - ")
    return v[a] - v[b]


def test(d):
    if len(d) < 3 or np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def window_diagnostics(rdir, W, E):
    h = [r for r in json.load(open(rdir / "history.json"))
         if r.get("event") == "eval" and "kd_frac_repel_exact" in r]
    bounds = [W + f * (E - W) for f in WINDOW_FRACTIONS]
    out = {}
    for name, lo, hi in zip(WINDOW_NAMES, bounds[:-1], bounds[1:]):
        sel = [r for r in h if lo < r["epoch"] <= hi]
        out[name] = {k: float(np.nanmean([r[f"kd_{k}"] for r in sel])) if sel else float("nan")
                     for k in ("frac_repel_exact", "frac_repel_sphere", "kernel_share_repel")}
    return out


def main(root="runs", ds="gowalla", suffix=SUFFIX):
    print(f"dataset: {ds}")
    data = {a: load(root, ds, t + suffix) for a, t in ARMS.items()}
    for a, v in data.items():
        print(f"  {a:7s} {ARMS[a] + suffix:52s} seeds: {len(v)}")
    seeds = sorted(set.intersection(*(set(v) for v in data.values())))
    if len(seeds) < 3:
        print(f"\nonly {len(seeds)} seeds present in all four arms -- nothing to test")
        return
    print(f"  paired seeds in all four arms: {len(seeds)}")

    missing = sorted({f"{a}/seed{s}: {sec}.{key}" for m, (sec, key) in METRICS.items()
                      for a in ARMS for s in seeds if key not in data[a][s][0][sec]})
    if missing:
        raise KeyError("pre-registered metrics missing from results.json:\n  " + "\n  ".join(missing[:10]))
    vals = {}
    for m, (sec, key) in METRICS.items():
        vals[m] = {a: np.array([data[a][s][0][sec][key] for s in seeds]) for a in ARMS}

    def row(c, m):
        d = contrast(vals[m], c)
        return dict(c=c, m=m, n=len(d), mean=float(d.mean()), lo=int((d < 0).sum()),
                    hi=int((d > 0).sum()), p=test(d))

    prim = row(*PRIMARY)
    print("\nPRIMARY (pre-registered, uncorrected)")
    print(f"  {prim['c']:15s} {prim['m']:10s} n={prim['n']:2d} mean {prim['mean']:+.5f} "
          f"{prim['lo']}/{prim['hi']} lower/higher  p={prim['p']:.4f}{'  *' if prim['p'] < 0.05 else ''}")

    sec = [row(c, m) for c, m in SECONDARY]
    for r, h in zip(sec, holm([r["p"] for r in sec])):
        r["holm"] = h
    print(f"\nSECONDARY (Holm across all {len(sec)} tests)")
    last = None
    for r in sec:
        if r["c"] != last:
            print(f"  {r['c']}"); last = r["c"]
        print(f"    {r['m']:10s} n={r['n']:2d} mean {r['mean']:+.5f} {r['lo']:2d}/{r['hi']:<2d} "
              f"p={r['p']:.4f} Holm={r['holm']:.4f}{'  *' if r['holm'] < 0.05 else ''}")

    er = vals["er_table"]
    total = er["dw"] - er["fresh"]
    print("\nDESCRIPTIVE -- share of emb_mr_dw's ER gain (dw - fresh), per seed")
    for a in ("weaken", "repel"):
        with np.errstate(divide="ignore", invalid="ignore"):
            share = (er[a] - er["fresh"]) / total
        print(f"  {a:7s} median {np.nanmedian(share):+.2f}  range [{np.nanmin(share):+.2f}, {np.nanmax(share):+.2f}]")
    print(f"  dw - fresh itself: mean {total.mean():+.4f}, {int((total > 0).sum())}/{len(total)} seeds positive")

    print("\nDESCRIPTIVE -- net-repulsive pairs during the MR phase (mean over seeds)")
    cfg = data["weaken"][seeds[0]][0]["config"]
    W, E = int(cfg["arm"]["warmup_epochs"]), int(cfg["epochs"])
    print(f"  {'arm':7s} {'window':6s} {'exact rule':>11s} {'sphere rule':>12s} {'kernel budget share':>20s}")
    for a in ("weaken", "repel"):
        per = [window_diagnostics(data[a][s][1], W, E) for s in seeds]
        for wn in WINDOW_NAMES:
            m = {k: np.nanmean([p[wn][k] for p in per]) for k in per[0][wn]}
            print(f"  {a:7s} {wn:6s} {100 * m['frac_repel_exact']:>10.2f}% {100 * m['frac_repel_sphere']:>11.2f}% "
                  f"{100 * m['kernel_share_repel']:>19.2f}%")


if __name__ == "__main__":
    main(*sys.argv[1:])
