"""lambda re-sweep under the corrected protocol: pre-registered analysis
(written 4 Oct 2026, before any sweep run existed).

Three purposes, one set of runs (Gowalla, seeds 2020-2029, W=100, E=1000,
k=20, rebuild 50, K=3, d=256, validation selection, budget-matched):

1. QUALIFICATION COMMITMENT. The qualification text promised a sensitivity
   analysis of lambda_manifold "com selecao final guiada por Recall@20 sobre
   uma particao de validacao dedicada". Rule fixed here: lambda* = the lambda
   with the highest mean VALIDATION recall@20 over the 10 seeds, candidates
   {0 (mr_off)} U GRID, ties to the smaller lambda. Test metrics are reported
   at lambda*. Boundary rule: if lambda* is the largest grid value, the grid is
   extended with 3e-4 and 1e-3 (10 seeds each) before a selection is declared;
   the analysis is then re-run with the extended --grid, so the dose-response
   Holm family below becomes the seven lambdas actually run.

2. DOSE-RESPONSE. For each lambda in GRID, emb_mr(lambda) - mr_off per seed on
   er_table, np_ref@20, recall@20, gini@20, tail_catalog_coverage@20; two-sided
   exact Wilcoxon, Holm across the GRID lambdas within each metric.

3. IS emb_mr_dw MORE THAN "LESS MR"? (stage 2)
   lambda' = the lambda at which emb_mr's mean er_table equals emb_mr_dw's
   (lambda = 1e-5). Rule: scan the stage-1 curve {0} U GRID in increasing
   lambda, take the FIRST segment whose end-point means bracket the target,
   interpolate linearly in lambda, round to 2 significant figures. If lambda'
   equals a grid value (or 0), those runs are reused. No bracketing segment:
   stage 2 is not run and that is reported.
   Stage 2 runs emb_mr at lambda', 10 seeds.
     PRIMARY (uncorrected): emb_mr_dw - emb_mr(lambda') on np_ref@20, two-sided.
     SECONDARY (Holm across 3): recall@20, gini@20, tail_catalog_coverage@20.
     MATCHING CHECK (not a hypothesis): er_table difference, reported.
   Descriptive only: the lambda matching emb_mr_dw on np_ref@20 by the same rule.
   If emb_mr_dw were simply less MR, the ER- and np_ref-matched lambdas would
   coincide.

lambda' and lambda* are computed only when every required arm has every
required seed; until then only seed counts are printed. Every run used must
also be healthy -- finite losses throughout history.json and finite reported
metrics -- because a run that diverged late would still report a plausible
earlier checkpoint. Unhealthy runs are listed and nothing is tested.

    python -m analysis.lambda_sweep                  # gowalla, defaults above
    python -m analysis.lambda_sweep --dataset yelp2018
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import numpy as np
from scipy.stats import wilcoxon

from src.train import _arm_tag

GRID = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
EXTENSION = (3e-4, 1e-3)
SEEDS = tuple(range(2020, 2030))
DW_LAMBDA = 1e-5
DOSE_METRICS = {"er_table": ("geometry", "er_table"), "np_ref@20": ("geometry", "np_ref@20"),
                "recall@20": ("test", "recall@20"), "gini@20": ("test", "gini@20"),
                "tail_cov@20": ("test", "tail_catalog_coverage@20")}
VAL_SELECT = ("val", "recall@20")
PRIMARY_2 = "np_ref@20"
SECONDARY_2 = ("recall@20", "gini@20", "tail_cov@20")


# ----------------------------------------------------------------- pure rules
def round_sig(x: float, sig: int = 2) -> float:
    if x == 0:
        return 0.0
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def matched_lambda(lams, means, target):
    """First segment (in increasing lambda) whose end-point means bracket
    ``target``; linear interpolation in lambda; 2 significant figures.
    Returns (lambda', (lo, hi)) or (None, None)."""
    order = np.argsort(lams)
    lams = [float(lams[i]) for i in order]
    means = [float(means[i]) for i in order]
    for i in range(len(lams) - 1):
        a, b = means[i], means[i + 1]
        if min(a, b) <= target <= max(a, b):
            lam = lams[i] if a == b else lams[i] + (target - a) / (b - a) * (lams[i + 1] - lams[i])
            return round_sig(lam), (lams[i], lams[i + 1])
    return None, None


def select_lambda(lams, val_means):
    """argmax of mean validation recall; ties go to the smaller lambda."""
    best = max(val_means)
    return min(l for l, v in zip(lams, val_means) if v == best)


def on_grid(lam, grid):
    """The grid value equal to ``lam`` (or 0.0), else None."""
    for g in (0.0,) + tuple(grid):
        if (g == 0 and lam == 0) or (g > 0 and abs(lam - g) <= 1e-9 * g):
            return g
    return None


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(ps) - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


def wtest(d):
    d = np.asarray(d, float)
    if len(d) < 3 or np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


# ------------------------------------------------------------------ loading
def load(root, ds, tag):
    out = {}
    for p in pathlib.Path(root, ds, tag).glob("seed*/results.json"):
        r = json.load(open(p))
        r["_dir"] = p.parent
        out[int(r["seed"])] = r
    return out


def unhealthy(r):
    """None if every reported metric and every logged loss is finite, else why not."""
    for sec in ("val", "test", "geometry"):
        for k, v in r.get(sec, {}).items():
            if isinstance(v, (int, float)) and not math.isfinite(v):
                return f"{sec}.{k} = {v}"
    hp = r["_dir"] / "history.json"
    if not hp.exists():
        return "history.json missing"
    for h in json.load(open(hp)):
        for k, v in h.items():
            if k.startswith("loss_") and isinstance(v, (int, float)) and not math.isfinite(v):
                return f"{k} = {v} at epoch {h.get('epoch')}"
    return None


def health_gate(runs):
    """runs: {label: {seed: result}}. Prints and returns True if all are healthy."""
    bad = [(lab, s, why) for lab, rs in runs.items() for s, r in sorted(rs.items())
           if (why := unhealthy(r)) is not None]
    if bad:
        print("\nUNHEALTHY RUNS -- nothing is tested:")
        for lab, s, why in bad:
            print(f"  {lab}/seed{s}: {why}")
        return False
    print(f"  health: {sum(len(rs) for rs in runs.values())} runs, all losses and metrics finite")
    return True


def value(r, sec_key):
    sec, key = sec_key
    if key not in r[sec]:
        raise KeyError(f"seed {r['seed']}: {sec}.{key} missing from results.json")
    return float(r[sec][key])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--dataset", default="gowalla")
    ap.add_argument("--grid", type=float, nargs="+", default=list(GRID))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--dw-lambda", type=float, default=DW_LAMBDA)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--rebuild", type=int, default=50)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    tag = lambda name, lam: _arm_tag(name, lam, a.k, "none", a.warmup, a.rebuild, a.layers, a.dim)
    lams = [0.0] + sorted(a.grid)
    tags = {0.0: tag("mr_off", 0.0), **{l: tag("emb_mr", l) for l in sorted(a.grid)}}
    dw_tag = tag("emb_mr_dw", a.dw_lambda)
    data = {l: load(a.runs, a.dataset, t) for l, t in tags.items()}
    dw = load(a.runs, a.dataset, dw_tag)
    seeds = sorted(a.seeds)

    print(f"dataset: {a.dataset}   required seeds: {seeds[0]}-{seeds[-1]} (n={len(seeds)})")
    complete = True
    for l in lams:
        have = sum(s in data[l] for s in seeds)
        complete &= have == len(seeds)
        print(f"  lambda={l:<8g} {tags[l]:44s} {have:2d}/{len(seeds)}")
    have = sum(s in dw for s in seeds)
    complete &= have == len(seeds)
    print(f"  emb_mr_dw  {dw_tag:44s} {have:2d}/{len(seeds)}")
    if not complete:
        print("\nINCOMPLETE -- no tests, no selection, no lambda' until every arm has every seed.")
        return
    req = lambda rs: {s: rs[s] for s in seeds}
    if not health_gate({**{tags[l]: req(data[l]) for l in lams}, dw_tag: req(dw)}):
        return

    M = {m: {l: np.array([value(data[l][s], sk) for s in seeds]) for l in lams}
         for m, sk in DOSE_METRICS.items()}
    V = {l: np.array([value(data[l][s], VAL_SELECT) for s in seeds]) for l in lams}
    D = {m: np.array([value(dw[s], sk) for s in seeds]) for m, sk in DOSE_METRICS.items()}

    print("\nSTAGE 1 -- means over seeds")
    print(f"  {'lambda':>8s} {'val R@20':>9s}" + "".join(f" {m:>12s}" for m in DOSE_METRICS))
    for l in lams:
        print(f"  {l:>8g} {V[l].mean():>9.5f}" + "".join(f" {M[m][l].mean():>12.5f}" for m in DOSE_METRICS))
    print(f"  {'dw':>8s} {np.mean([value(dw[s], VAL_SELECT) for s in seeds]):>9.5f}"
          + "".join(f" {D[m].mean():>12.5f}" for m in DOSE_METRICS))

    print("\nDOSE-RESPONSE -- emb_mr(lambda) - mr_off, two-sided Wilcoxon, Holm across lambdas within metric")
    rows = []
    for m in DOSE_METRICS:
        ds_ = [M[m][l] - M[m][0.0] for l in lams[1:]]
        ps = [wtest(d) for d in ds_]
        for l, d, p, h in zip(lams[1:], ds_, ps, holm(ps)):
            rows.append(dict(metric=m, lam=l, mean=float(d.mean()), lo=int((d < 0).sum()),
                             hi=int((d > 0).sum()), p=p, holm=float(h)))
    last = None
    for r in rows:
        if r["metric"] != last:
            print(f"  {r['metric']}"); last = r["metric"]
        print(f"    lambda={r['lam']:<8g} mean {r['mean']:+.5f} {r['lo']:2d}/{r['hi']:<2d} lower/higher "
              f"p={r['p']:.4f} Holm={r['holm']:.4f}{'  *' if r['holm'] < 0.05 else ''}")

    lam_star = select_lambda(lams, [V[l].mean() for l in lams])
    print(f"\nSELECTION (qualification rule): lambda* = {lam_star:g}  (highest mean validation recall@20)")
    if lam_star > 0:
        d = V[lam_star] - V[0.0]
        print(f"  lambda* vs lambda=0 on validation recall@20: {d.mean():+.5f}, {int((d < 0).sum())}/{int((d > 0).sum())} "
              f"lower/higher, p={wtest(d):.4f} (uncorrected; shows whether the choice is more than a tie-break)")
    print("  test at lambda*: " + ", ".join(f"{m} {M[m][lam_star].mean():.5f}" for m in DOSE_METRICS))
    at_top = lam_star == max(a.grid)
    ext = [x for x in EXTENSION if x > max(a.grid)] if at_top else []
    if at_top and not ext:
        print("  BOUNDARY: lambda* is the top of the extended grid. No further extension is "
              "pre-registered: report lambda* as a boundary selection.")
    if at_top and ext:
        print("  BOUNDARY: lambda* is the largest grid value -- pre-registered extension before a selection:")
        print(f"    python -m src.train -m dataset={a.dataset} seed={','.join(map(str, seeds))} arm=emb_mr "
              f"arm.lambda_manifold={','.join(f'{x:g}' for x in ext)}")
        print(f"    python -m analysis.lambda_sweep --dataset {a.dataset} --grid "
              f"{' '.join(f'{x:g}' for x in sorted(a.grid) + ext)}")

    print(f"\nYARDSTICK -- where emb_mr_dw (lambda={a.dw_lambda:g}) sits on the emb_mr curve")
    lam_er, seg_er = matched_lambda(lams, [M["er_table"][l].mean() for l in lams], D["er_table"].mean())
    lam_np, seg_np = matched_lambda(lams, [M["np_ref@20"][l].mean() for l in lams], D["np_ref@20"].mean())
    for name, lam, seg, m in (("ER-matched lambda'  ", lam_er, seg_er, "er_table"),
                              ("np_ref-matched (descr.)", lam_np, seg_np, "np_ref@20")):
        if lam is None:
            print(f"  {name}: no bracketing segment (dw mean {D[m].mean():.5f} lies outside the curve)")
        else:
            print(f"  {name}: {lam:g}  (segment {seg[0]:g}-{seg[1]:g}; dw mean {D[m].mean():.5f})")

    if a.out:
        out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{a.dataset}_curve.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["arm", "lambda", "seed", "val_recall@20", *DOSE_METRICS])
            for l in lams:
                for i, s in enumerate(seeds):
                    w.writerow(["mr_off" if l == 0 else "emb_mr", l, s, V[l][i], *[M[m][l][i] for m in DOSE_METRICS]])
            for i, s in enumerate(seeds):
                w.writerow(["emb_mr_dw", a.dw_lambda, s, value(dw[s], VAL_SELECT), *[D[m][i] for m in DOSE_METRICS]])
        print(f"\nwrote {out}/{a.dataset}_curve.csv")

    if lam_er is None:
        print("\nSTAGE 2 -- not run: no lambda on the curve matches emb_mr_dw's ER.")
        return
    reuse = on_grid(lam_er, a.grid)
    t2 = tags[reuse] if reuse is not None else tag("emb_mr", lam_er)
    run2 = data[reuse] if reuse is not None else load(a.runs, a.dataset, t2)
    have = sum(s in run2 for s in seeds)
    print(f"\nSTAGE 2 -- emb_mr_dw vs emb_mr at lambda' = {lam_er:g}  ({t2}: {have}/{len(seeds)} seeds"
          f"{', reused from stage 1' if reuse is not None else ''})")
    if have < len(seeds):
        print(f"  to run:  python -m src.train -m dataset={a.dataset} arm=emb_mr "
              f"arm.lambda_manifold={lam_er:g} seed={','.join(map(str, seeds))}")
        return
    if reuse is None and not health_gate({t2: {s: run2[s] for s in seeds}}):
        return
    R = {m: np.array([value(run2[s], sk) for s in seeds]) for m, sk in DOSE_METRICS.items()}
    diff = {m: D[m] - R[m] for m in DOSE_METRICS}
    show = lambda m, p, h=None: (f"    {m:12s} mean {diff[m].mean():+.5f} {int((diff[m] < 0).sum()):2d}/"
                                 f"{int((diff[m] > 0).sum()):<2d} dw lower/higher p={p:.4f}"
                                 + (f" Holm={h:.4f}" if h is not None else "")
                                 + ("  *" if (h if h is not None else p) < 0.05 else ""))
    print("  PRIMARY (pre-registered, uncorrected)")
    print(show(PRIMARY_2, wtest(diff[PRIMARY_2])))
    ps = [wtest(diff[m]) for m in SECONDARY_2]
    print(f"  SECONDARY (Holm across {len(SECONDARY_2)})")
    for m, p, h in zip(SECONDARY_2, ps, holm(ps)):
        print(show(m, p, float(h)))
    print("  MATCHING CHECK (not a hypothesis; should be near zero)")
    print(show("er_table", wtest(diff["er_table"])))


if __name__ == "__main__":
    main()
