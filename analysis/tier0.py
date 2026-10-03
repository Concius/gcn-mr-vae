"""Tier 0 interpretation of the differentiable-kernel result. CPU by default.

Uses only artefacts that already exist (history.json, best.pt, results.json);
nothing is retrained. Four views, all exploratory -- p-values are reported
uncorrected and are not part of any pre-registered test:

A. ER-gap trajectory vs the crossover share (history.json).
   Per seed, gap(t) = ER_table(emb_mr_dw) - ER_table(emb_mr) at every
   evaluation. The MR phase is split into early / mid / late windows; for each
   window we report how much the gap grew and the mean share of neighbour
   pairs past the repulsion crossover, as logged at each rebuild by the
   unit-sphere rule (1 - cos > sigma^2). CORRECTION (4 Oct 2026): off the unit
   sphere that rule can only UNDERCOUNT net-repulsive pairs (the exact rule is
   d2 > 2 sigma^2 |z_i||z_j|; see mr_layer.net_repulsive), so a share of ~0
   here does NOT show that no net-repulsive pairs remain.
   analysis/crossover_check.py re-measures the exact share.
   Sanity check: the gap must be 0 at epoch W -- both arms share the warm-up.

IMPORTANT for B-D: only the validation-selected checkpoint (best.pt) is saved,
and its epoch can differ between arms and seeds. Comparing checkpoints from
different epochs can reverse a sign purely through training time -- on the
synthetic replica it did. So every B-D comparison is printed twice: over all
seed pairs, and over only the pairs whose two runs selected the same epoch.
ER at the final epoch (from results.json, all seeds, fixed epoch) is added as
the clean fixed-epoch reference.

B. Norms vs angles (best.pt). The kernel-gradient term is purely rotational
   at the propagated level, so a dw effect driven by it should survive L2
   normalisation. Reports mean norms and ER on raw and normalised embeddings.
   Sanity check: ER recomputed here must match ER logged in results.json.

C. Alignment and uniformity (Wang & Isola 2020; DirectAU, Wang et al. 2022)
   on L2-normalised propagated embeddings. Alignment over training
   (user, item) pairs; uniformity = log E exp(-2 ||x - y||^2), users and items
   separately, averaged. Lower = better for both.

D. Singular-value spectrum (same subsample as ER): share of variance in the
   top 1 and top 10 directions, and stable rank ||Z||_F^2 / ||Z||_2^2 on the
   centred matrix (cf. Loveland et al. 2025). Self-check: ER rebuilt from
   this spectrum must equal effective_rank().

    python -m analysis.tier0                       # all three datasets
    python -m analysis.tier0 --datasets gowalla

Runtime is dominated by five 10,000 x 256 SVDs per checkpoint, which
effective_rank() always runs on CPU -- so --device cuda only speeds up the
(small) propagation step. Timing is printed per arm as it runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

from src.data.dataset import InteractionDataset
from src.metrics.geometry import effective_rank
from src.models.lightgcn import LightGCN

ARMS = ["mr_off", "emb_mr", "emb_mr_fresh", "emb_mr_dw"]
WINDOW_FRACTIONS = (0.0, 7 / 18, 13 / 18, 1.0)        # W=100,E=1000 -> 100, 450, 750, 1000
WINDOW_NAMES = ("early", "mid", "late")


# ------------------------------------------------------------------ helpers
def tags(mr_suffix: str, off_suffix: str) -> dict:
    return {a: (a + off_suffix if a == "mr_off" else a + mr_suffix) for a in ARMS}


def runs_of(root, ds, tag) -> dict:
    out = {}
    for p in sorted(pathlib.Path(root, ds, tag).glob("seed*/results.json")):
        out[json.load(open(p))["seed"]] = p.parent
    return out


def nearest(epochs, target):
    return min(epochs, key=lambda e: abs(e - target))


def paired(a: dict, b: dict, key: str, same_epoch_only: bool = False):
    seeds = sorted(set(a) & set(b))
    if same_epoch_only:
        seeds = [s for s in seeds if a[s]["best_epoch"] == b[s]["best_epoch"]]
    d = np.array([a[s][key] - b[s][key] for s in seeds])
    if len(d) < 3 or np.allclose(d, 0):
        return d, float("nan")
    return d, float(wilcoxon(d, alternative="two-sided").pvalue)


def fmt(d, p):
    if len(d) == 0:
        return f"{'--':>10s}  {'':5s}  {'':8s}"
    return f"{d.mean():>+10.4f}  {int((d < 0).sum()):>2d}/{int((d > 0).sum()):<2d}  p={p:.4f}"


# ------------------------------------------------------- A. trajectory
def trajectory(root, ds, T):
    dw, em = runs_of(root, ds, T["emb_mr_dw"]), runs_of(root, ds, T["emb_mr"])
    seeds = sorted(set(dw) & set(em))
    if len(seeds) < 3:
        return None
    cfg = json.load(open(dw[seeds[0]] / "results.json"))["config"]
    W, E = int(cfg["arm"]["warmup_epochs"]), int(cfg["epochs"])
    gaps, cross = {}, []
    for s in seeds:
        ha, hb = json.load(open(dw[s] / "history.json")), json.load(open(em[s] / "history.json"))
        ea = {h["epoch"]: h["er_table"] for h in ha if h.get("event") == "eval" and "er_table" in h}
        eb = {h["epoch"]: h["er_table"] for h in hb if h.get("event") == "eval" and "er_table" in h}
        gaps[s] = {e: ea[e] - eb[e] for e in set(ea) & set(eb)}
        cross += [(h["epoch"], h["frac_beyond_crossover"]) for h in ha
                  if h.get("event") == "laplacian_built" and "frac_beyond_crossover" in h]
    common = sorted(set.intersection(*(set(g) for g in gaps.values())))
    if not common:      # e.g. geometry.er_each_eval was off: no per-epoch ER logged
        return None
    bounds = [nearest(common, W + f * (E - W)) for f in WINDOW_FRACTIONS]
    rows = []
    for name, (lo, hi) in zip(WINDOW_NAMES, zip(bounds[:-1], bounds[1:])):
        growth = np.array([gaps[s][hi] - gaps[s][lo] for s in seeds])
        cr = [f for e, f in cross if lo <= e < hi]
        rows.append(dict(window=name, start=lo, end=hi, growth_mean=float(growth.mean()),
                         n_pos=int((growth > 0).sum()), n=len(seeds),
                         crossover_mean=float(np.mean(cr)) if cr else float("nan")))
    final = np.array([gaps[s][bounds[-1]] for s in seeds])
    start = np.array([gaps[s][bounds[0]] for s in seeds])
    curve = [(e, float(np.mean([gaps[s][e] for s in seeds]))) for e in common]
    return dict(W=W, E=E, start_gap_maxabs=float(np.abs(start).max()), final_gap=float(final.mean()),
                windows=rows, curve=curve)


# ---------------------------------------------------- B, C, D per checkpoint
def spectrum(Z, max_samples, seed, center):
    """Same subsampling and centring as effective_rank()."""
    z = Z.detach().float().cpu()
    if z.shape[0] > max_samples:
        g = torch.Generator().manual_seed(seed)
        z = z[torch.randperm(z.shape[0], generator=g)[:max_samples]]
    if center:
        z = z - z.mean(dim=0, keepdim=True)
    return torch.linalg.svdvals(z)


def uniformity(x, n, seed):
    x = F.normalize(x.float().cpu(), dim=1)
    if x.shape[0] > n:
        g = torch.Generator().manual_seed(seed)
        x = x[torch.randperm(x.shape[0], generator=g)[:n]]
    return float(torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log())


def run_metrics(rdir, data, graph, dev, n_align, n_unif):
    res = json.load(open(rdir / "results.json"))
    cfg = res["config"]
    g = cfg["geometry"]
    ms, sd, cen = int(g["er_max_samples"]), int(g["np_seed"]), bool(g.get("er_center", True))
    m = LightGCN(data.n_users, data.n_items, dim=int(cfg["model"]["dim"]),
                 n_layers=int(cfg["model"]["n_layers"]), graph=graph).to(dev)
    m.load_state_dict(torch.load(rdir / "best.pt", map_location=dev, weights_only=False)["model"])
    m.eval()
    with torch.no_grad():
        au, ai = m.computer()
        prop, ego = torch.cat([au, ai]), m.ego_embeddings()
    out = {"seed": res["seed"], "best_epoch": int(res["best_epoch"])}
    # fixed-epoch reference, logged at the final epoch for every run
    out["er_table_final"] = res["at_last_epoch"].get("er_table", float("nan"))
    out["er_prop_final"] = res["at_last_epoch"].get("er_prop", float("nan"))
    out["er_table"] = effective_rank(ego, ms, center=cen, seed=sd)
    out["er_prop"] = effective_rank(prop, ms, center=cen, seed=sd)
    out["check_vs_logged"] = max(abs(out["er_table"] - res["geometry"]["er_table"]),
                                 abs(out["er_prop"] - res["geometry"]["er_prop"]))
    out["er_prop_normalised"] = effective_rank(F.normalize(prop, dim=1), ms, center=cen, seed=sd)
    out["er_table_normalised"] = effective_rank(F.normalize(ego, dim=1), ms, center=cen, seed=sd)
    out["norm_prop"] = float(prop.norm(dim=1).mean())
    out["norm_ego"] = float(ego.norm(dim=1).mean())
    # C. alignment / uniformity
    rng = np.random.default_rng(0)
    pick = rng.choice(len(data.trainUser), size=min(n_align, len(data.trainUser)), replace=False)
    u = F.normalize(au[torch.from_numpy(data.trainUser[pick]).to(dev)].float(), dim=1)
    i = F.normalize(ai[torch.from_numpy(data.trainItem[pick]).to(dev)].float(), dim=1)
    out["alignment"] = float((u - i).norm(dim=1).pow(2).mean())
    out["uniformity"] = 0.5 * (uniformity(au, n_unif, 0) + uniformity(ai, n_unif, 1))
    # D. spectrum
    S = spectrum(prop, ms, sd, cen)
    S = S[S > 1e-10]
    p = S / S.sum()
    out["check_er_from_spectrum"] = abs(float(torch.exp(-(p * torch.log(p)).sum())) - out["er_prop"])
    v = S.pow(2)
    out["top1_var"] = float(v[0] / v.sum())
    out["top10_var"] = float(v[:10].sum() / v.sum())
    out["stable_rank"] = float(v.sum() / v[0])
    out["_spectrum"] = (v / v.sum()).numpy()
    return out


METRICS = ["er_table", "er_prop", "er_table_normalised", "er_prop_normalised", "norm_ego", "norm_prop",
           "alignment", "uniformity", "top1_var", "top10_var", "stable_rank"]
FINAL_METRICS = ["er_table_final", "er_prop_final"]
COMPARISONS = [("emb_mr_dw", "emb_mr"), ("emb_mr_dw", "emb_mr_fresh"), ("emb_mr_dw", "mr_off"),
               ("emb_mr", "mr_off")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--datasets", nargs="+", default=["gowalla", "yelp2018", "amazon-book"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mr-suffix", default="_w100_lam1e-05_k20_r50_K3_d256")
    ap.add_argument("--off-suffix", default="_w100_K3_d256")
    ap.add_argument("--n-align", type=int, default=200_000)
    ap.add_argument("--n-unif", type=int, default=5000)
    ap.add_argument("--out", default="runs/_tier0")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    T = tags(a.mr_suffix, a.off_suffix)
    outdir = pathlib.Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    flat, curves, spectra = [], [], []

    for ds in a.datasets:
        print(f"\n{'=' * 78}\n{ds}\n{'=' * 78}")
        # ---------------- A
        tr = trajectory(a.runs, ds, T)
        if tr:
            print(f"A. ER_table gap (dw - emb_mr) over the MR phase, W={tr['W']}, E={tr['E']}")
            print(f"   sanity: gap at epoch W = {tr['start_gap_maxabs']:.2e} (shared warm-up, must be ~0)")
            print(f"   {'window':6s} {'epochs':>11s} {'gap growth':>11s} {'seeds +':>8s} {'sphere-rule share':>18s}")
            for r in tr["windows"]:
                print(f"   {r['window']:6s} {r['start']:>4d}-{r['end']:<6d} {r['growth_mean']:>+11.4f} "
                      f"{r['n_pos']:>4d}/{r['n']:<3d} {100 * r['crossover_mean']:>17.3f}%")
            print(f"   final gap {tr['final_gap']:+.4f}; share accrued per window: " + ", ".join(
                f"{r['window']} {100 * r['growth_mean'] / tr['final_gap']:.0f}%" for r in tr["windows"]))
            curves += [dict(dataset=ds, epoch=e, gap=g) for e, g in tr["curve"]]
        # ---------------- B, C, D
        all_runs = {arm: runs_of(a.runs, ds, T[arm]) for arm in ARMS}
        splits = {tuple(json.load(open(r / "results.json"))["config"]["split"][k]
                        for k in ("val_frac", "split_seed", "min_train"))
                  for runs in all_runs.values() for r in runs.values()}
        if len(splits) > 1:
            raise ValueError(f"{ds}: runs were trained on different splits {sorted(splits)}; "
                             "propagation would not match training")
        data = graph = None
        per_arm = {}
        for arm in ARMS:
            runs = all_runs[arm]
            if not runs:
                continue
            if data is None:
                vf, ss, mt = next(iter(splits))
                data = InteractionDataset(ds, root=a.data, val_frac=float(vf), split_seed=int(ss),
                                          min_train=int(mt), verbose=False)
                graph = data.sparse_graph(dev)
            per_arm[arm] = {}
            t0 = time.time()
            for s, rdir in runs.items():
                m = run_metrics(rdir, data, graph, dev, a.n_align, a.n_unif)
                spectra.append(dict(dataset=ds, arm=arm, seed=s, spectrum=m.pop("_spectrum")))
                per_arm[arm][s] = m
                flat.append(dict(dataset=ds, arm=arm, **m))
            dt = time.time() - t0
            print(f"   [{arm}: {len(runs)} checkpoints in {dt:.0f}s, {dt / len(runs):.1f}s each]", flush=True)
        if not per_arm:
            continue
        chk = max(m["check_vs_logged"] for arm in per_arm.values() for m in arm.values())
        chk2 = max(m["check_er_from_spectrum"] for arm in per_arm.values() for m in arm.values())
        print(f"\nB-D. sanity: |ER recomputed - ER logged| max = {chk:.2e} "
              f"(0 if same device as training; ~1e-4 CPU vs GPU)")
        print(f"     sanity: |ER from spectrum - effective_rank| max = {chk2:.2e} (must be ~0)")
        print("\n   selected (best) epoch per arm, min / median / max:")
        for arm, runs in per_arm.items():
            be = [m["best_epoch"] for m in runs.values()]
            print(f"      {arm:14s} {min(be):>5d} {int(np.median(be)):>6d} {max(be):>5d}")
        print("\n   mean over seeds   " + "".join(f"{arm:>14s}" for arm in per_arm))
        for k in METRICS + FINAL_METRICS:
            print(f"   {k:20s}" + "".join(f"{np.mean([m[k] for m in per_arm[arm].values()]):>14.4f}"
                                          for arm in per_arm))
        print("\n   paired by seed (exploratory, uncorrected): mean delta, seeds lower/higher, p")
        for x, y in COMPARISONS:
            if x not in per_arm or y not in per_arm:
                continue
            common = sorted(set(per_arm[x]) & set(per_arm[y]))
            same = sum(per_arm[x][s]["best_epoch"] == per_arm[y][s]["best_epoch"] for s in common)
            print(f"   {x} - {y}    ({same}/{len(common)} seed pairs selected the same epoch)")
            print(f"      {'metric':20s} {'all pairs, selected checkpoints':>33s}   {'same-epoch pairs only':>27s}")
            for k in METRICS:
                print(f"      {k:20s} {fmt(*paired(per_arm[x], per_arm[y], k))}   "
                      f"{fmt(*paired(per_arm[x], per_arm[y], k, same_epoch_only=True))}")
            print("      fixed final epoch, all pairs:")
            for k in FINAL_METRICS:
                print(f"      {k:20s} {fmt(*paired(per_arm[x], per_arm[y], k))}")

    with open(outdir / "runs.csv", "w", newline="") as f:
        if flat:
            w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    with open(outdir / "trajectory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "epoch", "gap"]); w.writeheader(); w.writerows(curves)
    with open(outdir / "spectrum.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "arm", "seed", "component", "variance_share"])
        for r in spectra:
            for j, v in enumerate(r["spectrum"]):
                w.writerow([r["dataset"], r["arm"], r["seed"], j, f"{v:.6g}"])
    print(f"\nwrote {outdir}/runs.csv, trajectory.csv, spectrum.csv")


if __name__ == "__main__":
    main()
