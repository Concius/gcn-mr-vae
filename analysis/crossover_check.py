"""How many neighbour pairs are net-repulsive at the saved checkpoints, under
the exact rule versus the unit-sphere rule that earlier runs logged?

The logged ``frac_beyond_crossover`` uses 1 - cos > sigma^2, which is exact
only for unit-norm embeddings and otherwise can only undercount (see
``mr_layer.net_repulsive``). The Tier 0 working hypothesis -- that the
dw - emb_mr ER gap keeps growing where no net-repulsive pairs remain -- rests
on those logged numbers. This script re-measures them from the saved
checkpoints, on CPU, retraining nothing.

APPROXIMATION, stated up front: best.pt is the validation-selected checkpoint
(usually a late epoch), and the k-NN graph and sigma are rebuilt from it with
the run's own k. Training used the graph from the most recent rebuild, so
this measures the graph training *would* build at that point, not the exact
one in use. Early epochs cannot be reached this way (no early checkpoint is
saved); the new split runs log both shares at every evaluation instead.

    python -m analysis.crossover_check                       # all three datasets
    python -m analysis.crossover_check --datasets gowalla
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time

import numpy as np
import torch

from src.data.dataset import InteractionDataset
from src.models.lightgcn import LightGCN
from src.models.mr_layer import build_embedding_knn, kernel_force_diagnostics

ARMS = ["mr_off", "emb_mr", "emb_mr_fresh", "emb_mr_dw"]


def run_dirs(root, ds, tag):
    return {json.load(open(p))["seed"]: p.parent
            for p in sorted(pathlib.Path(root, ds, tag).glob("seed*/results.json"))
            if (p.parent / "best.pt").exists()}


def measure(rdir, data, graph, k, dev):
    cfg = json.load(open(rdir / "results.json"))["config"]
    m = LightGCN(data.n_users, data.n_items, dim=int(cfg["model"]["dim"]),
                 n_layers=int(cfg["model"]["n_layers"]), graph=graph).to(dev)
    m.load_state_dict(torch.load(rdir / "best.pt", map_location=dev, weights_only=False)["model"])
    m.eval()
    with torch.no_grad():
        Z = torch.cat(m.computer())
        g = build_embedding_knn(Z, k=k, device=dev, verbose=False)
        out = kernel_force_diagnostics(Z, g)
        nrm = Z.norm(dim=1)
        r = nrm.unsqueeze(1) / nrm[g.idx]
        r = torch.maximum(r, 1 / r)
    out.update(sigma=g.sigma, best_epoch=int(json.load(open(rdir / "results.json"))["best_epoch"]),
               norm_ratio_median=float(r.median()), norm_ratio_p90=float(r.flatten().kthvalue(
                   max(1, int(0.9 * r.numel()))).values))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--datasets", nargs="+", default=["gowalla", "yelp2018", "amazon-book"])
    ap.add_argument("--mr-suffix", default="_w100_lam1e-05_k20_r50_K3_d256")
    ap.add_argument("--off-suffix", default="_w100_K3_d256")
    ap.add_argument("--k", type=int, default=20, help="k of the rebuilt graph (the runs used 20)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/_crossover_check")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    rows = []
    for ds in a.datasets:
        tags = {arm: arm + (a.off_suffix if arm == "mr_off" else a.mr_suffix) for arm in ARMS}
        found = {arm: run_dirs(a.runs, ds, t) for arm, t in tags.items()}
        if not any(found.values()):
            continue
        splits = {tuple(json.load(open(d / "results.json"))["config"]["split"][x]
                        for x in ("val_frac", "split_seed", "min_train"))
                  for v in found.values() for d in v.values()}
        if len(splits) > 1:
            raise ValueError(f"{ds}: runs use different splits {sorted(splits)}")
        vf, ss, mt = next(iter(splits))
        data = InteractionDataset(ds, root=a.data, val_frac=float(vf), split_seed=int(ss),
                                  min_train=int(mt), verbose=False)
        graph = data.sparse_graph(dev)
        print(f"\n{ds}  (graph rebuilt at the selected checkpoint, k={a.k})")
        print(f"  {'arm':13s} {'n':>2s} {'best ep':>8s} {'sphere rule':>12s} {'exact rule':>11s} "
              f"{'kernel budget':>14s} {'norm ratio med/p90':>19s}")
        for arm, dirs in found.items():
            if not dirs:
                continue
            t0 = time.time()
            res = [dict(dataset=ds, arm=arm, seed=s, **measure(d, data, graph, a.k, dev))
                   for s, d in sorted(dirs.items())]
            rows += res
            f = lambda k: np.array([r[k] for r in res])
            print(f"  {arm:13s} {len(res):>2d} {int(np.median(f('best_epoch'))):>8d} "
                  f"{100 * f('frac_repel_sphere').mean():>11.2f}% {100 * f('frac_repel_exact').mean():>10.2f}% "
                  f"{100 * np.nanmean(f('kernel_share_repel')):>13.2f}% "
                  f"{np.median(f('norm_ratio_median')):>9.2f} / {np.median(f('norm_ratio_p90')):<6.2f}"
                  f"   [{time.time() - t0:.0f}s]", flush=True)
    if rows:
        out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
        with open(out / "checkpoints.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {out}/checkpoints.csv")
        print("columns: sphere rule = share logged by earlier runs; exact rule = true net-repulsive share;\n"
              "kernel budget = share of the kernel force carried by net-repulsive pairs.")


if __name__ == "__main__":
    main()
