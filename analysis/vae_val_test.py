"""Validation vs test along one VAE-CF trajectory, with list size and masking held equal.

Written 14 Oct 2026 (MIGRATION.md, "VAE-CF on Gowalla: diagnosing the gap"). On
Gowalla the validation-selected VAE-CF run kept validation Recall@20 flat while
test Recall@20 fell ~5% after epoch 20, so selection picked a late epoch. This
script asks whether that is about *which items* are held out, or about the
protocol's structure (validation users have fewer relevant items than test
users; test items are unmasked when ranking for validation).

It trains exactly as ``python -m src.train`` would (same config composition,
``set_seed``, ``build_vae_cf``, ``VAECFTrainer.train_epoch``; evaluation draws
no randomness, so reading the model at extra epochs cannot change the
trajectory) and, at each epoch in ``--at``, reads one set of scores five ways:

  val           relevant = validation items; masked = reduced train
                (the protocol's validation Recall@20)
  test          relevant = test items; masked = reduced train + validation
                (the protocol's test Recall@20), plus
  test_liang    the same hits / min(20, n_rel) (Liang et al.'s reference code)
  test_ndcg     NDCG@20, IDCG over min(20, n_rel)
  val_clean     relevant = validation items; masked = reduced train + test
  test_matched  relevant = a fixed random subset S_u of the test items with
                |S_u| = |val_u|; masked = reduced train + validation + (test - S_u)

``val_clean`` and ``test_matched`` have the same list sizes and the same masking
structure; they differ only in where the relevant items come from. Users: those
with at least one validation and one test item. ``val`` and ``test`` are
cross-checked against ``src.metrics.evaluate`` at the first reading.

    python -m analysis.vae_val_test --at 1,2,3,5,10,15,20,25,30,40,50 \\
        --out runs/_vae_val_test/gowalla_default_s2020.json \\
        dataset=gowalla arm=vae_cf seed=2020
    (any Hydra override works, e.g. arm.vae.dropout=0.2 arm.vae.beta.schedule=constant)
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from omegaconf import OmegaConf

from analysis.config_diff import composed
from src.data.dataset import InteractionDataset
from src.metrics.evaluate import evaluate
from src.sampling import dense_rows
from src.utils import get_device, set_seed
from src.vae_cf import VAECFTrainer, build_vae_cf

READS = ("val", "test", "test_liang", "test_ndcg", "val_clean", "test_matched")


def matched_subset(ds, seed: int = 12345) -> sp.csr_matrix:
    """One random subset of each user's test items, sized like the user's validation list."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for u in sorted(set(ds.valDict) & set(ds.testDict)):
        t = ds.testDict[u]
        pick = rng.choice(t, size=min(len(ds.valDict[u]), len(t)), replace=False)
        rows.extend([u] * len(pick))
        cols.extend(int(i) for i in pick)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(ds.n_users, ds.n_items))


@torch.no_grad()
def read_all(score, ds, S: sp.csr_matrix, users: np.ndarray, batch: int, device, k: int = 20) -> dict:
    """``score(users) -> (len(users), n_items)`` scores; the five reads, averaged over ``users``."""
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=device, dtype=torch.float64))
    idcg_at = torch.cumsum(disc, 0)
    sums = dict.fromkeys(READS, 0.0)
    neg = -float("inf")
    for s in range(0, len(users), batch):
        bu = users[s:s + batch]
        x = dense_rows(ds.UserItemNet, bu, device)
        tr = x > 0
        V = dense_rows(ds.ValNet, bu, device) > 0
        T = dense_rows(ds.TestNet, bu, device) > 0
        Sm = dense_rows(S, bu, device) > 0
        sc = score(bu)

        def hits(mask, rel):
            top = sc.masked_fill(mask, neg).topk(k, dim=1).indices
            return rel.gather(1, top).double()

        nV, nT, nS = V.sum(1).double(), T.sum(1).double(), Sm.sum(1).double()
        h = hits(tr, V)
        sums["val"] += float((h.sum(1) / nV).sum())
        h = hits(tr | V, T)
        sums["test"] += float((h.sum(1) / nT).sum())
        sums["test_liang"] += float((h.sum(1) / nT.clamp(max=k)).sum())
        sums["test_ndcg"] += float(((h * disc).sum(1) / idcg_at[nT.clamp(max=k).long() - 1]).sum())
        h = hits(tr | T, V)
        sums["val_clean"] += float((h.sum(1) / nV).sum())
        h = hits(tr | V | (T & ~Sm), Sm)
        sums["test_matched"] += float((h.sum(1) / nS).sum())
    return {key: v / len(users) for key, v in sums.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--at", required=True, help="comma-separated epochs (1-based) to read")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset-seed", type=int, default=12345)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    at = sorted({int(e) for e in args.at.split(",")})

    with tempfile.TemporaryDirectory() as scratch:
        cfg = OmegaConf.create(composed(list(args.overrides) + [f"paths.run_dir={scratch}"]))
        if str(cfg.arm.get("family", "lightgcn")) != "vae_cf":
            raise SystemExit("analysis.vae_val_test runs VAE-CF arms only (arm=vae_cf)")
        set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))
        device = get_device(str(cfg.device))
        ds = InteractionDataset(cfg.dataset.name, root=str(cfg.paths.data),
                                val_frac=float(cfg.split.val_frac), split_seed=int(cfg.split.split_seed),
                                min_train=int(cfg.split.min_train), short_head_frac=float(cfg.split.short_head_frac),
                                popularity_from=str(cfg.split.popularity_from))
        trainer = VAECFTrainer(cfg, ds, build_vae_cf(cfg, ds.n_items), device, scratch)
        S = matched_subset(ds, args.subset_seed)
        users = np.array(sorted(set(ds.valDict) & set(ds.testDict)), dtype=np.int64)
        print(f"{ds.name}: {len(users)} users with val and test items; mean |val| "
              f"{np.mean([len(ds.valDict[u]) for u in users]):.2f}, mean |test| "
              f"{np.mean([len(ds.testDict[u]) for u in users]):.2f}", flush=True)

        log, t0 = [], time.time()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(at[-1]):
            losses = trainer.train_epoch(epoch)
            if epoch + 1 not in at:
                continue
            trainer.model.eval()
            score = lambda bu: trainer.model.scores(dense_rows(ds.UserItemNet, bu, device))  # noqa: E731
            rec = {"epoch": epoch + 1, **read_all(score, ds, S, users, args.batch, device),
                   "beta": trainer.beta(trainer.step), "loss_nll": losses["nll"], "elapsed_s": time.time() - t0}
            if len(log) == 0:
                sc = trainer.scorer()
                rec["val_src_evaluate"] = evaluate(sc, ds, "val", [20], args.batch, device)["recall@20"]
                rec["test_src_evaluate"] = evaluate(sc, ds, "test", [20], args.batch, device)["recall@20"]
                trainer.model.train()
            log.append(rec)
            json.dump({"overrides": args.overrides, "subset_seed": args.subset_seed, "n_users": int(len(users)),
                       "reads": log}, open(out, "w"), indent=1)
            print("ep %4d  val %.4f  test %.4f (liang %.4f, ndcg %.4f)  |  val_clean %.4f  test_matched %.4f"
                  "  ratio %.3f" % (rec["epoch"], rec["val"], rec["test"], rec["test_liang"], rec["test_ndcg"],
                                    rec["val_clean"], rec["test_matched"], rec["val_clean"] / rec["test_matched"])
                  + (f"   [src.evaluate: val {rec['val_src_evaluate']:.4f} test {rec['test_src_evaluate']:.4f}]"
                     if "val_src_evaluate" in rec else ""), flush=True)


if __name__ == "__main__":
    main()
