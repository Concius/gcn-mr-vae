"""One trainer for every arm.

Replaces ``train_single_dataset`` (Cell 7), ``run_mr_experiment`` (Cell 8)
and ``run_mr_cooccurrence_experiment`` (Cell 9), which were the same loop
copy-pasted three times with different loss classes and different Drive
bookkeeping.

An *arm* is just a config: ``warmup_epochs`` (W), ``lambda_manifold``,
``laplacian`` source. Every arm trains for exactly ``epochs`` (E) epochs:

    epochs [0, W)   BPR + L2                    ("warmup" phase)
    epochs [W, E)   BPR + L2 [+ manifold term]  ("main" phase)

* ``baseline``  W=0,   lambda=0   -> E epochs of BPR from scratch
* ``mr_off``    W=100, lambda=0   -> E epochs of BPR from scratch, but with
                                     the same phase structure and the same
                                     epoch-W reference snapshot as emb_mr.
                                     This is the MR-off continuation control
                                     (P0 fix #2). Numerically it is the
                                     baseline protocol; it exists so the two
                                     arms differ in exactly one config value.
* ``emb_mr``    W=100, lambda>0, laplacian=embeddings, rebuilt every R epochs
* ``coocc_mr``  W=100, lambda>0, laplacian=cooccurrence (fixed)

P0 fix #1 (budget). There is no code path that loads a baseline checkpoint
and then trains for another E epochs. The only way to skip the warm-up phase
is ``warmstart.load``, which resumes model, optimiser *and* RNG state from a
checkpoint saved at epoch W by another run with ``warmstart.save=true``, and
then trains only epochs [W, E). Total budget is E either way.

P0 fix #3 (selection). ``eval.select_split`` defaults to ``val``. Test is
evaluated at the selected checkpoint at the end (always), and optionally at
every evaluation for curve plotting; it never influences selection.

P2 fix #7 (lambda scale). The manifold term is applied every minibatch, so
its per-epoch weight is ``n_batches * lambda``, and ``n_batches`` differs per
dataset (Gowalla ~25, Yelp ~37, Amazon-Book ~73 at batch 32768).
``arm.lambda_scale=per_epoch`` divides lambda by ``n_batches`` so the summed
per-epoch weight is dataset-invariant. ``none`` reproduces Chapter 5. The
gate should run with ``none`` so only the protocol changes.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .metrics.evaluate import evaluate
from .metrics.geometry import effective_rank, neighborhood_preservation, np_ref
from .models.losses import BPRTerm, BatchContext, CompositeLoss, L2EgoTerm, ManifoldTerm
from .models.mr_layer import build_cooccurrence_laplacian, build_embedding_laplacian
from .sampling import minibatch, n_batches, uniform_sample
from .utils import Timer, count_params, json_dump, rng_state_dict, rng_state_load


class Trainer:
    def __init__(self, cfg: DictConfig, dataset, model, device: torch.device,
                 run_dir: str | os.PathLike, wandb_run=None):
        self.cfg = cfg
        self.ds = dataset
        self.model = model.to(device)
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.wandb = wandb_run

        a = cfg.arm
        self.E = int(cfg.epochs)
        self.W = int(a.warmup_epochs)
        if not 0 <= self.W <= self.E:
            raise ValueError(f"warmup_epochs={self.W} must be in [0, epochs={self.E}]")
        self.lam = float(a.lambda_manifold)
        self.lap_source = str(a.laplacian)
        if self.lam > 0 and self.lap_source == "none":
            raise ValueError("lambda_manifold > 0 requires arm.laplacian in {embeddings, cooccurrence}")
        self.k = int(a.k_neighbors)
        self.rebuild_every = int(a.rebuild_every)
        self.knn_backend = str(a.knn_backend)
        self.sigma = None if a.sigma is None else float(a.sigma)

        self.batch_size = int(cfg.optim.batch_size)
        self.nb = n_batches(len(dataset.trainUser), self.batch_size)
        if str(a.lambda_scale) == "per_epoch":
            self.lam_eff = self.lam / self.nb
        elif str(a.lambda_scale) == "none":
            self.lam_eff = self.lam
        else:
            raise ValueError(a.lambda_scale)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(cfg.optim.lr), eps=1e-8)
        self.loss = CompositeLoss([BPRTerm(), L2EgoTerm(float(cfg.optim.decay)), ManifoldTerm(self.lam_eff)])

        self.sampler_rng = np.random.default_rng(int(cfg.seed))

        ev = cfg.eval
        self.topks = list(ev.topks)
        self.eval_every = int(ev.every)
        self.eval_bs = int(ev.batch_size)
        self.test_each_eval = bool(ev.test_each_eval)
        self.select_split = str(ev.select_split)
        self.select_metric = str(ev.select_metric)
        self.patience = int(ev.patience_evals)
        if self.select_split == "val" and not dataset.valDict:
            raise ValueError("eval.select_split=val but the dataset has no validation split "
                             "(split.val_frac=0). Use split.val_frac>0, or set "
                             "eval.select_split=test to knowingly reproduce the notebook protocol.")
        if self.select_split == "test":
            print("  !! WARNING: selecting checkpoints on TEST. This reproduces the notebook "
                  "protocol and is not valid for reporting.")

        self.laplacian: torch.Tensor | None = None
        self.ref_ego: torch.Tensor | None = None   # ego snapshot at epoch W
        self.history: list[dict] = []
        self.best = {"score": -1.0, "epoch": -1}
        self.evals_since_best = 0
        self.start_epoch = 0
        self.er_transition: float | None = None

    # ------------------------------------------------------------- warm-start
    def _warmstart_path(self) -> Path:
        return self.run_dir / f"warmstart_ep{self.W}.pt"

    def save_warmstart(self, epoch: int) -> Path:
        path = self._warmstart_path()
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "rng": rng_state_dict(),
            "sampler_rng": self.sampler_rng.bit_generator.state,
            "ref_ego": self.ref_ego.cpu() if self.ref_ego is not None else None,
            "history": self.history,
            "dataset": self.ds.summary(),
            "model_meta": self.model.state_for_checkpoint(),
        }, path)
        print(f"  warm-start checkpoint saved: {path}")
        return path

    def load_warmstart(self, path: str | os.PathLike) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck["epoch"] != self.W:
            raise ValueError(f"warm-start checkpoint is at epoch {ck['epoch']}, arm expects W={self.W}")
        if ck["dataset"]["name"] != self.ds.name or ck["dataset"]["val_frac"] != self.ds.val_frac \
                or ck["dataset"]["split_seed"] != self.ds.split_seed:
            raise ValueError("warm-start checkpoint was produced on a different dataset/split")
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        rng_state_load(ck["rng"])
        self.sampler_rng.bit_generator.state = ck["sampler_rng"]
        self.ref_ego = ck["ref_ego"].to(self.device) if ck["ref_ego"] is not None else None
        self.history = list(ck.get("history", []))
        self.start_epoch = self.W
        print(f"  resumed warm-start from {path} at epoch {self.W} (model+optimizer+RNG)")

    # ------------------------------------------------------------- laplacian
    @torch.no_grad()
    def _propagated(self):
        self.model.eval()
        au, ai = self.model.computer()
        self.model.train()
        return au, ai

    def _build_laplacian(self, epoch: int) -> None:
        if self.lam <= 0:
            return
        if self.lap_source == "embeddings":
            au, ai = self._propagated()
            self.laplacian = build_embedding_laplacian(torch.cat([au, ai]), k=self.k, sigma=self.sigma,
                                                       backend=self.knn_backend, device=self.device)
        elif self.lap_source == "cooccurrence":
            if self.laplacian is None:   # fixed: build once
                self.laplacian = build_cooccurrence_laplacian(self.ds.UserItemNet, k=self.k, device=self.device)
        else:
            raise ValueError(self.lap_source)
        self.history_event(epoch, "laplacian_built", source=self.lap_source)

    def history_event(self, epoch: int, event: str, **kw) -> None:
        self.history.append({"epoch": epoch, "event": event, **kw})

    # -------------------------------------------------------------- transition
    @torch.no_grad()
    def _on_transition(self, epoch: int) -> None:
        """Called once when epoch == W (before training that epoch)."""
        self.ref_ego = self.model.ego_embeddings().detach().clone()
        au, ai = self._propagated()
        self.er_transition = effective_rank(torch.cat([au, ai]), int(self.cfg.geometry.er_max_samples),
                                            seed=int(self.cfg.geometry.np_seed))
        print(f"\n--- epoch {epoch}: end of warm-up. ER(propagated)={self.er_transition:.2f} "
              f"({100*self.er_transition/self.model.dim:.1f}% of d). "
              f"{'MR ON, lambda_eff=%g' % self.lam_eff if self.lam > 0 else 'MR OFF (control)'} ---")
        self.history_event(epoch, "transition", effective_rank=self.er_transition)
        if bool(self.cfg.warmstart.save):
            self.save_warmstart(epoch)
        self._build_laplacian(epoch)
        active = {"bpr", "l2"} | ({"manifold"} if self.lam > 0 else set())
        self.loss.set_active(active)

    # ------------------------------------------------------------- one epoch
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        S = uniform_sample(self.ds.trainUser, self.ds.trainItem, self.ds.UserItemNet,
                           self.ds.n_items, self.sampler_rng)
        S = S[self.sampler_rng.permutation(len(S))]
        users = torch.from_numpy(S[:, 0]).to(self.device)
        pos = torch.from_numpy(S[:, 1]).to(self.device)
        neg = torch.from_numpy(S[:, 2]).to(self.device)

        acc: dict[str, float] = {}
        n = 0
        for bu, bp, bn in minibatch(users, pos, neg, batch_size=self.batch_size):
            au, ai = self.model.computer()
            ctx = BatchContext(self.model, bu, bp, bn, au, ai, epoch, laplacian=self.laplacian)
            total, logs = self.loss(ctx)
            self.opt.zero_grad(set_to_none=True)
            total.backward()
            self.opt.step()
            for k_, v in logs.items():
                acc[k_] = acc.get(k_, 0.0) + v
            n += 1
        return {k_: v / n for k_, v in acc.items()}

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate_point(self, epoch: int, phase: str, losses: dict) -> dict:
        self.model.eval()
        au, ai = self.model.computer()
        rec = {"epoch": epoch + 1, "phase": phase, "event": "eval", **{f"loss_{k}": v for k, v in losses.items()}}
        val = evaluate(self.model, self.ds, "val", self.topks, self.eval_bs, self.device, au, ai)
        rec.update({f"val_{k}": v for k, v in val.items()})
        if self.test_each_eval:
            test = evaluate(self.model, self.ds, "test", self.topks, self.eval_bs, self.device, au, ai)
            rec.update({f"test_{k}": v for k, v in test.items()})
        if bool(self.cfg.geometry.er_each_eval):
            rec["effective_rank"] = effective_rank(torch.cat([au, ai]), int(self.cfg.geometry.er_max_samples),
                                                   seed=int(self.cfg.geometry.np_seed))
        self.model.train()
        return rec

    def _selection_score(self, rec: dict) -> float:
        key = f"{self.select_split}_{self.select_metric}"
        if key not in rec:
            raise KeyError(f"selection metric {key} not in evaluation record; keys: {sorted(rec)}")
        return float(rec[key])

    def save_best(self, epoch: int, rec: dict) -> None:
        torch.save({"epoch": epoch + 1, "model": self.model.state_dict(), "record": rec,
                    "model_meta": self.model.state_for_checkpoint()}, self.run_dir / "best.pt")

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        cfg = self.cfg
        print(f"\n== {self.ds.name} | arm={cfg.arm.name} | seed={cfg.seed} | E={self.E} W={self.W} "
              f"lambda={self.lam:g} (eff {self.lam_eff:g}, {cfg.arm.lambda_scale}) "
              f"laplacian={self.lap_source} k={self.k} rebuild={self.rebuild_every} | "
              f"select on {self.select_split}:{self.select_metric} | params={count_params(self.model):,}")
        if cfg.warmstart.load:
            self.load_warmstart(cfg.warmstart.load)
        else:
            self.loss.set_active({"bpr", "l2"})

        t_start = time.time()
        for epoch in tqdm(range(self.start_epoch, self.E), desc=f"{self.ds.name}/{cfg.arm.name}/s{cfg.seed}",
                          initial=self.start_epoch, total=self.E):
            if epoch == self.W and self.start_epoch != self.W:
                self._on_transition(epoch)
            elif epoch == self.W and self.start_epoch == self.W:
                # resumed exactly at W: reference snapshot came from the checkpoint
                self._build_laplacian(epoch)
                self.loss.set_active({"bpr", "l2"} | ({"manifold"} if self.lam > 0 else set()))
            elif (self.lam > 0 and self.lap_source == "embeddings" and epoch > self.W
                  and self.rebuild_every > 0 and (epoch - self.W) % self.rebuild_every == 0):
                self._build_laplacian(epoch)

            phase = "warmup" if epoch < self.W else ("mr" if self.lam > 0 else "bpr")
            losses = self.train_epoch(epoch)

            is_last = epoch == self.E - 1
            end_of_warmup = epoch == self.W - 1
            if (epoch + 1) % self.eval_every == 0 or is_last or end_of_warmup:
                rec = self.evaluate_point(epoch, phase, losses)
                rec["elapsed_s"] = time.time() - t_start
                self.history.append(rec)
                json_dump(self.history, self.run_dir / "history.json")
                score = self._selection_score(rec)
                improved = score > self.best["score"]
                if improved:
                    self.best = {"score": score, "epoch": epoch + 1}
                    self.evals_since_best = 0
                    self.save_best(epoch, rec)
                else:
                    self.evals_since_best += 1
                msg = (f"ep {epoch+1:4d} [{phase:6s}] loss={losses.get('total', 0):.4f}"
                       + (f" mani={losses['manifold']:.3e}" if "manifold" in losses else "")
                       + f" | val {self.select_metric}={rec.get(f'val_{self.select_metric}', 0):.4f}"
                       + (f" | test {self.select_metric}={rec[f'test_{self.select_metric}']:.4f}"
                          if f"test_{self.select_metric}" in rec else "")
                       + (f" | ER={rec['effective_rank']:.1f}" if "effective_rank" in rec else "")
                       + (" *" if improved else ""))
                tqdm.write(msg)
                if self.wandb is not None:
                    self.wandb.log(rec, step=epoch + 1)
                if self.patience > 0 and self.evals_since_best >= self.patience:
                    tqdm.write(f"  early stop at epoch {epoch+1} ({self.patience} evals without improvement). "
                               f"NOTE: early stopping breaks budget matching across arms.")
                    self.history_event(epoch + 1, "early_stop")
                    break

        # -------------------------------------------------- final at best ckpt
        return self.finalize(time.time() - t_start)

    @torch.no_grad()
    def finalize(self, train_seconds: float) -> dict:
        ck = torch.load(self.run_dir / "best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        au, ai = self.model.computer()
        g = self.cfg.geometry
        val = evaluate(self.model, self.ds, "val", self.topks, self.eval_bs, self.device, au, ai)
        test = evaluate(self.model, self.ds, "test", self.topks, self.eval_bs, self.device, au, ai)

        geom = {"effective_rank": effective_rank(torch.cat([au, ai]), int(g.er_max_samples), seed=int(g.np_seed)),
                "effective_rank_pct": None, "effective_rank_transition": self.er_transition}
        geom["effective_rank_pct"] = 100.0 * geom["effective_rank"] / self.model.dim
        if self.ref_ego is not None:
            ego = self.model.ego_embeddings()
            for k in list(g.np_k):
                geom[f"np_vs_ref@{k}"] = neighborhood_preservation(
                    self.ref_ego, ego, k=int(k), max_samples=int(g.np_max_samples),
                    seed=int(g.np_seed), backend=self.knn_backend)
        for k in list(g.np_ref_k):
            geom[f"np_ref@{k}"] = np_ref(ai, self.ds.UserItemNet, k=int(k), max_samples=int(g.np_max_samples),
                                         seed=int(g.np_seed), backend=self.knn_backend)

        result = {
            "dataset": self.ds.name, "arm": str(self.cfg.arm.name),
            "arm_tag": str(self.cfg.paths.get("arm_tag", self.cfg.arm.name)), "seed": int(self.cfg.seed),
            "best_epoch": int(ck["epoch"]), "selected_on": f"{self.select_split}:{self.select_metric}",
            "epochs_budget": self.E, "warmup_epochs": self.W,
            "lambda_manifold": self.lam, "lambda_effective": self.lam_eff,
            "n_batches_per_epoch": self.nb,
            "val": val, "test": test, "geometry": geom,
            "train_seconds": train_seconds,
            "dataset_summary": self.ds.summary(),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "torch": torch.__version__,
            "device": str(self.device) + (f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else ""),
        }
        json_dump(result, self.run_dir / "results.json")
        m = self.select_metric
        nd = m.replace("recall", "ndcg") if m.startswith("recall") else m
        print(f"\n== done: best epoch {ck['epoch']} | val {m}={val.get(m, 0):.4f} | "
              f"test {m}={test.get(m, 0):.4f} {nd}={test.get(nd, 0):.4f} | "
              f"ER={geom['effective_rank']:.2f} | "
              + " ".join(f"{k}={v:.4f}" for k, v in geom.items() if k.startswith("np_ref")))
        if self.wandb is not None:
            self.wandb.summary.update({"best_epoch": ck["epoch"], **{f"final_test_{k}": v for k, v in test.items()},
                                       **{f"final_{k}": v for k, v in geom.items() if v is not None}})
        return result
