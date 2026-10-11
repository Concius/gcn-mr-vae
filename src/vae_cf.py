"""VAE-CF baseline (Mult-VAE^PR, Liang et al. 2018) under this repo's protocol.

A committed baseline (qualification §4.3.2). It consumes the user's training
interaction vector r_u directly; it is not LightGCN with a VAE attached, so it
has its own model class and this short loop. GCN-VAE / GCN-MR-VAE will *not*
get a loop of their own: they attach to ``Trainer`` as a loss term (Phase 3).

Same protocol as every LightGCN arm, so the numbers are comparable:

* same split (held-out items of known users; validation carved from train
  with the shared ``split_seed``), same ``evaluate()``, same masking of known
  positives, same diversity metrics;
* the encoder input at evaluation is the user's *reduced* training row --
  exactly the interactions LightGCN's graph is built from; validation items
  are never fed to the model, for val or for test scoring;
* checkpoint selection on validation Recall@20 over evaluations every
  ``eval.every`` epochs, test scored at the selected checkpoint;
* the same ``epochs`` budget E. Liang's beta annealing is per *update*
  (cap 0.2 reached after 40,000 updates); at 500 users per batch that is about
  epoch 670 on Gowalla, 630 on Yelp2018 and 380 on Amazon-Book, inside E=1000.
  (Liang et al. trained 200 epochs on datasets several times larger -- ML-20M:
  ~117k training users, ~47k updates -- i.e. comparable update counts.)

Not Liang's protocol: they hold out whole users and select on NDCG@100. That
difference is deliberate -- the comparison that matters is against the other
arms of this study, on this split.

Logged at every evaluation (and in results.json at the selected checkpoint and
at the last epoch): total KL, per-dimension KL, active units, beta, the
reconstruction NLL on training rows, and the effective rank of the users'
posterior means.

RNG: user order comes from ``stream_rng(seed, USER_BATCH_STREAM)``; dropout
masks and the reparameterisation noise from a ``torch.Generator`` seeded from
``(seed, VAE_NOISE_STREAM)``. Evaluation and diagnostics draw nothing from
either, so evaluation frequency cannot change the trajectory (tested).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .metrics.evaluate import evaluate
from .metrics.geometry import effective_rank, np_ref
from .models.vae import (BetaSchedule, MultVAE, gaussian_kl, kl_with_free_bits, latent_diagnostics,
                         multinomial_nll)
from .sampling import (USER_BATCH_STREAM, VAE_NOISE_STREAM, dense_rows, n_batches, stream_rng,
                       stream_torch_generator, user_batches, users_with_interactions)
from .utils import count_params, json_dump


def build_vae_cf(cfg: DictConfig, n_items: int) -> MultVAE:
    v = cfg.arm.vae
    return MultVAE(n_items, hidden=int(v.hidden), latent=int(v.latent), dropout=float(v.dropout))


class VAECFTrainer:
    def __init__(self, cfg: DictConfig, dataset, model: MultVAE, device: torch.device,
                 run_dir: str | os.PathLike, wandb_run=None):
        self.cfg, self.ds, self.device = cfg, dataset, device
        self.model = model.to(device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.wandb = wandb_run
        a, v = cfg.arm, cfg.arm.vae
        if float(a.lambda_manifold) != 0.0 or int(a.warmup_epochs) != 0:
            raise ValueError("arm=vae_cf has no manifold term and no warm-up phase "
                             "(lambda_manifold must be 0 and warmup_epochs 0)")
        if cfg.warmstart.load or bool(cfg.warmstart.save):
            raise ValueError("warm-start save/load is not supported for the VAE-CF baseline")
        self.E = int(cfg.epochs)
        self.batch_size = int(v.batch_size)
        self.weight_decay = float(v.get("weight_decay", 0.0))
        self.free_bits = float(v.get("free_bits", 0.0))
        self.au_threshold = float(v.get("au_threshold", 0.01))
        b = v.beta
        self.beta = BetaSchedule(schedule=str(b.schedule), cap=float(b.cap),
                                 anneal_steps=int(b.get("anneal_steps", 200_000)),
                                 cycle_steps=int(b.get("cycle_steps", 10_000)),
                                 ratio=float(b.get("ratio", 0.5)))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(v.lr))
        seed = int(cfg.seed)
        self.user_rng = stream_rng(seed, USER_BATCH_STREAM)
        self.noise = stream_torch_generator(seed, VAE_NOISE_STREAM, device)
        self.train_users = users_with_interactions(dataset.UserItemNet)
        self.nb = n_batches(len(self.train_users), self.batch_size)
        self.step = 0

        ev = cfg.eval
        self.topks = list(ev.topks)
        self.eval_every = int(ev.every)
        self.eval_bs = int(ev.batch_size)
        self.test_each_eval = bool(ev.test_each_eval)
        self.select_split = str(ev.select_split)
        self.select_metric = str(ev.select_metric)
        if int(ev.patience_evals) != 0:
            raise ValueError("early stopping is not supported for the VAE-CF baseline (budget matching)")
        sf = ev.get("select_from_epoch", "auto")
        self.select_from = 0 if str(sf) == "auto" else int(sf)
        if self.select_split == "val" and not dataset.valDict:
            raise ValueError("eval.select_split=val but the dataset has no validation split")
        self.history: list[dict] = []
        self.best = {"score": -1.0, "epoch": -1}

    # ------------------------------------------------------------- training
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        acc = {"total": 0.0, "nll": 0.0, "kl": 0.0, "beta": 0.0}
        n = 0
        for bu in user_batches(self.train_users, self.batch_size, self.user_rng):
            x = dense_rows(self.ds.UserItemNet, bu, self.device)
            logits, mu, logvar = self.model(x, self.noise)
            nll = multinomial_nll(logits, x)
            kl_elem = gaussian_kl(mu, logvar)
            kl = kl_with_free_bits(kl_elem, self.free_bits)
            beta = self.beta(self.step)
            total = nll + beta * kl
            if self.weight_decay > 0:
                total = total + self.weight_decay * self.model.weight_l2()
            self.opt.zero_grad(set_to_none=True)
            total.backward()
            self.opt.step()
            self.step += 1
            acc["total"] += total.item()
            acc["nll"] += nll.item()
            acc["kl"] += kl_elem.detach().sum(dim=1).mean().item()   # plain KL, also with free bits
            acc["beta"] += beta
            n += 1
        return {k: val / n for k, val in acc.items()}

    # ----------------------------------------------------------- evaluation
    def scorer(self):
        csr, model, dev = self.ds.UserItemNet, self.model, self.device

        def score(users: torch.Tensor) -> torch.Tensor:
            x = dense_rows(csr, users.cpu().numpy(), dev)
            return model.scores(x)          # posterior mean; fresh tensor
        return score

    @torch.no_grad()
    def _posteriors(self):
        self.model.eval()
        mus, lvs = [], []
        for s in range(0, len(self.train_users), self.eval_bs):
            bu = self.train_users[s:s + self.eval_bs]
            mu, lv = self.model.encode(dense_rows(self.ds.UserItemNet, bu, self.device), None)
            mus.append(mu)
            lvs.append(lv)
        return torch.cat(mus), torch.cat(lvs)

    @torch.no_grad()
    def _train_nll(self) -> float:
        """Reconstruction NLL of the training rows at the posterior mean."""
        tot, n = 0.0, 0
        for s in range(0, len(self.train_users), self.eval_bs):
            bu = self.train_users[s:s + self.eval_bs]
            x = dense_rows(self.ds.UserItemNet, bu, self.device)
            tot += float(multinomial_nll(self.model.scores(x), x)) * len(bu)
            n += len(bu)
        return tot / n

    @torch.no_grad()
    def latent_report(self, step: int | None = None) -> dict:
        """Collapse diagnostics for the current weights. ``step`` is the update
        count these weights were reached at (default: now); it sets the beta
        that is reported alongside.

        ``er_latent`` (ER of the posterior means) is scale-invariant, so it is
        *not* a collapse diagnostic: a collapsed posterior's tiny residual
        variation can have a high ER (on the 40-epoch synthetic check, beta=5
        gave the highest ER of four settings with 0 active units). Use KL and
        active units.
        """
        step = self.step if step is None else int(step)
        mu, lv = self._posteriors()
        d = latent_diagnostics(mu, lv, self.au_threshold)
        g = self.cfg.geometry
        d["er_latent"] = effective_rank(mu, int(g.er_max_samples), center=bool(g.er_center), seed=int(g.np_seed))
        d["nll_train"] = self._train_nll()
        d["beta"] = self.beta(step)
        d["updates"] = step
        return d

    @torch.no_grad()
    def geometry(self) -> dict:
        """ER and NP_ref of the decoder's item vectors (the VAE-CF analogue of
        item embeddings), and ER of the users' posterior means."""
        g = self.cfg.geometry
        items = self.model.item_vectors().detach()
        out = {"item_repr": "decoder output weights (n_items x hidden)",
               "er_item": effective_rank(items, int(g.er_max_samples), center=bool(g.er_center),
                                         seed=int(g.np_seed))}
        for k in list(g.np_ref_k):
            out[f"np_ref@{k}"] = np_ref(items, self.ds.UserItemNet, k=int(k),
                                        max_samples=int(g.np_max_samples), seed=int(g.np_seed),
                                        backend=str(self.cfg.arm.knn_backend))
        return out

    @torch.no_grad()
    def evaluate_point(self, epoch: int, losses: dict) -> dict:
        self.model.eval()
        rec = {"epoch": epoch + 1, "phase": "vae", "event": "eval", **{f"loss_{k}": v for k, v in losses.items()}}
        sc = self.scorer()
        rec.update({f"val_{k}": v for k, v in evaluate(sc, self.ds, "val", self.topks, self.eval_bs, self.device).items()})
        if self.test_each_eval:
            rec.update({f"test_{k}": v for k, v in evaluate(sc, self.ds, "test", self.topks, self.eval_bs, self.device).items()})
        lat = self.latent_report()
        rec.update({f"vae_{k}": v for k, v in lat.items() if k != "mu_var_per_dim"})
        self.model.train()
        return rec

    def save_best(self, epoch: int, rec: dict) -> None:
        torch.save({"epoch": epoch + 1, "model": self.model.state_dict(), "record": rec,
                    "updates": self.step, "model_meta": self.model.state_for_checkpoint()},
                   self.run_dir / "best.pt")

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        cfg = self.cfg
        print(f"\n== {self.ds.name} | arm={cfg.arm.name} (Mult-VAE^PR) | seed={cfg.seed} | E={self.E} | "
              f"users/batch={self.batch_size} ({self.nb} updates/epoch) | beta {self.beta.schedule} "
              f"cap={self.beta.cap:g} | select on {self.select_split}:{self.select_metric} | "
              f"params={count_params(self.model):,}")
        t0 = time.time()
        for epoch in tqdm(range(self.E), desc=f"{self.ds.name}/{cfg.arm.name}/s{cfg.seed}"):
            losses = self.train_epoch(epoch)
            if (epoch + 1) % self.eval_every == 0 or epoch == self.E - 1:
                rec = self.evaluate_point(epoch, losses)
                rec["elapsed_s"] = time.time() - t0
                self.history.append(rec)
                json_dump(self.history, self.run_dir / "history.json")
                key = f"{self.select_split}_{self.select_metric}"
                if key not in rec:
                    raise KeyError(f"selection metric {key} not in evaluation record")
                score = float(rec[key])
                eligible = epoch >= self.select_from
                improved = eligible and score > self.best["score"]
                if improved:
                    self.best = {"score": score, "epoch": epoch + 1}
                    self.save_best(epoch, rec)
                tqdm.write(f"ep {epoch+1:4d} [vae] loss={losses['total']:.4f} nll={losses['nll']:.3f} "
                           f"kl={rec['vae_kl']:.2f} AU={rec['vae_active_units']}/{rec['vae_latent_dim']} "
                           f"beta={rec['vae_beta']:.3f} | val {self.select_metric}={rec.get(f'val_{self.select_metric}', 0):.4f}"
                           + (f" | test {self.select_metric}={rec[f'test_{self.select_metric}']:.4f}"
                              if f"test_{self.select_metric}" in rec else "")
                           + (" *" if improved else ""))
                if self.wandb is not None:
                    self.wandb.log(rec, step=epoch + 1)
        return self.finalize(time.time() - t0)

    @torch.no_grad()
    def finalize(self, train_seconds: float) -> dict:
        if not (self.run_dir / "best.pt").exists():
            raise RuntimeError("no checkpoint was selected; check eval.every against epochs")
        geom_last = self.geometry()
        lat_last = self.latent_report()
        ck = torch.load(self.run_dir / "best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        sc = self.scorer()
        val = evaluate(sc, self.ds, "val", self.topks, self.eval_bs, self.device)
        test = evaluate(sc, self.ds, "test", self.topks, self.eval_bs, self.device)
        geom = self.geometry()
        lat = self.latent_report(step=int(ck["updates"]))   # beta in force at the checkpoint

        evals = [r for r in self.history if r.get("event") == "eval"]
        last = evals[-1] if evals else {}
        at_last = {k: v for k, v in last.items() if k.startswith(("val_", "test_")) or k == "epoch"}
        at_last["geometry"] = geom_last
        at_last["vae"] = lat_last

        result = {
            "dataset": self.ds.name, "arm": str(self.cfg.arm.name),
            "arm_tag": str(self.cfg.paths.get("arm_tag", self.cfg.arm.name)), "seed": int(self.cfg.seed),
            "best_epoch": int(ck["epoch"]), "selected_on": f"{self.select_split}:{self.select_metric}",
            "epochs_budget": self.E, "warmup_epochs": 0,
            "lambda_manifold": 0.0, "lambda_effective": 0.0,
            "n_batches_per_epoch": self.nb,
            "val": val, "test": test, "geometry": geom, "vae": lat, "at_last_epoch": at_last,
            "select_from_epoch": self.select_from, "train_seconds": train_seconds,
            "dataset_summary": self.ds.summary(),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "torch": torch.__version__,
            "device": str(self.device) + (f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else ""),
        }
        json_dump(result, self.run_dir / "results.json")
        m = self.select_metric
        print(f"\n== done: best epoch {ck['epoch']} | val {m}={val.get(m, 0):.4f} | test {m}={test.get(m, 0):.4f} | "
              f"KL={lat['kl']:.2f} AU={lat['active_units']}/{lat['latent_dim']} | ER_latent={lat['er_latent']:.1f} "
              + " ".join(f"{k}={v:.4f}" for k, v in geom.items() if k.startswith("np_ref")))
        if self.wandb is not None:
            self.wandb.summary.update({"best_epoch": ck["epoch"], **{f"final_test_{k}": v for k, v in test.items()}})
        return result
