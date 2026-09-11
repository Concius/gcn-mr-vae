"""Entry point.

    python -m src.train dataset=gowalla arm=emb_mr seed=2020
    python -m src.train -m dataset=gowalla arm=mr_off,emb_mr seed=2020,2021,2022,2023,2024

Run artifacts land in ``runs/<dataset>/<arm>/seed<seed>/``:
``config.yaml``, ``history.json``, ``best.pt``, ``results.json`` and, if
``warmstart.save=true``, ``warmstart_ep<W>.pt``.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .data.dataset import InteractionDataset
from .models.lightgcn import LightGCN
from .trainer import Trainer
from .utils import get_device, set_seed


def _arm_tag(name: str, lam: float, k: int, scale: str, warmup: int) -> str:
    """Directory-safe tag so sweeps over lambda/k/warmup never overwrite each other."""
    lam = float(lam)
    tag = f"{name}_w{int(warmup)}"
    if lam > 0:
        tag += f"_lam{lam:g}_k{int(k)}" + ("_pe" if str(scale) == "per_epoch" else "")
    return tag


OmegaConf.register_new_resolver("armtag", _arm_tag, replace=True)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _maybe_wandb(cfg: DictConfig, run_dir: Path):
    if not bool(cfg.logging.wandb):
        return None
    try:
        import wandb
    except ImportError:
        print("  logging.wandb=true but wandb is not installed; continuing without it")
        return None
    return wandb.init(project=str(cfg.logging.project), entity=cfg.logging.entity,
                      name=f"{cfg.dataset.name}-{cfg.arm.name}-s{cfg.seed}",
                      group=f"{cfg.dataset.name}-{cfg.arm.name}", dir=str(run_dir),
                      config=OmegaConf.to_container(cfg, resolve=True))


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    run_dir = Path(hydra.utils.to_absolute_path(str(cfg.paths.run_dir)))
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    print(OmegaConf.to_yaml(cfg))
    print(f"run_dir: {run_dir}  git: {_git_commit()}")

    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))
    device = get_device(str(cfg.device))

    ds = InteractionDataset(cfg.dataset.name, root=hydra.utils.to_absolute_path(str(cfg.paths.data)),
                            val_frac=float(cfg.split.val_frac), split_seed=int(cfg.split.split_seed),
                            min_train=int(cfg.split.min_train), short_head_frac=float(cfg.split.short_head_frac),
                            popularity_from=str(cfg.split.popularity_from))
    graph = ds.sparse_graph(device)
    model = LightGCN(ds.n_users, ds.n_items, dim=int(cfg.model.dim), n_layers=int(cfg.model.n_layers),
                     init_std=float(cfg.model.init_std), graph=graph)

    wb = _maybe_wandb(cfg, run_dir)
    trainer = Trainer(cfg, ds, model, device, run_dir, wandb_run=wb)
    result = trainer.run()
    if wb is not None:
        wb.finish()
    return float(result["test"][str(cfg.eval.select_metric)])


if __name__ == "__main__":
    main()
