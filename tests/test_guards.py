"""Guards that protect results but previously had no automated test.

Each was verified by hand when built; a mutation sweep on 3 Oct 2026 showed a
future edit could break any of them silently. Every test here is checked to
fail when the guard it covers is removed.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import src.train  # noqa: F401  registers the armtag resolver
from analysis.summarize import collect_results, paired_wilcoxon, summary_table
from src.data.dataset import InteractionDataset
from src.models.lightgcn import LightGCN
from src.trainer import Trainer
from src.utils import set_seed
from src.data.synthetic import make_synthetic
from tests.test_smoke import _cfg, _run


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("guarddata")
    make_synthetic(root / "synthetic", n_users=120, n_items=80, seed=0)
    return root


# ------------------------------------------------------------ warm-start
def _resume_into(synth, tmp_path, n_layers, weights_only):
    """Save a warm-start with the default depth, then load it into a model of
    depth ``n_layers``."""
    ca = _cfg(synth, "mr_off", tmp_path / "a")
    ca.warmstart.save = True
    _run(ca)
    ws = Path(ca.paths.run_dir) / f"warmstart_ep{int(ca.arm.warmup_epochs)}.pt"
    cb = _cfg(synth, "emb_mr", tmp_path / "b")
    cb.warmstart.weights_only = weights_only
    set_seed(int(cb.seed))
    ds = InteractionDataset("synthetic", root=cb.paths.data, val_frac=0.1, split_seed=2020, verbose=False)
    m = LightGCN(ds.n_users, ds.n_items, dim=cb.model.dim, n_layers=n_layers,
                 graph=ds.sparse_graph(torch.device("cpu")))
    return Trainer(cb, ds, m, torch.device("cpu"), cb.paths.run_dir), ws, int(ca.model.n_layers)


def test_resume_rejects_a_different_architecture(synth, tmp_path):
    t, ws, depth = _resume_into(synth, tmp_path, n_layers=99, weights_only=False)
    assert depth != 99
    with pytest.raises(ValueError, match="different architecture"):
        t.load_warmstart(ws)


def test_weights_only_resume_still_validates(synth, tmp_path):
    """The notebook-protocol hand-off must not skip the checks."""
    t, ws, _ = _resume_into(synth, tmp_path, n_layers=99, weights_only=True)
    with pytest.raises(ValueError, match="different architecture"):
        t.load_warmstart(ws)


# ------------------------------------------------------------ summarize
def _write(root, ds, tag, arm, seed, recall, folder=None):
    p = Path(root, ds, tag, folder or f"seed{seed}")
    p.mkdir(parents=True)
    json.dump({"dataset": ds, "arm": arm, "arm_tag": tag, "seed": seed, "best_epoch": 1,
               "selected_on": "val:recall@20", "epochs_budget": 1, "warmup_epochs": 0,
               "lambda_manifold": 0.0, "test": {"recall@20": recall}, "val": {},
               "geometry": {"er_table": 1.0}}, open(p / "results.json", "w"))


def test_summarize_refuses_an_ambiguous_arm_name(tmp_path):
    for s in range(2020, 2025):
        _write(tmp_path, "g", "mr_off_cfg", "mr_off", s, 0.10)
        _write(tmp_path, "g", "emb_mr_cfg_one", "emb_mr", s, 0.11)
        _write(tmp_path, "g", "emb_mr_cfg_two", "emb_mr", s, 0.12)
    df = collect_results(tmp_path)
    with pytest.raises(ValueError, match="matches 2 configurations"):
        paired_wilcoxon(df, "emb_mr", "mr_off", "test_recall@20")
    # the exact tag is accepted
    out = paired_wilcoxon(df, "emb_mr_cfg_one", "mr_off_cfg", "test_recall@20")
    assert int(out["n"].iloc[0]) == 5


def test_summary_refuses_repeated_seeds_in_one_group(tmp_path):
    """Two results claiming the same seed in one group would be averaged as one."""
    _write(tmp_path, "g", "emb_mr_cfg", "emb_mr", 2020, 0.11, folder="seed2020")
    _write(tmp_path, "g", "emb_mr_cfg", "emb_mr", 2020, 0.13, folder="seed2020_rerun")
    with pytest.raises(ValueError, match="repeated seeds"):
        summary_table(collect_results(tmp_path), metrics=["test_recall@20"])


def test_holm_corrects_only_over_tested_rows(tmp_path):
    rng = np.random.default_rng(0)
    for s in range(2020, 2025):          # testable dataset: 5 paired seeds
        _write(tmp_path, "g1", "off", "mr_off", s, 0.10)
        _write(tmp_path, "g1", "mr", "emb_mr", s, 0.11 + 0.001 * rng.random())
    for s in range(2020, 2022):          # untestable dataset: 2 paired seeds
        _write(tmp_path, "g2", "off", "mr_off", s, 0.10)
        _write(tmp_path, "g2", "mr", "emb_mr", s, 0.11)
    out = paired_wilcoxon(collect_results(tmp_path), "emb_mr", "mr_off", "test_recall@20")
    tested = out[out["p"].notna()]
    assert len(tested) == 1 and int(out["n_tests_corrected"].dropna().iloc[0]) == 1
    assert tested["p_holm"].iloc[0] == pytest.approx(tested["p"].iloc[0])


# ------------------------------------------------------------ clobber guard
def test_clobber_guard(tmp_path):
    from src.train import _assert_no_clobber
    base = OmegaConf.create({"seed": 2020, "optim": {"lr": 0.001},
                             "paths": {"run_dir": str(tmp_path)}, "logging": {"wandb": False}})
    OmegaConf.save(base, tmp_path / "config.yaml")
    _assert_no_clobber(base, tmp_path)                       # identical: allowed
    benign = OmegaConf.merge(base, {"paths": {"run_dir": "elsewhere"}, "logging": {"wandb": True}})
    _assert_no_clobber(benign, tmp_path)                     # ignored sections: allowed
    different = OmegaConf.merge(base, {"optim": {"lr": 0.05}})
    with pytest.raises(RuntimeError, match="optim.lr"):
        _assert_no_clobber(different, tmp_path)
