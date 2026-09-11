"""End-to-end on the synthetic dataset, CPU, a handful of epochs.

Checks that every arm runs, that results.json has the expected keys, that
selection happened on val, and that mr_off / emb_mr share the same warm-up.
"""
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.data.dataset import InteractionDataset
from src.data.synthetic import make_synthetic
from src.models.lightgcn import LightGCN
from src.trainer import Trainer
import src.train  # registers the armtag resolver
from src.utils import set_seed


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    make_synthetic(root / "synthetic", n_users=120, n_items=80, seed=0)
    return root


def _cfg(root, arm, tmp, **over):
    base = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "config.yaml")
    arm_cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "arm" / f"{arm}.yaml")
    cfg = OmegaConf.merge(base, {"dataset": {"name": "synthetic"}, "arm": OmegaConf.to_container(arm_cfg)})
    cfg = OmegaConf.merge(cfg, {
        "seed": 2020, "epochs": 6, "device": "cpu",
        "model": {"dim": 8, "n_layers": 2},
        "optim": {"batch_size": 256},
        "eval": {"every": 2, "batch_size": 64, "topks": [5, 10], "select_metric": "recall@10"},
        "geometry": {"np_max_samples": 50, "er_max_samples": 100, "np_k": [5], "np_ref_k": [5]},
        "arm": {"warmup_epochs": 2, "rebuild_every": 2, "k_neighbors": 5},
        "paths": {"data": str(root), "runs": str(tmp), "arm_tag": arm, "run_dir": str(tmp / arm)},
        "hydra": None,
    })
    cfg = OmegaConf.merge(cfg, over)
    OmegaConf.resolve(cfg)
    return cfg


def _run(cfg):
    set_seed(int(cfg.seed))
    ds = InteractionDataset("synthetic", root=cfg.paths.data, val_frac=0.1, split_seed=2020, verbose=False)
    dev = torch.device("cpu")
    model = LightGCN(ds.n_users, ds.n_items, dim=cfg.model.dim, n_layers=cfg.model.n_layers, graph=ds.sparse_graph(dev))
    return Trainer(cfg, ds, model, dev, cfg.paths.run_dir).run(), ds


@pytest.mark.parametrize("arm", ["baseline", "mr_off", "emb_mr", "coocc_mr"])
def test_arm_runs(synth, tmp_path, arm):
    res, ds = _run(_cfg(synth, arm, tmp_path))
    assert res["selected_on"] == "val:recall@10"
    for key in ("recall@10", "ndcg@10", "gini@10", "tail_share_user@10", "tail_catalog_coverage@10"):
        assert key in res["test"] and key in res["val"]
    assert "effective_rank" in res["geometry"] and "np_ref@5" in res["geometry"]
    assert (Path(res["config"]["paths"]["run_dir"]) / "best.pt").exists()
    hist = json.load(open(Path(res["config"]["paths"]["run_dir"]) / "history.json"))
    evals = [h for h in hist if h.get("event") == "eval"]
    assert len(evals) >= 3 and all("val_recall@10" in h for h in evals)
    if arm in ("emb_mr", "coocc_mr"):
        assert any(h.get("event") == "laplacian_built" for h in hist)
        assert any(h.get("loss_manifold", 0) > 0 for h in evals if h["phase"] == "mr")
    if arm == "mr_off":
        assert all(h["phase"] in ("warmup", "bpr") for h in evals)


def test_mr_off_and_emb_mr_share_warmup(synth, tmp_path):
    a, _ = _run(_cfg(synth, "mr_off", tmp_path, eval={"every": 1}))
    b, _ = _run(_cfg(synth, "emb_mr", tmp_path, eval={"every": 1}))
    ha = [h for h in json.load(open(Path(a["config"]["paths"]["run_dir"]) / "history.json")) if h.get("event") == "eval"]
    hb = [h for h in json.load(open(Path(b["config"]["paths"]["run_dir"]) / "history.json")) if h.get("event") == "eval"]
    # first two epochs (warm-up) must be identical between the control and the MR arm
    for x, y in zip(ha[:2], hb[:2]):
        assert x["phase"] == y["phase"] == "warmup"
        assert x["val_recall@10"] == pytest.approx(y["val_recall@10"], abs=1e-6)
    for conv in ("er_table_at_ref", "er_prop_at_ref"):
        assert a["geometry"][conv] == pytest.approx(b["geometry"][conv], abs=1e-4)


def test_warmstart_save_then_load(synth, tmp_path):
    cfg_a = _cfg(synth, "mr_off", tmp_path / "a", warmstart={"save": True})
    a, _ = _run(cfg_a)
    ws = Path(cfg_a.paths.run_dir) / "warmstart_ep2.pt"
    assert ws.exists()
    cfg_b = _cfg(synth, "emb_mr", tmp_path / "b", warmstart={"load": str(ws)})
    b, _ = _run(cfg_b)
    hb = json.load(open(Path(cfg_b.paths.run_dir) / "history.json"))
    assert min(h["epoch"] for h in hb if h.get("event") == "eval") <= 2   # inherited warm-up history
    assert b["best_epoch"] >= 1 and b["epochs_budget"] == 6


def test_selecting_on_test_is_refused_silently_never(synth, tmp_path):
    cfg = _cfg(synth, "baseline", tmp_path, eval={"select_split": "test"})
    res, _ = _run(cfg)
    assert res["selected_on"].startswith("test:")   # allowed only when asked for explicitly


def test_w0_captures_initial_reference(synth, tmp_path):
    """H1 needs NP and ER measured against the random initialisation.

    Regression test: with warmup_epochs=0 the trainer used to skip the
    reference snapshot entirely, silently dropping np_vs_ref and the ER
    baseline from every W=0 arm.
    """
    cfg = _cfg(synth, "baseline", tmp_path)
    cfg.arm.warmup_epochs = 0
    res, _ = _run(cfg)
    g = res["geometry"]
    assert g["er_table_at_ref"] is not None and g["er_prop_at_ref"] is not None
    assert "np_vs_ref@5" in g
    # a random init table is near-isotropic, so ER_tab at the reference is close to d
    assert g["er_table_at_ref"] > 0.9 * res["config"]["model"]["dim"]
    # training moves the neighbourhood structure away from the init
    assert 0.0 <= g["np_vs_ref@5"] < 1.0
    assert g["delta_er_table"] == pytest.approx(g["er_table"] - g["er_table_at_ref"], abs=1e-6)


def test_both_er_conventions_reported(synth, tmp_path):
    res, _ = _run(_cfg(synth, "mr_off", tmp_path))
    g = res["geometry"]
    for k in ("er_table", "er_prop", "er_table_pct", "er_prop_pct",
              "delta_er_table", "delta_er_prop"):
        assert k in g and g[k] is not None
    # propagation compresses the spectrum further than the table alone
    assert g["er_prop"] <= g["er_table"] + 1e-6
    assert g["effective_rank"] == pytest.approx(g["er_prop"])


def test_popularity_segmentation_independent_of_val_frac(synth):
    from src.data.dataset import InteractionDataset
    a = InteractionDataset("synthetic", root=str(synth), val_frac=0.0, verbose=False)
    b = InteractionDataset("synthetic", root=str(synth), val_frac=0.2, split_seed=7, verbose=False)
    assert a.popularity_groups["short_head"] == b.popularity_groups["short_head"]
    c = InteractionDataset("synthetic", root=str(synth), val_frac=0.2, split_seed=7,
                           popularity_from="train", verbose=False)
    assert c.summary()["popularity_from"] == "train"
