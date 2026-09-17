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


def test_selection_window_is_symmetric(synth, tmp_path):
    """A standalone control and a resumed treatment arm must have the SAME set
    of eligible checkpoints. Regression: `best` was not restored on resume, so
    a standalone arm could select from the shared warm-up prefix while a
    resumed arm could not — strictly more candidates for the control.
    """
    import json
    ca = _cfg(synth, "mr_off", tmp_path / "a")
    ca.eval.every = 1
    ca.warmstart.save = True
    a, _ = _run(ca)
    ws = Path(ca.paths.run_dir) / "warmstart_ep2.pt"
    cb = _cfg(synth, "emb_mr", tmp_path / "b")
    cb.eval.every = 1
    cb.warmstart.load = str(ws)
    b, _ = _run(cb)
    W = int(ca.arm.warmup_epochs)
    # neither arm may select a checkpoint from the shared prefix
    assert a["best_epoch"] > W and b["best_epoch"] > W
    assert a["select_from_epoch"] == b["select_from_epoch"] == W


def test_last_epoch_reading_is_reported(synth, tmp_path):
    """Geometry/diversity are not what selection optimises, so a fixed-epoch
    reading must also be available."""
    res, _ = _run(_cfg(synth, "emb_mr", tmp_path))
    al = res["at_last_epoch"]
    assert al["epoch"] == res["epochs_budget"]
    assert "er_table" in al and "er_prop" in al
    assert any(k.startswith("test_") for k in al) and any(k.startswith("val_") for k in al)


def test_baseline_w0_selects_from_all_epochs(synth, tmp_path):
    cfg = _cfg(synth, "baseline", tmp_path)
    cfg.arm.warmup_epochs = 0
    res, _ = _run(cfg)
    assert res["select_from_epoch"] == 0


def test_selection_window_rejects_a_prefix_peak(synth, tmp_path):
    """The shared prefix must be ineligible even when it holds the best score.

    The natural toy curve rises monotonically, so `best_epoch` lands on the
    last epoch whether or not the window is enforced -- a test built on it
    passes even with the eligibility check deleted. This scripts a curve that
    peaks *during* warm-up and then decays, which is the case the window
    exists for (late-training decay, where the control would otherwise report
    a prefix checkpoint the resumed treatment arm cannot reach).
    """
    from src.trainer import Trainer

    cfg = _cfg(synth, "mr_off", tmp_path)
    cfg.eval.every = 1
    W = int(cfg.arm.warmup_epochs)
    assert 0 < W < int(cfg.epochs)

    scripted = {}   # epoch (1-based) -> val score; peak inside the prefix

    class Scripted(Trainer):
        def evaluate_point(self, epoch, phase, losses):
            rec = super().evaluate_point(epoch, phase, losses)
            score = 1.0 - 0.1 * abs((epoch + 1) - 1)      # max at epoch 1
            rec[f"val_{self.select_metric}"] = score
            scripted[epoch + 1] = score
            return rec

    set_seed(int(cfg.seed))
    ds = InteractionDataset("synthetic", root=cfg.paths.data, val_frac=0.1,
                            split_seed=2020, verbose=False)
    dev = torch.device("cpu")
    model = LightGCN(ds.n_users, ds.n_items, dim=cfg.model.dim,
                     n_layers=cfg.model.n_layers, graph=ds.sparse_graph(dev))
    res = Scripted(cfg, ds, model, dev, cfg.paths.run_dir).run()

    best_overall = max(scripted, key=scripted.get)
    assert best_overall <= W, "fixture must put the global peak inside the prefix"
    assert res["best_epoch"] > W, (
        f"selected epoch {res['best_epoch']} is inside the shared prefix "
        f"(W={W}); the eligibility window is not being applied")
    eligible = {e: v for e, v in scripted.items() if e > W}
    assert res["best_epoch"] == max(eligible, key=eligible.get)


def test_resume_trains_exactly_the_remaining_budget(synth, tmp_path):
    """P0 #1: a resumed arm must train E - W epochs, never E.

    This is the budget confound the whole corrected protocol exists to remove:
    in the notebook the MR arm loaded a checkpoint and then trained a *full*
    extra run on top of it. Asserting on `epochs_budget` alone cannot catch a
    regression here, because that field just echoes the config; the number of
    optimiser passes actually taken is what matters, so it is counted.
    """
    from src.trainer import Trainer

    ca = _cfg(synth, "mr_off", tmp_path / "ctl")
    ca.warmstart.save = True
    E, W = int(ca.epochs), int(ca.arm.warmup_epochs)
    assert 0 < W < E

    counts = {}

    class Counting(Trainer):
        def train_epoch(self, epoch):
            key = str(self.cfg.arm.name)
            counts[key] = counts.get(key, 0) + 1
            return super().train_epoch(epoch)

    def run(cfg):
        set_seed(int(cfg.seed))
        ds = InteractionDataset("synthetic", root=cfg.paths.data, val_frac=0.1,
                                split_seed=2020, verbose=False)
        dev = torch.device("cpu")
        m = LightGCN(ds.n_users, ds.n_items, dim=cfg.model.dim,
                     n_layers=cfg.model.n_layers, graph=ds.sparse_graph(dev))
        return Counting(cfg, ds, m, dev, cfg.paths.run_dir).run()

    run(ca)
    assert counts["mr_off"] == E, f"control trained {counts['mr_off']} epochs, expected {E}"

    cb = _cfg(synth, "emb_mr", tmp_path / "mr")
    cb.warmstart.load = str(Path(ca.paths.run_dir) / f"warmstart_ep{W}.pt")
    res = run(cb)
    assert counts["emb_mr"] == E - W, (
        f"resumed arm trained {counts['emb_mr']} epochs; expected {E - W} "
        f"(E={E}, W={W}). Training E epochs on top of a warm-start is exactly "
        f"the budget confound the corrected protocol removes.")
    assert counts["mr_off"] == counts["emb_mr"] + W   # total budget matched


def test_mr_arm_refuses_a_warmup_that_never_ends(synth, tmp_path):
    """W == E with lambda > 0 would train pure BPR while labelling itself an MR
    arm, because the transition fires at epoch W and there is no epoch W."""
    from src.trainer import Trainer
    cfg = _cfg(synth, "emb_mr", tmp_path)
    cfg.arm.warmup_epochs = int(cfg.epochs)
    set_seed(int(cfg.seed))
    ds = InteractionDataset("synthetic", root=cfg.paths.data, val_frac=0.1,
                            split_seed=2020, verbose=False)
    dev = torch.device("cpu")
    m = LightGCN(ds.n_users, ds.n_items, dim=cfg.model.dim,
                 n_layers=cfg.model.n_layers, graph=ds.sparse_graph(dev))
    with pytest.raises(ValueError, match="MR phase would never start"):
        Trainer(cfg, ds, m, dev, cfg.paths.run_dir)
    # the same schedule is legitimate for a control arm (lambda = 0)
    cfg2 = _cfg(synth, "mr_off", tmp_path / "ctl")
    cfg2.arm.warmup_epochs = int(cfg2.epochs)
    m2 = LightGCN(ds.n_users, ds.n_items, dim=cfg2.model.dim,
                  n_layers=cfg2.model.n_layers, graph=ds.sparse_graph(dev))
    Trainer(cfg2, ds, m2, dev, cfg2.paths.run_dir)


def test_at_last_epoch_carries_geometry_including_np(synth, tmp_path):
    """The selection-independent reading must include the NP metrics, or H1/H3
    geometry claims have no fixed-epoch counterpart."""
    res, _ = _run(_cfg(synth, "emb_mr", tmp_path))
    g = res["at_last_epoch"]["geometry"]
    for key in ("er_table", "er_prop", "np_ref@5"):
        assert key in g and g[key] is not None, f"{key} missing from at_last_epoch"
    assert any(k.startswith("np_vs_ref@") for k in g)
    # it is a *different* reading from the selected checkpoint unless they coincide
    assert set(g) == set(res["geometry"])


def test_resume_rejects_a_different_split(synth, tmp_path):
    """Regression: only name/val_frac/split_seed were compared, so a checkpoint
    built with a different min_train resumed silently into another split."""
    from src.trainer import Trainer
    ca = _cfg(synth, "mr_off", tmp_path / "a")
    ca.warmstart.save = True
    set_seed(int(ca.seed))
    ds_a = InteractionDataset("synthetic", root=ca.paths.data, val_frac=0.1,
                              split_seed=2020, min_train=2, verbose=False)
    dev = torch.device("cpu")
    m = LightGCN(ds_a.n_users, ds_a.n_items, dim=ca.model.dim,
                 n_layers=ca.model.n_layers, graph=ds_a.sparse_graph(dev))
    Trainer(ca, ds_a, m, dev, ca.paths.run_dir).run()
    ws = Path(ca.paths.run_dir) / f"warmstart_ep{int(ca.arm.warmup_epochs)}.pt"
    assert ws.exists()

    ds_b = InteractionDataset("synthetic", root=ca.paths.data, val_frac=0.1,
                              split_seed=2020, min_train=15, verbose=False)
    assert len(ds_b.trainUser) != len(ds_a.trainUser), "fixture must change the split"
    cb = _cfg(synth, "emb_mr", tmp_path / "b")
    cb.warmstart.load = str(ws)
    m2 = LightGCN(ds_b.n_users, ds_b.n_items, dim=cb.model.dim,
                  n_layers=cb.model.n_layers, graph=ds_b.sparse_graph(dev))
    t = Trainer(cb, ds_b, m2, dev, cb.paths.run_dir)
    with pytest.raises(ValueError, match="different split"):
        t.load_warmstart(ws)
