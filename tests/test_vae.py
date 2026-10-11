"""Module 3, step 1: VAE pieces and the VAE-CF baseline (Mult-VAE^PR).

Unit tests pin the objective (KL, multinomial NLL, free bits, beta schedules)
and the collapse diagnostics against independent references; trainer tests
pin the protocol (same evaluate(), encoder fed the reduced training rows only,
selection on validation, RNG isolation); the collapse test deliberately breaks
the VAE (beta = 5) and requires the diagnostics to say so.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from omegaconf import OmegaConf

import src.train  # noqa: F401  registers the armtag resolver
from src.data.dataset import InteractionDataset
from src.data.synthetic import make_synthetic
from src.metrics.evaluate import evaluate
from src.models.vae import (BetaSchedule, MultVAE, gaussian_kl, kl_with_free_bits,
                            latent_diagnostics, multinomial_nll)
from src.sampling import (USER_BATCH_STREAM, dense_rows, stream_rng, stream_torch_generator,
                          user_batches, users_with_interactions)
from src.train import _arm_tag
from src.utils import set_seed
from src.vae_cf import VAECFTrainer, build_vae_cf

REPO = Path(__file__).resolve().parents[1]


# =================================================================== objective
def test_gaussian_kl_matches_torch_distributions():
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(7, 5, generator=g, dtype=torch.float64)
    logvar = torch.randn(7, 5, generator=g, dtype=torch.float64)
    q = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    ref = torch.distributions.kl_divergence(q, p)
    assert torch.allclose(gaussian_kl(mu, logvar), ref, atol=1e-12)
    assert torch.all(gaussian_kl(torch.zeros(3, 2), torch.zeros(3, 2)) == 0)


def test_multinomial_nll_matches_explicit_formula():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(4, 6, generator=g, dtype=torch.float64)
    x = (torch.rand(4, 6, generator=g) < 0.4).double()
    p = torch.softmax(logits, dim=1)
    ref = -(x * torch.log(p)).sum(dim=1).mean()
    assert torch.allclose(multinomial_nll(logits, x), ref, atol=1e-12)
    # one positive per row: the NLL is -log of that item's softmax probability
    one = torch.zeros(1, 6, dtype=torch.float64)
    one[0, 2] = 1
    assert torch.allclose(multinomial_nll(logits[:1], one), -torch.log(p[0, 2]))


def test_free_bits_zero_is_the_plain_kl_and_positive_floors_each_dimension():
    kl = torch.tensor([[0.0, 1.0, 4.0], [0.2, 3.0, 0.0]], requires_grad=True)
    assert torch.isclose(kl_with_free_bits(kl, 0.0), kl.sum(dim=1).mean())
    # per-dim batch means: 0.1, 2.0, 2.0 -> floored at 0.5: 0.5 + 2 + 2
    val = kl_with_free_bits(kl, 0.5)
    assert torch.isclose(val, torch.tensor(4.5))
    val.backward()
    assert torch.all(kl.grad[:, 0] == 0)          # below the floor: no KL gradient
    assert torch.all(kl.grad[:, 1:] == 0.5)        # above it: d mean / d elem = 1/batch


def test_beta_schedules():
    lin = BetaSchedule("linear", cap=0.2, anneal_steps=1000)
    assert lin(0) == 0.0 and lin(100) == pytest.approx(0.1) and lin(200) == pytest.approx(0.2)
    assert lin(10_000) == pytest.approx(0.2)
    liang = BetaSchedule()            # defaults = Liang et al.: cap 0.2 after 40,000 updates
    assert liang(39_999) < 0.2 and liang(40_000) == pytest.approx(0.2)
    assert BetaSchedule("constant", cap=5.0)(0) == 5.0
    cyc = BetaSchedule("cyclical", cap=1.0, cycle_steps=100, ratio=0.5)
    assert cyc(0) == 0.0 and cyc(25) == pytest.approx(0.5) and cyc(50) == 1.0 and cyc(99) == 1.0
    assert cyc(100) == 0.0 and cyc(125) == pytest.approx(0.5)
    for bad in (dict(schedule="nope"), dict(cap=-1), dict(schedule="linear", anneal_steps=0),
                dict(schedule="cyclical", ratio=0.0)):
        with pytest.raises(ValueError):
            BetaSchedule(**bad)


def test_latent_diagnostics_counts_active_units_and_decomposes_kl():
    g = torch.Generator().manual_seed(2)
    n = 4000
    mu = torch.zeros(n, 5, dtype=torch.float64)
    mu[:, 0] = torch.randn(n, generator=g, dtype=torch.float64)            # var ~ 1
    mu[:, 1] = 0.5 * torch.randn(n, generator=g, dtype=torch.float64)      # var ~ 0.25
    mu[:, 2] = 0.2 * torch.randn(n, generator=g, dtype=torch.float64)      # var ~ 0.04
    mu[:, 3] = 0.05 * torch.randn(n, generator=g, dtype=torch.float64)     # var ~ 0.0025 -> inactive
    logvar = torch.zeros(n, 5, dtype=torch.float64)                         # dim 4: exactly the prior
    d = latent_diagnostics(mu, logvar, au_threshold=0.01)
    assert d["active_units"] == 3 and d["latent_dim"] == 5
    assert sum(d["kl_per_dim"]) == pytest.approx(d["kl"])
    assert d["kl_per_dim"][4] == pytest.approx(0.0)
    assert d["kl_per_dim"][0] == pytest.approx(0.5 * float((mu[:, 0] ** 2).mean()))
    # the threshold is strict and population variance is used
    same = torch.full((10, 1), 3.0)
    assert latent_diagnostics(same, torch.zeros(10, 1))["active_units"] == 0


# ======================================================================= model
def test_multvae_eval_pass_is_deterministic_posterior_mean_and_scale_invariant():
    torch.manual_seed(0)
    m = MultVAE(n_items=30, hidden=8, latent=4, dropout=0.5).eval()
    x = (torch.rand(5, 30) < 0.3).float()
    x[0, :3] = 1
    s1, s2 = m.scores(x), m.scores(x)
    assert torch.equal(s1, s2)
    mu, _ = m.encode(x, None)
    assert torch.equal(s1, m.decode(mu))
    assert torch.allclose(m.scores(3.0 * x), s1, atol=1e-6)     # input is L2-normalised


def test_multvae_training_pass_draws_only_from_the_given_generator():
    torch.manual_seed(0)
    m = MultVAE(n_items=30, hidden=8, latent=4, dropout=0.5)
    x = (torch.rand(5, 30) < 0.3).float()
    a = m(x, torch.Generator().manual_seed(7))[0]
    torch.manual_seed(123)                                        # global stream must not matter
    b = m(x, torch.Generator().manual_seed(7))[0]
    c = m(x, torch.Generator().manual_seed(8))[0]
    assert torch.equal(a, b) and not torch.equal(a, c)
    nodrop = MultVAE(n_items=30, hidden=8, latent=4, dropout=0.0)
    nodrop.load_state_dict(m.state_dict())
    mu_d, _ = m.encode(x, torch.Generator().manual_seed(7))
    mu_n, _ = nodrop.encode(x, torch.Generator().manual_seed(7))
    assert not torch.equal(mu_d, mu_n)                            # dropout is applied in training


def test_input_dropout_is_inverted_and_unbiased():
    m = MultVAE(n_items=6, hidden=4, latent=2, dropout=0.5)
    h = torch.tensor([[0.5, 0.25, 0.0, 1.0, 0.75, 0.1]], dtype=torch.float64).repeat(20000, 1)
    out = m.input_dropout(h, torch.Generator().manual_seed(3))
    kept = out != 0
    assert torch.allclose(out[kept], (h * 2)[kept])                  # survivors scaled by 1/keep
    assert abs(kept[:, 0].double().mean().item() - 0.5) < 0.02         # keep probability 0.5
    assert torch.allclose(out.mean(dim=0), h[0], atol=0.02)           # unbiased


def test_multvae_reparameterisation_is_mu_plus_eps_times_std():
    """z = mu + eps * exp(logvar / 2), with eps drawn from the same generator
    right after the dropout mask."""
    torch.manual_seed(0)
    m = MultVAE(n_items=30, hidden=8, latent=4, dropout=0.5)
    x = (torch.rand(5, 30) < 0.3).float()
    logits = m(x, torch.Generator().manual_seed(11))[0]
    g = torch.Generator().manual_seed(11)
    mu, logvar = m.encode(x, g)
    eps = torch.randn(mu.shape, generator=g)
    assert torch.allclose(logits, m.decode(mu + eps * torch.exp(0.5 * logvar)), atol=1e-6)


def test_multvae_initialisation_follows_liang():
    torch.manual_seed(0)
    m = MultVAE(n_items=500, hidden=60, latent=20)
    for layer in (m.enc_in, m.enc_out, m.dec_in, m.dec_out):
        assert layer.bias.abs().max() <= 0.002
        fan_in, fan_out = layer.weight.shape[1], layer.weight.shape[0]
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        assert layer.weight.abs().max() <= bound + 1e-7
    assert m.enc_out.out_features == 2 * 20 and m.item_vectors().shape == (500, 60)


# ==================================================================== sampling
def test_user_batches_cover_every_user_once_and_are_seeded():
    users = np.array([3, 5, 8, 9, 11, 20, 21], dtype=np.int64)
    a = np.concatenate(list(user_batches(users, 3, stream_rng(2020, USER_BATCH_STREAM))))
    b = np.concatenate(list(user_batches(users, 3, stream_rng(2020, USER_BATCH_STREAM))))
    assert sorted(a.tolist()) == users.tolist() and np.array_equal(a, b)
    assert not np.array_equal(a, users)                     # shuffled, not in id order
    assert [len(x) for x in user_batches(users, 3, stream_rng(0, 1))] == [3, 3, 1]
    # the stream is not the BPR sampler's stream (default_rng(seed))
    assert not np.array_equal(stream_rng(2020, USER_BATCH_STREAM).permutation(50),
                              np.random.default_rng(2020).permutation(50))
    g1 = stream_torch_generator(2020, 2, "cpu")
    g2 = stream_torch_generator(2020, 2, "cpu")
    g3 = stream_torch_generator(2020, 1, "cpu")
    assert torch.equal(torch.rand(4, generator=g1), torch.rand(4, generator=g2))
    assert not torch.equal(torch.rand(4, generator=stream_torch_generator(2020, 2, "cpu")),
                           torch.rand(4, generator=g3))


def test_dense_rows_and_users_with_interactions():
    csr = sp.csr_matrix(np.array([[1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    assert users_with_interactions(csr).tolist() == [0, 2]
    rows = dense_rows(csr, np.array([2, 0, 1]), "cpu")
    assert torch.equal(rows, torch.tensor(csr[[2, 0, 1]].toarray()))


# ===================================================================== trainer
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("vdata")
    make_synthetic(root / "synthetic", n_users=300, n_items=200, seed=0)
    return root


def _cfg(root, tmp, **over):
    base = OmegaConf.load(REPO / "configs" / "config.yaml")
    arm = OmegaConf.load(REPO / "configs" / "arm" / "vae_cf.yaml")
    cfg = OmegaConf.merge(base, {"dataset": {"name": "synthetic"}, "arm": OmegaConf.to_container(arm)})
    cfg = OmegaConf.merge(cfg, {
        "seed": 2020, "epochs": 10, "device": "cpu",
        "eval": {"every": 5, "batch_size": 128, "topks": [10, 20], "select_metric": "recall@20"},
        "geometry": {"np_max_samples": 100, "er_max_samples": 500, "np_k": [10], "np_ref_k": [10]},
        "arm": {"vae": {"hidden": 32, "latent": 8, "batch_size": 64,
                        "beta": {"schedule": "linear", "cap": 0.2, "anneal_steps": 500}}},
        "paths": {"data": str(root), "runs": str(tmp), "arm_tag": "vae_cf", "run_dir": str(tmp / "vae_cf")},
        "hydra": None,
    })
    cfg = OmegaConf.merge(cfg, over)
    OmegaConf.resolve(cfg)
    return cfg


def _trainer(cfg):
    set_seed(int(cfg.seed))
    ds = InteractionDataset("synthetic", root=cfg.paths.data, val_frac=0.1, split_seed=2020, verbose=False)
    return VAECFTrainer(cfg, ds, build_vae_cf(cfg, ds.n_items), torch.device("cpu"), cfg.paths.run_dir), ds


def test_vae_cf_run_writes_the_shared_result_schema(synth, tmp_path):
    t, ds = _trainer(_cfg(synth, tmp_path))
    res = t.run()
    rd = Path(res["config"]["paths"]["run_dir"])
    assert (rd / "best.pt").exists() and (rd / "results.json").exists()
    assert res["selected_on"] == "val:recall@20" and res["select_from_epoch"] == 0
    for key in ("recall@20", "ndcg@20", "gini@20", "tail_catalog_coverage@20"):
        assert key in res["val"] and key in res["test"]
    assert {"np_ref@10", "er_item"} <= set(res["geometry"])
    v = res["vae"]
    assert len(v["kl_per_dim"]) == 8 and sum(v["kl_per_dim"]) == pytest.approx(v["kl"], rel=1e-5)
    assert 0 <= v["active_units"] <= 8 and "er_latent" in v and "nll_train" in v
    assert "vae" in res["at_last_epoch"] and "geometry" in res["at_last_epoch"]
    hist = json.load(open(rd / "history.json"))
    evals = [h for h in hist if h["event"] == "eval"]
    assert [h["epoch"] for h in evals] == [5, 10]
    assert all({"vae_kl", "vae_active_units", "vae_beta", "vae_kl_per_dim"} <= set(h) for h in evals)
    # the beta reported at the checkpoint is the one in force at that update
    ck = torch.load(rd / "best.pt", weights_only=False)
    assert v["beta"] == pytest.approx(t.beta(ck["updates"]))
    assert ck["updates"] == ck["epoch"] * t.nb


def test_encoder_sees_only_reduced_training_rows(synth, tmp_path):
    """Validation items must never reach the encoder, for val or test scoring:
    the scorer's output must equal scoring the reduced training rows."""
    t, ds = _trainer(_cfg(synth, tmp_path))
    t.model.eval()
    users = torch.arange(ds.n_users)
    got = t.scorer()(users)
    want = t.model.scores(dense_rows(ds.UserItemNet, np.arange(ds.n_users), "cpu"))
    assert torch.equal(got, want)
    assert ds.ValNet.nnz > 0       # there is something that could have leaked


def test_scorer_returns_fresh_tensors(synth, tmp_path):
    t, ds = _trainer(_cfg(synth, tmp_path))
    t.model.eval()
    sc = t.scorer()
    a = evaluate(sc, ds, "val", [10], 128)
    b = evaluate(sc, ds, "val", [10], 128)
    assert a == b


def test_evaluation_frequency_cannot_change_the_trajectory(synth, tmp_path):
    a, _ = _trainer(_cfg(synth, tmp_path / "a", eval={"every": 1}))
    b, _ = _trainer(_cfg(synth, tmp_path / "b", eval={"every": 5}))
    ha = {h["epoch"]: h for h in (a.run(), a.history)[1]}
    hb = {h["epoch"]: h for h in (b.run(), b.history)[1]}
    for ep in (5, 10):
        for k in ("loss_total", "loss_nll", "loss_kl", "val_recall@20", "vae_kl"):
            assert ha[ep][k] == hb[ep][k], (ep, k)


def test_checkpoint_report_uses_the_checkpoints_update_count(synth, tmp_path, monkeypatch):
    """Script validation so the best checkpoint is epoch 5 of 10: the reported
    beta and update count must be those at epoch 5, not at the end."""
    import src.vae_cf as vc
    real = vc.evaluate
    script = iter([0.9, 0.1])

    def scripted(scorer, ds, split, *a, **k):
        out = real(scorer, ds, split, *a, **k)
        if split == "val":
            out["recall@20"] = next(script, out["recall@20"])
        return out
    monkeypatch.setattr(vc, "evaluate", scripted)
    t, _ = _trainer(_cfg(synth, tmp_path))
    res = t.run()
    assert res["best_epoch"] == 5
    assert res["vae"]["updates"] == 5 * t.nb and t.step == 10 * t.nb
    assert res["vae"]["beta"] == pytest.approx(t.beta(5 * t.nb))
    assert res["at_last_epoch"]["vae"]["updates"] == 10 * t.nb


def test_free_bits_and_weight_decay_reach_the_training_loss(synth, tmp_path):
    def first_epoch_loss(**v):
        t, _ = _trainer(_cfg(synth, tmp_path / str(sorted(v.items())), epochs=1, eval={"every": 1},
                             arm={"vae": v}))
        return t.train_epoch(0)
    base = first_epoch_loss()
    fb = first_epoch_loss(free_bits=5.0)
    wd = first_epoch_loss(weight_decay=1.0)
    assert fb["total"] > base["total"] and wd["total"] > base["total"]
    assert fb["nll"] == pytest.approx(base["nll"], rel=0.2)    # same model, only the penalty differs


def test_deliberate_collapse_is_caught_by_the_diagnostics(synth, tmp_path):
    """beta = 5 (constant) collapses Mult-VAE on the synthetic data; the
    diagnostics must report it (no active units, KL near zero) and accuracy
    must fall to roughly popularity. Under the Mult-VAE^PR cap (0.2) the same
    model uses its latent space. Measured on CPU (seed 2020, 40 epochs):
    beta=5 -> KL 0.014, AU 0/16, val R@20 0.128 (popularity 0.107);
    beta=0.2 linear -> KL 11.7, AU 16/16, val R@20 0.495."""
    def over(beta):
        return {"epochs": 40, "eval": {"every": 5, "batch_size": 512},
                "arm": {"vae": {"hidden": 64, "latent": 16, "batch_size": 64, "beta": beta}}}
    ok, ds = _trainer(_cfg(synth, tmp_path / "ok", **over({"schedule": "linear", "cap": 0.2, "anneal_steps": 2000})))
    bad, _ = _trainer(_cfg(synth, tmp_path / "bad", **over({"schedule": "constant", "cap": 5.0})))
    r_ok, r_bad = ok.run(), bad.run()
    pop = torch.tensor(np.asarray(ds.UserItemNet.sum(0)).ravel(), dtype=torch.float32)
    pop_recall = evaluate(lambda u: pop.unsqueeze(0).repeat(len(u), 1), ds, "val", [20], 512)["recall@20"]
    assert r_bad["vae"]["active_units"] == 0 and r_bad["vae"]["kl"] < 0.1
    assert r_ok["vae"]["active_units"] >= 12 and r_ok["vae"]["kl"] > 2.0
    assert r_bad["val"]["recall@20"] < pop_recall + 0.1 < r_ok["val"]["recall@20"]


def test_vae_cf_refuses_settings_it_does_not_support(synth, tmp_path):
    for over in ({"arm": {"lambda_manifold": 1e-5}}, {"arm": {"warmup_epochs": 100}},
                 {"warmstart": {"save": True}}, {"eval": {"patience_evals": 3}}):
        with pytest.raises(ValueError):
            _trainer(_cfg(synth, tmp_path, **over))


# ============================================================ entry point / tags
def test_folder_names_existing_arms_unchanged_and_vae_cf_named_by_arm():
    from analysis.config_diff import composed
    expect = {("arm=emb_mr",): "emb_mr_w100_lam1e-05_k20_r50_K3_d256",
              ("arm=mr_off",): "mr_off_w100_K3_d256",
              ("arm=emb_mr", "arm.lambda_manifold=1e-4"): "emb_mr_w100_lam0.0001_k20_r50_K3_d256",
              ("arm=emb_mr_dw",): "emb_mr_dw_w100_lam1e-05_k20_r50_K3_d256",
              ("arm=coocc_mr",): "coocc_mr_w100_lam0.0001_k20_K3_d256",
              ("arm=mr_off", "model.n_layers=0"): "mr_off_w100_K0_d256",
              ("arm=vae_cf",): "vae_cf",
              ("arm=vae_cf", "arm.name=vae_cf_cap1", "arm.vae.beta.cap=1.0"): "vae_cf_cap1"}
    for ov, tag in expect.items():
        assert composed(["dataset=gowalla", "seed=2020", *ov])["paths"]["arm_tag"] == tag, ov
    assert _arm_tag("emb_mr", 1e-5, 20, "none", 100, 50, 3, 256) == "emb_mr_w100_lam1e-05_k20_r50_K3_d256"


def test_cli_runs_vae_cf_end_to_end(synth, tmp_path):
    env = dict(os.environ)
    cmd = [sys.executable, "-m", "src.train", "dataset=synthetic", "arm=vae_cf", "seed=2021", "epochs=4",
           "device=cpu", f"paths.data={synth}", f"paths.runs={tmp_path}", "eval.every=2", "eval.batch_size=128",
           "eval.topks=[10,20]", "arm.vae.hidden=16", "arm.vae.latent=4", "arm.vae.batch_size=64",
           "geometry.np_max_samples=50", "geometry.np_ref_k=[5]", "geometry.np_k=[5]",
           f"hydra.run.dir={tmp_path}/hydra"]
    out = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.load(open(tmp_path / "synthetic" / "vae_cf" / "seed2021" / "results.json"))
    assert res["arm"] == "vae_cf" and res["epochs_budget"] == 4 and "kl" in res["vae"]
