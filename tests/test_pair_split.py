"""Pair-split of the kernel gradient (emb_mr_dw_weaken / emb_mr_dw_repel).

Claims pinned here:
  * the exact rule d2 > 2 sigma^2 |z_i||z_j| gives the sign of a pair's net
    angular force in ``grad`` mode; the unit-sphere rule only undercounts;
  * the split modes have the same loss value as fresh / grad, and their
    gradients add up: weaken + repel - fresh == grad;
  * routing is real: each split mode differs from both fresh and grad when the
    pairs are mixed, and collapses to fresh or grad at the limits;
  * the diagnostics agree with the rule they report.
"""
import contextlib
import io
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import src.train  # noqa: F401  registers the armtag resolver
from src.data.dataset import InteractionDataset
from src.data.synthetic import make_synthetic
from src.models.lightgcn import LightGCN
from src.models.mr_layer import (SPLIT_MODES, KnnGraph, _pairwise_manifold_loss_reference,
                                 build_embedding_knn, kernel_force_diagnostics,
                                 net_repulsive, pairwise_manifold_loss)
from src.trainer import Trainer
from src.utils import set_seed

CPU = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[1]


def _emb(n=400, d=12, seed=0, norm_spread=0.5, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g, dtype=dtype)
    return z * torch.exp(norm_spread * torch.randn(n, 1, generator=g, dtype=dtype))


def _graph(emb, k=8, sigma=None):
    g = build_embedding_knn(emb.float(), k=k, device=CPU, verbose=False)
    return KnnGraph(idx=g.idx, w0=g.w0.to(emb.dtype), sigma=g.sigma if sigma is None else sigma)


def _grad(Z, g, mode, fn=pairwise_manifold_loss):
    Z = Z.detach().clone().requires_grad_(True)
    v = fn(Z, g, mode)
    v.backward()
    return v.detach(), Z.grad


def _pair_d2_nrm(Z, g):
    nrm = Z.norm(dim=1)
    d2 = ((Z.unsqueeze(1) - Z[g.idx]) ** 2).sum(-1)
    return d2, nrm


# ------------------------------------------------------------ the rule itself
def test_exact_rule_gives_the_sign_of_the_net_angular_force():
    """Two-node graphs with very different norms: the descent component of
    z_0's gradient along the tangent towards z_1 must be negative (moving
    away) exactly when the rule says net-repulsive."""
    rng = torch.Generator().manual_seed(1)
    sigma, agree, n_rep, n = 0.5, 0, 0, 400
    for _ in range(n):
        base = torch.randn(6, generator=rng, dtype=torch.float64)
        noise = torch.rand((), generator=rng, dtype=torch.float64) * 1.5
        z = torch.stack([base, base + noise * torch.randn(6, generator=rng, dtype=torch.float64)])
        z = z * torch.exp(0.5 * torch.randn(2, 1, generator=rng, dtype=torch.float64))
        g = KnnGraph(idx=torch.tensor([[1], [0]]), w0=torch.ones(2, 1, dtype=torch.float64), sigma=sigma)
        _, gr = _grad(z, g, "grad")
        zi, zj = z[0] / z[0].norm(), z[1] / z[1].norm()
        tangent = zj - (zj @ zi) * zi
        moves_away = float(-gr[0] @ tangent) < 0
        d2, nrm = _pair_d2_nrm(z, g)
        rule = bool(net_repulsive(d2, nrm, g)[0, 0])
        agree += moves_away == rule
        n_rep += rule
    assert agree == n
    assert 0.1 * n < n_rep < 0.9 * n, "fixture must contain both kinds of pair"


def test_sphere_rule_only_undercounts_and_is_exact_on_the_sphere():
    Z = _emb(norm_spread=0.6)
    g = _graph(Z, sigma=0.6)
    d2, nrm = _pair_d2_nrm(Z, g)
    exact = net_repulsive(d2, nrm, g)
    cos = (Z.unsqueeze(1) * Z[g.idx]).sum(-1) / (nrm.unsqueeze(1) * nrm[g.idx])
    sphere = (1 - cos) > g.sigma ** 2
    assert not (sphere & ~exact).any(), "sphere rule flagged a pair the exact rule does not"
    assert (exact & ~sphere).any(), "fixture must show the undercount"
    U = Z / Z.norm(dim=1, keepdim=True)
    d2u, nu = _pair_d2_nrm(U, g)
    cosu = (U.unsqueeze(1) * U[g.idx]).sum(-1)
    disagree = (net_repulsive(d2u, nu, g) != ((1 - cosu) > g.sigma ** 2)).float().mean().item()
    assert disagree < 1e-3, f"rules disagree on the unit sphere for {disagree:.4f} of pairs"


# ------------------------------------------------------- value and gradient
@pytest.fixture(scope="module")
def mixed():
    """Embeddings and graph with a genuine mix of the two pair classes."""
    Z = _emb(norm_spread=0.4)
    g = _graph(Z)
    d2, nrm = _pair_d2_nrm(Z, g)
    frac = net_repulsive(d2, nrm, g).float().mean().item()
    assert 0.05 < frac < 0.95, f"fixture not mixed: {frac:.3f}"
    return Z, g


def test_split_modes_share_the_loss_value(mixed):
    Z, g = mixed
    vals = {m: _grad(Z, g, m)[0].item() for m in ("fresh", "grad") + SPLIT_MODES}
    for m, v in vals.items():
        assert v == pytest.approx(vals["grad"], rel=1e-14), m


def test_gradients_add_up_exactly(mixed):
    """weaken routes the kernel term on one class, repel on the other, so
    weaken + repel - fresh must reproduce grad."""
    Z, g = mixed
    G = {m: _grad(Z, g, m)[1] for m in ("fresh", "grad") + SPLIT_MODES}
    recon = G["grad_weaken"] + G["grad_repel"] - G["fresh"]
    assert torch.allclose(recon, G["grad"], rtol=1e-12, atol=1e-15)


def test_routing_is_real_on_a_mixed_graph(mixed):
    Z, g = mixed
    G = {m: _grad(Z, g, m)[1] for m in ("fresh", "grad") + SPLIT_MODES}
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    for m in SPLIT_MODES:
        assert rel(G[m], G["fresh"]) > 1e-4, f"{m} equals fresh"
        assert rel(G[m], G["grad"]) > 1e-4, f"{m} equals grad"


def _limit_graph(Z, all_repel):
    """sigma chosen from the data so that every pair is (or none is)
    net-repulsive while the kernel weights stay far from underflow."""
    g = _graph(Z)
    d2, nrm = _pair_d2_nrm(Z, g)
    ratio = d2 / (nrm.unsqueeze(1) * nrm[g.idx])          # repulsive iff ratio > 2 sigma^2
    s2 = 0.45 * ratio.min().item() if all_repel else 0.55 * ratio.max().item()
    return KnnGraph(idx=g.idx, w0=g.w0, sigma=s2 ** 0.5)


@pytest.mark.parametrize("all_repel", [True, False])
def test_limits_collapse_to_fresh_or_grad(all_repel):
    """Every pair net-repulsive -> repel == grad, weaken == fresh; none -> the reverse."""
    Z = _emb()
    g = _limit_graph(Z, all_repel)
    d2, nrm = _pair_d2_nrm(Z, g)
    assert net_repulsive(d2, nrm, g).all() if all_repel else not net_repulsive(d2, nrm, g).any()
    G = {m: _grad(Z, g, m)[1] for m in ("fresh", "grad") + SPLIT_MODES}
    # the equalities below are only meaningful if fresh and grad really differ
    assert ((G["grad"] - G["fresh"]).norm() / G["fresh"].norm()).item() > 1e-3
    same_as_grad, same_as_fresh = ("grad_repel", "grad_weaken") if all_repel else ("grad_weaken", "grad_repel")
    assert torch.equal(G[same_as_grad], G["grad"])
    assert torch.equal(G[same_as_fresh], G["fresh"])


@pytest.mark.parametrize("mode", SPLIT_MODES)
def test_split_fast_path_matches_reference(mixed, mode):
    Z, g = mixed
    va, ga = _grad(Z, g, mode)
    vb, gb = _grad(Z, g, mode, fn=_pairwise_manifold_loss_reference)
    assert va.item() == pytest.approx(vb.item(), rel=1e-12)
    assert torch.allclose(ga, gb, rtol=1e-10, atol=1e-13)


def test_routed_gradient_is_not_the_derivative_of_the_value(mixed):
    """By design: the split modes stop the kernel gradient on some pairs, so
    their gradient differs from the true derivative of the (shared) value,
    which is the grad-mode gradient. A gradcheck would therefore fail; this
    pins that the difference is exactly the stopped kernel term."""
    Z, g = mixed
    G = {m: _grad(Z, g, m)[1] for m in ("fresh", "grad") + SPLIT_MODES}
    stopped_in_weaken = G["grad"] - G["grad_weaken"]     # kernel term on repel pairs
    routed_in_repel = G["grad_repel"] - G["fresh"]       # the same term
    assert torch.allclose(stopped_in_weaken, routed_in_repel, rtol=1e-12, atol=1e-15)
    assert stopped_in_weaken.norm() > 0


# ------------------------------------------------------------ diagnostics
def test_diagnostics_agree_with_the_rule(mixed):
    Z, g = mixed
    d = kernel_force_diagnostics(Z, g)
    d2, nrm = _pair_d2_nrm(Z, g)
    assert d["frac_repel_exact"] == pytest.approx(net_repulsive(d2, nrm, g).double().mean().item())
    assert d["frac_repel_sphere"] <= d["frac_repel_exact"] + 1e-12
    assert 0.0 < d["kernel_share_repel"] < 1.0 and d["kernel_to_attraction"] > 0


def test_diagnostics_limits_and_build_time_fields():
    Z = _emb()
    lo = kernel_force_diagnostics(Z, _limit_graph(Z, all_repel=True))
    hi = kernel_force_diagnostics(Z, _limit_graph(Z, all_repel=False))
    assert lo["frac_repel_exact"] == 1.0 and lo["kernel_share_repel"] == pytest.approx(1.0)
    assert hi["frac_repel_exact"] == 0.0 and hi["kernel_share_repel"] == 0.0
    # every pair net-repulsive means kernel > attraction pair by pair, so the
    # summed ratio must exceed 1; no pair net-repulsive means it is below 1
    assert lo["kernel_to_attraction"] > 1.0 > hi["kernel_to_attraction"]
    g = build_embedding_knn(Z.float(), k=8, device=CPU, verbose=False)
    at_build = kernel_force_diagnostics(Z.float(), g)
    assert g.frac_repel_exact == pytest.approx(at_build["frac_repel_exact"])
    assert g.frac_beyond_crossover == pytest.approx(at_build["frac_repel_sphere"], abs=2e-3)


def test_kernel_to_attraction_on_a_single_pair_is_the_rule_ratio():
    """For one symmetric pair the budget ratio must be exactly
    d2 / (2 sigma^2 |z_i||z_j|) -- the quantity whose crossing of 1 is the
    net-repulsion rule. Pins the kernel and attraction formulas together."""
    rng = torch.Generator().manual_seed(3)
    for _ in range(50):
        base = torch.randn(5, generator=rng, dtype=torch.float64)
        z = torch.stack([base, base + 0.8 * torch.randn(5, generator=rng, dtype=torch.float64)])
        z = z * torch.exp(0.5 * torch.randn(2, 1, generator=rng, dtype=torch.float64))
        g = KnnGraph(idx=torch.tensor([[1], [0]]), w0=torch.ones(2, 1, dtype=torch.float64), sigma=0.6)
        d2 = ((z[0] - z[1]) ** 2).sum()
        expected = (d2 / (2 * g.sigma ** 2 * z[0].norm() * z[1].norm())).item()
        assert kernel_force_diagnostics(z, g)["kernel_to_attraction"] == pytest.approx(expected, rel=1e-9)


def test_net_repulsive_share_equals_kernel_beating_attraction(mixed):
    """Per endpoint, net-repulsive means the kernel's tangential magnitude
    exceeds the attraction's -- the definition the diagnostics budget uses."""
    Z, g = mixed
    d2, nrm = _pair_d2_nrm(Z, g)
    ni, nj = nrm.unsqueeze(1), nrm[g.idx]
    cos = ((Z.unsqueeze(1) * Z[g.idx]).sum(-1) / (ni * nj)).clamp(-1, 1)
    w = torch.exp(-(1 - cos) / g.sigma ** 2)
    sin = (1 - cos ** 2).clamp_min(0).sqrt()
    K_i = d2 * w * sin / (g.sigma ** 2 * ni)
    A_i = 2 * w * sin * nj
    assert torch.equal(K_i > A_i, net_repulsive(d2, nrm, g))


# ------------------------------------------------------------ trainer level
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("splitdata")
    make_synthetic(root / "synthetic", n_users=150, n_items=100, seed=0)
    return root


def _train(root, tmp, arm, lam=1.0):
    arm_cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "arm" / f"{arm}.yaml"))
    arm_cfg["lambda_manifold"] = lam
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "configs" / "config.yaml"),
                          {"dataset": {"name": "synthetic"}, "arm": arm_cfg})
    cfg = OmegaConf.merge(cfg, {
        "seed": 2020, "epochs": 8, "device": "cpu",
        "model": {"dim": 8, "n_layers": 2}, "optim": {"batch_size": 256},
        "eval": {"every": 1, "batch_size": 64, "topks": [5, 10], "select_metric": "recall@10"},
        "geometry": {"np_max_samples": 50, "er_max_samples": 100, "np_k": [5], "np_ref_k": [5]},
        "arm": {"warmup_epochs": 3, "rebuild_every": 2, "k_neighbors": 5},
        "paths": {"data": str(root), "runs": str(tmp), "arm_tag": arm, "run_dir": str(tmp / arm)},
        "hydra": None})
    OmegaConf.resolve(cfg)
    set_seed(2020)
    ds = InteractionDataset("synthetic", root=str(root), val_frac=0.1, split_seed=2020, verbose=False)
    m = LightGCN(ds.n_users, ds.n_items, dim=8, n_layers=2, graph=ds.sparse_graph(CPU))
    t = Trainer(cfg, ds, m, CPU, cfg.paths.run_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        t.run()
    with torch.no_grad():
        z = torch.cat(t.model.computer())
    return t, z


@pytest.fixture(scope="module")
def trained(synth, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("splitruns")
    return {a: _train(synth, tmp, a) for a in ("emb_mr_fresh", "emb_mr_dw", "emb_mr_dw_weaken", "emb_mr_dw_repel")}


def test_split_arms_take_the_pairwise_path_and_log_diagnostics(trained):
    for arm, mode in (("emb_mr_dw_weaken", "grad_weaken"), ("emb_mr_dw_repel", "grad_repel")):
        t, _ = trained[arm]
        assert t.kernel == mode and t.knn is not None and t.laplacian is None
        built = [h for h in t.history if h.get("event") == "laplacian_built"]
        assert built and all(h["kernel"] == mode and 0 <= h["frac_repel_exact"] <= 1 for h in built)
        mr = [h for h in t.history if h.get("event") == "eval" and h["phase"] == "mr"]
        assert mr and all(0 <= h["kd_frac_repel_exact"] <= 1 and 0 <= h["kd_kernel_share_repel"] <= 1
                          for h in mr)
        assert all(h["kd_frac_repel_sphere"] <= h["kd_frac_repel_exact"] + 1e-9 for h in mr)


def test_split_arms_train_differently(trained):
    """With lambda = 1 the four corners of the 2x2 must all end in different
    places, or the factorial cannot separate anything."""
    z = {a: v[1] for a, v in trained.items()}
    rep = [h["kd_frac_repel_exact"] for h in trained["emb_mr_dw_repel"][0].history
           if h.get("event") == "eval" and h["phase"] == "mr"]
    assert max(rep) > 0, "fixture never produced a net-repulsive pair"
    arms = list(z)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            assert (z[a] - z[b]).abs().max().item() > 1e-5, f"{a} == {b}"


def test_warmup_is_shared_by_all_four_corners(trained):
    """The arms differ only after W: every warm-up evaluation must match."""
    def warm(arm):
        return [h["val_recall@10"] for h in trained[arm][0].history
                if h.get("event") == "eval" and h["phase"] == "warmup"]
    ref = warm("emb_mr_fresh")
    assert ref and all(warm(a) == ref for a in trained)


def test_logging_does_not_change_training(synth, tmp_path):
    """Diagnostics are computed at evaluation and must be side-effect free:
    a run with them switched off ends at exactly the same embeddings."""
    from src import trainer as T
    t1, z1 = _train(synth, tmp_path / "a", "emb_mr_dw_repel")
    orig = T.kernel_force_diagnostics
    T.kernel_force_diagnostics = lambda Z, g: {}
    try:
        t2, z2 = _train(synth, tmp_path / "b", "emb_mr_dw_repel")
    finally:
        T.kernel_force_diagnostics = orig
    assert torch.equal(z1, z2)
