"""Differentiable kernel weights (post-gate plan, step 2).

The pairwise manifold term must (a) reproduce the existing sparse-Laplacian
term exactly when its weights are frozen, (b) differ from it only in the ways
each mode is meant to, and (c) produce the gradient it claims to.
"""
import math

import numpy as np
import pytest
import torch

from src.models.mr_layer import (KnnGraph, build_embedding_knn, build_embedding_laplacian,
                                 laplacian_from_knn, manifold_loss, pairwise_manifold_loss)

CPU = torch.device("cpu")


def _emb(n=300, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def _graph(emb, k=8):
    return build_embedding_knn(emb, k=k, device=CPU, verbose=False)


# ---------------------------------------------------------------- exactness
def test_frozen_matches_laplacian_identity_exactly_in_float64():
    """tr(Z^T L Z) = 1/2 sum_i sum_{j in kNN(i)} w_ij ||z_i - z_j||^2 is an
    identity. Build the reference L in float64 from the same weights and check
    value and gradient to ~machine precision, at a point away from the build."""
    emb = _emb()
    g = _graph(emb)
    n = emb.shape[0]
    L = laplacian_from_knn(g.idx.numpy(), g.w0.double().numpy(), n).toarray()
    L = torch.from_numpy(L)                                    # float64, dense
    g64 = KnnGraph(idx=g.idx, w0=g.w0.double(), sigma=g.sigma)

    Z1 = (emb + 0.3 * _emb(seed=1)).double().requires_grad_(True)
    Z2 = Z1.detach().clone().requires_grad_(True)
    ref = (Z1 * (L @ Z1)).sum() / n
    new = pairwise_manifold_loss(Z2, g64, "frozen")
    ref.backward(); new.backward()
    assert new.item() == pytest.approx(ref.item(), rel=1e-12)
    assert torch.allclose(Z1.grad, Z2.grad, rtol=1e-10, atol=1e-12)


def test_frozen_matches_production_emb_mr_term_in_float32():
    """Against the actual sparse path emb_mr uses (float32 Laplacian)."""
    emb = _emb()
    L = build_embedding_laplacian(emb, k=8, device=CPU, verbose=False)
    g = _graph(emb)
    Z1 = (emb + 0.3 * _emb(seed=1)).requires_grad_(True)
    Z2 = Z1.detach().clone().requires_grad_(True)
    ref = manifold_loss(Z1, L); new = pairwise_manifold_loss(Z2, g, "frozen")
    ref.backward(); new.backward()
    assert new.item() == pytest.approx(ref.item(), rel=1e-5)
    assert torch.allclose(Z1.grad, Z2.grad, rtol=1e-4, atol=1e-7)


def test_graph_builders_agree_on_neighbours_weights_and_sigma():
    """build_embedding_knn must carry exactly what build_embedding_laplacian uses."""
    emb = _emb()
    g = _graph(emb)
    L = build_embedding_laplacian(emb, k=8, device=CPU, verbose=False).to_dense()
    rebuilt = laplacian_from_knn(g.idx.numpy(), g.w0.numpy(), emb.shape[0]).toarray()
    assert np.allclose(L.numpy(), rebuilt, atol=1e-7)


# ------------------------------------------------------------ mode semantics
def test_fresh_equals_frozen_at_the_build_point():
    """Recomputed weights, evaluated where they were built, are the build weights."""
    emb = _emb()
    g = _graph(emb)
    a = pairwise_manifold_loss(emb, g, "frozen")
    b = pairwise_manifold_loss(emb, g, "fresh")
    assert b.item() == pytest.approx(a.item(), rel=1e-5)


def test_fresh_and_frozen_diverge_away_from_the_build_point():
    emb = _emb()
    g = _graph(emb)
    Z = emb + 0.5 * _emb(seed=2)
    a = pairwise_manifold_loss(Z, g, "frozen").item()
    b = pairwise_manifold_loss(Z, g, "fresh").item()
    assert abs(a - b) / abs(a) > 1e-3


def test_grad_and_fresh_share_value_but_not_gradient():
    """The only difference between the two: whether d(loss)/dZ includes the
    term through the weights."""
    emb = _emb()
    g = _graph(emb)
    Z1 = (emb + 0.3 * _emb(seed=1)).requires_grad_(True)
    Z2 = Z1.detach().clone().requires_grad_(True)
    f = pairwise_manifold_loss(Z1, g, "fresh"); d = pairwise_manifold_loss(Z2, g, "grad")
    assert d.item() == pytest.approx(f.item(), rel=1e-7)
    f.backward(); d.backward()
    rel = (Z1.grad - Z2.grad).norm() / Z1.grad.norm()
    assert rel > 1e-3, f"gradients should differ, relative difference {rel:.2e}"


def test_grad_mode_gradient_is_correct():
    """Finite-difference check of the full gradient, including through the kernel."""
    torch.manual_seed(0)
    emb = torch.randn(30, 5, dtype=torch.float64)
    g = build_embedding_knn(emb.float(), k=4, device=CPU, verbose=False)
    g = KnnGraph(idx=g.idx, w0=g.w0.double(), sigma=g.sigma)
    Z = (emb + 0.1 * torch.randn_like(emb)).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda z: pairwise_manifold_loss(z, g, "grad"), (Z,),
                                    eps=1e-6, atol=1e-6)


# ------------------------------------------------- attraction vs repulsion
def _pair_energy_slope(theta_deg, sigma, mode):
    """Two points on the unit circle at angle theta, each other's only neighbour.
    On the unit sphere the kernel's cosine distance and the energy's Euclidean
    distance coincide: x = ||z_i - z_j||^2 = 2(1 - cos theta). Returns
    d(energy)/d(theta): > 0 means the gradient pulls the pair together."""
    th = torch.tensor(math.radians(theta_deg), dtype=torch.float64, requires_grad=True)
    Z = torch.stack([torch.stack([torch.ones((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)]),
                     torch.stack([torch.cos(th), torch.sin(th)])])
    g = KnnGraph(idx=torch.tensor([[1], [0]]), w0=torch.ones(2, 1, dtype=torch.float64), sigma=sigma)
    pairwise_manifold_loss(Z, g, mode).backward()
    return th.grad.item()


def test_grad_mode_attracts_close_pairs_and_repels_distant_ones():
    """Per pair the energy is x * exp(-x / 2 sigma^2), whose derivative changes
    sign at x = 2 sigma^2. sigma = 0.5 puts the crossover at 41.4 degrees."""
    assert _pair_energy_slope(20, 0.5, "grad") > 0     # x = 0.12 < 0.5: attract
    assert _pair_energy_slope(80, 0.5, "grad") < 0     # x = 1.65 > 0.5: repel


def test_detached_modes_only_ever_attract():
    for mode in ("frozen", "fresh"):
        assert _pair_energy_slope(20, 0.5, mode) > 0
        assert _pair_energy_slope(80, 0.5, mode) > 0


def test_unknown_mode_rejected():
    emb = _emb(); g = _graph(emb)
    with pytest.raises(ValueError, match="mode must be one of"):
        pairwise_manifold_loss(emb, g, "gradient")


# =========================================================== trainer level
import contextlib
import io
from pathlib import Path

from omegaconf import OmegaConf

import src.train  # noqa: F401  registers the armtag resolver
from src.data.dataset import InteractionDataset
from src.data.synthetic import make_synthetic
from src.models.lightgcn import LightGCN
from src.trainer import Trainer
from src.utils import set_seed

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("kwdata")
    make_synthetic(root / "synthetic", n_users=120, n_items=80, seed=0)
    return root


def _train(root, tmp, arm, lam=None, kernel=None, laplacian=None):
    arm_cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "arm" / f"{arm}.yaml"))
    if lam is not None: arm_cfg["lambda_manifold"] = lam
    if kernel is not None: arm_cfg["kernel_weights"] = kernel
    if laplacian is not None: arm_cfg["laplacian"] = laplacian
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "configs" / "config.yaml"),
                          {"dataset": {"name": "synthetic"}, "arm": arm_cfg})
    tag = f"{arm}_{kernel}_{lam}"
    cfg = OmegaConf.merge(cfg, {
        "seed": 2020, "epochs": 6, "device": "cpu",
        "model": {"dim": 8, "n_layers": 2}, "optim": {"batch_size": 256},
        "eval": {"every": 1, "batch_size": 64, "topks": [5, 10], "select_metric": "recall@10"},
        "geometry": {"np_max_samples": 50, "er_max_samples": 100, "np_k": [5], "np_ref_k": [5]},
        "arm": {"warmup_epochs": 2, "rebuild_every": 2, "k_neighbors": 5},
        "paths": {"data": str(root), "runs": str(tmp), "arm_tag": tag, "run_dir": str(tmp / tag)},
        "hydra": None})
    OmegaConf.resolve(cfg)
    set_seed(2020)
    ds = InteractionDataset("synthetic", root=str(root), val_frac=0.1, split_seed=2020, verbose=False)
    m = LightGCN(ds.n_users, ds.n_items, dim=8, n_layers=2, graph=ds.sparse_graph(CPU))
    t = Trainer(cfg, ds, m, CPU, cfg.paths.run_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        res = t.run()
    return t, res


def _final_embeddings(t):
    with torch.no_grad():
        return torch.cat(t.model.computer())


def test_frozen_mode_reproduces_a_full_emb_mr_training_run(synth, tmp_path):
    """End-to-end: the whole trainer, not just one loss call. lambda = 1 so the
    manifold term materially changes training; a lambda = 0 run proves the
    comparison could detect a missing or broken term."""
    t_lap, _ = _train(synth, tmp_path, "emb_mr", lam=1.0)
    t_frz, _ = _train(synth, tmp_path, "emb_mr", lam=1.0, kernel="frozen")
    t_off, _ = _train(synth, tmp_path, "emb_mr", lam=0.0)
    assert t_lap.laplacian is not None and t_lap.knn is None
    assert t_frz.knn is not None and t_frz.laplacian is None
    same = (_final_embeddings(t_lap) - _final_embeddings(t_frz)).abs().max().item()
    diff = (_final_embeddings(t_lap) - _final_embeddings(t_off)).abs().max().item()
    assert diff > 1e-3, "lambda=1 must change training, or this test proves nothing"
    assert same < 1e-6, f"frozen deviates from emb_mr by {same:.2e}"
    # warm-up evaluations carry no manifold term, so the key is absent there
    e_lap = [h.get("loss_manifold", 0.0) for h in t_lap.history if h.get("event") == "eval"]
    e_frz = [h.get("loss_manifold", 0.0) for h in t_frz.history if h.get("event") == "eval"]
    assert any(e > 0 for e in e_lap), "no manifold energy was logged at all"
    assert np.allclose(e_lap, e_frz, rtol=1e-5)


def test_new_arms_take_the_pairwise_path_and_train(synth, tmp_path):
    for arm, mode in (("emb_mr_dw", "grad"), ("emb_mr_fresh", "fresh")):
        t, res = _train(synth, tmp_path, arm)
        assert t.kernel == mode and t.knn is not None and t.laplacian is None
        built = [h for h in t.history if h.get("event") == "laplacian_built"]
        assert built and all(h["kernel"] == mode for h in built)
        assert all(0.0 <= h["frac_beyond_crossover"] <= 1.0 and h["sigma"] > 0 for h in built)
        mr = [h for h in t.history if h.get("event") == "eval" and h["phase"] == "mr"]
        assert mr and all(h["loss_manifold"] > 0 for h in mr)
        assert "er_table" in res["geometry"] and "recall@10" in res["test"]


def test_gradient_through_kernel_changes_training(synth, tmp_path):
    """At the trainer level, grad and fresh differ only in the kernel gradient;
    with lambda = 1 that must produce different embeddings."""
    t_f, _ = _train(synth, tmp_path, "emb_mr_fresh", lam=1.0)
    t_g, _ = _train(synth, tmp_path, "emb_mr_dw", lam=1.0)
    assert (_final_embeddings(t_f) - _final_embeddings(t_g)).abs().max().item() > 1e-4


def test_invalid_kernel_configurations_rejected(synth, tmp_path):
    with pytest.raises(ValueError, match="kernel_weights"):
        _train(synth, tmp_path, "emb_mr", kernel="gradient")
    with pytest.raises(ValueError, match="requires"):
        _train(synth, tmp_path, "coocc_mr", kernel="grad")


def test_new_arms_get_their_own_run_directories():
    tags = {a: src.train._arm_tag(a, 1e-5, 20, "none", 100, 50, 3, 256)
            for a in ("emb_mr", "emb_mr_dw", "emb_mr_fresh")}
    assert len(set(tags.values())) == 3
    assert tags["emb_mr"] == "emb_mr_w100_lam1e-05_k20_r50_K3_d256"   # unchanged


def test_crossover_fraction_matches_its_definition():
    emb = _emb()
    g = _graph(emb)
    from src.models.mr_layer import knn_cosine_torch
    d, _ = knn_cosine_torch(emb, 8)
    assert g.frac_beyond_crossover == pytest.approx(float(np.mean(2 * d > 2 * g.sigma ** 2)))


# ------------------------------------------- fast path vs plain-autograd reference
from src.models.mr_layer import _pairwise_manifold_loss_reference, knn_dots


@pytest.mark.parametrize("mode", ["frozen", "fresh", "grad"])
def test_fast_path_matches_reference_exactly(mode):
    """The custom backward must give the same value and gradient as plain
    autograd through a gather, in every mode (float64)."""
    emb = _emb()
    g = _graph(emb)
    g64 = KnnGraph(idx=g.idx, w0=g.w0.double(), sigma=g.sigma)
    Z1 = (emb + 0.3 * _emb(seed=1)).double().requires_grad_(True)
    Z2 = Z1.detach().clone().requires_grad_(True)
    a = pairwise_manifold_loss(Z1, g64, mode); b = _pairwise_manifold_loss_reference(Z2, g64, mode)
    a.backward(); b.backward()
    assert a.item() == pytest.approx(b.item(), rel=1e-12)
    assert torch.allclose(Z1.grad, Z2.grad, rtol=1e-10, atol=1e-13)


def test_knn_dots_gradcheck_and_chunking():
    torch.manual_seed(1)
    Z = torch.randn(40, 6, dtype=torch.float64, requires_grad=True)
    idx = torch.randint(0, 40, (40, 5))
    assert torch.autograd.gradcheck(lambda z: knn_dots(z, idx, 7), (Z,), eps=1e-6, atol=1e-7)
    # chunk size must not change the result
    assert torch.equal(knn_dots(Z.detach(), idx, 3), knn_dots(Z.detach(), idx, 4096))


def test_csr_structures_are_canonical():
    """Columns sorted within every row, for both S and S^T -- so the backward
    never relies on a backend accepting unsorted CSR (untestable without GPU)."""
    from src.models.mr_layer import csr_structure
    emb = _emb(); g = _graph(emb)
    crow_s, cols_s, perm_s, crow_t, col_t, perm_t = csr_structure(g.idx)
    n = emb.shape[0]
    for crow, cols in ((crow_s, cols_s), (crow_t, col_t)):
        assert crow[0] == 0 and crow[-1] == cols.numel() and crow.shape[0] == n + 1
        for i in range(n):
            seg = cols[crow[i]:crow[i + 1]]
            assert torch.all(seg[1:] > seg[:-1]), f"row {i} not strictly increasing"
    # both permutations are true permutations of the flat gradient
    for p in (perm_s, perm_t):
        assert torch.equal(torch.sort(p).values, torch.arange(g.idx.numel()))
