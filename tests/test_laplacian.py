"""Laplacian invariants and manifold loss."""
import numpy as np
import scipy.sparse as sp
import torch
import pytest

from src.models.mr_layer import (build_embedding_laplacian, build_cooccurrence_laplacian,
                                 knn_cosine_torch, knn_cosine_sklearn, manifold_loss,
                                 laplacian_from_knn, gaussian_weights)


def _rand_emb(n=200, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def test_torch_and_sklearn_knn_agree():
    x = _rand_emb()
    _, i_t = knn_cosine_torch(x, 5)
    _, i_s = knn_cosine_sklearn(x, 5)
    agree = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(i_t, i_s)])
    assert agree > 0.99


def test_laplacian_rows_sum_to_zero_and_symmetric():
    x = _rand_emb()
    L = build_embedding_laplacian(x, k=5, backend="torch", device=torch.device("cpu"), verbose=False)
    Ld = L.to_dense()
    assert torch.allclose(Ld, Ld.t(), atol=1e-6)
    assert torch.allclose(Ld.sum(1), torch.zeros(Ld.shape[0]), atol=1e-5)
    assert (torch.diag(Ld) >= 0).all()


def test_manifold_loss_equals_trace_form():
    x = _rand_emb(100, 8)
    L = build_embedding_laplacian(x, k=4, backend="torch", device=torch.device("cpu"), verbose=False)
    z = _rand_emb(100, 8, seed=1)
    direct = torch.trace(z.t() @ L.to_dense() @ z) / z.shape[0]
    assert manifold_loss(z, L).item() == pytest.approx(direct.item(), rel=1e-5)
    # Dirichlet energy is non-negative for a proper Laplacian
    assert manifold_loss(z, L).item() >= -1e-6


def test_cooccurrence_block_diagonal():
    rng = np.random.default_rng(0)
    R = sp.random(60, 40, density=0.1, format="csr", random_state=0)
    R.data[:] = 1.0
    L = build_cooccurrence_laplacian(R, k=3, device=torch.device("cpu"), verbose=False).to_dense()
    assert L.shape == (100, 100)
    assert torch.count_nonzero(L[:60, 60:]) == 0 and torch.count_nonzero(L[60:, :60]) == 0
    assert torch.allclose(L.sum(1), torch.zeros(100), atol=1e-5)


def test_gaussian_weights_in_unit_interval():
    d = np.random.default_rng(0).random((10, 5))
    w, sigma = gaussian_weights(d)
    assert sigma > 0 and (w > 0).all() and (w <= 1).all()
