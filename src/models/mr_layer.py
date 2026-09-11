"""Manifold Regularisation layer (Belkin, Niyogi & Sindhwani 2006).

Ported from notebook Cell 8 (``build_knn_laplacian``, ``compute_manifold_loss``)
and Cell 9 (``build_cooccurrence_laplacian``, ``_build_knn_laplacian_from_sparse``).
The June 2026 audit validated both the Laplacian (L = D - W, W symmetrised as
(W + W^T)/2, Gaussian kernel with median-heuristic sigma) and the Dirichlet
energy ``tr(Z^T L Z) / n``. Those computations are unchanged.

What changed:

* one builder, two neighbour sources (``embeddings`` rebuilt periodically,
  or ``cooccurrence`` fixed from R), instead of two near-duplicate cells;
* the k-NN search over dense embeddings has a ``torch`` backend (chunked
  cosine similarity + topk on the GPU) next to the original ``sklearn`` brute
  force search. Same neighbours up to ties; tens of seconds instead of a
  minute or more per rebuild on Amazon-Book, and no CPU round trip. The
  ``sklearn`` backend is kept so the notebook's numbers can be reproduced
  exactly if anyone asks.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import torch


# --------------------------------------------------------------------------- kNN
def knn_cosine_torch(emb: torch.Tensor, k: int, chunk: int = 2048):
    """k nearest neighbours by cosine similarity, self excluded.

    Returns ``(cos_dist, idx)`` as numpy arrays of shape ``(n, k)`` with
    ``cos_dist = 1 - cos_sim`` (the same quantity sklearn's ``metric='cosine'``
    reports).
    """
    with torch.no_grad():
        x = emb.detach().float()
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        n = x.shape[0]
        dists = torch.empty((n, k), dtype=torch.float32)
        idxs = torch.empty((n, k), dtype=torch.int64)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sim = x[s:e] @ x.t()                          # (b, n)
            ar = torch.arange(s, e, device=x.device)
            sim[torch.arange(e - s, device=x.device), ar] = -float("inf")  # drop self
            top_sim, top_idx = torch.topk(sim, k, dim=1)
            dists[s:e] = (1.0 - top_sim).cpu()
            idxs[s:e] = top_idx.cpu()
    return dists.numpy(), idxs.numpy()


def knn_cosine_sklearn(emb: torch.Tensor, k: int):
    """Exact notebook path (Cell 8): sklearn brute-force cosine k-NN."""
    from sklearn.neighbors import NearestNeighbors
    x = emb.detach().cpu().numpy().astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    nn_ = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn_.fit(x)
    d, i = nn_.kneighbors(x)
    return d[:, 1:], i[:, 1:]


def knn_cosine_sparse(X: sp.csr_matrix, k: int):
    """Cosine k-NN over sparse feature rows (co-occurrence graph, Cell 9)."""
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    nn_ = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine", algorithm="brute", n_jobs=-1)
    nn_.fit(X)
    d, i = nn_.kneighbors(X)
    return d[:, 1:], i[:, 1:]


# --------------------------------------------------------------------- Laplacian
def gaussian_weights(cos_dist: np.ndarray, sigma: float | None = None,
                     positive_only_median: bool = False):
    """w_ij = exp(-||z_i - z_j||^2 / (2 sigma^2)) on the unit sphere, where
    ``||z_i - z_j||^2 = 2 * cos_dist``. sigma defaults to the median heuristic.

    ``positive_only_median`` reproduces Cell 9's guard (median over strictly
    positive distances) for the co-occurrence graph, where duplicate rows
    produce many exact zeros.
    """
    sq = 2.0 * cos_dist
    if sigma is None:
        pool = sq[sq > 0] if positive_only_median else sq
        sigma = float(np.sqrt(np.median(pool))) if pool.size else 1e-5
        sigma = max(sigma, 1e-5)
    return np.exp(-sq / (2.0 * sigma ** 2)).astype(np.float32), float(sigma)


def laplacian_from_knn(idx: np.ndarray, w: np.ndarray, n: int) -> sp.coo_matrix:
    """L = D - W with W = (W_knn + W_knn^T) / 2."""
    k = idx.shape[1]
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
    W = (W + W.T) / 2.0
    deg = np.asarray(W.sum(axis=1)).ravel()
    return (sp.diags(deg) - W).tocoo()


def scipy_to_torch_sparse(M: sp.coo_matrix, device: torch.device) -> torch.Tensor:
    idx = torch.from_numpy(np.vstack([M.row, M.col]).astype(np.int64))
    val = torch.from_numpy(M.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, torch.Size(M.shape)).coalesce().to(device)


def build_embedding_laplacian(emb: torch.Tensor, k: int = 20, sigma: float | None = None,
                              backend: str = "torch", device: torch.device | None = None,
                              verbose: bool = True) -> torch.Tensor:
    """Emb-MR Laplacian: k-NN over the current (propagated) embeddings."""
    device = device or emb.device
    t0 = time.time()
    if backend == "torch":
        d, i = knn_cosine_torch(emb, k)
    elif backend == "sklearn":
        d, i = knn_cosine_sklearn(emb, k)
    else:
        raise ValueError(f"unknown knn backend {backend!r}")
    w, sigma = gaussian_weights(d, sigma)
    L = laplacian_from_knn(i, w, emb.shape[0])
    Lt = scipy_to_torch_sparse(L, device)
    if verbose:
        print(f"    kNN Laplacian (n={emb.shape[0]:,}, k={k}, {backend}) "
              f"nnz={Lt._nnz():,} sigma={sigma:.4f} in {time.time()-t0:.1f}s")
    return Lt


def build_cooccurrence_laplacian(user_item_csr: sp.csr_matrix, k: int = 20,
                                 device: torch.device | None = None,
                                 verbose: bool = True) -> torch.Tensor:
    """CoOcc-MR Laplacian (GRALS-style): block_diag(L_user, L_item) from R.

    Fixed for the whole run; depends only on the training interactions.
    """
    t0 = time.time()
    R = user_item_csr.tocsr()
    du, iu = knn_cosine_sparse(R, k)
    wu, su = gaussian_weights(du, positive_only_median=True)
    L_u = laplacian_from_knn(iu, wu, R.shape[0])
    di, ii = knn_cosine_sparse(R.T.tocsr(), k)
    wi, si = gaussian_weights(di, positive_only_median=True)
    L_i = laplacian_from_knn(ii, wi, R.shape[1])
    L = sp.block_diag([L_u, L_i], format="coo")
    Lt = scipy_to_torch_sparse(L, device or torch.device("cpu"))
    if verbose:
        print(f"    co-occurrence Laplacian nnz={Lt._nnz():,} "
              f"sigma_u={su:.4f} sigma_i={si:.4f} in {time.time()-t0:.1f}s")
    return Lt


# -------------------------------------------------------------------------- loss
def manifold_loss(Z: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Dirichlet energy tr(Z^T L Z) / n, computed as sum(Z * (L Z)) / n."""
    return (Z * torch.sparse.mm(L, Z)).sum() / Z.shape[0]
