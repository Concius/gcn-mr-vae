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
from dataclasses import dataclass, field

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


@dataclass
class KnnGraph:
    """k-NN graph for the pairwise manifold term.

    ``idx``   (n, k) neighbour ids, fixed until the next rebuild.
    ``w0``    (n, k) Gaussian kernel weights computed at build time -- exactly
              the values the sparse Laplacian path would use.
    ``sigma`` median-heuristic kernel width, fixed until the next rebuild.
    ``frac_beyond_crossover`` share of neighbour pairs with 1 - cos > sigma^2
              at build time: the unit-sphere form of the net-repulsion rule.
              Off the sphere it can only UNDERCOUNT (see ``net_repulsive``);
              kept because earlier runs logged it.
    ``frac_repel_exact`` share of pairs that are net-repulsive in ``grad`` mode
              under the exact rule, at build time.
    """
    idx: torch.Tensor
    w0: torch.Tensor
    sigma: float
    frac_beyond_crossover: float = float("nan")
    frac_repel_exact: float = float("nan")
    _csr: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def csr_structure(self) -> tuple:
        """Sparsity pattern of S and S^T for the knn_dots backward. Depends only
        on ``idx``, so it is built once per rebuild and reused every batch."""
        if self._csr is None:
            self._csr = csr_structure(self.idx)
        return self._csr


def build_embedding_knn(emb: torch.Tensor, k: int = 20, sigma: float | None = None,
                        backend: str = "torch", device: torch.device | None = None,
                        verbose: bool = True) -> KnnGraph:
    """Same neighbours, same weights and same sigma as
    :func:`build_embedding_laplacian`, kept in pairwise form instead of being
    assembled into a sparse matrix, so the weights can be recomputed (and
    differentiated) from the current embeddings."""
    device = device or emb.device
    t0 = time.time()
    if backend == "torch":
        d, i = knn_cosine_torch(emb, k)
    elif backend == "sklearn":
        d, i = knn_cosine_sklearn(emb, k)
    else:
        raise ValueError(f"unknown knn backend {backend!r}")
    w, sigma = gaussian_weights(d, sigma)
    g = KnnGraph(idx=torch.from_numpy(np.ascontiguousarray(i)).long().to(device),
                 w0=torch.from_numpy(np.ascontiguousarray(w)).to(device),
                 sigma=float(sigma),
                 frac_beyond_crossover=float(np.mean(2.0 * d > 2.0 * sigma ** 2)))
    with torch.no_grad():
        g.frac_repel_exact = float(kernel_force_diagnostics(emb.to(device), g)["frac_repel_exact"])
    if verbose:
        print(f"    kNN graph (n={emb.shape[0]:,}, k={k}, {backend}) pairs={g.idx.numel():,} "
              f"sigma={sigma:.4f} net-repulsive={100*g.frac_repel_exact:.2f}% "
              f"(sphere rule {100*g.frac_beyond_crossover:.2f}%) in {time.time()-t0:.1f}s")
    return g


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


KERNEL_MODES = ("frozen", "fresh", "grad", "grad_weaken", "grad_repel")
SPLIT_MODES = ("grad_weaken", "grad_repel")


def csr_structure(idx: torch.Tensor) -> tuple:
    """Canonical CSR patterns (column indices sorted within every row) for
    S (row i holds idx[i, :]) and for S^T, plus the permutations that map the
    flat (n*k,) gradient onto each pattern.

    Sorted indices are required so the backward does not depend on how a given
    backend treats unsorted CSR: PyTorch's CPU multiply tolerates them, but
    cuSPARSE routines may assume sorted columns, and that cannot be tested
    without the GPU. Built once per rebuild; per batch only values are permuted.
    """
    n, k = idx.shape
    dev = idx.device
    order = torch.argsort(idx, dim=1)                              # sort each row's neighbours
    perm_s = ((torch.arange(n, device=dev) * k).unsqueeze(1) + order).reshape(-1)
    cols_s = idx.gather(1, order).reshape(-1).contiguous()
    crow_s = torch.arange(0, n * k + 1, k, device=dev)
    rows = torch.arange(n, device=dev).repeat_interleave(k)
    flat_cols = idx.reshape(-1)
    perm_t = torch.argsort(flat_cols, stable=True)                 # group by column; stable keeps
    crow_t = torch.zeros(n + 1, dtype=torch.int64, device=dev)     # rows ascending within a group
    crow_t[1:] = torch.cumsum(torch.bincount(flat_cols, minlength=n), 0)
    col_t = rows[perm_t].contiguous()
    return crow_s, cols_s, perm_s, crow_t, col_t, perm_t


class _KnnDots(torch.autograd.Function):
    """dots[i, m] = z_i . z_{idx[i, m]} for the k-NN pairs.

    Autograd through a plain gather would store an (n, k, d) tensor and then
    scatter its gradient back into Z, which dominates the cost (measured on
    Gowalla scale: 4.5 s of a 6 s step). Instead the forward is computed in row
    chunks without keeping the gathered tensor, and the backward uses

        dL/dZ = S Z + S^T Z,   S[i, idx[i, m]] = dL/d dots[i, m],

    with the sparsity pattern precomputed once per rebuild (csr_structure).
    """

    @staticmethod
    def forward(ctx, Z, idx, chunk, struct):
        n = idx.shape[0]
        out = torch.empty(idx.shape, dtype=Z.dtype, device=Z.device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            out[s:e] = torch.bmm(Z[idx[s:e]], Z[s:e].unsqueeze(2)).squeeze(2)
        ctx.save_for_backward(Z)
        ctx.struct = struct
        return out

    @staticmethod
    def backward(ctx, grad_dots):
        (Z,) = ctx.saved_tensors
        crow_s, cols_s, perm_s, crow_t, col_t, perm_t = ctx.struct
        n = Z.shape[0]
        v = grad_dots.reshape(-1).to(Z.dtype)
        S = torch.sparse_csr_tensor(crow_s, cols_s, v[perm_s].contiguous(), (n, n))
        St = torch.sparse_csr_tensor(crow_t, col_t, v[perm_t].contiguous(), (n, n))
        return S @ Z + St @ Z, None, None, None


def knn_dots(Z: torch.Tensor, idx: torch.Tensor, chunk: int = 4096,
             struct: tuple | None = None) -> torch.Tensor:
    if struct is None:
        struct = csr_structure(idx)
    return _KnnDots.apply(Z, idx, chunk, struct)


def _energy_from_dots(Z: torch.Tensor, g: KnnGraph, mode: str, dots: torch.Tensor) -> torch.Tensor:
    n = Z.shape[0]
    sq = (Z * Z).sum(1)                                    # (n,)    ||z_i||^2
    d2 = (sq.unsqueeze(1) + sq[g.idx] - 2.0 * dots).clamp_min(0.0)
    if mode == "frozen":
        w = g.w0.to(Z.dtype)
    else:
        nrm = sq.clamp_min(1e-16).sqrt()                   # clamp before sqrt: finite grad at 0
        cos = dots / (nrm.unsqueeze(1) * nrm[g.idx])
        w = torch.exp(-(2.0 * (1.0 - cos)) / (2.0 * g.sigma ** 2))
        if mode == "fresh":
            w = w.detach()
        elif mode in SPLIT_MODES:
            # Same value as `grad` everywhere; the gradient through w is kept
            # only on the routed pairs and stopped on the rest (which therefore
            # behave exactly as in `fresh`).
            repel = net_repulsive(d2.detach(), nrm.detach(), g)
            route = repel if mode == "grad_repel" else ~repel
            w = torch.where(route, w, w.detach())
    return 0.5 * (w * d2).sum() / n


def net_repulsive(d2: torch.Tensor, nrm: torch.Tensor, g: KnnGraph) -> torch.Tensor:
    """Pairs whose own net angular force is repulsive in ``grad`` mode.

    Per pair, E = w(cos) ||z_i - z_j||^2 with w = exp(-(1 - cos)/sigma^2).
    The kernel term d2 * grad(w) is purely tangential and pushes the pair apart
    in angle; the tangential part of the attraction w * grad(d2) pulls it
    together. On z_i their coefficients along the direction towards z_j are
    d2 w / (sigma^2 |z_i|) (away) and 2 w |z_j| (towards), so the pair is
    net-repulsive iff

        d2 > 2 sigma^2 |z_i| |z_j|,

    a condition symmetric in i and j, so both endpoints agree. Dividing by
    |z_i||z_j| gives 2(1 - cos) + (r + 1/r - 2) > 2 sigma^2 with r = |z_i|/|z_j|:
    on the unit sphere this is 1 - cos > sigma^2 (``frac_beyond_crossover``),
    and off it the extra term is >= 0, so the sphere rule only undercounts.

    This is the sign of the pair term's force in isolation. A node's actual
    motion also depends on its other pairs, on BPR, on back-propagation through
    LightGCN, and on Adam.
    """
    return d2 > 2.0 * g.sigma ** 2 * nrm.unsqueeze(1) * nrm[g.idx]


@torch.no_grad()
def kernel_force_diagnostics(Z: torch.Tensor, g: KnnGraph) -> dict:
    """Descriptive force budget for the current embeddings and graph.

    Per pair, tangential magnitudes summed over both endpoints (the common
    factor 0.5/n is dropped; it cancels in every ratio reported):
        kernel    K = d2 * w * sin(theta) * (1/|z_i| + 1/|z_j|) / sigma^2
        attraction A = 2 * w * sin(theta) * (|z_i| + |z_j|)
    These are sums of per-pair magnitudes, not the norm of the net gradient.

    Returns shares of pairs that are net-repulsive under the exact and the
    unit-sphere rule, the share of the kernel budget carried by net-repulsive
    pairs, and the overall kernel / attraction ratio.
    """
    dots = knn_dots(Z, g.idx, struct=g.csr_structure())
    sq = (Z * Z).sum(1)
    d2 = (sq.unsqueeze(1) + sq[g.idx] - 2.0 * dots).clamp_min(0.0)
    nrm = sq.clamp_min(1e-16).sqrt()
    ni, nj = nrm.unsqueeze(1), nrm[g.idx]
    cos = (dots / (ni * nj)).clamp(-1.0, 1.0)
    s2 = g.sigma ** 2
    w = torch.exp(-(1.0 - cos) / s2)
    sin = (1.0 - cos * cos).clamp_min(0.0).sqrt()
    K = d2 * w * sin * (1.0 / ni + 1.0 / nj) / s2
    A = 2.0 * w * sin * (ni + nj)
    repel = net_repulsive(d2, nrm, g)
    k_tot = K.sum()
    return {
        "frac_repel_exact": float(repel.float().mean()),
        "frac_repel_sphere": float(((1.0 - cos) > s2).float().mean()),
        # undefined (not zero) when the kernel force vanishes, e.g. weights underflow
        "kernel_share_repel": float(K[repel].sum() / k_tot) if k_tot > 0 else float("nan"),
        "kernel_to_attraction": float(k_tot / A.sum()) if A.sum() > 0 else float("nan"),
    }


def pairwise_manifold_loss(Z: torch.Tensor, g: KnnGraph, mode: str) -> torch.Tensor:
    """The same Dirichlet energy in pairwise form.

    For L = D - W with W = (W_knn + W_knn^T)/2 (as built by
    :func:`laplacian_from_knn`), the identity

        tr(Z^T L Z) = 1/2 * sum_i sum_{j in kNN(i)} w_ij * ||z_i - z_j||^2

    holds exactly, so with ``mode='frozen'`` this returns the value and the
    gradient of :func:`manifold_loss` (verified in tests/test_kernel_weights.py).

    ``mode``:
      * ``frozen`` -- w_ij are the build-time weights ``g.w0`` (constants).
      * ``fresh``  -- w_ij recomputed from the current Z, then detached.
      * ``grad``   -- w_ij recomputed from the current Z and kept in the graph,
                      so the gradient also flows through the kernel.
      * ``grad_weaken`` / ``grad_repel`` -- same value as ``grad``, but the
                      gradient through the kernel is kept only on pairs that
                      are not / are net-repulsive (:func:`net_repulsive`,
                      re-evaluated every call) and stopped on the others.
                      ``grad_weaken + grad_repel - fresh == grad`` for the
                      gradient, exactly. Like ``fresh``, a routed gradient is
                      not the derivative of the reported value: it descends a
                      surrogate in which w is held fixed on unrouted pairs.
    Recomputed weights use the build-time formula: cosine distance on the
    embeddings, w = exp(-2(1 - cos)/(2 sigma^2)), with ``g.sigma`` held fixed.

    Every quantity is expressed through the pair dot products z_i . z_j and the
    squared norms, so the only (n, k, d) interaction goes through
    :func:`knn_dots`, whose backward is a sparse-dense product.
    """
    if mode not in KERNEL_MODES:
        raise ValueError(f"mode must be one of {KERNEL_MODES}, got {mode!r}")
    return _energy_from_dots(Z, g, mode, knn_dots(Z, g.idx, struct=g.csr_structure()))


def _pairwise_manifold_loss_reference(Z: torch.Tensor, g: KnnGraph, mode: str) -> torch.Tensor:
    """Plain-autograd reference (gathers an (n, k, d) tensor). Slow; kept only
    so the tests can check the fast path against it."""
    if mode not in KERNEL_MODES:
        raise ValueError(f"mode must be one of {KERNEL_MODES}, got {mode!r}")
    dots = torch.bmm(Z[g.idx], Z.unsqueeze(2)).squeeze(2)
    return _energy_from_dots(Z, g, mode, dots)
