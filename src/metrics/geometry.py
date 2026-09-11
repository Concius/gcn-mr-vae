"""Geometric diagnostics (ported from Cells 6, 8 and Block 6/7).

* ``effective_rank``: Roy & Vetterli (2007) on the *propagated* all-node
  matrix, mean-centred, subsampled to ``max_samples`` rows. P2 fix #8: the
  centring is now an explicit, documented parameter; Eq. 3.2 in the text
  should state that rows are mean-centred before the SVD.
* ``neighborhood_preservation``: NP@k between two snapshots of the *ego*
  embedding table (the notebook's B.1 metric).
* ``np_ref``: NP@k of *propagated item* embeddings against the external
  Jaccard item-item graph from the training interactions (Block 6). This is
  the non-circular reference used in the qualification text. P2 fix #6: it is
  computed for every arm at the selected checkpoint with a fixed subsample
  seed, so all arms are measured against the same reference.

RNG hygiene: all subsampling uses local generators; nothing here touches the
global torch/numpy RNG streams that training uses.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from ..models.mr_layer import knn_cosine_sklearn, knn_cosine_torch


@torch.no_grad()
def effective_rank(Z: torch.Tensor, max_samples: int = 10000, center: bool = True,
                   seed: int = 0) -> float:
    z = Z.detach().float().cpu()
    if z.shape[0] > max_samples:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(z.shape[0], generator=g)[:max_samples]
        z = z[idx]
    if center:
        z = z - z.mean(dim=0, keepdim=True)
    try:
        S = torch.linalg.svdvals(z)
    except Exception:  # pragma: no cover - fallback mirrors the notebook
        _, S, _ = torch.svd_lowrank(z, q=min(z.shape[1], 128))
    S = S[S > 1e-10]
    p = S / S.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()).item())


def _knn_idx(emb: torch.Tensor, k: int, backend: str) -> np.ndarray:
    if backend == "torch":
        return knn_cosine_torch(emb, k)[1]
    return knn_cosine_sklearn(emb, k)[1]


def _mean_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    # rows are neighbour id lists; overlap = |a_i ∩ b_i| / k
    n = a.shape[0]
    a_s = np.sort(a, axis=1)
    b_s = np.sort(b, axis=1)
    overlap = 0
    for i in range(n):
        overlap += len(np.intersect1d(a_s[i], b_s[i], assume_unique=True))
    return overlap / (n * k)


@torch.no_grad()
def neighborhood_preservation(emb_before: torch.Tensor, emb_after: torch.Tensor,
                              k: int = 10, max_samples: int = 5000, seed: int = 42,
                              backend: str = "torch") -> float:
    """NP@k between two embedding snapshots of the same nodes."""
    n = emb_before.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = torch.from_numpy(np.sort(rng.choice(n, max_samples, replace=False)))
        emb_before, emb_after = emb_before[idx.to(emb_before.device)], emb_after[idx.to(emb_after.device)]
    nb = _knn_idx(emb_before, k, backend)
    na = _knn_idx(emb_after, k, backend)
    return float(_mean_overlap(nb, na, k))


def jaccard_item_knn(user_item_csr: sp.csr_matrix, item_indices: np.ndarray, k: int) -> np.ndarray:
    """k-NN by Jaccard similarity of user sets, restricted to ``item_indices``
    (returns indices *relative* to ``item_indices``). Exact port of Block 6."""
    sub = user_item_csr[:, item_indices]
    co = (sub.T @ sub).toarray().astype(np.float64)
    deg = np.diag(co).copy()
    union = deg[:, None] + deg[None, :] - co
    with np.errstate(divide="ignore", invalid="ignore"):
        jacc = np.where(union > 0, co / union, 0.0)
    np.fill_diagonal(jacc, -1.0)
    return np.argpartition(-jacc, kth=k, axis=1)[:, :k]


@torch.no_grad()
def np_ref(item_emb: torch.Tensor, user_item_csr: sp.csr_matrix, k: int = 20,
           max_samples: int = 5000, seed: int = 42, backend: str = "torch") -> float:
    """NP_ref@k: overlap between embedding k-NN and Jaccard k-NN over items."""
    n = item_emb.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, max_samples, replace=False)) if n > max_samples else np.arange(n)
    nbr_ref = jaccard_item_knn(user_item_csr, idx, k)
    sub = item_emb[torch.from_numpy(idx).to(item_emb.device)]
    nbr_emb = _knn_idx(sub, k, backend)
    return float(_mean_overlap(nbr_ref, nbr_emb, k))
