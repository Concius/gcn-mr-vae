"""Ranking metrics over a batch, vectorised (ported from Cell 6).

Inputs are a boolean hit matrix ``hits`` of shape ``(n_users, max_k)`` where
``hits[u, r]`` says whether the item at rank ``r`` (0-based) is relevant, and
``n_rel`` of shape ``(n_users,)`` with the number of relevant items per user.

P1 fix #4 (NDCG). The notebook's IDCG summed discounts over the number of
*hits in the top-k*, so any list whose hits sat at the top scored 1.0
regardless of how many relevant items were missed. The correct ideal list has
``min(k, n_rel)`` relevant items at the top:

    IDCG@k = sum_{i=1}^{min(k, n_rel)} 1 / log2(i + 1)

Recall and precision are unchanged and were validated in the audit.
"""
from __future__ import annotations

import numpy as np


def discounts(max_k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, max_k + 2))


def recall_at_k(hits: np.ndarray, n_rel: np.ndarray, k: int) -> np.ndarray:
    """Per-user recall@k; users with ``n_rel == 0`` get 0."""
    h = hits[:, :k].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(n_rel > 0, h / np.maximum(n_rel, 1), 0.0)
    return r


def precision_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    return hits[:, :k].sum(axis=1) / float(k)


def ndcg_at_k(hits: np.ndarray, n_rel: np.ndarray, k: int) -> np.ndarray:
    """Per-user NDCG@k with the corrected IDCG."""
    disc = discounts(k)
    dcg = (hits[:, :k] * disc[None, :]).sum(axis=1)
    cum = np.cumsum(disc)
    ideal_len = np.minimum(k, n_rel).astype(np.int64)
    idcg = np.where(ideal_len > 0, cum[np.maximum(ideal_len, 1) - 1], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1.0), 0.0)
    return out


def ndcg_single(relevance: np.ndarray, k: int, n_rel: int) -> float:
    """Scalar convenience for tests / spot checks."""
    r = np.asarray(relevance, dtype=bool)[None, :k]
    if r.shape[1] < k:
        r = np.pad(r, ((0, 0), (0, k - r.shape[1])))
    return float(ndcg_at_k(r, np.array([n_rel]), k)[0])
