"""Gini and long-tail metrics (ported from Cell 6, B.2 and B.3).

P1 fix #5. The notebook's ``tail_coverage`` was the *mean over users of the
fraction of their top-k that is long-tail*, while the dissertation's Eq. 2.18
defines *catalog* coverage. Both are computed here under distinct names so the
text and the code can be made to agree by choosing one, not by re-running:

* ``tail_share_user``      : mean_u  |topk(u) ∩ LongTail| / k         (notebook)
* ``tail_catalog_coverage``: |∪_u topk(u) ∩ LongTail| / |LongTail|     (Eq. 2.18 style)
* ``catalog_coverage``     : |∪_u topk(u)| / |Items|                   (bonus, standard)
"""
from __future__ import annotations

import numpy as np


def gini_from_counts(counts: np.ndarray) -> float:
    """Gini of recommendation frequency across the catalog (0 = uniform)."""
    c = np.sort(np.asarray(counts, dtype=np.float64))
    total = c.sum()
    if total == 0:
        return 0.0
    n = len(c)
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * c) / (n * total))


class DiversityAccumulator:
    """Accumulates per-k recommendation counts and per-user tail shares."""

    def __init__(self, n_items: int, topks: list[int], long_tail_mask: np.ndarray):
        self.topks = topks
        self.n_items = n_items
        self.lt_mask = long_tail_mask.astype(bool)
        self.rec_counts = {k: np.zeros(n_items, dtype=np.int64) for k in topks}
        self.tail_share_sum = {k: 0.0 for k in topks}
        self.n_users = 0

    def update(self, topk_items: np.ndarray) -> None:
        """``topk_items``: (batch, max_k) int array of recommended item ids."""
        lt_hits = self.lt_mask[topk_items]
        self.n_users += topk_items.shape[0]
        for k in self.topks:
            np.add.at(self.rec_counts[k], topk_items[:, :k].ravel(), 1)
            self.tail_share_sum[k] += lt_hits[:, :k].mean(axis=1).sum()

    def finalize(self) -> dict[str, float]:
        out: dict[str, float] = {}
        n_lt = int(self.lt_mask.sum())
        for k in self.topks:
            counts = self.rec_counts[k]
            recommended = counts > 0
            out[f"gini@{k}"] = gini_from_counts(counts)
            out[f"tail_share_user@{k}"] = float(self.tail_share_sum[k] / max(self.n_users, 1))
            out[f"tail_catalog_coverage@{k}"] = float((recommended & self.lt_mask).sum() / max(n_lt, 1))
            out[f"catalog_coverage@{k}"] = float(recommended.sum() / self.n_items)
        return out
