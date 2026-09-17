"""Full-ranking evaluation on the validation or test split (ported from
Cell 6 ``Test``).

Scoring is delegated to a ``Scorer``: a callable that maps a batch of user
ids (``LongTensor`` on ``device``) to a dense ``(batch, n_items)`` float score
matrix on the same device. ``evaluate`` never touches the model; the caller
decides how R̂ is produced:

* LightGCN / MF-BPR: ``dot_product_scorer(all_users, all_items)`` over
  propagated embeddings, computed once by the caller.
* VAE (Module 3): a closure around ``decoder(mu_u)`` — score with the
  posterior *mean*, not a sample, at evaluation time (Liang et al. 2018).
* VAE-CF baseline: a closure that pulls ``r_u`` rows from ``dataset.UserItemNet``.

Exclusion of known positives stays here, because it is protocol, not model.
Note that ``evaluate`` masks known positives **in place** on the tensor the
scorer returns, so a scorer must hand back a freshly allocated tensor, never a
cached or shared one (``dot_product_scorer`` returns a matmul result, which is
fresh).

Protocol:

* scores are computed for every item, then the user's *known positives* are
  masked out before top-k: training items when evaluating on ``val``,
  training **and** validation items when evaluating on ``test``. Not masking
  validation items at test time would let known positives occupy top-k slots
  and depress every arm's test numbers by the same amount, which is noise,
  not signal;
* the caller propagates once per evaluation and passes the result in through
  the scorer, rather than re-propagating per batch as the notebook's
  ``getUsersRating`` did;
* the notebook's per-user Python loop is replaced by an integer-key
  ``np.isin`` against the ground-truth pairs, giving the hit matrix in one
  vectorised step. Same numbers, minutes faster on Amazon-Book;
* short-head / long-tail breakdowns keep the notebook's convention: a user
  counts toward the ``_short`` average iff they have at least one short-head
  relevant item (likewise ``_long``).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch

from .diversity import DiversityAccumulator
from .ranking import ndcg_at_k, precision_at_k, recall_at_k


Scorer = Callable[[torch.Tensor], torch.Tensor]


def dot_product_scorer(all_users: torch.Tensor, all_items: torch.Tensor) -> Scorer:
    """R̂ = U Iᵀ over (propagated) embeddings. The notebook's scoring rule."""
    def score(users: torch.Tensor) -> torch.Tensor:
        return all_users[users] @ all_items.t()
    return score


def _exclusion_from_csr(csr: sp.csr_matrix, batch_users: np.ndarray):
    sub = csr[batch_users].tocoo()
    return sub.row, sub.col


@torch.no_grad()
def evaluate(scorer: Scorer, dataset, split: str = "val", topks=(10, 20, 50),
             batch_size: int = 4096, device: torch.device = torch.device("cpu")) -> dict[str, float]:
    if split == "val":
        gt_csr, gt_dict = dataset.ValNet, dataset.valDict
        exclude = [dataset.UserItemNet]
    elif split == "test":
        gt_csr, gt_dict = dataset.TestNet, dataset.testDict
        exclude = [dataset.UserItemNet, dataset.ValNet]
    else:
        raise ValueError(split)
    if not gt_dict:
        return {}

    topks = list(topks)
    max_k = max(topks)
    users = np.array(sorted(gt_dict.keys()), dtype=np.int64)

    short_mask = np.zeros(dataset.n_items, dtype=bool)
    short_mask[list(dataset.popularity_groups["short_head"])] = True
    long_mask = dataset.long_tail_mask

    sums = {}
    for grp in ("", "_short", "_long"):
        for k in topks:
            for m in ("recall", "precision", "ndcg"):
                sums[f"{m}{grp}@{k}"] = 0.0
    n_users = {"": 0, "_short": 0, "_long": 0}
    div = DiversityAccumulator(dataset.n_items, topks, long_mask)

    for s in range(0, len(users), batch_size):
        bu = users[s:s + batch_size]
        rating = scorer(torch.from_numpy(bu).to(device))
        if rating.shape != (len(bu), dataset.n_items):
            raise ValueError(f"scorer returned {tuple(rating.shape)}, expected {(len(bu), dataset.n_items)}")
        for csr in exclude:
            r, c = _exclusion_from_csr(csr, bu)
            if len(r):
                rating[torch.from_numpy(r).to(device), torch.from_numpy(c).to(device)] = -float("inf")
        topk = torch.topk(rating, k=max_k, dim=1).indices.cpu().numpy()   # (b, max_k)
        del rating

        gt_rows = gt_csr[bu].tocoo()                                        # (b, n_items) sparse
        n_items = dataset.n_items
        gt_keys = gt_rows.row.astype(np.int64) * n_items + gt_rows.col.astype(np.int64)
        topk_keys = np.arange(len(bu), dtype=np.int64)[:, None] * n_items + topk.astype(np.int64)
        hits = np.isin(topk_keys, gt_keys)                                  # (b, max_k) bool
        n_rel = np.bincount(gt_rows.row, minlength=len(bu)).astype(np.int64)
        gt_rows = gt_rows.tocsr()

        hits_s = hits & short_mask[topk]
        hits_l = hits & long_mask[topk]
        n_rel_s = np.asarray((gt_rows @ short_mask.astype(np.float32))).ravel().astype(np.int64)
        n_rel_l = np.asarray((gt_rows @ long_mask.astype(np.float32))).ravel().astype(np.int64)

        for grp, h, nr in (("", hits, n_rel), ("_short", hits_s, n_rel_s), ("_long", hits_l, n_rel_l)):
            valid = nr > 0
            n_users[grp] += int(valid.sum())
            for k in topks:
                sums[f"recall{grp}@{k}"] += recall_at_k(h, nr, k)[valid].sum()
                sums[f"precision{grp}@{k}"] += precision_at_k(h, k)[valid].sum()
                sums[f"ndcg{grp}@{k}"] += ndcg_at_k(h, nr, k)[valid].sum()
        div.update(topk)

    out: dict[str, float] = {}
    for key, val in sums.items():
        grp = "_short" if "_short" in key else "_long" if "_long" in key else ""
        out[key] = float(val / n_users[grp]) if n_users[grp] else 0.0
    out.update(div.finalize())
    out["n_users_eval"] = int(n_users[""])
    return out
