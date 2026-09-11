"""Full-ranking evaluation on the validation or test split (ported from
Cell 6 ``Test``).

Protocol:

* scores are computed for every item, then the user's *known positives* are
  masked out before top-k: training items when evaluating on ``val``,
  training **and** validation items when evaluating on ``test``. Not masking
  validation items at test time would let known positives occupy top-k slots
  and depress every arm's test numbers by the same amount, which is noise,
  not signal;
* propagation (``model.computer()``) runs once per evaluation, not once per
  batch as in the notebook (``getUsersRating`` re-propagated every call);
* per-user loops are replaced by CSR fancy-indexing into a hit matrix. Same
  numbers, minutes faster on Amazon-Book;
* short-head / long-tail breakdowns keep the notebook's convention: a user
  counts toward the ``_short`` average iff they have at least one short-head
  relevant item (likewise ``_long``).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .diversity import DiversityAccumulator
from .ranking import ndcg_at_k, precision_at_k, recall_at_k


def _exclusion_from_csr(csr: sp.csr_matrix, batch_users: np.ndarray):
    sub = csr[batch_users].tocoo()
    return sub.row, sub.col


@torch.no_grad()
def evaluate(model, dataset, split: str = "val", topks=(10, 20, 50),
             batch_size: int = 4096, device: torch.device | None = None,
             all_users=None, all_items=None) -> dict[str, float]:
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

    device = device or next(model.parameters()).device
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

    model.eval()
    if all_users is None:
        all_users, all_items = model.computer()

    for s in range(0, len(users), batch_size):
        bu = users[s:s + batch_size]
        rating = all_users[torch.from_numpy(bu).to(device)] @ all_items.t()
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
