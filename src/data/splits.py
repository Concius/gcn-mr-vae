"""Validation holdout carved out of the training interactions.

P0 fix #3 from the code audit: the notebook selected checkpoints on the
*test* set. Here a validation set is carved out of ``train.txt`` and used for
checkpoint selection; ``test.txt`` is only read at the end.

Design choices (state them in the dissertation):

* The split is per user: a fraction ``val_frac`` of each user's training
  items (at least one) is held out. Users with fewer than ``min_train``
  training items are left untouched so nobody ends up with an empty
  training profile.
* ``split_seed`` is independent of the run seed. All runs and all seeds see
  the *same* validation set, so paired-by-seed comparisons compare arms on
  identical data; the run seed then only controls initialisation and
  negative sampling.
* The user-item graph (and therefore LightGCN's propagation) is built from
  the reduced training set only. Validation items never enter the adjacency.
  This is the transductive-leak trap; it means absolute numbers will sit a
  little below the notebook's Chapter 5 numbers, which were both trained on
  100% of train and selected on test.
"""
from __future__ import annotations

import numpy as np


def holdout_validation(train_user: np.ndarray, train_item: np.ndarray,
                       val_frac: float = 0.1, split_seed: int = 2020,
                       min_train: int = 2):
    """Split flat (user, item) arrays into train and validation.

    Returns ``(tr_user, tr_item, va_user, va_item)`` as int64 arrays.
    ``val_frac <= 0`` returns everything as train and empty validation
    arrays, which reproduces the notebook protocol (selection on test).
    """
    train_user = np.asarray(train_user, dtype=np.int64)
    train_item = np.asarray(train_item, dtype=np.int64)
    if val_frac <= 0:
        return train_user, train_item, np.empty(0, np.int64), np.empty(0, np.int64)
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in (0, 1)")

    rng = np.random.default_rng(split_seed)
    order = np.argsort(train_user, kind="stable")
    users_sorted = train_user[order]
    items_sorted = train_item[order]
    uniq, start = np.unique(users_sorted, return_index=True)
    end = np.append(start[1:], len(users_sorted))

    is_val = np.zeros(len(users_sorted), dtype=bool)
    for s, e in zip(start, end):
        n = e - s
        if n < min_train:
            continue
        n_val = max(1, int(round(val_frac * n)))
        n_val = min(n_val, n - 1)  # keep at least one training item
        pick = rng.choice(n, size=n_val, replace=False)
        is_val[s + pick] = True

    tr_user, tr_item = users_sorted[~is_val], items_sorted[~is_val]
    va_user, va_item = users_sorted[is_val], items_sorted[is_val]
    return tr_user, tr_item, va_user, va_item


def to_dict(users: np.ndarray, items: np.ndarray) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for u, i in zip(users.tolist(), items.tolist()):
        out.setdefault(u, []).append(i)
    return out
