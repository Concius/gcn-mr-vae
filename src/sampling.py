"""Uniform BPR negative sampling (ported from Cell 5).

Semantics are identical to ``UniformSample_vectorized``: one negative per
training pair, drawn uniformly over the catalog and re-drawn until it is not
one of the user's training positives. The collision check is done with a
single CSR lookup instead of a Python loop over every pair; on Amazon-Book
(~2.4M pairs per epoch) that loop was a noticeable share of epoch time.

Held-out items: collisions are checked against the *reduced* training matrix,
so an item moved into the validation split can be drawn as a negative for its
own user. This is deliberate. Excluding validation items would require the
sampler to consult held-out labels, which leaks them into training; standard
practice is to exclude only training positives. The effect is tiny and
identical for every arm (on Gowalla a user has ~3 validation items out of
40,981, so under 0.01% of draws), and it applies to the control and the
treatment alike.

RNG hygiene: sampling uses its own ``numpy.random.Generator`` seeded from the
run seed. Evaluation and geometric metrics use separate generators, so
evaluating more or less often never changes the negative-sampling stream.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def uniform_sample(train_user: np.ndarray, train_item: np.ndarray,
                   user_item_csr: sp.csr_matrix, n_items: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Return an ``(N, 3)`` int64 array of ``(user, pos, neg)`` triples."""
    n = len(train_user)
    neg = rng.integers(0, n_items, size=n, dtype=np.int64)
    collide = np.asarray(user_item_csr[train_user, neg]).ravel() > 0
    while collide.any():
        idx = np.flatnonzero(collide)
        neg[idx] = rng.integers(0, n_items, size=len(idx), dtype=np.int64)
        collide[idx] = np.asarray(user_item_csr[train_user[idx], neg[idx]]).ravel() > 0
    return np.stack([train_user, train_item, neg], axis=1)


def minibatch(*arrays, batch_size: int):
    n = len(arrays[0])
    for i in range(0, n, batch_size):
        yield tuple(a[i:i + batch_size] for a in arrays)


def n_batches(n: int, batch_size: int) -> int:
    return (n + batch_size - 1) // batch_size
