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
import torch


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


# ------------------------------------------------------------ user batches
# Separate, named random streams. The BPR sampler above is seeded with the
# bare run seed (``default_rng(seed)``, unchanged so existing runs reproduce).
# Streams added for Module 3 are derived from (seed, stream id), which gives
# statistically independent generators: a run that uses both BPR triples and
# user batches (Phase 3) draws them from different streams, and adding the
# second cannot shift the first.
USER_BATCH_STREAM = 1
VAE_NOISE_STREAM = 2


def stream_rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(stream)])


def stream_torch_generator(seed: int, stream: int, device) -> torch.Generator:
    s = int(np.random.SeedSequence([int(seed), int(stream)]).generate_state(1, dtype=np.uint64)[0] >> 1)
    return torch.Generator(device=device).manual_seed(s)


def users_with_interactions(user_item_csr: sp.csr_matrix) -> np.ndarray:
    return np.flatnonzero(np.diff(user_item_csr.indptr) > 0).astype(np.int64)


def user_batches(users: np.ndarray, batch_size: int, rng: np.random.Generator):
    """One epoch of user-level minibatches: a fresh permutation of ``users``
    from ``rng``, cut into consecutive batches. Every user appears exactly once
    per epoch (the VAE-CF convention of Liang et al.)."""
    perm = users[rng.permutation(len(users))]
    for i in range(0, len(perm), batch_size):
        yield perm[i:i + batch_size]


def dense_rows(user_item_csr: sp.csr_matrix, users: np.ndarray, device) -> torch.Tensor:
    """Rows of the interaction matrix as a dense float32 ``(len(users), n_items)``
    tensor on ``device``, built on the device from the CSR coordinates (no
    host-side dense copy)."""
    sub = user_item_csr[users].tocoo()
    out = torch.zeros((len(users), user_item_csr.shape[1]), dtype=torch.float32, device=device)
    if sub.nnz:
        r = torch.from_numpy(sub.row.astype(np.int64)).to(device)
        c = torch.from_numpy(sub.col.astype(np.int64)).to(device)
        out[r, c] = torch.from_numpy(sub.data.astype(np.float32)).to(device)
    return out


def minibatch(*arrays, batch_size: int):
    n = len(arrays[0])
    for i in range(0, n, batch_size):
        yield tuple(a[i:i + batch_size] for a in arrays)


def n_batches(n: int, batch_size: int) -> int:
    return (n + batch_size - 1) // batch_size
