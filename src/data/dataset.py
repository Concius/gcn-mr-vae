"""Interaction dataset in the LightGCN ``train.txt`` / ``test.txt`` format.

Ported from notebook Cell 3 (``Loader``). Differences:

* validation split (see ``splits.py``) -- ``valDict`` and ``ValNet``;
* the normalised adjacency is built with ``scipy.sparse.bmat`` instead of
  the lil/dok loop (same matrix, seconds instead of minutes on Amazon-Book);
* the adjacency cache filename encodes the split so a validation-split
  graph never collides with the original ``s_pre_adj_mat.npz``;
* no global ``config``: the device is passed explicitly.

The ml-1m sequential path from the notebook was not ported; nothing in
Chapter 5 uses it.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from .splits import holdout_validation, to_dict


def _read_lightgcn_file(path: str | os.PathLike):
    users, items = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                continue
            its = [int(x) for x in parts[1:] if x]
            if not its:
                continue
            uid = int(parts[0])
            users.extend([uid] * len(its))
            items.extend(its)
    return np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)


class InteractionDataset:
    def __init__(self, name: str, root: str | os.PathLike = "data",
                 val_frac: float = 0.1, split_seed: int = 2020,
                 min_train: int = 2, short_head_frac: float = 0.2,
                 popularity_from: str = "full_train", verbose: bool = True):
        self.name = name
        self.path = Path(root) / name
        self.val_frac = val_frac
        self.split_seed = split_seed
        self.min_train = min_train
        self.short_head_frac = short_head_frac
        self.popularity_from = popularity_from

        full_tr_u, full_tr_i = _read_lightgcn_file(self.path / "train.txt")
        te_u, te_i = _read_lightgcn_file(self.path / "test.txt")

        # id space is defined by the union of train and test, as in the notebook
        self.n_users = int(max(full_tr_u.max(), te_u.max())) + 1
        self.n_items = int(max(full_tr_i.max(), te_i.max())) + 1

        tr_u, tr_i, va_u, va_i = holdout_validation(
            full_tr_u, full_tr_i, val_frac=val_frac,
            split_seed=split_seed, min_train=min_train)

        self.trainUser, self.trainItem = tr_u, tr_i
        self.fullTrainItem = full_tr_i
        self.valUser, self.valItem = va_u, va_i
        self.testUser, self.testItem = te_u, te_i
        self.n_train_full = len(full_tr_u)

        self.UserItemNet = sp.csr_matrix(
            (np.ones(len(tr_u), dtype=np.float32), (tr_u, tr_i)),
            shape=(self.n_users, self.n_items))
        self.ValNet = sp.csr_matrix(
            (np.ones(len(va_u), dtype=np.float32), (va_u, va_i)),
            shape=(self.n_users, self.n_items))
        self.TestNet = sp.csr_matrix(
            (np.ones(len(te_u), dtype=np.float32), (te_u, te_i)),
            shape=(self.n_users, self.n_items))

        self.valDict = to_dict(va_u, va_i)
        self.testDict = to_dict(te_u, te_i)

        self._graph = None
        self._popularity_groups = None
        self._long_tail_mask = None

        if verbose:
            print(f"[{name}] users={self.n_users:,} items={self.n_items:,} "
                  f"train={len(tr_u):,} (of {self.n_train_full:,}) "
                  f"val={len(va_u):,} test={len(te_u):,}")

    # ------------------------------------------------------------------ props
    @property
    def m_items(self) -> int:  # notebook name, kept for readability of diffs
        return self.n_items

    @property
    def trainDataSize(self) -> int:
        return len(self.trainUser)

    def all_pos(self, users):
        return [self.UserItemNet[u].indices for u in users]

    # ------------------------------------------------------------------ graph
    def _adj_cache_path(self) -> Path:
        tag = f"val{self.val_frac:g}_s{self.split_seed}" if self.val_frac > 0 else "full"
        return self.path / f"norm_adj_{tag}.npz"

    def normalized_adjacency(self) -> sp.csr_matrix:
        """D^-1/2 (A) D^-1/2 for the bipartite graph of *training* edges."""
        cache = self._adj_cache_path()
        if cache.exists():
            return sp.load_npz(cache).tocsr()
        R = self.UserItemNet.tocsr()
        A = sp.bmat([[None, R], [R.T, None]], format="csr", dtype=np.float32)
        rowsum = np.asarray(A.sum(axis=1)).flatten()
        d_inv = np.power(rowsum, -0.5, where=rowsum > 0, out=np.zeros_like(rowsum))
        D = sp.diags(d_inv)
        norm_adj = (D @ A @ D).tocsr().astype(np.float32)
        sp.save_npz(cache, norm_adj)
        return norm_adj

    def sparse_graph(self, device: torch.device) -> torch.Tensor:
        if self._graph is None:
            coo = self.normalized_adjacency().tocoo()
            idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
            val = torch.from_numpy(coo.data.astype(np.float32))
            self._graph = torch.sparse_coo_tensor(
                idx, val, torch.Size(coo.shape), dtype=torch.float32).coalesce()
        if self._graph.device.type != device.type:
            self._graph = self._graph.to(device)
        return self._graph

    # ------------------------------------------------------------- popularity
    @property
    def popularity_groups(self) -> dict[str, set[int]]:
        """Top ``short_head_frac`` of items by training interaction count.

        ``popularity_from='full_train'`` (default) segments on the original
        ``train.txt``, matching Chapter 5 and keeping the segmentation
        independent of ``val_frac``; ``'train'`` uses the reduced training set.
        Items never seen in training belong to neither group.
        """
        if self._popularity_groups is None:
            src = self.fullTrainItem if self.popularity_from == "full_train" else self.trainItem
            counts = np.bincount(src, minlength=self.n_items)
            present = np.flatnonzero(counts > 0)
            order = present[np.argsort(-counts[present], kind="stable")]
            n_short = int(len(order) * self.short_head_frac)
            short = set(order[:n_short].tolist())
            long = set(order[n_short:].tolist())
            self._popularity_groups = {"short_head": short, "long_tail": long}
        return self._popularity_groups

    @property
    def long_tail_mask(self) -> np.ndarray:
        if self._long_tail_mask is None:
            m = np.zeros(self.n_items, dtype=bool)
            m[list(self.popularity_groups["long_tail"])] = True
            self._long_tail_mask = m
        return self._long_tail_mask

    def summary(self) -> dict:
        return {
            "name": self.name, "n_users": self.n_users, "n_items": self.n_items,
            "n_train": int(len(self.trainUser)), "n_train_full": int(self.n_train_full),
            "n_val": int(len(self.valUser)), "n_test": int(len(self.testUser)),
            "val_frac": self.val_frac, "split_seed": self.split_seed,
            "min_train": self.min_train, "short_head_frac": self.short_head_frac,
            "popularity_from": self.popularity_from,
        }
