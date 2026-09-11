"""LightGCN encoder (He et al., SIGIR 2020), ported from notebook Cell 5.

The propagation is unchanged and was validated against the official
implementation in the June 2026 audit:

    E^(l+1) = Â E^(l),   E = mean(E^(0), ..., E^(K))

``n_layers=0`` gives the ego embeddings back unchanged, i.e. plain MF-BPR.
That is the MF-BPR baseline; no separate model class is needed.

Differences from the notebook class:

* constructor takes explicit sizes/hyperparameters instead of a global
  ``config`` and a dataset object;
* ``users_rating`` returns raw dot products. The notebook applied a sigmoid
  before ranking, which is monotone and therefore never changes any top-k;
* the ``keep_prob``/``A_split`` dropout knobs, which were always disabled in
  Chapter 5, were not ported.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 256,
                 n_layers: int = 3, init_std: float = 0.1,
                 graph: torch.Tensor | None = None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.embedding_user = nn.Embedding(n_users, dim)
        self.embedding_item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.embedding_user.weight, std=init_std)
        nn.init.normal_(self.embedding_item.weight, std=init_std)
        # sparse normalised adjacency; registered as a buffer-like attribute
        # (not a Parameter, not saved in state_dict)
        self.graph = graph

    # ---------------------------------------------------------------- forward
    def ego_embeddings(self) -> torch.Tensor:
        """Concatenated raw embedding table (users then items)."""
        return torch.cat([self.embedding_user.weight, self.embedding_item.weight])

    def computer(self):
        """Propagate and return ``(all_users, all_items)``.

        Name kept from the reference implementation so that Chapter 5 text
        that references ``computer()`` still reads correctly.
        """
        all_emb = self.ego_embeddings()
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)
        out = torch.stack(embs, dim=1).mean(dim=1)
        return torch.split(out, [self.n_users, self.n_items])

    @torch.no_grad()
    def users_rating(self, users: torch.Tensor, all_users=None, all_items=None):
        if all_users is None:
            all_users, all_items = self.computer()
        return all_users[users] @ all_items.t()

    def state_for_checkpoint(self) -> dict:
        return {"n_users": self.n_users, "n_items": self.n_items,
                "dim": self.dim, "n_layers": self.n_layers}
