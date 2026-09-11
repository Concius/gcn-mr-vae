"""Composable training objective.

The notebook had three near-identical classes (``BPRLoss``, ``BPRLoss_MR``,
``BPRLoss_MR_CoOccurrence``) whose ``stageOne`` differed only in whether a
manifold term was added. Here there is one objective:

    L = L_BPR + decay * L_reg  [+ lambda_m * tr(Z^T L Z)/n]  [+ extra terms]

Every term reads from a shared ``BatchContext`` (embeddings are propagated
once per batch) and returns a scalar plus a dict of numbers to log. Adding
the VAE means adding one ``LossTerm`` (see ``vae.py``), not a fourth class
and a fourth training loop.

Numerics of BPR and the L2 term are exactly the notebook's:

    L_BPR = mean(softplus(s_neg - s_pos))
    L_reg = 1/2 * (||e_u||^2 + ||e_p||^2 + ||e_n||^2) / |batch|   (ego embeddings)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F

from .mr_layer import manifold_loss


@dataclass
class BatchContext:
    """Everything a loss term may need for one minibatch."""
    model: torch.nn.Module
    users: torch.Tensor
    pos: torch.Tensor
    neg: torch.Tensor
    all_users: torch.Tensor      # propagated, (n_users, d)
    all_items: torch.Tensor      # propagated, (n_items, d)
    epoch: int
    laplacian: torch.Tensor | None = None
    extras: dict = field(default_factory=dict)

    @property
    def all_emb(self) -> torch.Tensor:
        return torch.cat([self.all_users, self.all_items])


class LossTerm(Protocol):
    name: str

    def __call__(self, ctx: BatchContext) -> tuple[torch.Tensor, dict]: ...


# ----------------------------------------------------------------------- terms
class BPRTerm:
    name = "bpr"

    def __call__(self, ctx: BatchContext):
        u = ctx.all_users[ctx.users]
        p = ctx.all_items[ctx.pos]
        n = ctx.all_items[ctx.neg]
        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)
        loss = F.softplus(neg_scores - pos_scores).mean()
        return loss, {"bpr": loss.item()}


class L2EgoTerm:
    """L2 on the *ego* embeddings of the batch, scaled by ``decay``."""
    name = "l2"

    def __init__(self, decay: float):
        self.decay = decay

    def __call__(self, ctx: BatchContext):
        m = ctx.model
        eu = m.embedding_user(ctx.users)
        ep = m.embedding_item(ctx.pos)
        en = m.embedding_item(ctx.neg)
        reg = 0.5 * (eu.norm(2).pow(2) + ep.norm(2).pow(2) + en.norm(2).pow(2)) / float(len(ctx.users))
        return self.decay * reg, {"l2": reg.item()}


class ManifoldTerm:
    """lambda * tr(Z^T L Z)/n over the full propagated embedding matrix.

    ``lam`` is the *effective* weight applied in every batch. The trainer is
    responsible for the ``lambda_scale`` policy (see trainer docs): with
    ``lambda_scale=none`` this is the notebook behaviour, where the term is
    applied once per minibatch and therefore ``n_batches`` times per epoch.
    """
    name = "manifold"

    def __init__(self, lam: float):
        self.lam = lam

    def __call__(self, ctx: BatchContext):
        if ctx.laplacian is None or self.lam <= 0:
            return ctx.all_users.new_zeros(()), {"manifold": 0.0}
        m = manifold_loss(ctx.all_emb, ctx.laplacian)
        return self.lam * m, {"manifold": m.item()}


# -------------------------------------------------------------------- composer
class CompositeLoss:
    """Sum of terms. ``active`` lets the trainer switch terms on per phase."""

    def __init__(self, terms: list[LossTerm]):
        self.terms = terms
        self.active: set[str] = {t.name for t in terms}

    def set_active(self, names: set[str]) -> None:
        self.active = set(names)

    def __call__(self, ctx: BatchContext) -> tuple[torch.Tensor, dict]:
        total = ctx.all_users.new_zeros(())
        logs: dict[str, float] = {}
        for t in self.terms:
            if t.name not in self.active:
                continue
            val, lg = t(ctx)
            total = total + val
            logs.update(lg)
        logs["total"] = total.item()
        return total, logs
