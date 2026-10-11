"""Variational pieces for Module 3.

What is here (October track, step 1 -- nothing in this file depends on the
open GCN-MR-VAE design decisions):

* ``gaussian_kl``          KL(N(mu, sigma^2) || N(0, I)) per sample and dimension.
* ``multinomial_nll``      the Mult-VAE reconstruction term.
* ``kl_with_free_bits``    the KL term as it enters the loss, with an optional
                           per-dimension floor (free bits; Kingma et al. 2016).
* ``BetaSchedule``         linear annealing (Liang et al. 2018), constant, or
                           cyclical (Fu et al. 2019).
* ``latent_diagnostics``   total KL, per-dimension KL and active units
                           (Burda et al. 2016) over a set of posteriors. H4 is
                           *defined* by KL (qualification §1.2.2), so these are
                           reported results, not debugging aids.
* ``MultVAE``              Mult-VAE^PR (Liang et al., WWW 2018), the VAE-CF
                           baseline. Not part of PyTorch; reimplemented from
                           the paper and its reference code.

What is not here yet: ``VAETerm``, the loss term that attaches a VAE to the
LightGCN trainer for GCN-VAE / GCN-MR-VAE (Phase 3). It depends on the encoder
input and the decoder choice, which are still open, so it keeps raising.

Mult-VAE^PR, as in the reference implementation:

* encoder q(z|x): L2-normalise x, input dropout (keep 0.5, survivors scaled by
  1/keep), Linear(I, 600) + tanh, Linear(600, 2 * 200) -> (mu, logvar);
* decoder p(x|z): Linear(200, 600) + tanh, Linear(600, I) -> logits;
* loss per batch: -mean_u sum_i x_ui log softmax(logits)_ui
                  + beta * mean_u sum_d KL_ud  (+ lam * sum ||W||^2, lam = 0);
* beta annealed linearly in *updates*: beta = min(cap, step / 200000), cap 0.2;
* weights Xavier-uniform, biases truncated normal (std 0.001);
* evaluation scores with the posterior mean (no sampling, no dropout).

Randomness (dropout masks and the reparameterisation noise) is drawn from an
explicit ``torch.Generator`` passed by the caller, never from the global torch
stream, so evaluating or measuring more often cannot change training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- objective
def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Elementwise KL(N(mu, exp(logvar)) || N(0, 1)), shape of ``mu``.

    Summing over the last dimension gives the KL of the diagonal Gaussian
    posterior against the standard normal prior.
    """
    return 0.5 * (-logvar + logvar.exp() + mu.pow(2) - 1.0)


def multinomial_nll(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """-mean_u sum_i x_ui log softmax(logits_u)_i (Liang et al. 2018, Eq. 3)."""
    return -(F.log_softmax(logits, dim=1) * x).sum(dim=1).mean()


def kl_with_free_bits(kl_elem: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """KL term as it enters the loss, from per-sample, per-dimension KL.

    ``free_bits = 0``: mean over the batch of the summed KL (the plain ELBO).
    ``free_bits > 0``: sum over dimensions of max(free_bits, batch-mean KL of
    that dimension) -- dimensions below the floor receive no KL gradient
    (Kingma et al. 2016, applied per latent dimension).
    """
    per_dim = kl_elem.mean(dim=0)
    if free_bits > 0:
        per_dim = torch.clamp(per_dim, min=float(free_bits))
    return per_dim.sum()


@dataclass
class BetaSchedule:
    """KL weight as a function of the update counter (0-based).

    * ``linear``   beta = min(cap, step / anneal_steps)        (Liang et al.)
    * ``constant`` beta = cap
    * ``cyclical`` within each cycle of ``cycle_steps`` updates, beta rises
                   linearly from 0 to cap over the first ``ratio`` of the
                   cycle and stays at cap for the rest (Fu et al. 2019).
    """
    schedule: str = "linear"
    cap: float = 0.2
    anneal_steps: int = 200_000
    cycle_steps: int = 10_000
    ratio: float = 0.5

    def __post_init__(self):
        if self.schedule not in ("linear", "constant", "cyclical"):
            raise ValueError(f"beta schedule {self.schedule!r}; expected linear|constant|cyclical")
        if self.cap < 0:
            raise ValueError("beta cap must be >= 0")
        if self.schedule == "linear" and self.anneal_steps <= 0:
            raise ValueError("linear beta schedule needs anneal_steps > 0")
        if self.schedule == "cyclical" and (self.cycle_steps <= 0 or not 0 < self.ratio <= 1):
            raise ValueError("cyclical beta schedule needs cycle_steps > 0 and 0 < ratio <= 1")

    def __call__(self, step: int) -> float:
        if self.schedule == "constant":
            return float(self.cap)
        if self.schedule == "linear":
            return float(min(self.cap, step / self.anneal_steps))
        tau = (step % self.cycle_steps) / self.cycle_steps
        return float(self.cap * min(1.0, tau / self.ratio))


# --------------------------------------------------------------- diagnostics
@torch.no_grad()
def latent_diagnostics(mu: torch.Tensor, logvar: torch.Tensor,
                       au_threshold: float = 0.01) -> dict:
    """Posterior-collapse diagnostics over a set of users.

    * ``kl``: mean over users of sum_d KL_ud -- the quantity H4 compares;
    * ``kl_per_dim``: mean over users of KL_ud, one value per dimension
      (sums to ``kl``);
    * ``active_units``: number of dimensions whose posterior mean varies across
      users, Var_u(mu_ud) > ``au_threshold`` (Burda et al. 2016, threshold
      0.01). Population variance (divides by n). A collapsed dimension has
      mu_ud ~ 0 for every user and so carries no information about the input.
    """
    kl = gaussian_kl(mu.float(), logvar.float())
    mu_var = mu.float().var(dim=0, unbiased=False)
    return {
        "kl": float(kl.sum(dim=1).mean()),
        "kl_per_dim": kl.mean(dim=0).cpu().tolist(),
        "active_units": int((mu_var > au_threshold).sum()),
        "latent_dim": int(mu.shape[1]),
        "au_threshold": float(au_threshold),
        "mu_var_per_dim": mu_var.cpu().tolist(),
    }


# --------------------------------------------------------------------- model
def _init_liang(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.trunc_normal_(layer.bias, std=0.001, a=-0.002, b=0.002)


class MultVAE(nn.Module):
    """Mult-VAE^PR (Liang et al., WWW 2018) with one hidden layer each side.

    ``forward(x, generator)`` is the training pass (dropout + sampled z);
    ``scores(x)`` is the evaluation pass (no dropout, z = mu).
    """

    family = "vae_cf"

    def __init__(self, n_items: int, hidden: int = 600, latent: int = 200,
                 dropout: float = 0.5):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_items, self.hidden, self.latent, self.dropout = n_items, hidden, latent, dropout
        self.enc_in = nn.Linear(n_items, hidden)
        self.enc_out = nn.Linear(hidden, 2 * latent)
        self.dec_in = nn.Linear(latent, hidden)
        self.dec_out = nn.Linear(hidden, n_items)
        for layer in (self.enc_in, self.enc_out, self.dec_in, self.dec_out):
            _init_liang(layer)

    # -- pieces
    def input_dropout(self, h: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Inverted dropout: survivors scaled by 1/keep, so E[output] = h."""
        if self.dropout <= 0:
            return h
        keep = 1.0 - self.dropout
        mask = torch.rand(h.shape, generator=generator, device=h.device) < keep
        return h * mask / keep

    def encode(self, x: torch.Tensor, generator: torch.Generator | None = None):
        """Posterior parameters. Dropout is applied only when a generator is
        given (training); evaluation passes ``None`` and is deterministic."""
        h = F.normalize(x, p=2, dim=1)
        if generator is not None:
            h = self.input_dropout(h, generator)
        h = torch.tanh(self.enc_in(h))
        out = self.enc_out(h)
        return out[:, :self.latent], out[:, self.latent:]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_out(torch.tanh(self.dec_in(z)))

    # -- passes
    def forward(self, x: torch.Tensor, generator: torch.Generator):
        mu, logvar = self.encode(x, generator)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        z = mu + eps * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def scores(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x, None)
        return self.decode(mu)

    def weight_l2(self) -> torch.Tensor:
        return sum(m.weight.pow(2).sum() for m in (self.enc_in, self.enc_out, self.dec_in, self.dec_out))

    def item_vectors(self) -> torch.Tensor:
        """Decoder output weights, one row per item: logit_i = w_i . h + b_i.
        The VAE-CF analogue of an item embedding, used for ER and NP_ref."""
        return self.dec_out.weight

    def state_for_checkpoint(self) -> dict:
        return {"family": self.family, "n_items": self.n_items, "hidden": self.hidden,
                "latent": self.latent, "dropout": self.dropout}


class VAETerm:
    """Phase-3 loss term for GCN-VAE / GCN-MR-VAE. Not implemented yet: it
    depends on the encoder input (user rows of the propagated Z vs raw r_u)
    and the decoder (tied to the item rows of Z vs an MLP), which are still
    open. See VAE_ROADMAP.md."""
    name = "vae"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VAETerm (GCN-VAE / GCN-MR-VAE, Phase 3) is not implemented yet; the "
            "standalone VAE-CF baseline is src/vae_cf.py with arm=vae_cf.")
