"""VAE module: interface only. NOT IMPLEMENTED.

This file exists so the place where Module 3 attaches is fixed now, while
the loss composition is fresh, and so nobody adds a fourth training loop
later. It deliberately raises; the plan (Aug 2 chat, October track) is to
build the VAE against a tiny synthetic dataset first, with KL-per-dimension
and active-unit logging from epoch 1, before it ever touches Gowalla.

Intended shape when implemented:

    class VAETerm(LossTerm):
        name = "vae"
        def __init__(self, encoder, decoder, beta_schedule, free_bits=0.0): ...
        def __call__(self, ctx):
            z_in = ctx.all_emb (or a user/item slice)     # Z_MR from the MR layer
            mu, logvar = encoder(z_in)
            ...
            return recon + beta * kl, {"recon": ..., "kl": ..., "kl_per_dim": ...,
                                       "active_units": ...}

Collapse defences to wire in from day one (from the resource map):
cyclical KL annealing (Fu et al. 2019), free bits (Kingma et al. 2016),
per-dim KL / active-unit logging, and a beta=0 warm-up when attaching.
"""
from __future__ import annotations


class VAETerm:
    name = "vae"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VAE module is scheduled for the October track. "
            "See src/models/vae.py docstring for the intended interface.")
