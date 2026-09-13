# Module 3 (VAE) — readiness assessment and roadmap

Written 11 Sep 2026, against the qualification text of 12 Aug 2026 (§4.2.2,
§4.2.3, §6.3, Tabela 6) and the repo at commit `47b7d42` + post-port audit.

Cronograma position: VAE integration is scheduled **Ago–Out 2026 and Nov 2026 –
Jan 2027**, overlapping the corrected re-run. Defence deadline is **10 Sep 2027**
(regimental maximum). The November push is the main build.

---

## Verdict

The repo is in good shape for this. The port decision that pays off here is
the composable loss: `VAETerm` is a drop-in, not a fourth training loop.
**Three blocking gaps and four smaller ones**; none need a redesign, and the
largest is a one-hour refactor that gets much more expensive if left to
November.

### Already right — reuse, no work

| Thing | Why it matters for the VAE |
|---|---|
| `CompositeLoss` / `LossTerm` (`src/models/losses.py`) | `VAETerm` registers alongside BPR/L2/manifold; `set_active` switches it on at Phase 3 |
| `BatchContext.extras` | passes `r_u`, `mu`, `logvar` between terms without changing any signature |
| Budget matching + val selection + warm-start resume | H4's comparison (GCN-VAE vs GCN-MR-VAE) is exactly the two-arm design already built; both branch from the same Phase-2 checkpoint |
| `src/metrics/geometry.py` | ER / NP / NP_ref take any matrix — they work on `Z_MR` *and* on the VAE latent `z` unchanged |
| `analysis/summarize.py` | arm-agnostic; H4's paired Wilcoxon comes free once the runs exist |
| `src/data/synthetic.py` | the plan (Feb 2026 chat, §6.3 risk register) says build the VAE on tiny data first — the generator is there |
| MF-BPR baseline | `model.n_layers=0` already reduces to MF-BPR; regression-tested |

---

## Blocking gaps

### B1. `evaluate()` hardcoded dot-product scoring — **DONE 11 Sep 2026**

`evaluate()` now takes a `Scorer` callable, `users -> (batch, n_items)`, and
never touches the model. `dot_product_scorer(all_users, all_items)` is the
default and equals the notebook's rule. The trainer exposes one seam,
`Trainer.scorer(all_users, all_items)`; Module 3 overrides it to return a
closure around `decoder(mu_u)` and nothing else in the evaluation path changes.
Exclusion of known positives stays inside `evaluate()` (protocol, not model).

Verified: old vs new `evaluate()` on real Gowalla, val + test, **40/40 metrics
bit-identical** per split; `audit_equivalence.py` still 24/24;
`tests/test_scorer.py` pins the contract (custom scorer honoured, train masked
on val, val masked on test, shape validated).

### B2. One phase boundary; Phase 3 needs a second

`src/trainer.py` — `self.W`, `_on_transition`, and `phase = "warmup" if epoch < self.W else ...`

The schedule is hardcoded to two phases. §4.2.3 specifies three: warm-start
(100) → MR (~1000) → **VAE 20 epochs + 10 epochs joint fine-tuning at reduced
LR**. Needs boundaries `[W, W+M]`, a per-phase active loss set, and a per-phase
learning rate.

Fix: replace the scalar `W` with a list of phase specs. The transition
bookkeeping (`_on_transition`, reference snapshot, warm-start save) generalises
to "on entering phase *p*".

### B3. Optimizer is built once over `model.parameters()`

`src/trainer.py:95`. Encoder and decoder parameters do not exist at
construction. Attaching mid-run needs `opt.add_param_group`, and the Phase-3
LR reduction is a param-group LR change, not a new optimizer (a fresh Adam
would reset moments and re-introduce a budget confound of exactly the kind
P0 #1 removed). `save_warmstart` / `load_warmstart` must also carry the
encoder/decoder state, and `LightGCN.state_for_checkpoint` is currently
LightGCN-only.

---

## Smaller gaps

**S1. No user-level batching.** VAE-CF reconstruction is a multinomial
likelihood over the user's interaction vector `r_u`. `src/sampling.py` yields
`(u, pos, neg)` triples. Need an iterator over `UserItemNet` rows. Not hard —
but it is a second sampling stream, so it needs its own seeded generator to
preserve the RNG hygiene the port established.

**S2. No collapse diagnostics — and H4 is *defined* by one of them.**
§1.2.2: H4 is validated by "divergência de Kullback-Leibler mais alta que a do
pipeline sem a camada" **and** Recall@20 superior. So KL is not a debugging
aid, it is a reported result. Needed from epoch 1: total KL, **per-dimension
KL**, active units (Burda), and the β schedule value. §6.3 commits to KL
annealing if collapse appears, and to reporting collapse as a finding if it
persists.

**S3. No VAE-CF baseline model.** VAE-CF (Liang et al. 2018) consumes sparse
`r_u` directly — it is *not* LightGCN with the VAE attached. Separate model
class. It is a committed baseline (see below).

**S4. Nothing measures the thing H4 is actually about.** The claim is that
geometrically regularized inputs give the encoder enough signal to resist
collapse. That is a statement about `Z_MR` vs `Z_GCN` as encoder *inputs*.
Worth logging ER and NP_ref of the encoder input alongside KL, so the
mechanism is visible and not just the outcome.

---

## Roadmap

### Phase A — now to end of October: finish Module 2, prepare the seam

1. **Run the gate** (`scripts/run_gate.sh`, W=1000/E=2000, 5 seeds, Gowalla).
   This is the September deliverable and it gates everything downstream.
2. ~~B1 refactor~~ done.
3. **B2 + B3** as one change: phase list, param groups, extended checkpoint
   schema. Add a `vae` phase that is a no-op until `VAETerm` exists, so the
   plumbing is tested before the model lands.
4. Extend the gate to Yelp2018 and Amazon-Book if time allows (Amazon-Book is
   ~3× slower per epoch; consider `eval.test_each_eval=false`).

Exit criterion: three phases run end-to-end with an empty Phase 3, all tests
green, gate results in hand.

### Phase B — November: build the VAE

5. **Synthetic first.** `python -m src.data.synthetic` → 300 users. Implement
   `VAETerm` against it, with KL/per-dim-KL/active-units logging from the first
   epoch. Induce collapse deliberately (β=5) and confirm the diagnostics catch
   it. This is the Feb 2026 plan and it is still the right order: a VAE that
   silently collapses on Gowalla is indistinguishable from a VAE that is wired
   up wrong.
6. **λ_VAE = 1** as specified (§4.2.3: neutral unit weight on the ELBO),
   recalibrate only if collapse appears.
7. **Attach to Gowalla** from the Phase-2 checkpoint. Phase 3 = 20 epochs VAE +
   10 joint fine-tuning, reduced LR.
8. **Cyclical KL annealing + free bits** behind config flags, off by default,
   so turning them on is a documented intervention rather than a silent change.

Exit criterion: GCN-MR-VAE trains on Gowalla without collapse, or collapses
with diagnostics good enough to say *why*.

### Phase C — December to January: experiments

9. **The H4 pair**: `gcn_vae` (no manifold term) vs `gcn_mr_vae`, both branching
   from the same Phase-2 checkpoint, both same budget, 5 seeds. Report KL and
   Recall@20 — that is H4's stated criterion, both halves.
10. **Remaining baselines**: VAE-CF (S3) and MF-BPR (already supported).
11. **Ablations**: λ_manifold, k, rebuild frequency (§4.2.3 commits to these).
12. **Correlation analysis** between geometric and ranking deltas — §4.3.4,
    explicitly conditional on schedule.

---

## Details from earlier chats and the text worth not losing

- **DirectAU is no longer a committed baseline.** You decided to remove it
  rather than hedge it, applied at six sites including inside the Figura 7
  TikZ. Committed set: **MF-BPR, LightGCN, VAE-CF, GCN-VAE (ablation)**.
  DirectAU remains cited as literature and as a "possibilidade adicional"
  only. Don't let it creep back into a results table by habit.
- **Phase 3 is short**: 20 + 10 epochs, reduced LR (§4.2.3, Figura 6c). The VAE
  is a fine-tuning stage, not a co-trained-from-scratch module.
- **GCN-VAE must share encoder, warm-start schedule and hyperparameters** with
  the full pipeline (§4.3.2) — differing only by the MR term. Same
  single-variable discipline as the current gate.
- **Second experiment** (sparse users, ≤5 interactions) is explicitly
  conditional on schedule, and is warm-start-with-little-data, *not* true
  cold-start — the MR layer needs pre-existing embeddings.
- **If the accuracy gain is real but below practical relevance** (§6.3), the
  contribution shifts to diversity metrics and geometric interpretability.
  That is already written into the text; it is a planned landing, not a
  retreat.
- **If collapse cannot be beaten**: the negative result is publishable —
  "geometric correction of encoder inputs is insufficient to prevent posterior
  collapse; the mechanism is decoder capacity, not encoder quality." The
  nuclear fallback is the two-stage non-end-to-end version (train LightGCN,
  apply MR post-hoc on frozen embeddings, train VAE separately), which still
  tests the core hypothesis.
- **Publication**: iSys / A4 submission planned; BraSNAM paper presented
  July 2026.

---

## One thing to decide before November

Whether the VAE consumes **user embeddings only** or the full `Z_MR`
(users + items). §4.2.2 says the VAE receives `Z_MR` and that "cada usuário
vira uma distribuição, não um ponto", which reads user-side. The decoder's
output shape, the reconstruction target, and whether item embeddings stay
deterministic all follow from this. It is a two-line difference in the code and
a chapter-level difference in the writing, so settle it before building.
