# Migration log: `LightGCN_Manifold_refactored (6).ipynb` → `gcn-mr-vae`

Ported 11 Sep 2026 from the notebook version of 20 May 2026. The model code
(LightGCN propagation, BPR, adjacency, k-NN Laplacian, manifold loss,
Recall@K) was validated in the June 2026 audit and is unchanged in
substance. Everything that changed is either (a) a protocol fix from the
audit, (b) removal of Colab/Drive scaffolding, or (c) de-duplication.

## Cell → module map

| Notebook cell | Content | Now |
|---|---|---|
| 1 | Drive mount, `SAVE_DIR` | deleted (local `runs/`) |
| 2 | imports, `Config`, `set_seed`, GPU helpers | `src/utils.py`; `Config` → `configs/config.yaml` |
| 3 | `download_file`, `setup_datasets`, `Loader`, `create_sample_data` | `src/data/download.py`, `src/data/dataset.py`, `src/data/synthetic.py` |
| 4 | `make_save_tag`, `build_config_snapshot`, `save_experiment_results`, `tag_exists_in_drive` | replaced by the run directory (`config.yaml` + `results.json` per run) |
| 5 | `LightGCN`, `BPRLoss`, `UniformSample_vectorized`, `minibatch`, `BPR_train_original` | `src/models/lightgcn.py`, `src/models/losses.py`, `src/sampling.py`, `src/trainer.py` |
| 6 | `RecallPrecision_ATk`, `NDCGatK_r`, `compute_gini_from_counts`, `compute_neighborhood_preservation`, `Test` | `src/metrics/{ranking,diversity,geometry,evaluate}.py` |
| 7 | `train_single_dataset` (baseline harness) | `src/trainer.py` with `arm=baseline` |
| 8 | `build_knn_laplacian`, `compute_manifold_loss`, `compute_effective_rank`, `BPRLoss_MR`, `BPR_train_with_MR`, `run_mr_experiment` | `src/models/mr_layer.py`, `src/models/losses.py`, `src/metrics/geometry.py`, `src/trainer.py` with `arm=emb_mr` |
| 9 | `build_cooccurrence_laplacian`, `BPRLoss_MR_CoOccurrence`, `run_mr_cooccurrence_experiment` | `src/models/mr_layer.py` (`build_cooccurrence_laplacian`), `arm=coocc_mr` |
| 10–11 (Seção 3) | `parse_save_tag`, `inventory_drive`, `recover_metrics_*`, `_find_history` | deleted (Colab tax; nothing to recover when runs write their own files) |
| 12–14 (Seção 4) | exploratory single runs | command-line overrides |
| 15.1–15.30 (Bloco C) | 30 copy-paste cells, one per (arm × dataset × seed) | `python -m src.train -m ...` / `scripts/run_gate.sh` |
| 16 (Seção 6) | plots, `wilcoxon_test_block_c`, `build_cap5_table` | `analysis/summarize.py` (plots: `notebooks/`) |
| Block 6 | `compute_np_ref` (Jaccard), `find_checkpoint`, `build_canonical_pool` | `src/metrics/geometry.py::np_ref`, computed inside every run; checkpoint discovery deleted |
| Block 6b | per-epoch test curve with RNG protection | `eval.every=1 eval.test_each_eval=true`; RNG isolation is structural now |
| Block 7 | K-sweep with `measure_initial_er` | `python -m src.train -m arm=baseline model.n_layers=2,3,4,5`; ER at init is `effective_rank_transition` with `warmup_epochs=0` |
| Block 8 | raw-table ER | `geometry.py::effective_rank` on `model.ego_embeddings()` |

## Fixes applied (from the June and August audits)

### P0: the re-run is invalid without these

1. **Warm-start budget confound (fixed).** `run_mr_experiment` looked for
   `live_best_baseline_{ds}_seed{s}.pth`, loaded it, and set
   `warmup_epochs = 0`; the diagnostic confirmed all 15 Emb-MR runs took this
   path. Emb-MR therefore trained for `best_epoch(baseline) + 1000` epochs on
   top of a checkpoint the baseline had already selected on test, with a
   *fresh* Adam state. There is no such path now. Every arm trains exactly
   `epochs` epochs from the seed's initialisation. The only way to skip
   warm-up is `warmstart.load`, which restores model, optimizer and RNG state
   from a checkpoint saved at epoch W and continues to the same E.
2. **MR-off continuation control (added).** `arm=mr_off` is `emb_mr` with
   `lambda_manifold=0`: same W, same E, same epoch-W reference snapshot, same
   negative-sampling stream. The August plan described three arms (baseline
   from scratch, warm-start + BPR, warm-start + MR); under matched budgets the
   first two are the same run, so the gate has two arms. `baseline` is kept
   for runs with no phase structure (e.g. the K-sweep).
3. **Validation-set checkpoint selection (added).** `src/data/splits.py`
   carves `split.val_frac` of each user's training items into a validation
   set (fixed `split_seed`, shared by all arms and seeds). Selection uses
   `val_recall@20`; test is scored once at the selected checkpoint. The
   adjacency is built from the reduced training set only. Consequence:
   absolute numbers will sit below Chapter 5's, which trained on 100% of
   train and selected on test. This is expected and must be said in the text.

### P1: blocks reporting numbers

4. **NDCG IDCG (fixed).** `NDCGatK_r` normalised by the DCG of the *hits
   found*, not of the ideal list; any list with its hits at the top scored
   1.0. Now `IDCG@k = Σ_{i=1}^{min(k,|GT|)} 1/log2(i+1)`. `tests/test_metrics.py`
   contains the notebook implementation for contrast.
5. **Tail-coverage definition (disambiguated).** The notebook computed the
   mean per-user share of long-tail items in the top-k; Eq. 2.18 defines
   catalog coverage. Both are reported: `tail_share_user@k` (notebook) and
   `tail_catalog_coverage@k` (Eq. 2.18). Pick one in the text.

### P2: consistency

6. **NP reference (fixed).** Baseline measured NP against random init;
   Emb-MR measured it against the loaded baseline checkpoint. Now every arm
   reports (a) `np_vs_ref@k` against its own epoch-W ego snapshot, which is
   the *same* snapshot for `mr_off` and `emb_mr`, and (b) `np_ref@k` against
   the external Jaccard item graph, with a fixed subsample seed, for all arms.
7. **λ per-batch vs per-epoch (made explicit).** The manifold term is applied
   every minibatch, so its per-epoch weight is `n_batches × λ` (≈25λ Gowalla,
   ≈37λ Yelp, ≈73λ Amazon-Book at batch 32768). `arm.lambda_scale=per_epoch`
   divides λ by `n_batches`. Default `none` reproduces Chapter 5; the gate
   should use `none` so only the protocol changes. `results.json` records
   `n_batches_per_epoch` and `lambda_effective`.
8. **ER mean-centering (documented).** `effective_rank(center=True)` is the
   explicit default; Eq. 3.2 should state that rows are mean-centred before
   the SVD. ER is computed on propagated embeddings (as in Ch. 5).

### Structural

- Four functions were defined twice in different cells
  (`compute_neighborhood_preservation`, `compute_manifold_loss`,
  `compute_effective_rank`, `compute_np_ref`). They were semantically identical,
  so no results were affected; the hazard is gone with modules.
- Three loss classes and three training loops → `CompositeLoss` of
  `LossTerm`s plus one `Trainer`. The VAE attaches as a fourth term
  (`src/models/vae.py`, interface only).
- Global `config` object with override/restore → per-run Hydra config,
  saved to `runs/.../config.yaml`.
- `torch.cuda.empty_cache()` every 10 batches → removed (it was a slowdown, not a fix).
- Evaluation: propagation once per evaluation instead of once per batch;
  hit matrix vectorised; exclusion of train (val) / train+val (test) positives.
- Negative sampling: same distribution, collision check via one CSR lookup.
- RNG hygiene: sampling, evaluation subsampling and geometry subsampling use
  separate generators. Evaluating more often never changes the training
  trajectory (Block 6b's snapshot/restore hack is unnecessary).
- k-NN for the Emb-MR Laplacian has a `torch` backend (chunked cosine + topk
  on GPU); `arm.knn_backend=sklearn` is the notebook's exact path.
- Not ported: ml-1m sequential loader, `create_sample_data` (replaced by
  `synthetic.py`), `keep_prob`/`A_split` dropout (never enabled), the sigmoid
  in `getUsersRating` (monotone; no effect on rankings).

## Things that still need a decision from the author

- `split.val_frac` (0.1 default) and whether `min_train=2` is the right floor.
- Which tail metric Eq. 2.18 means.
- `eval.every`: 10 by default (the notebook used 10 until epoch 100 then 50).
  With val selection, coarser evaluation adds selection noise; 10 is cheap on
  Gowalla/Yelp, consider 25 on Amazon-Book.
- Whether to bump to 10 seeds for the full sweep (Wilcoxon floor at n=5 is p=0.031).


---

# Post-port audit (11 Sep 2026)

Re-checked the port against the notebook *and* against the qualification text
(`Marinho — Arquiteturas Geométricas Variacionais`, 12 Aug 2026 build) to find
anything that would skew or destroy comparability with the Chapter 5 experiments.

## Numerical equivalence: verified, not assumed

`audit_equivalence.py` reimplements the notebook's `BPRLoss_MR.stageOne` and
its `Test()` per-user loop verbatim and runs both against the ported code on
identical inputs and identical weights. Run it with `python audit_equivalence.py`.

**24/24 checks bit-identical**, including:

- total loss and the **full gradient** w.r.t. the embedding tables at
  λ ∈ {0, 1e-5, 1e-2} — max|difference| = `0.00e+00`;
- `recall@{10,20}`, `precision@{10,20}` and their short-head/long-tail splits;
- `gini@{10,20}` and the notebook's tail metric (now `tail_share_user@k`);
- LightGCN mean-over-layers propagation, and `n_layers=0` reducing to MF-BPR.

The only intended divergence is NDCG. On the audit fixture the fixed IDCG gives
**NDCG@10 −49.0%** and **NDCG@20 −44.1%** relative to the notebook. Any future
NDCG number will therefore be far below the notebook's; that is the bug being
removed, not a regression. This also retroactively justifies dropping NDCG
from the qualification text.

## Issues found and fixed

**A. `warmup_epochs=0` never captured the initial reference (real bug, H1-critical).**
The phase-transition branch treated a fresh W=0 run as if it were resumed from
a warm-start checkpoint, so `_on_transition` never ran: no reference snapshot,
no `np_vs_ref`, no ER baseline. Since `arm=baseline` is W=0, **every baseline run
silently omitted the H1 measurements** — and H1 is validated precisely by
"queda do Effective Rank em relação ao valor medido na inicialização" and
"preservação de vizinhança … quando a própria inicialização é tomada como
referência" (§1.2.2). The smoke tests missed it because they overrode
`warmup_epochs=2` for every arm. Fixed by keying the branch on
`warmstart.load` instead of comparing `start_epoch` to `W`; regression test
`test_w0_captures_initial_reference`.

**B. Only one Effective Rank convention was reported (gap).**
Table 2 reports ER on the **learned embedding table** (ER_tab, ER_init ≈ 255.2
of d = 256), and Table 3 contrasts ER_tab against ER_prop (post-propagation).
The port reported only the propagated value, so Chapter 5's headline H1 numbers
were not reproducible from `results.json`. Both are now computed everywhere
(`er_table`, `er_prop`, plus `_pct`, `_at_ref` and `delta_` variants);
`effective_rank` is kept as an alias for `er_prop`. Sanity check: the ported
`effective_rank` on a random (70839, 256) table returns **255.16**, against the
paper's 255.2 ± 0.0 — the metric is faithful.

**C. Popularity segmentation depended on `val_frac` (comparability).**
Computing item popularity on the reduced training set moved **678 items (8.27%
of the short head)** across the 20/80 boundary on Gowalla, which shifts every
Gini and tail number away from Chapter 5 for reasons unrelated to the method.
`split.popularity_from` now defaults to `full_train` (the original `train.txt`,
Chapter 5's convention), making the segmentation independent of `val_frac`.
Regression test `test_popularity_segmentation_independent_of_val_frac`.

## Confirmed correct, no change needed

- **Tail-Coverage.** Eq. 2.18 is |I_tail ∩ ∪_u L_u| / |I_tail| — catalog
  coverage over tail items. That is exactly `tail_catalog_coverage@k`. Use that
  key in the text; `tail_share_user@k` is the notebook's different quantity,
  kept only so the old numbers remain reproducible.
- **Negative sampling.** The notebook's collision loop re-indexes into the
  previous collided subset after the first pass, so it resamples the wrong
  positions and leaves a few true positives labelled as negatives: measured at
  **15 of 810,128 per epoch on Gowalla (0.002%)**. The port leaves 0. The
  difference is far below seed noise; noted for completeness only.
- **Sigmoid on scores.** Dropping it cannot change any ranking (monotone), and
  the equivalence test confirms identical recall.
- **RNG.** The notebook's `compute_effective_rank` drew an *unseeded*
  `torch.randperm`, so every ER measurement perturbed the global RNG and the
  training trajectory depended on how often you evaluated. The port uses local
  seeded generators for sampling, ER and NP subsampling, so evaluation
  frequency can no longer affect training. This is why Block 6b's
  snapshot/restore hack is unnecessary here.

## Still expected to differ from Chapter 5 (by design)

Absolute numbers will move, for reasons already documented above: 90% training
edges instead of 100%, validation-based selection, equalised budgets, and the
NDCG fix. Within the new runs all arms share these conditions, so the
comparisons are valid; only the cross-reference to the old tables shifts.

## Scorer seam (11 Sep 2026, prep for Module 3)

`evaluate(model, ...)` became `evaluate(scorer, ...)`. The default
`dot_product_scorer` reproduces the notebook's `U Iᵀ` exactly (old vs new on
real Gowalla: 40/40 metrics bit-identical on both splits; notebook harness
24/24). `LightGCN.users_rating` was removed so there is a single scoring path.
See `VAE_ROADMAP.md` B1.

## Hyperparameter cross-check (11 Sep 2026)

Every method hyperparameter was diffed against the notebook's `Config` class,
the MR harness defaults, the Bloco C launch cells, and Tabela 7 of the
qualification. All match, with one correction applied:

**`coocc_mr.yaml` had lambda = 1e-5; the correct value is 1e-4.** The notebook's
`run_mr_cooccurrence_experiment` defaulted to `lambda_manifold=0.0001` and
Tabela 7 records "lambda_manifold (CoOcc-MR) 10^-4, sondagem preliminar",
against 10^-5 for Emb-MR. The port had silently inherited Emb-MR's value.
No results were affected (CoOcc-MR has not been re-run), but a CoOcc run
launched from the config as shipped would not have reproduced Section 5.4.

Four differences remain and are intentional protocol changes, documented in
`docs/CODEBASE_GUIDE.md` Appendix A.2: evaluation user-batch size (memory
only), evaluation every 10 epochs throughout, early stopping disabled, and
validation-based checkpoint selection.

## Fairness audit (11 Sep 2026) — checking for bias against the MR arm

Re-read the design specifically for choices that would systematically
disadvantage the treatment relative to the control. One real defect found.

**Asymmetric checkpoint-selection window (fixed).** `self.best` was not
restored by `load_warmstart`, and a resumed arm starts at epoch W. So a
standalone control running 0→E could select its best checkpoint from *any*
epoch, including the warm-up prefix, while an MR arm resumed at W could only
select from [W, E). On the gate (W=1000, E=2000) that is half the candidates.
It bites whenever validation peaks during warm-up and declines afterwards —
i.e. exactly the late-stage BPR overfitting regime the gate runs into. The
control would report its warm-up peak; the MR arm would be forced to report a
lower post-treatment epoch, and the difference would be read as MR hurting.

Fixed by excluding the shared prefix from selection for **both** arms
(`eval.select_from_epoch: auto` = W). This is also the more principled rule:
epochs before W are bit-identical between arms that branch from the same
warm-start, so a checkpoint drawn from them carries no information about the
treatment. `results.json` records `select_from_epoch`. Regression test
`test_selection_window_is_symmetric`.

**Selection-independent reading added.** Every reported number was taken at the
checkpoint that maximised validation Recall@20. Geometry (ER, NP_ref) and
diversity (Gini, Tail-Coverage) are not what selection optimises, so reporting
them only at the accuracy peak can understate an arm whose geometric effect is
still growing after its accuracy has plateaued. `results.json` now also carries
`at_last_epoch`: the same metrics at epoch E, a fixed point identical for every
arm by construction. Use the selected checkpoint for accuracy claims (H2's
"not inferior") and either — stated explicitly — for geometric and diversity
claims (H1, H3).

**Checked and found fair, no change:** equal budgets, shared warm-start and
negative-sampling stream, the same validation split and popularity segmentation
for all arms, the NDCG fix (applies to both arms), fixed geometry subsampling
seeds, early stopping disabled (which protects a slow-improving MR arm), and
both tail definitions reported. `mr_off` selecting its own val peak rather than
its endpoint is generous to the control, and deliberately so: it is the
conservative comparison.

**Open item, not a code issue.** λ = 1e-5 was chosen by a sensitivity sweep run
under the *old* protocol, where MR received extra epochs on a converged,
test-selected checkpoint. The control has no comparably tuned hyperparameter.
Re-sweeping λ on the validation set under the corrected protocol
(`-m arm.lambda_manifold=1e-6,1e-5,1e-4`) is the symmetric thing to do before
concluding anything from a null result; a λ tuned for the old regime may simply
be the wrong strength for the new one.

## Second-pass review: is anything here over-engineered? (11 Sep 2026)

Re-read every deviation from the notebook, asking whether it was *needed* or
merely critical, and whether it costs accuracy. Summary: the code is faithful
and nothing in it lowers results relative to the notebook beyond removing the
confound itself. One recommendation (not code) was over-engineered and has been
simplified.

**Measured, not assumed: the validation split is close to free.** Gowalla,
seed 2020, d = 64, 25 epochs, same test set, only `val_frac` varied:

| val_frac | train edges | test Recall@20 | vs full |
|---|---|---|---|
| 0.00 | 810,128 | 0.09762 | — |
| 0.05 | 762,857 | 0.09582 | −1.85% |
| 0.10 | 728,432 | 0.09666 | −0.99% |

Not monotone in the data removed, so the spread is noise rather than signal.
The drop against Chapter 5's published numbers comes from removing the extra
1,000 epochs the MR arm received, which is the artefact being corrected.

**Simplified: the gate is W = 100, E = 1000, one command.** The earlier
recommendation of W = 1000 / E = 2000 was chosen to preserve the notebook's
*realised* condition (MR acting on a converged model) — but that condition was
itself produced by the budget bug. Tabela 7 and §4.2.1 specify activation at
epoch 100 with a 1,000-epoch horizon; that is the design of record and the
corrected re-run should use it. Halves the compute (≈5 h instead of ≈7.3 h for
five seeds) and removes the warm-start dance entirely: with the same seed both
arms share initialisation and the negative-sampling stream, and epochs 0–99 are
BPR-only for both, so the warm-up is bit-identical by construction — verified
on real Gowalla data, not assumed. `scripts/run_gate_converged.sh` keeps the
W = 1000 variant as an optional follow-up if the primary gate is null.

**Classification of every change.**

*Necessary — the comparison is invalid or a number is wrong without it:*
budget equalisation; the `mr_off` control; validation-based selection;
the NDCG IDCG fix; W = 0 reference capture; the symmetric selection window.

*Free and additive — new columns, no behavioural change:* both tail
definitions; both ER conventions; `popularity_from=full_train` (which
*prevents* a change); `at_last_epoch`; NP reference consistency.

*Explicit but inert — the default reproduces the notebook exactly:*
`lambda_scale=none`; `er_center=true`; `knn_backend` (torch and sklearn agree
on >99% of neighbours).

*Verified faithful:* loss and full gradient bit-identical; all ranking and
diversity metrics bit-identical; propagation identical; `effective_rank` on a
random table returns 255.16 against the text's 255.2.

Nothing in the list makes the method look worse than it is. The measured cost
of the entire corrected protocol, on the primary accuracy metric, is within
noise.

## Third pass: full line-by-line comb (11 Sep 2026)

Read every module line by line, verified config-to-code wiring in both
directions, simulated the evaluation schedule across edge cases, and re-tested
each earlier finding for false alarms. Four real defects found and fixed; one
earlier concern confirmed as a false alarm.

**1. Sweep runs silently overwrote each other (real, high impact).**
`paths.arm_tag` encoded only name, W, lambda, k and lambda_scale. So
`-m model.n_layers=2,3,4,5` — the K-depth sweep recommended in both the README
and the guide — produced the tag `baseline_w0` four times and every run wrote
to the same directory, each replacing the last. The same held for
`arm.rebuild_every` (a committed ablation in §4.2.3) and `model.dim`. Fixed by
adding rebuild period, depth and dimension to the tag
(`emb_mr_w100_lam1e-05_k20_r50_K3_d256`), and by adding `_assert_no_clobber`,
which refuses to start if the run directory already holds a *different*
configuration and prints the differing keys. Re-running an identical config is
still allowed. The guard ignores `paths`, `hydra`, `warmstart`, `logging`,
`device` and `deterministic`, none of which change what the experiment is.

**2. Warm-start resume would have crashed on GPU (real, GPU-only).**
`torch.load(map_location=device)` moves *every* tensor in a checkpoint to that
device, including the saved RNG ByteTensors — and `torch.set_rng_state` accepts
only a CPU ByteTensor ("This function only works for CPU", per its docstring).
CPU tests could never catch it. `rng_state_load` now coerces states back to CPU;
two regression tests cover the round trip. This would have surfaced only when
running `run_gate_converged.sh` on the RTX 5060 Ti.

**3. Adjacency cache key omitted `min_train`.** Changing `split.min_train`
changes which edges are held out but would have reused a stale cached graph.
Now in the key.

**4. Dead imports** (`defaultdict`, `Timer`) removed; the in-place masking of
the scorer's return value is now documented, since Module 3 must return a
freshly allocated tensor.

**False alarm, corrected.** While reading `evaluate.py` I suspected that
`n_rel` (counting COO entries) and the short/long-head counts (summing matrix
values) would disagree if a dataset contained duplicate `(user, item)` pairs,
since SciPy sums duplicates. Checked directly: Gowalla and Yelp2018 contain
**zero** duplicates in both train and test, all stored values are exactly 1.0,
and `n_rel == n_short + n_long` holds exactly on the real test set. Not a bug.
A one-line binarisation was added anyway so the property is guaranteed for any
future dataset.

**Re-verified, all confirmed real (no false alarms):** the W=0 reference bug
(demonstrated on Gowalla, `er_table_at_ref` was absent); the ER-convention gap
against Table 2; the popularity-segmentation drift (678 items); the CoOcc
lambda — cell 19 sets `LAMBDA_MANIFOLD = 1e-4` explicitly while the fifteen
Bloco C cells use `LAMBDA = 1e-05`, matching Tabela 7; and the selection-window
asymmetry.

**Checks that came back clean.** Every config key is read by the code and every
key the code reads exists in the config (verified programmatically, both
directions). All seventeen modules import. `vae.py` raises as intended. The
evaluation schedule yields at least one selectable checkpoint under the gate
config and seven edge cases including `W == E` and `eval.every > epochs`.
Loss and gradient still bit-identical to the notebook (24/24). Two arms stay
bit-identical through warm-up on real Gowalla without any warm-start file.
The full CLI path — multirun, distinct folders, `results.json`,
`analysis.summarize` — runs end to end on Gowalla.
