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
| 16 (Seção 6) | plots, `wilcoxon_test_block_c`, `build_cap5_table` | `analysis/summarize.py` for the tables and the test. **The plotting code was not ported**; `notebooks/` is an empty place to put figure notebooks, and `history.json` carries every series the notebook plotted (per-epoch val/test metrics, losses, both ER conventions). |
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
   `val_recall@20`; the reported test numbers come from the selected
   checkpoint. With the shipped default `eval.test_each_eval: true` test is
   also scored at every evaluation for the curve, never for selection. The
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

- Four functions were defined twice in different cells of the migrated
  notebook (`LightGCN_Manifold_refactored (6).ipynb`):
  `compute_neighborhood_preservation` (cells 8, 10), `compute_manifold_loss`
  and `compute_effective_rank` (cells 10, 11), `compute_np_ref` (cells 56,
  58). They were semantically identical, so no results were affected; the
  hazard is gone with modules. Later notebook revisions de-duplicated some of
  these, so a different `.ipynb` may show three rather than four.
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

**4. Dead imports** (`defaultdict`, `Timer`) removed, and the now-unused
`Timer` class itself deleted from `utils.py`; the in-place masking of
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

## Fourth pass: response to the external swarm audit (Kimi), 17 Sep 2026

An independent multi-agent audit was run against the notebook, the package and
this log. It reproduced the equivalence claims (24/24 bit-identical, NDCG
-48.96%/-44.08%, 34/34 tests, ER 255.16) and found four real periphery defects.
All are fixed; two of them were regressions introduced by the third pass.

**1. `paired_wilcoxon` could report a false significance (high).** Selecting an
arm by its bare name fell back to matching `arm`, which after the third pass's
`arm_tag` expansion can match several configurations. With one extra tag it
raised a cryptic broadcast error; with an extra tag on *both* arms it silently
pooled them and reported `n=5` while testing 10 pairs, yielding p=0.000977 --
below the 0.031 floor attainable with five seeds. Ambiguous labels are now
refused with a message listing the candidate tags, and repeated seeds are
rejected. Exact tags, and bare names while they remain unambiguous, work as
before.

**2. `run_gate_converged.sh` had a stale warm-start path (medium).** The third
pass added `_K{layers}_d{dim}` to `arm_tag` but did not update the script that
hardcodes the control's directory, so the optional follow-up would have failed
on its second command. Fixed in the script and in all four documents, with a
comment noting the path must track `model.n_layers`/`model.dim`.

**3. Holm correction counted untestable rows (low).** Datasets skipped for too
few paired seeds got `p=1.0` and inflated the multiplier. Only tested rows are
corrected now, and `n_tests_corrected` is reported.

**4. Undisclosed and inaccurate documentation (low).** Validation items can be
drawn as BPR negatives, because collisions are checked against the reduced
training matrix; excluding them would consult held-out labels during training,
so the behaviour is deliberate -- now documented in `sampling.py`, with the
magnitude (under 0.01% of draws on Gowalla, identical across arms). The claim
that plots live in `notebooks/` is corrected: plotting was **not** ported, and
`history.json` carries every series the notebook plotted. Warm-start resume now
validates the architecture explicitly instead of relying on a shape error.

### Vacuous tests, and how they were found

The audit flagged two exclusion tests as vacuous. Rather than patch them by
inspection, every safety-critical guarantee was **mutation-tested**: the line
implementing it was deliberately broken and the suite re-run. This found that
the first replacement test was *also* vacuous, and that a third test the audit
had not flagged -- `test_selection_window_is_symmetric` -- passed with the
eligibility check deleted, because the toy validation curve rises monotonically
so the best epoch is the last one either way.

The cause in every case was the same: at `k=10` on the fixture, users have few
enough held-out items that they all fit in the list whether or not the masked
items are removed. The rewritten tests assert at `k=1`, where a single
displaced slot changes the metric exactly, and the selection test now scripts a
validation curve that peaks *inside* the warm-up prefix.

Mutation results (each row breaks one guarantee; the suite must fail):

| mutation | caught |
|---|---|
| val positives not masked on test | yes |
| train positives not masked on val | yes |
| selection window disabled | yes |
| NDCG ideal list = hits found | yes |
| W=0 reference snapshot skipped | yes |
| manifold term silently zeroed | yes (2 tests) |
| popularity from reduced train | yes |
| RNG state not coerced to CPU | **no -- CPU-only limitation** |
| warm-start trains E instead of E-W | yes (added after this sweep) |
| Laplacian not symmetrised | yes |
| validation split not disjoint | yes |

The last is unverifiable without a GPU: on CPU the saved RNG state is already a
CPU tensor, so removing the coercion is a no-op. The guard matters only when
`torch.load(map_location=cuda)` moves it, so it is exercised the first time a
warm-start is resumed on the RTX 5060 Ti and cannot be covered here.

## Verification of the fourth pass (17 Sep 2026)

Every finding re-checked by execution, not inspection. 13/13 verification
checks pass: ambiguous arm labels refused with the candidate tags listed,
exact tags giving n=5 and p=0.03125 (the true five-seed floor), duplicate
seeds refused, Holm correcting only tested rows, the converged script's
warm-start path matching `arm_tag`, and all four disclosures in place.

**Two further problems found while verifying.**

*A stale path survived in the guide's command cheat-sheet.* The earlier fix
replaced `runs/gowalla/mr_off_w1000/...` but the cheat-sheet used
`runs/<ds>/mr_off_w1000/...`, which the pattern missed. No occurrence of the
old tag now remains in any `.md`, `.sh` or `.py` file.

*Three documented numbers were wrong.* The per-epoch batch counts
("Gowalla ~25, Yelp ~37, Amazon-Book ~73") were computed on the **full**
training set, but every run uses the reduced one. Measured from the real
files: 23 / 34 / 66 with the default 10% split, against 25 / 38 / 73 without
it. Corrected in both places, with a pointer to `n_batches_per_epoch` in
`results.json` as the authoritative value. Two statements in
`evaluate.py`'s docstring also predated the scorer refactor (it no longer
propagates, and the hit matrix comes from an integer-key `np.isin` rather than
CSR fancy-indexing); both corrected.

**The most important guarantee in the project was untested.** Mutation testing
revealed that changing `self.start_epoch = self.W` to `0` in `load_warmstart`
— which makes a resumed arm train a full E epochs *on top of* the warm-start,
recreating precisely the budget confound P0 #1 exists to remove — passed all
35 tests. `epochs_budget` in `results.json` only echoes the config, so nothing
observed the optimiser passes actually taken. `test_resume_trains_exactly_the_remaining_budget`
now counts `train_epoch` calls and asserts the control takes E while the
resumed arm takes E - W, and that the two sum correctly. Verified to fail
under the mutation.

Ten of eleven guarantees are now mutation-covered. The exception remains the
GPU RNG coercion, which is a no-op on CPU and can only be exercised on the
target hardware.

## Fifth pass: remaining items from the follow-up audit (18 Sep 2026)

The external follow-up verified the fourth pass (8/8 fixes reproduce, 36/36
tests, 24/24 equivalence) and left three code items and several documentation
items open. All are now closed.

**1. `warmup_epochs == epochs` with λ>0 trained pure BPR under an MR label.**
The transition fires *at* epoch W, so W == E meant it never fired: the run
completed, every epoch logged `[warmup]`, no Laplacian was built, and
`results.json` recorded `arm=emb_mr, lambda_manifold=1e-05`. The constructor
now refuses that combination, while still allowing it for a λ=0 control, where
it is a meaningful (if pointless) schedule. Mutation-tested.

**2. `at_last_epoch` lacked the geometry.** The selection-independent reading
carried only `er_*` from the evaluation record, so the NP metrics that H1 and
H3 lean on had no fixed-epoch counterpart. The geometry block was factored into
`Trainer._geometry` and is now computed twice from identical code: once at the
selected checkpoint, and once at the final epoch *before* `best.pt` is loaded,
so both readings contain the same keys (`er_*`, `delta_er_*`, `np_vs_ref@k`,
`np_ref@k`). Mutation-tested.

**3. Resume validated architecture but not the split.** Only name, `val_frac`
and `split_seed` were compared, so a checkpoint built with a different
`min_train` resumed silently into a different split. `load_warmstart` now
compares every field that defines the training data — `n_users`, `n_items`,
`n_train`, `n_val`, `val_frac`, `split_seed`, `min_train` — and names the
mismatching fields. Note that dropping `min_train` from that list alone does
not change behaviour: any `min_train` that actually alters the split also
alters `n_train`/`n_val`, and when it does not the splits are byte-identical
and accepting the resume is correct. That mutation is therefore benign rather
than an uncaught defect, and is reported as such below.

### Documentation corrections

- *"test is scored once at the selected checkpoint"* was misleading: the
  shipped default `eval.test_each_eval: true` scores test at every evaluation
  for the curve. Reworded in README, MIGRATION and the guide to say the
  *reported* numbers come from the selected checkpoint and the per-eval reading
  never influences selection.
- *"Four functions were defined twice"* is correct for the notebook actually
  migrated, `LightGCN_Manifold_refactored (6).ipynb`, with the defining cells
  now listed. Later revisions de-duplicated some, so another `.ipynb` may show
  three.
- The unused `Timer` class was deleted, making the "dead code removed" claim
  true of the class and not only the import.

### Previously undisclosed differences, now recorded

- **Evaluation grid offset.** The notebook evaluated after epochs 1, 11,
  21, … (`epoch % 10 == 0` with a 0-based counter, after training); the port
  evaluates after epochs 10, 20, 30, …. A grid offset, identical for every arm.
- **Popularity tie-breaking.** The notebook ordered items by count descending
  with ties broken by first appearance in `trainItem` (dict insertion order);
  the port breaks ties by ascending item id (`np.argsort(..., kind="stable")`
  over `flatnonzero`). Measured on Gowalla: the boundary does sit inside a tie
  group (count 22 on both sides) yet the two orderings produce **identical**
  short-head sets — 0 of 8,196 items differ. Ascending id was kept because it
  does not depend on file ordering.
- **k-NN backend for the geometry metrics.** NP and NP_ref follow
  `arm.knn_backend`, which defaults to `torch`; the notebook used sklearn. The
  two agree on >99% of neighbours, and `knn_backend=sklearn` reproduces the
  notebook path exactly.
- **NP subsample seed** is fixed at `geometry.np_seed=42` for every run seed,
  so all arms are measured on the same nodes. The notebook subsampled with the
  run seed.
- **Standard deviations** in `analysis/summarize.py` come from pandas
  (`ddof=1`); the notebook's `build_cap5_table` used `np.std` (`ddof=0`). With
  five seeds this raises the reported sd by about 12%.

### Mutation coverage: 12 of 14 guarantees

Caught: budget equalisation, selection window, val-masked-on-test,
train-masked-on-val, NDCG ideal list, W=0 reference snapshot, manifold term
applied, popularity source, Laplacian symmetrisation, validation-split
disjointness, the W≥E guard, and `at_last_epoch` geometry.

Not caught, both explained rather than papered over: the GPU RNG coercion (a
no-op on CPU, exercisable only on the target GPU), and `min_train` in the
resume check (a benign mutation — see item 3).

## Differentiable kernel weights (post-gate plan, step 2)

New arms `emb_mr_dw` and `emb_mr_fresh`; `emb_mr` and every other arm are
unchanged. All claims below were verified by running them before hand-over.

**Design.** Making the kernel weights differentiable necessarily recomputes
them from the current embeddings every batch, whereas `emb_mr` computes them
once per rebuild and freezes them for 50 epochs. Compared against `emb_mr`
alone, "weights become current" and "gradient flows through the weights" would
be confounded. So the weights come in three modes on one pairwise code path:

| mode | weights | role |
|---|---|---|
| `frozen` | build-time, constant | must reproduce `emb_mr` exactly (verification only) |
| `fresh` | recomputed each batch, detached | arm `emb_mr_fresh`: the control |
| `grad` | recomputed each batch, differentiable | arm `emb_mr_dw`: the experiment |

Neighbour sets, sigma (median heuristic), lambda, W, E and the 50-epoch
rebuild are identical across all three. `emb_mr` keeps its original sparse
Laplacian path; the new arms select the pairwise path via
`arm.kernel_weights`, a key absent from the original arm files so their saved
configs stay byte-identical.

**Correctness.** Via the exact identity
tr(ZᵀLZ) = ½ Σᵢ Σⱼ∈kNN(i) wᵢⱼ ‖zᵢ − zⱼ‖², the `frozen` mode reproduces the sparse
term: value error 0 and gradient error 2.8×10⁻¹⁷ in float64; value identical and
gradient 1.3×10⁻⁷ relative against `emb_mr`'s actual float32 term. Over a full
training run at lambda = 1, `frozen` matches `emb_mr` to 7.5×10⁻⁹ in the final
embeddings while removing MR moves them by 2×10⁻² — so the check discriminates.
`fresh` and `grad` share their value exactly and differ in gradient by 30%.
gradcheck passes; a sign test on the unit circle confirms `grad` attracts close
pairs and repels pairs beyond 2σ², while detached modes only ever attract.

**Performance.** Plain autograd through the neighbour gather cost 6.1 s per call
at Gowalla scale on CPU (vs 0.6 s for the sparse term), 4.5 s of it in
scattering an (n, k, d) gradient. A custom backward (`_KnnDots`) computes the
same gradient as S·Z + Sᵀ·Z with a CSR pattern built once per rebuild, bringing
it to 2.0 s with no extra memory beyond the graph build. It is tested against
the plain-autograd reference (`_pairwise_manifold_loss_reference`) to 10⁻¹² in
every mode. CSR column indices are kept sorted: PyTorch's CPU multiply
tolerates unsorted ones, but cuSPARSE on the GPU may not, and that cannot be
tested here. GPU speed is unmeasured; time the first MR epochs on the RTX 5060 Ti.

**Diagnostic.** Each rebuild logs sigma and `frac_beyond_crossover`, the share
of neighbour pairs past the point where `grad` turns repulsive (exact on the
unit sphere, approximate otherwise). In a short CPU run on Gowalla at d = 64 it
was 0.00% and 0.56% — early, small embeddings, so not a result; it is logged so
the real run can say whether repulsion ever engages.

**Tests.** 21 new (62 total); `audit_equivalence.py` still 24/24. Mutation
testing: 20 of 20 deliberate bugs caught — 7 in the pairwise term, 6 in the
trainer/loss wiring, 7 in the custom backward and its CSR structure.

**Analysis.** `analysis/compare_kernel.py`, pre-registered: primary test is
`emb_mr_dw` vs `emb_mr_fresh` on er_table (uncorrected, single test);
everything else Holm-corrected together. Verified on a replica with planted
effects.

### Extension to Yelp2018 and Amazon-Book (pre-registered 29 Sep 2026, before running)

Gowalla result (full design, 10 seeds): emb_mr_dw raises ER over emb_mr_fresh by
+0.620 in 10/10 seeds (p = 0.002); fresh vs emb_mr accounts for only +0.024.
MR's ER gap to the control shrinks from -0.963 (emb_mr) to -0.319 (emb_mr_dw);
np_ref gap shrinks ~40%; Gini and tail coverage are unchanged.

For Yelp2018 and Amazon-Book only emb_mr_dw is run (no emb_mr_fresh), on the
evidence that freshness was ~4% of the effect on Gowalla. That transfer is an
assumption and is stated as such. `analysis/compare_kernel.py` selects the
design from which arms exist, never from results:

- **full** (emb_mr_fresh present): primary dw vs fresh on er_table; secondary
  family unchanged from the Gowalla analysis (14 tests).
- **dw-only**: primary dw vs emb_mr on er_table (includes the small freshness
  component); secondary dw vs emb_mr (other metrics) and dw vs mr_off (9 tests).

Usage: `python -m analysis.compare_kernel runs <dataset>`.

## Tier 0 interpretation analysis (3 Oct 2026)

`analysis/tier0.py` — exploratory, from existing artefacts only. Results are a
**working hypothesis**, not a conclusion: the open experiments (pair-split
kernel gradient, Fast Differentiable Sorting) bear directly on them, and every
p-value here is uncorrected.

**Caught during testing.** Only the validation-selected checkpoint is saved,
and its epoch can differ between paired runs. On the synthetic replica that
reversed a sign. Every checkpoint comparison is therefore reported over all
pairs *and* over same-epoch pairs only, with ER at the fixed final epoch as
reference. On the real runs the three columns agree in sign everywhere.

**Observed (10/10 seeds, all three datasets).** Emb-MR changes look like an
alignment force: alignment better, uniformity worse, rank lower. The
differentiable kernel moves embeddings back along that trade-off. Effects
survive L2 normalisation (angular), as expected from the kernel-gradient term
being purely rotational at the propagated level. The dw - emb_mr ER gap forms
73-93% in epochs 100-450, but keeps growing in all seeds where no net-repulsive
pairs remain, so net repulsion is not necessary for the effect. Attribution of
the early bulk is open: split the kernel gradient by pair type.

**Runtime.** ~1-3 s per checkpoint on the RTX 5060 Ti desktop (CPU), about
2.5 min in total. The 49 s measured during development was a one-core sandbox.

**Correction: determinism.** Earlier entries describe the shared warm-up of two
arms as "bit-identical by construction" and the pipeline as bit-deterministic.
That was verified on CPU only. On the RTX 5060 Ti the end-of-warm-up ER of two
arms differs by 2-3e-4 (out of ~200), i.e. reproducible to ~1e-6 relative --
consistent with `deterministic: false` and non-deterministic CUDA sparse
matmul. No conclusion changes (effects are 0.2-1.4 in ER), but "bit-identical"
must not be claimed for GPU runs. The notebook-equivalence results
(`audit_equivalence.py`, 24/24) are exact and unaffected: they run on CPU.

## Second full comb before the hand-off (3 Oct 2026)

Static analysis, config wiring in both directions (including `.get()` reads),
a line-by-line read of everything added since the last audit, every script run
end to end, and a fresh mutation sweep.

**My mutation harness was wrong at first.** It counted any output containing
"error" as a caught mutation, and PyTorch's warning text ("Memory errors (e.g.
SEGFAULT)") appears in every passing run, so everything showed CAUGHT. Spotted
because a mutation known to be uncatchable on CPU (the GPU RNG coercion) was
reported caught. The harness now uses pytest's exit code and was validated with
controls: a harmless edit and the CPU no-op are not caught, a real bug is. Earlier
sweeps in this log used a different check and are unaffected.

**Six guarantees had no automated test** -- each verified by hand when built:
resume rejecting a different architecture; `weights_only` resumes still
validating; `summarize` refusing an ambiguous arm name and repeated seeds; Holm
counting only tested rows; the clobber guard. `tests/test_guards.py` adds a test
for each, and each was confirmed to fail when its guard is removed. 68 tests.

**Differentiable-kernel mutations re-run on the current code.** The original 20
predated the fast-backward rewrite, so some patterns targeted code that no longer
exists. Re-run against current code: 20/20 caught. Project total: 42/42, plus the
GPU RNG coercion (untestable on CPU).

**Scripts.** `run_gate_converged.sh` would have collided with the notebook-protocol
replication in `runs/gowalla/mr_off_w1000_K3_d256` (different protocol; the clobber
guard would refuse). It now writes to `*_converged_w1000`; tested end to end on
synthetic data with epochs scaled down (6 runs, 3 resumed, exit 0). The guide's
cheat-sheet had the same colliding path and now points to the script.
`run_sweep.sh` and the converged script ended with a bare `summarize` call, which
is ambiguous on Gowalla; `summarize` now prints the ambiguity as a message after
the table instead of a traceback.

**Analysis scripts.** `tier0.py` built datasets with default split settings; it
now reads the split from the runs and refuses if they disagree (verified: refuses
a mixed-split replica; results on a same-split replica unchanged to 4 decimals).
`compare_kernel.py` now gives p = 1 for a metric identical in every pair instead
of risking an exception or a NaN in Holm.

**Clean-ups.** Dead variables and unused imports removed; pyflakes is clean apart
from one intentional side-effect import (`import src.train`, which registers the
run-folder resolver). Config wiring verified complete in both directions.

## Pair-split of the kernel gradient (4 Oct 2026) -- pre-registered before running

Question carried over from Tier 0: does `emb_mr_dw`'s effective-rank gain come
from the few neighbour pairs that the kernel gradient turns net-repulsive, or
from the many pairs whose attraction it merely weakens?

### The net-repulsion rule, and a correction to the logged diagnostic

Per pair, E = w(cos) ||z_i - z_j||^2 with w = exp(-(1 - cos)/sigma^2). The
kernel term d2 * grad(w) is purely tangential and pushes the pair apart in
angle; the tangential part of the attraction pulls it together. Comparing the
two coefficients gives the exact rule: **net-repulsive iff d2 > 2 sigma^2
|z_i||z_j|**, symmetric in i and j. Checked against autograd on 20,000 random
pairs with varied norms: 20,000/20,000 agree.

Dividing by |z_i||z_j| gives 2(1 - cos) + (r + 1/r - 2) > 2 sigma^2 with
r = |z_i|/|z_j|. The rule logged until now as `frac_beyond_crossover`
(1 - cos > sigma^2) is the unit-sphere case; off the sphere the extra term is
>= 0, so **the logged share can only undercount**. On a trained synthetic
LightGCN (neighbour norm ratio median 1.15, 90th percentile 1.43) the logged
rule gave 6.5% and the exact rule 16.6%. Synthetic, so only illustrative.

Consequence: the Tier 0 statement that the dw - emb_mr ER gap "keeps growing
where no net-repulsive pairs remain" rests on the undercounting share and is
**suspended** until re-measured. `analysis/crossover_check.py` does that from
the saved checkpoints on CPU (graph rebuilt at the selected checkpoint, so an
approximation of the graph in use; early epochs are not reachable that way).

### Design

Two arms, routing the gradient through the kernel weights with
`torch.where(mask, w, w.detach())`: same loss value as `fresh` and `grad`
everywhere, kernel gradient kept only on the routed pairs.

- `emb_mr_dw_weaken` (mode `grad_weaken`): kernel gradient only on pairs that
  stay net-attractive.
- `emb_mr_dw_repel` (mode `grad_repel`): kernel gradient only on net-repulsive pairs.

With the existing `emb_mr_fresh` (no kernel gradient) and `emb_mr_dw` (all
pairs) this is a 2x2 factorial, so both main effects and the interaction are
estimable. Gradients add exactly: weaken + repel - fresh == grad.

Choices made in advance:
- **Exact rule**, not the unit-sphere one.
- **Re-evaluated every batch**, not frozen at rebuild: an arm is a routing
  rule, so the two classes can evolve differently across arms. Inherent, and
  stated.
- The classification is the sign of **the pair term's own force** on the
  propagated embeddings. A node's actual update also reflects its other pairs,
  BPR, back-propagation through LightGCN and Adam (cf. Islam & Fleischer, TMLR,
  on per-pair attraction/repulsion "shapes" versus net motion).
- Like `fresh`, a routed gradient is not the derivative of the reported loss;
  it descends a surrogate with w held fixed on unrouted pairs. The masking
  technique is what Cloud et al. (2024) call gradient routing.

### Pre-registration (`analysis/compare_split.py`)

Gowalla, 10 seeds (2020-2029), paired by seed, two-sided exact Wilcoxon.
- **Primary:** weaken - repel on er_table, uncorrected.
- **Secondary**, Holm-corrected together (11 tests): weaken - fresh, repel -
  fresh and the interaction (dw - weaken - repel + fresh) on er_table,
  np_ref@20 and recall@20; weaken - repel on np_ref@20 and recall@20.
- **Descriptive:** each class's share of dw's ER gain per seed; net-repulsive
  share (exact and sphere rule) and the kernel-budget share of net-repulsive
  pairs per MR window.
- **Caveat fixed in advance:** the weaken class is the larger one, so a
  positive primary can partly reflect class size. The kernel-budget share is
  reported alongside; no size-normalised test is run.
- `emb_mr_fresh` and `emb_mr_dw` are the existing runs; reusing them is valid
  because their code paths are unchanged (below).

### Verification

- `frozen`, `fresh` and `grad`: value and gradient **bit-identical** to the
  previous implementation, float32 and float64. The existing runs stand.
- 18 new tests in `tests/test_pair_split.py` (86 in total): exact rule
  vs autograd; sphere rule only undercounts and coincides on the sphere; equal
  values; exact additivity; routing differs from both fresh and grad on mixed
  pairs; limits collapse to fresh / grad with non-trivial gradients (sigma
  chosen from data -- a first version using sigma = 1e-3 underflowed every
  weight and passed vacuously); fast path vs plain-autograd reference;
  diagnostics vs the rule, including an exact single-pair identity; trainer
  level: pairwise path, logging, all four corners distinct, shared warm-up,
  and logging proven side-effect free.
- A gradcheck on the split modes was dropped as conceptually wrong (a routed
  gradient is by design not the derivative of the value) and replaced by the
  property that holds: the stopped term equals the routed term exactly.
- Mutation sweep (harness validated with a harmless-edit control, which was
  not caught): 15/15 caught -- routing, rule, diagnostics and trainer wiring.
  Two diagnostic mutations were initially missed and are now pinned.
- `audit_equivalence.py` 24/24. Routing costs nothing measurable; diagnostics
  cost about one forward pass per evaluation.
- **Not verified:** the routed modes on the GPU. They use the same CSR
  backward that has already run on the RTX 5060 Ti; only `torch.where` and the
  mask are new.

Logged per evaluation (pairwise arms, new runs only): `kd_frac_repel_exact`,
`kd_frac_repel_sphere`, `kd_kernel_share_repel`, `kd_kernel_to_attraction`;
each rebuild event also carries `frac_repel_exact`.
