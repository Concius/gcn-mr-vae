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
