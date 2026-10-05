# Handoff — state of the project, 4 Oct 2026 (pair-split results in; λ re-sweep pre-registered)

## Project
- MSc dissertation **GCN-MR-VAE** (LightGCN + Manifold Regularization + VAE for
  collaborative filtering), UNESP, FAPESP 2025/07727-0. Advisor: **Pedronette**.
- Qualification passed Sep 2026. Defence deadline **10 Sep 2027** (regimental maximum).
- Repo: github.com/Concius/gcn-mr-vae. Ported from `LightGCN_Manifold_refactored (6).ipynb`.
- Hypotheses: H1 LightGCN training distorts geometry (validated); H2 MR raises ER/NP
  without hurting Recall; H3 MR improves diversity; H4 MR-regularised inputs help
  the VAE resist posterior collapse -- operationally, KL above the GCN-VAE control's
  KL together with Recall@20 (see `VAE_ROADMAP.md`).
- Committed baselines: MF-BPR, LightGCN, VAE-CF, GCN-VAE. DirectAU was removed as a
  committed baseline (it remains relevant as theory and as a possible MR alternative).

## Environment and workflow
- Pop!_OS desktop, RTX 5060 Ti 16 GB, PyTorch cu128, venv at `~/Projects/gcn-mr-vae/.venv`.
- Long runs in tmux. Paste commands **one line at a time**; never type into a running
  session; check progress from another terminal (`tail -5 runs/<log>`). Copying a
  tmux pane with select-all produces misleading interleaved lines.
- Downloads re-save as `name (1).zip`; quote such paths, and the unsuffixed file may be stale.
- Training never downloads data (`python -m src.data.download <ds>`); `run_gate.sh` checks.
- Clobber guard: an identical config re-run overwrites its own folder; a different config
  into the same folder is refused. Variants of one config need `paths.run_dir=...`.
- Back up after every batch: `tar czf ~/gcn_results_$(date +%Y%m%d).tar.gz
  --exclude='*.pt' --exclude='*.npz' runs/` and copy to Drive.
- Measured hours per run (mr_off / emb_mr): Gowalla 0.48 / 0.66, Yelp 0.93 / 1.18,
  Amazon 3.45 / 4.48. `emb_mr_dw` costs about the same as `emb_mr`.

## Repository
- One `Trainer` for every arm; composable loss; Hydra configs; arms `baseline`, `mr_off`,
  `emb_mr`, `coocc_mr`, `emb_mr_fresh`, `emb_mr_dw`, `emb_mr_dw_weaken`, `emb_mr_dw_repel`.
- 113 tests; `audit_equivalence.py` 24/24 against the notebook (CPU, exact). Mutation
  sweeps (harness validated with controls): 42/42 guarantees on 3 Oct, plus 15/15 for the
  pair-split code and 21/21 for the λ-sweep analysis on 4 Oct; the GPU RNG coercion is
  untestable on CPU (a no-op there).
- Analysis: `analysis/summarize.py` (groups by run directory, `--dataset`, refuses
  ambiguous arm names), `analysis/compare_kernel.py` and `analysis/compare_split.py`
  and `analysis/lambda_sweep.py` (pre-registered), `analysis/tier0.py`,
  `analysis/crossover_check.py` (CPU), `analysis/config_diff.py` (does a new command
  reproduce an existing run's config? run before reusing old runs).
- On Gowalla the bare name `emb_mr` matches several configurations: use exact tags.

## Results

### Established
1. **Chapter 5 gains were a protocol artefact.** Under the notebook protocol (full edges,
   test selection, W=1000 / E=2000, fresh Adam) Emb-MR gives +1.64% Recall on Gowalla,
   reproducing Chapter 5 (+1.53%; baseline 0.1877 identical). Decomposition from the same
   epoch-1000 checkpoint (3 seeds): 1000 extra BPR epochs alone give +1.33% Recall, ER +6.28,
   np_ref +0.029; MR's own contribution is +0.29% Recall, ER −0.70, np_ref −0.007; resetting
   Adam contributes nothing.
2. **Corrected protocol, 10 seeds × 3 datasets (emb_mr vs mr_off; W=100, E=1000, λ=1e-5,
   validation selection, budget-matched).** ER and np_ref lower in 30/30 seed pairs,
   significant under every Holm grouping (per metric, per hypothesis, all 15 tests).
   Gini and tail coverage worse on Gowalla and Yelp (10/10, all groupings). Amazon: tail
   coverage better 10/10 (all groupings), Gini better 9/1 (grouping-dependent).
   Recall: Gowalla n.s. (7/3), Yelp slightly better 8/2 (grouping-dependent), Amazon worse
   10/10 (all groupings). Effects are small (ER shifts ~0.3–0.4%).
3. **Rebuild frequency is not the lever** (Gowalla, 3 seeds): every schedule lowers ER;
   rebuilding every epoch equals every 10; differences between schedules are within seed noise.
4. **Differentiable kernel weights (`emb_mr_dw`).** Gowalla, full design: dw − fresh
   +0.620 ER, 10/10 (pre-registered primary, p=0.002); fresh − emb_mr only +0.024.
   Yelp and Amazon, dw-only design: dw − emb_mr +0.409 and +0.631 ER, 10/10 (primaries).
   Removes 59–77% of MR's ER penalty and ~40% of its np_ref penalty, but MR stays below the
   control on both. Diversity worse than the control everywhere (on Amazon dw loses
   emb_mr's diversity gains). Recall level with the control except Amazon (slightly worse, 9/10).
5. **The kernel-gradient term is purely rotational** at the propagated level (radial share
   2.5e-15, verified numerically).
6. **Determinism:** GPU runs are reproducible to ~1e-6 relative, not bitwise
   (end-of-warm-up ER gap 2–3e-4). Bit-identical holds on CPU only.
7. **Pair-split of the kernel gradient (Gowalla, 10 seeds, pre-registered; commit
   a707398).** Routing the kernel gradient only to pairs it merely weakens
   (`emb_mr_dw_weaken`) reproduces ~86% of `emb_mr_dw`'s ER gain (median share 0.87,
   range 0.79–0.90); in that arm the MR term never exerts a net-repulsive pair force,
   so net repulsion is not necessary for most of the effect. Primary weaken − repel
   +0.415 ER, 10/10, p=0.002. Net-repulsive pairs (`emb_mr_dw_repel`) contribute a
   smaller real part (+0.12 ER, 10/10) and nothing detectable on np_ref or Recall;
   interaction −0.035 (10/10 negative, slightly sub-additive). Weakening also raises
   np_ref (+0.0025, 10/10) at a negligible Recall cost (−0.0002, 9/1). Repelled pairs
   carry ~34% of the kernel force early, ~17% over the MR phase, for 14–19% of the gain:
   not disproportionately potent (descriptive). Details in `MIGRATION.md`.
8. **The logged crossover share undercounts.** Exact rule d2 > 2σ²|z_i||z_j|. Early MR
   window: 18.2% net-repulsive vs 5.7% logged; late: 1.5% vs 0.00%. At the saved late
   checkpoints: Gowalla ~1.3%, Yelp ~0.2%, Amazon ~1.0% (similar in `mr_off`).

### Working hypothesis (Tier 0, exploratory, uncorrected — not a conclusion)
On all three datasets (10/10 seeds), Emb-MR's changes look like an alignment force
(alignment better, uniformity worse, rank lower), and the differentiable kernel appears to
move embeddings back along that trade-off; the effects survive L2 normalisation. The
dw − emb_mr ER gap forms mostly in epochs 100–450 but keeps growing where no net-repulsive
pairs remain, so net repulsion is not necessary for the effect. **Revised 4 Oct:** that
inference used the logged share, which undercounts (item 8); "no net-repulsive pairs
remain" was false (~1.5% remain late). It is superseded by the pair-split (item 7), which
shows directly that most of the effect needs no net repulsion. The pair-split is
consistent with "the kernel gradient mostly relaxes MR's pull", but attributes by pair
class, not by mechanism: the alignment/uniformity reading stays a hypothesis.

## Literature anchors (found by search this session, except where noted)
- Böhm, Berens & Kobak, JMLR 2022 — attraction–repulsion spectrum; Laplacian Eigenmaps at
  the pure-attraction end.
- Wang & Isola, ICML 2020 — alignment and uniformity. DirectAU (Wang et al., KDD 2022) —
  both matter for CF.
- Loveland et al., WWW 2025 — alignment induces rank collapse, uniformity raises rank; stable rank.
- Di Giovanni et al., TMLR 2023 — attraction/repulsion via eigenvalue signs; smoothing vs sharpening.
- Chen et al. (nCL), WSDM 2024 — dimensional collapse in CF, singular-value spectra.
- Li, Han & Wu 2018; Cai & Wang 2020 — GCN propagation as Laplacian smoothing (§6.3 risk).
  Cited in the qualification text; not re-checked here.
- Liang et al., WWW 2018 — Mult-VAE^PR: multinomial likelihood, β annealed from 0.
- Islam & Fleischer, TMLR — per-pair attraction/repulsion "shapes"; one term can be
  attractive below a distance and repulsive above it, like x·exp(−x/2σ²) here.
- Cloud et al. 2024 (arXiv 2410.04332) — gradient routing: selective stop-gradient masks,
  the technique the pair-split uses.

## Next steps (in order)
1. ~~Pair-split kernel gradient~~ — done 4 Oct (Established, item 7). Optional: repeat
   on Yelp/Amazon (fresh, weaken, repel would be new there).
2. **Fast Differentiable Sorting** (§4.1.2's named remedy; after the λ re-sweep, whose
   curve is the yardstick it must beat) — differentiable neighbour
   *selection*. A full differentiable k-NN over 70k+ nodes cannot fit in memory; needs a
   candidate-restricted design.
3. **λ re-sweep under the corrected protocol** — pre-registered 4 Oct, NEXT TO RUN
   (`MIGRATION.md`, `analysis/lambda_sweep.py`). Fulfils the qualification's §4.2.3
   commitment (λ selected by validation Recall@20), gives the dose-response, and is the
   control for item 7: stage 2 runs `emb_mr` at the λ′ that matches `emb_mr_dw` on ER and
   tests np_ref. Gowalla first (~26 h + ~7 h); Yelp (~47 h + ~12 h) and Amazon (~180 h)
   are a later decision.
4. **CoOcc-MR (maybe)** — λ=1e-4 (notebook cell 19). np_ref is circular for it (both built
   from R); judge on ER and Recall or pick another external reference.
5. If MR cannot be fixed: a uniformity term (DirectAU) or stable-rank regularisation (Loveland).
6. **Module 3:** Mult-VAE^PR (not in PyTorch) as the VAE-CF baseline and template; trainer
   seams B2/B3 from `VAE_ROADMAP.md`; then GCN-VAE and GCN-MR-VAE; H4 budget-matched,
   10 seeds, KL / active-unit logging.
7. **Writing:** Chapter 5 rewrite (Jan–Feb). §6.3's diversity fallback no longer holds;
   §6.5's "MR produz ganhos nos três datasets" is false under the corrected protocol;
   §6.4's differentiable diversity terms become the natural follow-up.
8. **Publication:** the confound finding suits a reproducibility track (check RecSys CFP); iSys.

## Decisions pending
- Holm family for reporting (per hypothesis is the most defensible; at 10 seeds unanimous
  results survive every grouping) — state the choice.
- GCN-MR-VAE encoder input: raw interaction vector or LightGCN/MR user embedding.
- Whether to run `emb_mr_fresh` on Yelp/Amazon (currently assumed freshness ~4% transfers).
- Module-2 framing, to discuss with Pedronette.

## Do not claim
- "Bit-identical" for GPU runs.
- Chapter 5's MR gains as an MR effect.
- The alignment/uniformity mechanism as established.
- The direction of the ER effect as pre-registered (first observed on Gowalla).
- That no net-repulsive pairs remain late in training: the logged share undercounts.
- That net-repulsive pairs play no part: repel − fresh is +0.12 ER, 10/10.
- The pair-split attribution beyond Gowalla, or `emb_mr_dw` as more than "less MR"
  (untested until the λ re-sweep).

## `runs/` map
- `<ds>/mr_off_w100_K3_d256`, `<ds>/emb_mr_w100_lam1e-05_k20_r50_K3_d256` — corrected gate,
  10 seeds, all three datasets.
- `<ds>/emb_mr_dw_w100_lam1e-05_k20_r50_K3_d256` — 10 seeds, all three datasets;
  `gowalla/emb_mr_fresh_...` — 10 seeds.
- `gowalla/emb_mr_w100_lam1e-05_k20_{r1,r10}_K3_d256`, `gowalla/emb_mr_w100_lam1e-05_k20_K3_d256`
  (never rebuilt) — rebuild ablation, 3 seeds.
- `gowalla/*_ablationbatch` — seeds 2020–2022 of the gate config run twice (reproducibility).
- `gowalla/baseline_w1000_K3_d256`, `gowalla/emb_mr_w1000_lam1e-05_k20_r50_K3_d256`,
  `gowalla/emb_mr_w1000_adamrestored`, `gowalla/mr_off_w1000_K3_d256` — notebook-protocol
  replication and decomposition, 3 seeds. (`run_gate_converged.sh` writes to
  `*_converged_w1000` instead, so it cannot collide with these.)
- `_tier0/` and `tier0.log` — Tier 0 outputs.
- `gowalla/emb_mr_dw_{weaken,repel}_w100_lam1e-05_k20_r50_K3_d256` — pair-split, 10 seeds;
  `split.log`. `_crossover_check/checkpoints.csv` and `crossover_check.log` — pre-check.

## Standing rules
Verify claims by running them before stating them; one variable per experiment;
pre-register the primary test; mutation-test new code; say plainly when something is
untested or only verified on CPU; separate established results from working hypotheses.
