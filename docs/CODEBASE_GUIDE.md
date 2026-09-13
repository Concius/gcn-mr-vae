# GCN-MR-VAE — The Complete Codebase Guide

*Version 1.0 — 11 September 2026 — describes the repository as of the post-port audit and the scorer seam.*

---

## How to read this document

This guide is written in layers. Every section starts with a **plain-language** explanation that assumes no programming or machine-learning background, then goes **into detail** for readers who want to follow the code, and closes with **why it is built this way**, which is the part a future maintainer (or an examiner) most needs.

If you are a non-programmer, read Parts 1, 2, 4, 6 and the Glossary. You will understand what the software does, why the experiments are designed the way they are, and what changed from the original notebook, without needing to open a single source file.

If you are going to modify the code, read everything, in order. Part 3 is the file-by-file tour; Part 7 tells you how to prove you have not broken anything.

The document is organised around four questions:

| Part | Question it answers |
|---|---|
| 1–2 | *What is this software for, and what ideas do I need to understand it?* |
| 3 | *What does each file do?* |
| 4–5 | *Why are the experiments shaped this way, and how do I run them?* |
| 6–7 | *What changed from the Colab notebook, and how do we know nothing broke?* |

Appendices give a reference for every configuration key, every output field, and every command.

---

# Part 1 — What this project is

## 1.1 The problem, in one paragraph

A recommender system looks at what people have interacted with in the past — places they have checked in to, restaurants they have reviewed, books they have bought — and predicts what they will want next. The datasets used here (Gowalla, Yelp2018, Amazon-Book) contain only *implicit feedback*: we know a user interacted with an item, not how much they liked it. The task is called **Top-K recommendation**: for each user, produce a ranked list of K items they have not yet seen, and hope the things they actually go on to interact with are near the top.

## 1.2 The three modules

The dissertation proposes a pipeline of three modules, each addressing a specific weakness of the previous one.

**Module 1 — LightGCN (the encoder).** Every user and every item is represented by a vector of 256 numbers, called an *embedding*. LightGCN improves those vectors by letting them absorb information from their neighbours in the user–item graph: a user's embedding is nudged towards the embeddings of the items they interacted with, and those items towards the users who interacted with them, repeated for three rounds. This is *graph convolution*, and it is what makes the model *collaborative* — you benefit from the taste of people similar to you.

**Module 2 — the Manifold Regularisation (MR) layer.** The dissertation's diagnosis (Hypothesis H1) is that graph convolution over-compresses the embedding space: after training, the 256 dimensions are used unevenly, and the local neighbourhood structure the model started with is almost entirely discarded. The MR layer adds a penalty during training that says: *if two embeddings are close now, keep them close*. It does this by building a graph of each embedding's 20 nearest neighbours and penalising abrupt changes across that graph. The penalty is small, and the neighbour graph is rebuilt every 50 epochs so it tracks the embeddings as they learn.

**Module 3 — the VAE (not yet implemented).** A Variational Autoencoder turns each user's embedding into a *distribution* rather than a single point, capturing uncertainty about their preferences. It is scheduled for the November 2026 – January 2027 window. The code has a fixed attachment point for it (see `src/models/vae.py` and `VAE_ROADMAP.md`), but nothing in this repository trains a VAE yet.

## 1.3 The four hypotheses the code is built to test

The experiments exist to accept or reject four falsifiable claims, stated in §1.2.2 of the qualification text:

- **H1 — Distortions exist.** Training LightGCN measurably lowers *Effective Rank* (how many of the 256 dimensions are really used) and pushes *Neighbourhood Preservation* against the initialisation towards zero. *Validated by:* ER dropping relative to its value at random initialisation, and NP-vs-init near zero.
- **H2 — MR fixes geometry without hurting accuracy.** With the MR layer, Effective Rank is higher and neighbourhoods are more faithful to an *external* reference graph, while Recall@20 is not worse. *Refuted if* the geometric gain costs statistically significant accuracy.
- **H3 — MR reduces popularity bias.** Gini coefficient lower, Tail-Coverage higher, both significant by paired Wilcoxon over five seeds.
- **H4 — MR helps the VAE.** The full pipeline shows a higher KL divergence (the latent variable is actually used) *and* higher Recall@20 than the same pipeline without the MR layer. Not testable until Module 3 exists.

Every metric the code computes maps onto one of these. When you see `er_table` or `np_vs_ref@10` in a results file, that is H1. `np_ref@20` and `test_recall@20` are H2. `test_gini@20` and `test_tail_catalog_coverage@20` are H3.

## 1.4 What the code does today, and what it does not

**Does:** loads the three benchmark datasets; carves a validation set; trains LightGCN with or without the MR term under a strictly controlled protocol; evaluates all recommendation, diversity and geometric metrics; writes every result to disk; aggregates runs and performs the paired statistical test.

**Does not (yet):** train a VAE; run the VAE-CF baseline; run the sparse-user experiment. These are Phase B and C of `VAE_ROADMAP.md`.

---

# Part 2 — The science in plain language

You do not need to follow the mathematics to use the code, but you do need the vocabulary. Every term here appears in a filename, a config key, or an output column.

## 2.1 Users, items, interactions, the graph

The data is a list of pairs: *user 17 interacted with item 4032*. Nothing else. Stacked into a table with one row per user and one column per item, this is the **interaction matrix** `R`, mostly zeros (Gowalla is 99.93% empty). The same data drawn as dots and lines — every user a dot on the left, every item a dot on the right, a line for each interaction — is the **bipartite graph**. The code stores both views: `dataset.UserItemNet` is the matrix, `dataset.sparse_graph()` is the graph in the normalised form LightGCN needs.

The files come in the standard LightGCN layout: `train.txt` and `test.txt`, one line per user, the user id followed by the item ids they interacted with.

## 2.2 Embeddings

An embedding is a list of numbers that stands in for a user or an item. Nothing about it is meaningful on its own; what matters is the *geometry* — which embeddings are near each other, which directions the cloud of points stretches along. Recommendation is then just: score every item for a user by how well the two embeddings align (a dot product), and rank. The whole model is one table of 29,858 user vectors and one of 40,981 item vectors, 256 numbers each. These raw tables are the **ego embeddings**. In the code, `model.ego_embeddings()` returns them stacked.

## 2.3 LightGCN propagation

Start from the ego embeddings. Replace each user's vector with the (degree-normalised) average of its neighbouring items' vectors, and each item's with the average of its neighbouring users'. That is one *layer*. Do it three times, keeping every intermediate result, and average all four (the original plus three propagated versions). The result is the **propagated embedding**, and it is what the model scores with.

In the code this is `LightGCN.computer()` — the name is kept from the reference implementation so that Chapter 5 of the dissertation, which mentions it, still reads correctly. Setting `model.n_layers=0` skips propagation entirely and gives plain matrix factorisation (**MF-BPR**), which is one of the committed baselines.

## 2.4 How the model learns: BPR

The training signal is **Bayesian Personalised Ranking**. Take a user, an item they interacted with (the *positive*), and a random item they did not (the *negative*). The model should score the positive higher than the negative. The loss is `softplus(score_neg − score_pos)`, averaged over a batch of such triples; it is near zero when the positive wins comfortably and grows when it does not. A small **L2 penalty** on the ego embeddings keeps them from growing without bound. That is the entire objective for Module 1.

Negatives are drawn uniformly from the catalogue and re-drawn if they happen to be one of the user's positives. `src/sampling.py` does this.

## 2.5 Manifold regularisation

Three ideas, in order.

**k-nearest-neighbour graph.** For every embedding (users and items together, 70,839 points on Gowalla), find the 20 most similar others by cosine similarity. Connect them. Weight each connection by how close the pair is, using a Gaussian kernel whose width is the median distance — so "close" is defined relative to the data, not by a magic number.

**Laplacian.** From that weighted graph build the matrix `L = D − W`, where `W` holds the connection weights and `D` is a diagonal with each node's total weight. This is the standard graph Laplacian; its one property that matters here is that the quantity `tr(ZᵀLZ)` — the **Dirichlet energy** — is small when connected embeddings are similar and large when they are not.

**The penalty.** Add `λ · tr(ZᵀLZ) / n` to the loss, with λ = 10⁻⁵. Its gradient is `2LZ / n`, closed form, so nothing about it interferes with backpropagation. The graph is rebuilt every 50 epochs from the *current* embeddings, which is why the MR phase cannot start at epoch 0 — a neighbour graph over random vectors would lock in meaningless neighbourhoods.

The variant called **Emb-MR** builds the k-NN graph from the learned embeddings. The variant called **CoOcc-MR** builds it once from the interaction matrix instead (users similar if they share items, items similar if they share users) and never rebuilds. The dissertation's main result uses Emb-MR; CoOcc-MR is the comparison that addresses the "redundant with message passing" risk in §6.3.

## 2.6 The metrics

**Recall@K** — of the items the user actually interacted with in the held-out set, what fraction appear in the top K? Higher is better. The primary accuracy metric; checkpoints are selected on it.

**Precision@K** — of the K recommended, what fraction were hits.

**NDCG@K** — like recall but rewards putting hits *higher* in the list. Normalised so 1.0 means a perfect ordering. *This metric was miscomputed in the notebook; see Part 6.*

**Short-head / long-tail splits** — the 20% most-interacted items are the *short head*, the rest the *long tail* (Pareto split, Celma 2010). Each accuracy metric is also reported restricted to short-head and to long-tail ground-truth items, so you can see whether gains come from popular or obscure items.

**Gini coefficient** — how concentrated recommendations are across the catalogue. 0 means every item is recommended equally often; values near 0.9 are typical and mean a handful of items dominate. Lower is better for diversity.

**Tail-Coverage@K** (Eq. 2.18 of the text) — what fraction of the long-tail catalogue appears in *at least one* user's top-K. The code calls this `tail_catalog_coverage@k`. A different quantity, the mean per-user share of long-tail items in their list, is what the notebook computed; the code reports it too as `tail_share_user@k`, so both are available and the text can cite the right one.

**Effective Rank (ER)** — treat the embedding table as a cloud of points, ask how many dimensions it really spans. Computed from the singular values: ER = exp(entropy of normalised singular values). A perfectly isotropic cloud in 256 dimensions scores ≈ 255; a cloud squashed onto a plane scores ≈ 2. Rows are mean-centred first (Eq. 3.2). The dissertation uses two conventions and the code reports both: **ER_tab** on the ego table (Table 2's headline numbers, ER_init ≈ 255.2), and **ER_prop** on the propagated embeddings (Table 3). Propagation compresses further, so ER_prop ≤ ER_tab.

**Neighbourhood Preservation (NP@k)** — take two snapshots of the same embeddings, find each point's k nearest neighbours in both, report the average overlap. 1.0 means neighbourhoods are unchanged; 0 means completely replaced. Two uses: **NP-vs-reference** compares the final ego table against the snapshot taken when the MR phase started (for the baseline that is random initialisation, which is H1's measurement), and **NP_ref** compares the final propagated *item* embeddings against a fixed external reference — the Jaccard similarity of items' user sets in the training data. NP_ref is H2's geometry criterion because it does not depend on where training started.

## 2.7 Why five seeds and a paired test

Every experiment is run five times with different random seeds (2020–2024). A seed fixes the random initialisation and the negative-sampling stream, so two arms run with the same seed start from the same place and see the same negatives; only the treatment differs. Results are then compared *pair by pair*, seed 2020 against seed 2020, with a one-sided **Wilcoxon signed-rank test**. With five pairs the smallest attainable p-value is 1/32 = 0.031, which is why the text describes p = 0.0313 as certifying directional consistency rather than effect size. Ten seeds bring the floor to ~0.001. `analysis/summarize.py` performs the test and applies Holm–Bonferroni across datasets.

---

# Part 3 — Tour of the repository

## 3.1 The layout, and the one rule behind it

```
gcn-mr-vae/
├── configs/            every experimental choice, as YAML (Hydra)
│   ├── config.yaml       root: seed, epochs, model, optimiser, split, eval, geometry, paths
│   ├── dataset/          gowalla.yaml, yelp2018.yaml, amazon-book.yaml, synthetic.yaml
│   └── arm/              baseline.yaml, mr_off.yaml, emb_mr.yaml, coocc_mr.yaml
├── src/                the software
│   ├── data/             loading, splitting, downloading, synthetic data
│   ├── models/           LightGCN, the MR layer, the composable loss, the VAE stub
│   ├── metrics/          ranking, diversity, geometry, and the evaluator
│   ├── sampling.py       BPR negative sampling
│   ├── trainer.py        the training loop — one class for every arm
│   ├── train.py          the command-line entry point
│   └── utils.py          seeding, device, JSON helpers, RNG state capture
├── analysis/summarize.py   collects results, runs the paired Wilcoxon
├── scripts/            run_gate.sh, run_sweep.sh, run_tests.sh
├── tests/              29 automated tests (see Part 7)
├── audit_equivalence.py    proves the port matches the notebook numerically
├── docs/               this guide
├── MIGRATION.md        cell-by-cell map from the notebook, fix log, audit log
├── VAE_ROADMAP.md      Module 3 readiness and the Nov–Jan build plan
├── data/               datasets land here (git-ignored)
└── runs/               every experiment writes its own folder here (git-ignored)
```

The one rule: **an experiment is a configuration, not a function.** The notebook had one training function per experimental condition and thirty copy-pasted cells to launch them. Here, the same `Trainer` runs every condition, and what distinguishes `baseline` from `emb_mr` is four lines in a YAML file. If you cannot express a new experiment as a config change, that is a signal the code needs a new *capability*, not a new *copy*.

## 3.2 `configs/` — where every decision lives

**Plain language.** Every knob — how many epochs, how strong the MR penalty, how big a batch, where to save — is a named value in a text file. The program reads these files at startup. You change an experiment by editing a value or by overriding it on the command line; you never edit Python to change a setting.

**In detail.** The system is [Hydra](https://hydra.cc). `config.yaml` is the root and declares *defaults*: pick `dataset/gowalla.yaml` and `arm/emb_mr.yaml` unless told otherwise. Any key can be overridden as `key=value` on the command line, nested keys as `section.key=value`, and lists as `'key=[a,b]'`. The `-m` flag turns comma-separated values into a sweep: `arm=mr_off,emb_mr seed=2020,2021` launches four runs.

An **arm** file describes one experimental condition. All four share the same eight keys so that any two arms can be diffed field by field:

| key | meaning |
|---|---|
| `name` | label used in output folders and tables |
| `warmup_epochs` | W — epochs of pure BPR before the treatment (if any) switches on |
| `lambda_manifold` | λ — strength of the MR penalty; 0 disables it |
| `laplacian` | `none`, `embeddings` (Emb-MR) or `cooccurrence` (CoOcc-MR) |
| `k_neighbors` | k in the k-NN graph |
| `rebuild_every` | epochs between graph rebuilds (Emb-MR only) |
| `lambda_scale` | `none` (λ per batch, as the notebook) or `per_epoch` (λ ÷ batches per epoch) |
| `knn_backend` | `torch` (GPU, default) or `sklearn` (the notebook's exact routine) |

The root config's `paths.arm_tag` is computed from those values (`emb_mr_w100_lam1e-05_k20`), so two sweeps over λ never overwrite each other's folders.

**Why.** Because the single most damaging class of bug in the notebook was *hidden state*: a global `config` object mutated by one cell and read by another, and a function that quietly changed `warmup_epochs` to 0 when it found a file on disk. A YAML file that is saved verbatim into every run folder cannot be mutated behind your back, and `results.json` records exactly what ran.

## 3.3 `src/utils.py` — small shared tools

Seeding (`set_seed`), device selection (`get_device`, which raises a clear error if CUDA is requested but absent), a `Timer`, JSON helpers that know how to serialise NumPy and Torch objects, and `rng_state_dict` / `rng_state_load`, which capture and restore the Python, NumPy and Torch random-number states so a warm-start checkpoint can be resumed *exactly*.

## 3.4 `src/data/`

### `splits.py` — carving out validation

**Plain language.** The original datasets come with a training file and a test file. To choose the best model without peeking at the test answers, we set aside 10% of each user's training interactions as a *validation* set. The split is fixed — every run, every seed, every arm sees the same validation items — so comparisons are on identical data.

**In detail.** `holdout_validation(train_user, train_item, val_frac=0.1, split_seed=2020, min_train=2)` returns four arrays. For each user with at least `min_train` items it moves `max(1, round(val_frac · n))` items to validation, never leaving fewer than one in training. Users with fewer than `min_train` items are untouched. `val_frac=0` returns everything as train and reproduces the notebook's (no-validation) protocol. Pure NumPy; the unit tests check determinism, disjointness and the edge cases.

**Why a fixed `split_seed`, separate from the run seed.** If the validation set changed with the seed, seed-2020's `mr_off` and seed-2021's `emb_mr` would be scored on different data. Fixing it means the seed controls only initialisation and sampling — the things that *should* vary between repetitions.

### `dataset.py` — `InteractionDataset`

**Plain language.** Reads the two text files, applies the validation split, builds the matrix and graph views, and works out which items are popular. Everything else in the program asks this object for data.

**In detail.** Construction reads `train.txt` and `test.txt`, sizes the id space from the union of both (as the notebook did), applies `holdout_validation`, and builds three sparse matrices: `UserItemNet` (reduced training), `ValNet`, `TestNet`. Two dictionaries, `valDict` and `testDict`, map user → list of held-out items for evaluation.

`normalized_adjacency()` builds `D^{-1/2} A D^{-1/2}` for the bipartite graph of *reduced-training* edges only, with `scipy.sparse.bmat` — the same matrix the notebook built with a slow element-by-element loop, in seconds instead of minutes. It is cached to `data/<name>/norm_adj_val0.1_s2020.npz`; the filename encodes the split so it can never collide with the notebook's `s_pre_adj_mat.npz`. `sparse_graph(device)` converts it to a Torch sparse tensor once.

`popularity_groups` splits items into short head (top `short_head_frac` = 20% by interaction count) and long tail. By default (`popularity_from: full_train`) the count is taken over the *original* `train.txt`, so the segmentation is a property of the dataset, identical to Chapter 5, and does not shift when `val_frac` changes. (Computing it on the reduced set moved 678 items — 8.27% of the short head — across the boundary on Gowalla, which would have changed every Gini and tail number for reasons unrelated to the method.) Items never seen in training belong to neither group.

**Why validation items are excluded from the graph.** If they stayed in the adjacency, the model would propagate information from the very interactions it is being scored on. That is transductive leakage. The cost is that the corrected runs train on 89.9% of the edges the notebook used, so absolute numbers sit slightly below Chapter 5's. That is expected and must be stated in the text.

### `download.py`

Fetches the three datasets from the official LightGCN-PyTorch repository. `python -m src.data.download gowalla`.

### `synthetic.py`

Generates a tiny dataset (300 users, 200 items by default) with *clustered* preferences, so the k-NN and Jaccard neighbourhoods are meaningful rather than noise. Used by the test suite and intended as the first target for the VAE in November — a model that silently collapses on Gowalla is indistinguishable from one that is wired wrong; on 300 users you can see the difference in seconds.

## 3.5 `src/sampling.py` — BPR negatives

`uniform_sample(train_user, train_item, user_item_csr, n_items, rng)` returns `(N, 3)` triples `(user, positive, negative)`, one per training pair, negatives drawn uniformly and re-drawn until they are not one of the user's positives. The collision check is a single sparse-matrix lookup instead of the notebook's Python loop (a measurable share of epoch time on Amazon-Book's 2.4 million pairs). The `rng` is a dedicated `numpy.random.Generator`, seeded from the run seed and used for nothing else. `minibatch` slices arrays; `n_batches` counts them.

**Why the dedicated generator.** The notebook drew negatives from NumPy's *global* random state, which the Effective Rank routine also consumed (an unseeded `torch.randperm`). Consequence: evaluating more often changed the sequence of negatives and therefore the training trajectory. Here every consumer of randomness has its own stream.

## 3.6 `src/models/`

### `lightgcn.py` — the encoder

Sixty-six lines. Two embedding tables initialised `N(0, 0.1²)`, the sparse normalised adjacency, and `computer()`, which propagates `n_layers` times and averages. `ego_embeddings()` returns the raw stacked tables. There is deliberately *no scoring method* on the model — scoring lives in the evaluator (§3.7) so that Module 3 can replace it without touching the encoder. The dropout knobs the notebook carried but never enabled were not ported.

### `mr_layer.py` — building Laplacians, computing the penalty

One builder, two neighbour sources, one loss function.

`knn_cosine_torch(emb, k, chunk=2048)` normalises the embeddings, multiplies chunks of 2,048 rows against the full matrix on the GPU, masks each row's self-similarity to −∞, and takes the top-k. Returns cosine *distances* (1 − similarity) and indices — the same quantities scikit-learn reports, in under a second on Gowalla's 70,839 nodes instead of a minute or more, with no trip to the CPU. `knn_cosine_sklearn` is the notebook's routine, kept so the original numbers can be reproduced if anyone asks; the two agree on >99% of neighbours (ties aside). `knn_cosine_sparse` handles the co-occurrence case over sparse rows of `R`.

`gaussian_weights(cos_dist, sigma=None)` converts distances to kernel weights `exp(−d² / 2σ²)` where `d² = 2 · cos_dist` on the unit sphere and σ defaults to the median distance (the *median heuristic*). A flag reproduces Cell 9's guard of taking the median over strictly positive distances only, which matters for the co-occurrence graph where duplicate rows produce many exact zeros.

`laplacian_from_knn(idx, w, n)` assembles `W`, symmetrises it as `(W + Wᵀ)/2`, and returns `L = D − W` as a SciPy matrix; `scipy_to_torch_sparse` moves it to the GPU. The unit tests check that rows sum to zero, that `L` is symmetric, and that the diagonal is non-negative.

`build_embedding_laplacian` (Emb-MR) and `build_cooccurrence_laplacian` (CoOcc-MR, block-diagonal `[L_user, L_item]`) are the two public entry points. `manifold_loss(Z, L)` computes `sum(Z ⊙ (L Z)) / n`, which equals `tr(ZᵀLZ) / n` without forming the dense product — verified against the explicit trace in the tests.

### `losses.py` — the composable objective

**Plain language.** The training objective is a sum of named terms. Each term is a small object that knows how to compute its value and what to log. A conductor object adds up whichever terms are switched on. Adding the VAE means adding one more term to the list, not writing a new training loop.

**In detail.** `BatchContext` carries everything a term might need for one batch: the model, the `(user, pos, neg)` index tensors, the propagated embeddings (computed *once* per batch and shared), the epoch, the current Laplacian, and an `extras` dict for anything Module 3 wants to pass around. Three terms exist:

- `BPRTerm`: `mean(softplus(s_neg − s_pos))` on propagated embeddings.
- `L2EgoTerm(decay)`: `decay · ½(‖e_u‖² + ‖e_p‖² + ‖e_n‖²) / batch` on *ego* embeddings — the reference implementation's regulariser, exactly.
- `ManifoldTerm(lam)`: `lam · manifold_loss(all_emb, L)`, or zero if no Laplacian is set.

`CompositeLoss(terms)` sums the active ones and returns the total plus a flat dict of logged values. `set_active({"bpr", "l2"})` is what the trainer calls in the warm-up phase; `set_active({"bpr", "l2", "manifold"})` when MR switches on. `LossTerm` is a `Protocol`, so any object with a `name` and a `__call__(ctx) → (tensor, dict)` qualifies.

**Why.** The notebook had `BPRLoss`, `BPRLoss_MR` and `BPRLoss_MR_CoOccurrence`, three classes whose `stageOne` methods differed only in whether a manifold term was added, each with its own Adam optimiser, each driven by its own copy of the training loop. Three copies of the same logic is three places for a fix to be forgotten. The audit harness (Part 7) reimplements the notebook's `stageOne` verbatim and confirms the composed loss and its **gradient** match to `0.00e+00`.

### `vae.py` — interface only

Raises `NotImplementedError` on construction. The docstring records the intended shape (a `VAETerm` reading `ctx.all_emb`, returning `recon + β·KL` and logging per-dimension KL and active units) and the collapse defences to wire in from day one. It exists so the attachment point is fixed now and nobody writes a fourth training loop later.

## 3.7 `src/metrics/`

### `ranking.py`

Vectorised over a batch: `recall_at_k`, `precision_at_k`, `ndcg_at_k`, all taking a boolean *hit matrix* (rows = users, columns = ranks) and a per-user count of relevant items. NDCG's ideal list has `min(k, n_rel)` relevant items at the top — the correct definition. The notebook's version normalised by the discounted gain of the hits *found*, so any list whose hits happened to sit at the top scored 1.0 regardless of how many relevant items it missed. `tests/test_metrics.py` carries the notebook's version for contrast; on the audit fixture the fix lowers NDCG@10 by 49% and NDCG@20 by 44%.

### `diversity.py`

`gini_from_counts` and a `DiversityAccumulator` that, fed each batch's top-k item ids, tallies recommendation counts per item and per-user long-tail shares, then reports `gini@k`, `tail_share_user@k` (the notebook's quantity), `tail_catalog_coverage@k` (Eq. 2.18) and `catalog_coverage@k`.

### `geometry.py`

`effective_rank(Z, max_samples=10000, center=True, seed)` subsamples rows with a *seeded* generator, mean-centres, takes singular values, and returns `exp(H(p))`. On a random `(70839, 256)` table it returns 255.16 against the text's 255.2 ± 0.0, which is the sanity check that the implementation matches the paper's convention. `neighborhood_preservation(a, b, k, ...)` is NP between two snapshots; `np_ref(item_emb, R, k, ...)` builds the Jaccard item-item k-NN from the training matrix and compares. All subsampling uses fixed seeds so every arm is measured on the same nodes.

### `evaluate.py` — the evaluator

**Plain language.** Given a way to score every item for a batch of users, this function hides the items each user has already seen, takes the top 50, checks them against the held-out answers, and accumulates every metric.

**In detail.** The signature is `evaluate(scorer, dataset, split, topks, batch_size, device)`. The first argument is a **Scorer**: any callable that maps a tensor of user ids to a dense `(batch, n_items)` score matrix. `dot_product_scorer(all_users, all_items)` is the default and is the notebook's rule `U Iᵀ`. The function never touches the model.

For each batch it: calls the scorer; sets to −∞ the user's *training* positives (on `val`) or training **and** validation positives (on `test`); takes the top `max(topks)` by `torch.topk`; encodes `(row, item)` pairs as integers and uses `np.isin` against the ground-truth pairs to get a hit matrix; computes every metric in `ranking.py` for the overall, short-head and long-tail views; and feeds the diversity accumulator. A user counts toward the short-head average only if they have at least one short-head relevant item (the notebook's convention). Propagation happens once per evaluation, not once per batch.

**Why a scorer, and why exclusion lives here.** The VAE produces scores from a decoder, not a dot product. If scoring were hard-wired, every metric would need reworking in November. With a scorer, Module 3 supplies one closure (`decoder(mu_u)` — the posterior *mean*, per Liang et al. 2018, not a sample) and nothing else changes. Exclusion stays in `evaluate` because it is part of the *protocol*: whatever the scores, known positives must not occupy top-k slots, and on the test split that includes validation positives. Excluding test items during validation would leak test knowledge into selection and is deliberately *not* done — it is why validation Recall sits below test Recall (test positives compete for slots at validation time; validation positives are masked at test time), a gap that is expected and identical across arms.

The refactor from `evaluate(model, …)` to `evaluate(scorer, …)` was verified by running both versions on real Gowalla data: 40 metrics per split, both splits, zero differences.

## 3.8 `src/trainer.py` — the heart

**Plain language.** One class runs every experiment. You tell it the total number of epochs and at which epoch the treatment (if any) switches on. It trains, evaluates on a schedule, remembers the best checkpoint according to the validation set, and at the end reloads that checkpoint to compute the final numbers and write them to disk.

**The lifecycle of a run.**

1. **Construction** reads the config into plain attributes: `E` (total epochs), `W` (warm-up epochs), `lam`, the Laplacian source, `k`, rebuild period, batch size, and the effective λ (`lam / n_batches` if `lambda_scale=per_epoch`, else `lam`). It builds one Adam optimiser over the model, the composite loss with all three terms, the sampler's random generator, and validates the selection settings (refusing `select_split=val` with no validation set; warning loudly if `select_split=test`).

2. **`run()`** either resumes a warm-start checkpoint or activates `{bpr, l2}` and starts at epoch 0. Then, for each epoch:
   - If `epoch == W` and the run was not resumed, **`_on_transition`** fires: it snapshots the ego embeddings as the reference (`ref_ego`), measures ER under both conventions (`er_ref`), records a `reference_snapshot` event, optionally saves the warm-start checkpoint, builds the Laplacian if λ > 0, and activates the manifold term. For `W = 0` this happens before any training, so the reference is the random initialisation — which is exactly what H1 measures against.
   - If `epoch == W` and the run *was* resumed, only the Laplacian build and term activation happen; the reference came from the checkpoint.
   - Otherwise, if Emb-MR is active and `(epoch − W) % rebuild_every == 0`, the Laplacian is rebuilt from current embeddings.
   - **`train_epoch`** samples negatives, shuffles, and for each minibatch propagates once, builds a `BatchContext`, computes the composite loss, and steps the optimiser. Returns the mean of each logged term.
   - On the evaluation schedule (every `eval.every` epochs, plus the last epoch of warm-up and the final epoch), **`evaluate_point`** propagates once, builds the scorer via `self.scorer(au, ai)`, evaluates `val` (and `test` if `test_each_eval`), measures ER, and appends a record to `history`. The record's `val_<select_metric>` is compared to the best so far; on improvement `best.pt` is written. `history.json` is rewritten after every evaluation so a crash leaves a trace.
   - Optional early stopping after `patience_evals` evaluations without improvement — off by default, because it breaks budget matching.

3. **`finalize`** reloads `best.pt`, propagates, evaluates both splits once more (so the reported test number always corresponds to the selected checkpoint even if `test_each_eval` was off), computes ER under both conventions with deltas against `er_ref`, NP against `ref_ego`, and NP_ref against the Jaccard graph, and writes `results.json`.

**Warm-start sharing.** `save_warmstart` writes model, optimiser state, all RNG states, the sampler's generator state, `ref_ego`, `er_ref`, the history so far, and a dataset fingerprint. `load_warmstart` refuses a checkpoint from a different dataset or split or at the wrong epoch, restores all of it, and sets `start_epoch = W`. Two arms resumed from the same file share a bit-identical prefix and the same sequence of negatives thereafter; the *only* difference between them is the treatment.

**`Trainer.scorer(all_users, all_items)`** is a one-line method returning `dot_product_scorer`. It is the single seam Module 3 overrides.

## 3.9 `src/train.py` — the entry point

Registers the `armtag` resolver so run folders are named from hyperparameters, composes the config through Hydra, saves it verbatim to the run folder, seeds, loads the dataset, builds the graph and model, optionally opens a Weights & Biases run, and hands off to `Trainer.run()`. `python -m src.train dataset=gowalla arm=emb_mr seed=2020`.

## 3.10 `analysis/summarize.py`

`collect_results(runs_root)` walks every `results.json` into a pandas table (one row per run) with the test metrics, validation metrics and geometry flattened into columns. `summary_table` groups by dataset and arm tag, reporting mean, standard deviation and n. `paired_wilcoxon(df, arm_a, arm_b, metric)` pairs runs by seed within each dataset, tests `arm_a > arm_b` one-sided, and applies Holm–Bonferroni across datasets. The command line does all three: `python -m analysis.summarize runs --a emb_mr --b mr_off --metric recall@20`.

This replaces the notebook's Drive inventory, checkpoint-discovery and "recover metrics from filenames" machinery entirely: there is nothing to recover when every run writes its own results.

## 3.11 `scripts/`

`run_gate.sh` launches the September gate (two arms × five seeds on Gowalla) and summarises. `run_sweep.sh` does all three datasets. `run_tests.sh` is `pytest tests -q`. All accept extra Hydra overrides as arguments.

## 3.12 `tests/` and `audit_equivalence.py`

Described in Part 7.

---

# Part 4 — The experimental protocol, and why the code is shaped by it

This part explains the *design* the code enforces. The four rules below are the reason the trainer looks the way it does; each corresponds to a specific defect in the notebook's protocol that the June and August 2026 audits identified.

## 4.1 The arms

An **arm** is one experimental condition. The four defined arms:

| arm | warm-up (BPR only) | then, to epoch E | Laplacian | what it is for |
|---|---|---|---|---|
| `baseline` | 0 | BPR | none | from-scratch reference; K-depth sweep; H1 (reference = random init) |
| `mr_off` | W | BPR | none | **the control** — identical to `emb_mr` except λ = 0 |
| `emb_mr` | W | BPR + MR | k-NN on embeddings, rebuilt every 50 | the proposed method (H2, H3) |
| `coocc_mr` | W | BPR + MR | fixed, from `R` | the redundancy check (§6.3) |

The primary comparison is `emb_mr` against `mr_off`. They share `W`, `E`, the reference snapshot, and — when resumed from the same warm-start file — the exact model, optimiser and random state at epoch W. They differ in one config value. Any difference in outcome is attributable to the manifold term.

Under matched budgets, `baseline` for E epochs and `mr_off` for E epochs are numerically the same protocol (both are E epochs of BPR); `mr_off` exists so that the control has the same *phase structure* and the same reference point as the treatment. The August plan described "three arms"; that framing dates from before the budget fix and the first two collapsed into one.

## 4.2 Rule 1 — equal budgets

**The defect.** The notebook's MR run looked for the baseline's *best* checkpoint on disk, loaded it, silently set `warmup_epochs = 0`, and trained 1,000 more epochs with a fresh Adam optimiser. All fifteen Chapter 5 runs took this path. The MR arm therefore had `best_epoch(baseline) + 1000` epochs of training on top of a starting point the baseline had already selected on the test set, while the baseline had at most 1,000. "MR helps" and "MR got more training" were indistinguishable.

**The rule.** Every arm trains for exactly `epochs` (E) epochs from the seed's initialisation. There is no code path that loads a checkpoint and adds epochs. The *only* way to skip the warm-up is `warmstart.load`, which resumes at epoch W and trains to the same E; total budget is E either way.

**What this means for W.** The notebook's realised protocol was "MR acts on a converged model". The corrected protocol lets you keep that — set `W` to a converged point (1,000 on Gowalla) and `E = 2000` for *both* arms — while removing the confound, because the control now also gets W + (E − W). W must be a fixed number, not each seed's best epoch: selecting the hand-off point on test is exactly the contaminated step.

## 4.3 Rule 2 — the MR-off control

**The defect.** Nothing in the notebook continued the baseline checkpoint with BPR only. The MR arm was compared to a baseline trained under a different schedule from a different starting point.

**The rule.** `mr_off` is `emb_mr` with `lambda_manifold = 0`. Same W, same E, same everything. It answers "what would have happened if we had kept training normally instead of switching MR on?" — the counterfactual the comparison needs.

## 4.4 Rule 3 — validation-set selection

**The defect.** The notebook's best-checkpoint logic tracked Recall@20 on the *test* set. Every reported number was optimistically biased by selection.

**The rule.** `split.val_frac` of each user's training items (default 10%) is held out as validation, with a fixed `split_seed` shared by all runs. Checkpoints are selected on `val_recall@20`. Test is scored once at the selected checkpoint (and optionally at every evaluation for curve plotting, never for selection). Validation items are excluded from the adjacency matrix.

**What it costs — measured, not assumed.** Holding everything else fixed
(Gowalla, seed 2020, d = 64, 25 epochs, scored on the same test set), test
Recall@20 was 0.09762 at `val_frac=0`, 0.09582 at 0.05 and 0.09666 at 0.1 —
a spread of about 1%, not monotone in the amount of data removed, i.e. within
run-to-run noise. **The validation split is close to free.** The gap to
Chapter 5's published numbers therefore comes almost entirely from removing
the extra 1,000 epochs the MR arm used to receive, which is the confound being
corrected, not a cost of the new protocol.

The remaining tax can still be measured directly if you want the number for
the text: run `arm=baseline` once with `split.val_frac=0 eval.select_split=test`
(the notebook protocol; the trainer prints a warning) and once with defaults,
same seed.

## 4.5 Rule 4 — one variable at a time

The gate should change only the protocol. Two knobs exist that change the *method* and must stay at their Chapter 5 values for the gate, then be varied as ablations:

- `arm.lambda_scale=none` applies λ every minibatch, so the per-epoch weight is `n_batches · λ` and differs by dataset (≈23 batches on Gowalla, ≈37 Yelp, ≈73 Amazon-Book at batch 32,768). `per_epoch` divides λ by `n_batches` so the per-epoch weight is dataset-invariant. `results.json` records `n_batches_per_epoch` and `lambda_effective` either way.
- `eval.patience_evals=0`. Early stopping would let arms stop at different epochs, breaking Rule 1.

## 4.6 Symmetric checkpoint selection

Epochs before W are the *shared prefix*: two arms branching from the same
warm-start are bit-identical there. A checkpoint drawn from the prefix
therefore says nothing about the treatment. More importantly, a resumed arm
never trained those epochs and cannot select from them — so letting a
standalone arm do so would hand the control strictly more candidates.
`eval.select_from_epoch` (default `auto` = W) excludes the prefix for every
arm. `results.json` records the value used.

Related: the reported numbers come from the checkpoint that maximised
validation Recall@20. Geometry and diversity are not what selection optimises,
so `results.json` also carries `at_last_epoch` — the same metrics at epoch E,
a fixed point identical across arms. Accuracy claims should cite the selected
checkpoint; geometric and diversity claims may cite either, stated explicitly.

## 4.7 RNG hygiene

Negative sampling, evaluation subsampling, ER subsampling and NP subsampling each use a separate seeded generator. Evaluating more or less often cannot change the training trajectory. The subsampling seeds are fixed across arms (`geometry.np_seed`), so every arm's geometry is measured on the same nodes and the paired comparison holds for geometric metrics as well as ranking ones.

---

# Part 5 — Using the code

## 5.1 Installation (Pop!_OS, RTX 5060 Ti)

The GPU is Blackwell (sm_120) and needs a CUDA 12.8+ PyTorch wheel; the default PyPI wheel will not run on it.

```bash
git clone https://github.com/Concius/gcn-mr-vae && cd gcn-mr-vae
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0))"   # should name the 5060 Ti
python -m src.data.download                                       # all three datasets → data/
python -m pytest tests -q                                         # 29 tests, ~10 s, CPU
python audit_equivalence.py                                       # 24/24 vs the notebook
```

If `python` is not found, `sudo apt install python-is-python3`.

## 5.2 Running one experiment

```bash
python -m src.train dataset=gowalla arm=emb_mr seed=2020
```

You will see the resolved config printed in full, the dataset summary (`users=29,858 items=40,981 train=728,432 (of 810,128) val=81,696 test=217,242`), a header line summarising the arm, and then one line per evaluation:

```
ep   10 [warmup] loss=0.1540 | val recall@20=0.0890 | test recall@20=0.1059 | ER_tab=… ER_prop=154.9 *
--- epoch 10: end of warm-up. ER_tab=… ER_prop=154.94 (d=256). MR ON, lambda_eff=1e-05 ---
    kNN Laplacian (n=70,839, k=20, torch) nnz=2,804,167 sigma=0.3791 in 0.9s
ep   20 [mr    ] loss=0.0995 mani=3.578e+01 | val recall@20=0.1042 | test recall@20=0.1171 | ER_prop=144.9 *
```

A trailing `*` marks a new best on the selection metric. The final line reports the selected checkpoint's numbers.

Timing on the 5060 Ti (Gowalla, d = 256): about 1.5 s per BPR epoch, 2.3 s per MR epoch, 2 s per evaluation, 0.9 s per Laplacian rebuild. A 1,000-epoch run is ~28 min (`mr_off`) or ~41 min (`emb_mr`).

## 5.3 Running the gate

The gate is the protocol Tabela 7 specifies — W = 100, E = 1000 — with the
budget confound removed. One command:

```bash
tmux new -s gate
bash scripts/run_gate.sh          # or: SEEDS=2020,2021 bash scripts/run_gate.sh
# Ctrl-b d to detach; tmux attach -t gate to return
```

which is just:

```bash
python -m src.train -m dataset=gowalla arm=mr_off,emb_mr seed=2020,2021,2022,2023,2024
python -m analysis.summarize runs --a emb_mr --b mr_off --metric recall@20
```

**No warm-start files are needed.** With the same seed both arms have the same
initialisation and the same negative-sampling stream, and epochs 0–99 are
BPR-only for both, so the warm-up is bit-identical by construction — verified
on real Gowalla data, not assumed (`test_mr_off_and_emb_mr_share_warmup`).
About 25 min (`mr_off`) plus 37 min (`emb_mr`) per seed, ≈ 5 hours for five
seeds.

**Optional follow-up.** `scripts/run_gate_converged.sh` moves the activation
epoch to 1000 with E = 2000, testing whether MR needs a converged model to act
on. Run it only if the primary gate is null: it costs ~7.3 hours and ~220 MB of
warm-start checkpoints per seed (worth sharing the prefix at that length; not
worth it at W = 100).

## 5.4 Reading the outputs

Each run writes to `runs/<dataset>/<arm_tag>/seed<seed>/`:

| file | contents |
|---|---|
| `config.yaml` | the fully resolved configuration, verbatim |
| `history.json` | a list of records: one per evaluation, plus events (`reference_snapshot`, `laplacian_built`, `early_stop`) |
| `best.pt` | the selected checkpoint (model weights, epoch, the evaluation record that selected it) |
| `results.json` | the final report — see Appendix B |
| `warmstart_ep<W>.pt` | only with `warmstart.save=true` |

`history.json` is what you plot. Filter to `event == "eval"`; each record has `epoch`, `phase`, every `loss_*`, every `val_*` and `test_*` metric, and `er_table` / `er_prop`.

`results.json` is what you tabulate. `test` and `val` are dicts of every metric; `geometry` has both ER conventions, their values at the reference point, the deltas, NP-vs-reference and NP_ref.

## 5.5 Common overrides

```bash
python -m src.train dataset=yelp2018 arm=emb_mr seed=2021 arm.lambda_manifold=1e-4 arm.k_neighbors=10
python -m src.train dataset=gowalla arm=baseline model.n_layers=0                  # MF-BPR
python -m src.train -m dataset=gowalla arm=baseline model.n_layers=2,3,4,5           # depth sweep
python -m src.train -m dataset=gowalla arm=emb_mr arm.lambda_manifold=1e-6,1e-5,1e-4 # λ sweep (distinct folders)
python -m src.train dataset=gowalla arm=emb_mr eval.test_each_eval=false             # halve eval cost
python -m src.train dataset=gowalla arm=emb_mr arm.lambda_scale=per_epoch            # ablation
python -m src.train dataset=gowalla arm=baseline split.val_frac=0 eval.select_split=test  # notebook protocol (warns)
python -m src.train dataset=synthetic arm=emb_mr device=cpu epochs=10 model.dim=16   # smoke run
```

## 5.6 Troubleshooting

*`device=cuda requested but CUDA is not available`* — the PyTorch wheel is not the cu128 build, or the driver is too old. Reinstall from the cu128 index.

*`eval.select_split=val but the dataset has no validation split`* — you set `split.val_frac=0`. Either restore it or set `eval.select_split=test` knowingly.

*`selection metric val_recall@20 not in evaluation record`* — `eval.topks` does not include 20. Add it or change `eval.select_metric`.

*`warm-start checkpoint was produced on a different dataset/split`* — the checkpoint fingerprint does not match; you are mixing runs with different `val_frac` or `split_seed`.

*`scorer returned (b, n) expected (b, n_items)`* — a custom scorer (Module 3) returned the wrong shape.

*Out of memory during evaluation on Amazon-Book* — lower `eval.batch_size` (4096 × 91,599 floats ≈ 1.5 GB per batch).

---

# Part 6 — From the Colab notebook to this repository

## 6.1 What the notebook was

`LightGCN_Manifold_refactored (6).ipynb` — 60 cells, 53 of them code, 260,000 characters — was the entire experimental apparatus for the qualification. It ran on Google Colab against an A100 with Google Drive as the only persistent storage. It produced every number in Chapter 5.

It had grown in the way research notebooks do. The model code (Cells 5, 8, 9) was sound and had been validated against the official LightGCN implementation. Around it had accreted: a 40,000-character subsystem for recovering results from Drive after Colab timeouts (Cells 10–11, 52–55); thirty near-identical cells launching one run each for every combination of arm, dataset and seed (Cells 21–50); a global `config` object mutated by override-and-restore patterns; four functions defined twice in different cells, the later definition silently shadowing the earlier; and three training loops that were copies of each other with one line changed.

None of that was wrong, exactly. It was the shape a tool takes when its author is fighting a 12-hour session limit and a flaky file system while trying to get a qualification document out. But it was also the shape in which two protocol defects could hide for months.

## 6.2 Why migrate rather than adopt a framework

The August 2026 planning discussion weighed moving to RecBole (a mature recommendation framework with LightGCN, VAE-CF and budget-equalised protocols built in) against porting the existing code. The port won for three reasons: the model code was already validated and the risk was in the harness, not the model; the MR layer's full-matrix Laplacian term is awkward to express inside RecBole's per-batch `calculate_loss`; and a port to a clean structure is the same amount of work as learning a framework, but leaves the author owning every line they will have to defend.

## 6.3 What was kept unchanged — and how we know

The scientific content is the same code, moved:

| component | notebook | repository | proof of equivalence |
|---|---|---|---|
| LightGCN propagation | Cell 5 `LightGCN.computer` | `models/lightgcn.py` | matches an explicit re-implementation to 1e-6; `n_layers=0` reduces to ego |
| BPR + L2 loss | Cell 5 `BPRLoss.stageOne` | `models/losses.py` | loss *and full gradient* identical to `0.00e+00` |
| k-NN Laplacian, `L = D − W` | Cell 8 `build_knn_laplacian` | `models/mr_layer.py` | rows sum to zero, symmetric; torch and sklearn backends agree on >99% of neighbours |
| manifold term `tr(ZᵀLZ)/n` | Cell 8 `compute_manifold_loss` | `models/mr_layer.py` | equals explicit trace to 1e-5; gradient identical |
| co-occurrence Laplacian | Cell 9 | `models/mr_layer.py` | block-diagonal, rows sum to zero |
| Recall, Precision, short/long splits | Cell 6 `Test` | `metrics/evaluate.py` | identical to 1e-9 against a verbatim reimplementation of the notebook's loop |
| Gini | Cell 6 | `metrics/diversity.py` | identical |
| Notebook's tail metric | Cell 6 | `tail_share_user@k` | identical |
| Effective Rank | Cell 8 / Block 7–8 | `metrics/geometry.py` | 255.16 on random init vs the text's 255.2 |
| NP, NP_ref | Cell 6 / Block 6 | `metrics/geometry.py` | same algorithm; deterministic subsampling |

The word "proof" is used deliberately. `audit_equivalence.py` contains verbatim reimplementations of the notebook's `stageOne` and `Test()` and runs them against the ported code on identical weights and inputs. Twenty-four checks, all bit-identical except NDCG, which differs *because it was fixed*.

## 6.4 What was fixed

Eight defects from the June and August audits, plus three found in the post-port audit against the qualification text.

**P0 — the re-run is invalid without these**

1. **Warm-start budget confound.** *Notebook:* MR runs loaded the baseline's best checkpoint and trained 1,000 further epochs with a fresh optimiser. *Repository:* no such path exists; every arm trains exactly E epochs; the only resume mechanism carries model, optimiser and RNG and trains to the same E.
2. **No MR-off control.** *Notebook:* the MR arm was compared to a baseline on a different schedule. *Repository:* `mr_off`, identical to `emb_mr` but λ = 0.
3. **Checkpoint selection on test.** *Notebook:* best Recall@20 on the test set. *Repository:* validation split, fixed across seeds, excluded from the adjacency; test scored once at the selected checkpoint.

**P1 — blocks reporting**

4. **NDCG.** *Notebook:* ideal DCG computed over the hits found, so a list whose hits sat at the top scored 1.0 regardless of misses. *Repository:* ideal list is `min(k, |GT|)` relevant items. On the audit fixture: −49% at @10, −44% at @20. Any future NDCG will be far below the notebook's; that is the bug leaving.
5. **Tail metric definition.** *Notebook:* mean per-user long-tail share. *Text (Eq. 2.18):* catalogue coverage of long-tail items. *Repository:* both, as `tail_share_user@k` and `tail_catalog_coverage@k`; the text should cite the latter.

**P2 — consistency**

6. **NP reference.** *Notebook:* baseline measured NP against random init; MR arm against the loaded baseline checkpoint — different references. *Repository:* every arm reports NP against its own epoch-W snapshot (identical for `mr_off`/`emb_mr`) and NP_ref against the external Jaccard graph with a fixed subsample.
7. **λ per batch vs per epoch.** *Notebook:* implicit; the effective per-epoch weight varied 3× across datasets. *Repository:* `arm.lambda_scale` makes it explicit; `results.json` records `n_batches_per_epoch` and `lambda_effective`.
8. **ER mean-centring.** *Notebook:* centred, undocumented. *Repository:* `geometry.er_center: true`, explicit; Eq. 3.2 should say so.

**Post-port audit (against the qualification PDF)**

9. **W = 0 dropped the H1 reference.** A branch in the trainer mistook a fresh `warmup_epochs=0` run for a resumed one and skipped the reference snapshot, so every `baseline` run silently omitted ER-at-init and NP-vs-init — the two quantities H1 is validated by. Fixed; regression-tested.
10. **Only one ER convention.** The text's Table 2 uses ER on the embedding table; the port reported only post-propagation. Both now reported everywhere.
11. **Popularity segmentation depended on `val_frac`.** Computing it on the reduced training set moved 8.27% of the short head. Now computed on the original `train.txt` by default.

Three smaller items were found and left as notes: the notebook's negative-sampling collision loop re-indexed into the wrong array after the first pass, leaving 15 false negatives per epoch on Gowalla out of 810,128 (0.002%; the port leaves 0); `compute_effective_rank` drew an unseeded `torch.randperm`, so evaluation frequency perturbed training; and `torch.cuda.empty_cache()` was called every ten batches, which slows training and fixes nothing.

## 6.5 What was removed

- The Colab/Drive scaffolding: mounting, `SAVE_DIR`, save tags, `tag_exists_in_drive`, history-file discovery by filename.
- The Section 3 recovery subsystem (`parse_save_tag`, `inventory_drive`, `recover_metrics_for_folder`, `find_checkpoint`, `build_canonical_pool`). There is nothing to recover when every run writes its own `results.json`.
- The thirty Bloco C cells. Replaced by `-m` sweeps and two shell scripts.
- The global `config` object and every override/restore pattern.
- `BPRLoss_MR` and `BPRLoss_MR_CoOccurrence` as separate classes, and their two duplicate training loops.
- The duplicate definitions of `compute_neighborhood_preservation`, `compute_manifold_loss`, `compute_effective_rank`, `compute_np_ref`.
- The ml-1m sequential loader and `create_sample_data`; the never-enabled dropout knobs; the sigmoid on scores (monotone, no effect on any ranking).
- `LightGCN.users_rating` — dead code and a second scoring path.

## 6.6 What was added

- Validation split (`data/splits.py`) and the exclusion logic that goes with it.
- The `mr_off` and `coocc_mr` arms as configs.
- Warm-start save/load with full RNG capture, and the `resumed` distinction.
- A GPU k-NN backend.
- Vectorised evaluation (hit matrix by integer-key `isin` instead of per-user loops).
- Both ER conventions with deltas; NP_ref inside every run.
- Per-run folders with config snapshot, history and results; hyperparameter-derived folder names.
- `analysis/summarize.py` with Holm–Bonferroni.
- The scorer seam for Module 3.
- Twenty-nine automated tests and the equivalence harness.
- `synthetic.py` for fast iteration.

## 6.7 What this means for comparing with Chapter 5

The numbers will move. Expect: lower absolute Recall (90% of edges; no selection on test), much lower NDCG (the fix), slightly different Gini and tail numbers only if `popularity_from` is changed from its default. Within the new runs all arms share these conditions and comparisons are valid. The cross-reference to the old tables is what shifts, and the protocol tax can be measured as described in §4.4.

---

# Part 7 — Verification

## 7.1 The test suite

`python -m pytest tests -q` — 29 tests, about 10 seconds on CPU.

| file | what it pins down |
|---|---|
| `test_metrics.py` | NDCG fix (perfect list = 1; missed items penalised; ideal capped at k; vectorised = scalar); recall, precision; Gini; the two tail definitions differ |
| `test_laplacian.py` | torch and sklearn k-NN agree; `L` symmetric, rows sum to zero, diagonal ≥ 0; manifold loss = explicit trace; co-occurrence block-diagonal; kernel weights in (0, 1] |
| `test_split.py` | validation split deterministic, disjoint from train; users below `min_train` untouched; `val_frac=0` = no split |
| `test_smoke.py` | every arm runs end-to-end; `mr_off` and `emb_mr` produce identical warm-up curves and identical ER at the transition; warm-start save→load resumes with the correct budget; selection on test only when explicitly asked; **W = 0 captures the initial reference** (regression for defect 9); both ER conventions reported; popularity segmentation independent of `val_frac` |
| `test_scorer.py` | default scorer = inline dot product; a custom scorer is honoured; train positives masked on val; val positives masked on test; shape validated |

## 7.2 The equivalence harness

`python audit_equivalence.py` builds a synthetic dataset, a model, and a Laplacian, then runs the notebook's own `stageOne` and `Test()` (reimplemented verbatim in the script) against the ported loss and evaluator on identical inputs. It prints one line per check. Expected output ends with `24 checks passed, 0 drifted` and two `[FIX ]` lines showing NDCG diverging as intended. Run it after any change to `losses.py`, `mr_layer.py`, `evaluate.py`, `ranking.py` or `diversity.py`.

## 7.3 What the tests cannot tell you

They run on CPU against a toy dataset. They cannot detect a CUDA-specific numerical issue, an out-of-memory condition on Amazon-Book, or a wrong per-epoch time estimate. The 30-epoch Gowalla run on the actual GPU is the check for those, and it reproduced the official dataset statistics exactly (29,858 / 40,981 / 810,128 / 217,242).

---

# Part 8 — What comes next

`VAE_ROADMAP.md` is the authoritative plan. In brief: the scorer seam (this document, §3.7) is done; the remaining blockers for Module 3 are a three-phase schedule in the trainer (§4.2.3 of the text: warm-up → MR → VAE 20 epochs + 10 fine-tuning at reduced LR) and optimiser parameter groups so the encoder/decoder can be attached mid-run without resetting Adam's moments. Build the VAE against `synthetic.py` first, with per-dimension KL and active-unit logging from the first epoch, because H4 is *defined* by KL divergence being higher with the MR layer than without.

---

# Part 9 — Glossary

**Adam** — the optimisation algorithm that adjusts embeddings after each batch; keeps running averages ("moments") of gradients, which is why a fresh optimiser mid-run is not the same as continuing.
**Arm** — one experimental condition, defined by a config file in `configs/arm/`.
**Batch / minibatch** — a slice of training triples processed together; 32,768 here.
**BPR** — Bayesian Personalised Ranking; the loss that teaches the model to score a user's positives above random negatives.
**Budget** — total training epochs. Equal across arms by rule.
**Checkpoint** — a saved copy of the model weights at some epoch.
**Co-occurrence graph** — users linked if they share items, items linked if they share users; built from `R` alone.
**Dirichlet energy** — `tr(ZᵀLZ)`; small when connected embeddings are similar.
**Ego embeddings** — the raw embedding tables before propagation.
**Effective Rank (ER)** — how many dimensions a point cloud effectively spans; `exp(entropy of normalised singular values)`.
**Epoch** — one pass over all training triples.
**Gate** — the corrected Gowalla comparison that decides whether the MR effect survives the fixed protocol.
**Gini** — concentration of recommendations across items; lower = more diverse.
**Hydra** — the configuration system; composes YAML files and command-line overrides.
**Implicit feedback** — interactions without ratings.
**k-NN graph** — each point connected to its k most similar others.
**Laplacian** — `L = D − W`; the matrix that turns a graph into a smoothness penalty.
**λ (lambda)** — weight of the manifold term; 10⁻⁵.
**LightGCN** — the graph-convolutional encoder; propagates embeddings over the user–item graph.
**Long tail / short head** — the 80% least / 20% most popular items.
**Manifold regularisation (MR)** — the penalty that keeps neighbouring embeddings close.
**Median heuristic** — setting the Gaussian kernel width to the median pairwise distance.
**Negative** — an item a user did not interact with, drawn at random for BPR.
**NP, NP_ref** — Neighbourhood Preservation between two snapshots; against an external Jaccard reference.
**Phase** — a stretch of epochs with a fixed set of active loss terms: `warmup`, `mr` or `bpr`.
**Propagated embeddings** — the output of `computer()`; what the model scores with.
**Reference snapshot** — the ego embeddings captured at epoch W, against which NP-vs-reference is measured.
**Run folder** — `runs/<dataset>/<arm_tag>/seed<seed>/`.
**Scorer** — a callable mapping user ids to a score for every item; the seam Module 3 replaces.
**Seed** — the number that fixes all randomness for a run; five are used.
**Split** — train / validation / test partition of the interactions.
**Warm-start** — the phase of BPR-only training before the treatment switches on; also the checkpoint saved at its end.
**Wilcoxon signed-rank test** — the paired non-parametric test used to compare arms across seeds.

---

# Appendix A — Configuration reference

Every key in `configs/config.yaml`, its default, and what it does.

| key | default | meaning |
|---|---|---|
| `seed` | 2020 | run seed: initialisation and negative sampling |
| `epochs` | 1000 | total budget E for every arm |
| `device` | cuda | `cuda` / `cpu` / `auto` |
| `deterministic` | false | cuDNN deterministic mode (slower; sparse matmul stays non-deterministic on CUDA) |
| `model.dim` | 256 | embedding dimension d |
| `model.n_layers` | 3 | propagation depth K; 0 = MF-BPR |
| `model.init_std` | 0.1 | std of the Gaussian initialisation |
| `optim.lr` | 1e-3 | Adam learning rate |
| `optim.decay` | 1e-4 | L2 weight on ego embeddings |
| `optim.batch_size` | 32768 | training triples per step |
| `split.val_frac` | 0.1 | fraction of each user's training items held out as validation; 0 = none |
| `split.split_seed` | 2020 | seed of the validation split; fixed across runs |
| `split.min_train` | 2 | users with fewer training items are not split |
| `split.short_head_frac` | 0.2 | top fraction of items by popularity = short head |
| `split.popularity_from` | full_train | `full_train` (original `train.txt`, Chapter 5 convention) or `train` (reduced) |
| `eval.topks` | [10, 20, 50] | cut-offs; must include the one in `select_metric` |
| `eval.batch_size` | 4096 | users scored per batch (memory: batch × n_items × 4 bytes) |
| `eval.every` | 10 | evaluate every N epochs, plus end of warm-up and last epoch |
| `eval.test_each_eval` | true | also score test at every evaluation (for curves; never for selection) |
| `eval.select_split` | val | `val` or `test` (the latter warns; notebook protocol) |
| `eval.select_metric` | recall@20 | metric that picks the best checkpoint |
| `eval.patience_evals` | 0 | early stopping after N evaluations without improvement; 0 = off |
| `eval.select_from_epoch` | auto | first epoch eligible for selection; `auto` = W (excludes the shared prefix) |
| `warmstart.save` | false | write `warmstart_ep<W>.pt` at epoch W |
| `warmstart.load` | null | path to resume from; skips epochs [0, W) |
| `geometry.er_each_eval` | true | measure ER at every evaluation |
| `geometry.er_center` | true | mean-centre rows before the SVD |
| `geometry.er_max_samples` | 10000 | rows subsampled for ER |
| `geometry.np_max_samples` | 5000 | nodes subsampled for NP / NP_ref |
| `geometry.np_seed` | 42 | subsampling seed, fixed across arms |
| `geometry.np_k` | [10, 20] | k for NP vs reference snapshot |
| `geometry.np_ref_k` | [10, 20] | k for NP_ref vs Jaccard graph |
| `logging.wandb` | false | enable Weights & Biases (if installed) |
| `logging.project` / `entity` | gcn-mr-vae / null | W&B project / entity |
| `paths.data` | data | dataset root |
| `paths.runs` | runs | run-folder root |
| `paths.arm_tag` | computed | `<name>_w<W>[_lam<λ>_k<k>[_pe]]` |
| `paths.run_dir` | computed | `runs/<dataset>/<arm_tag>/seed<seed>` |

Arm keys (`configs/arm/*.yaml`) are listed in §3.2.

## Appendix A.2 — Hyperparameters vs. the notebook and Tabela 7

Everything the method depends on carries over unchanged. The four deliberate
differences are protocol, not method, and each is justified in Part 4.

| parameter | notebook / Tabela 7 | repository | status |
|---|---|---|---|
| embedding dim d | 256 | 256 | same |
| propagation layers K | 3 | 3 | same |
| initialisation std | 0.1 | 0.1 | same |
| optimiser | Adam | Adam | same |
| learning rate | 1e-3 | 1e-3 | same |
| L2 decay | 1e-4 | 1e-4 | same |
| BPR batch size | 32,768 | 32,768 | same |
| training horizon | 1,000 | 1,000 | same |
| MR activation epoch (W) | 100 | 100 | same |
| k-NN neighbours k | 20 | 20 | same |
| k-NN rebuild period | 50 | 50 (`emb_mr`) | same |
| CoOcc rebuild | never (fixed graph) | `rebuild_every: 0` | same |
| λ Emb-MR | 1e-5 | 1e-5 | same |
| **λ CoOcc-MR** | **1e-4** | **1e-4** | same — *note it differs from Emb-MR's; see the comment in `coocc_mr.yaml`* |
| seeds | 2020–2024 | 2020–2024 | same |
| top-K cut-offs | [10, 20, 50] | [10, 20, 50] | same |
| σ (kernel width) | median heuristic | median heuristic (`sigma: null`) | same |
| eval user batch | 12,288 | 4,096 | **differs — memory only**, sized for a 16 GB GPU rather than a 40 GB A100; no effect on any number |
| evaluation schedule | every 10 to epoch 100, then every 50 | every 10 throughout | **differs — protocol**; finer selection granularity, identical across arms |
| early-stop patience | 150 (≈15 evaluations) | 0 (disabled) | **differs — protocol**; early stopping would let arms stop at different epochs, breaking equal budgets (§4.2) |
| checkpoint selection | test set | validation set | **differs — protocol** (§4.4) |

The gate additionally runs with `W = 1000, E = 2000` rather than `W = 100,
E = 1000`. That is a deliberate choice, not a drift: it preserves the
notebook's *realised* condition — MR acting on a converged model — while
giving the control the same total budget. Both values are available; W is a
config key.

# Appendix B — `results.json` schema

```
dataset               "gowalla"
arm                   "emb_mr"
arm_tag               "emb_mr_w1000_lam1e-05_k20"
seed                  2020
best_epoch            1-based epoch of the selected checkpoint
selected_on           "val:recall@20"
epochs_budget         E
warmup_epochs         W
lambda_manifold       λ as configured
lambda_effective      λ actually applied per batch
n_batches_per_epoch   ceil(n_train / batch_size)
val                   { recall@K, precision@K, ndcg@K, *_short@K, *_long@K,
                        gini@K, tail_share_user@K, tail_catalog_coverage@K,
                        catalog_coverage@K, n_users_eval }   for each K
test                  same keys, on the test split
geometry              er_table, er_prop, er_table_pct, er_prop_pct,
                      er_table_at_ref, er_prop_at_ref,
                      delta_er_table, delta_er_prop, delta_er_table_pct, delta_er_prop_pct,
                      effective_rank (alias of er_prop),
                      np_vs_ref@k for k in np_k, np_ref@k for k in np_ref_k
select_from_epoch     first epoch eligible for checkpoint selection
at_last_epoch         val_*/test_*/er_* at epoch E — a selection-independent reading
train_seconds         wall-clock of the training loop
dataset_summary       sizes, split parameters, popularity_from
config                the full resolved config
torch                 version string
device                "cuda (NVIDIA GeForce RTX 5060 Ti)"
```

Mapping to hypotheses: H1 → `er_table_at_ref`, `delta_er_table`, `np_vs_ref@k` (for `baseline`); H2 → `er_table`, `np_ref@k`, `test.recall@20`; H3 → `test.gini@20`, `test.tail_catalog_coverage@20`.

# Appendix C — `history.json` schema

A list. Evaluation records have `event: "eval"` and:

```
epoch          1-based
phase          "warmup" | "mr" | "bpr"
loss_total, loss_bpr, loss_l2, loss_manifold   epoch means
val_<metric>   every metric on the validation split
test_<metric>  every metric on test (if test_each_eval)
er_table, er_prop   (if er_each_eval)
elapsed_s      seconds since training started
```

Event records have `event` ∈ {`reference_snapshot` (with `er_table`, `er_prop`), `laplacian_built` (with `source`), `early_stop`} and an `epoch`.

# Appendix D — Command cheat-sheet

```bash
# setup
python -m src.data.download [gowalla|yelp2018|amazon-book]
python -m pytest tests -q
python audit_equivalence.py

# one run
python -m src.train dataset=<ds> arm=<arm> seed=<s> [overrides]

# sweeps (each combination gets its own folder)
python -m src.train -m dataset=gowalla arm=mr_off,emb_mr seed=2020,2021,2022,2023,2024

# the gate (see §5.3)
bash scripts/run_gate.sh

# shared warm-up
... arm=mr_off warmstart.save=true epochs=2000 arm.warmup_epochs=1000
... arm=emb_mr 'warmstart.load=runs/<ds>/mr_off_w1000/seed${seed}/warmstart_ep1000.pt' epochs=2000 arm.warmup_epochs=1000

# results
python -m analysis.summarize runs [--a emb_mr --b mr_off --metric recall@20]

# long runs
tmux new -s <name>   # launch, Ctrl-b d, later: tmux attach -t <name>
```

---

*Maintained alongside the code. When behaviour changes, change this file in the same commit.*
