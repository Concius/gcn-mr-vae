# gcn-mr-vae

LightGCN + Manifold Regularisation (+ VAE, in progress) for collaborative
filtering. Code for the MSc dissertation *GCN-MR-VAE* (UNESP / FAPESP
2025/07727-0). Ported from the Colab notebook `LightGCN_Manifold_refactored`
with the protocol fixes from the June/August 2026 code audits applied. See
[`MIGRATION.md`](MIGRATION.md) for the cell-by-cell map and the fix log.

## Install (local, RTX 5060 Ti / Pop!_OS)

The 5060 Ti is Blackwell (sm_120); it needs a CUDA 12.8+ PyTorch wheel.

```bash
git clone https://github.com/Concius/gcn-mr-vae && cd gcn-mr-vae
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.randn(2,2,device='cuda')@torch.randn(2,2,device='cuda'))"
python -m src.data.download            # gowalla, yelp2018, amazon-book -> data/
python -m pytest tests -q              # ~10 s, CPU
```

## Run

```bash
# one run
python -m src.train dataset=gowalla arm=emb_mr seed=2020

# the September gate: control vs MR, 5 seeds, budget-matched, val-selected
bash scripts/run_gate.sh               # or: SEEDS=2020,2021 bash scripts/run_gate.sh

# anything is overridable from the command line
python -m src.train dataset=yelp2018 arm=emb_mr seed=2021 arm.lambda_manifold=1e-4 arm.k_neighbors=10
python -m src.train dataset=gowalla arm=baseline model.n_layers=0     # MF-BPR
python -m src.train -m dataset=gowalla arm=emb_mr seed=2020 arm.lambda_manifold=1e-6,1e-5,1e-4   # sweep

# results
python -m analysis.summarize runs --a emb_mr --b mr_off --metric recall@20
```

Long runs: `tmux new -s gate`, launch, `Ctrl-b d`, come back later.
Each run writes `runs/<dataset>/<arm>/seed<seed>/{config.yaml, history.json, best.pt, results.json}`.

## Arms

| arm        | warm-up (BPR only) | then                | Laplacian              |
|------------|--------------------|---------------------|------------------------|
| `baseline` | 0                  | BPR to epoch E      | none                   |
| `mr_off`   | 100                | BPR to epoch E      | none (control)         |
| `emb_mr`   | 100                | BPR + MR to epoch E | k-NN on embeddings, rebuilt every 50 |
| `coocc_mr` | 100                | BPR + MR to epoch E | fixed, from R (GRALS-style) |

Every arm trains for exactly `epochs` (default 1000). `mr_off` and `emb_mr`
differ in one config value (`arm.lambda_manifold`); their first 100 epochs are
the same run. That is the single-variable isolation of the MR effect.

Checkpoints are selected on a validation split carved from `train.txt`
(`split.val_frac=0.1`, fixed `split.split_seed`); the test set is scored once
at the selected checkpoint. Validation items are excluded from the adjacency.

## Protocol knobs you may want to change (and what they do)

* `arm.lambda_scale=per_epoch` divides λ by the number of minibatches so the
  per-epoch manifold weight is the same on every dataset. Default `none`
  reproduces Chapter 5. Run the gate with `none`; run `per_epoch` as an
  ablation afterwards.
* `eval.test_each_eval=false` halves evaluation time (test only at the end).
* `warmstart.save=true` on the `mr_off` run writes `warmstart_ep100.pt`
  (model + optimizer + RNG); `warmstart.load=<path>` on `emb_mr` resumes from it
  so both arms share a bit-identical prefix and the warm-up is trained once.
* `eval.select_split=test` reproduces the notebook's selection-on-test. It
  prints a warning. Do not report numbers produced this way.

## Layout

```
configs/       Hydra: config.yaml, dataset/*.yaml, arm/*.yaml
src/
  data/        dataset.py (loader + val split + graph), splits.py, download.py, synthetic.py
  models/      lightgcn.py, mr_layer.py (Laplacians, manifold loss), losses.py (composable), vae.py (interface, TODO)
  metrics/     ranking.py (NDCG fixed), diversity.py, geometry.py (ER, NP, NP_ref), evaluate.py
  sampling.py  BPR negative sampling
  trainer.py   one trainer for all arms
  train.py     entry point
analysis/      summarize.py: collect results, paired Wilcoxon (Holm across datasets)
scripts/       run_gate.sh, run_sweep.sh, run_tests.sh
tests/         metrics, Laplacian, split, end-to-end smoke on a synthetic dataset
notebooks/     plots only
```
