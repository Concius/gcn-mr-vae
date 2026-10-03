#!/usr/bin/env bash
# OPTIONAL follow-up, only if the primary gate comes back null.
# Tests whether MR needs a *converged* model to act on, by moving the activation
# epoch from 100 to 1000 while keeping budgets equal (E=2000 for both arms).
# Here the shared prefix is 1000 epochs, so it is worth training once and
# sharing it: warmstart.save on the control, warmstart.load on the MR arm.
# ~7.3 h for 5 seeds, and ~220 MB per warm-start file.
# NOTE: the warm-start path below embeds the control's arm_tag, which encodes
# K and d. If you change model.n_layers or model.dim, update it to match
# (run the control first and copy the directory name it prints).
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-2020,2021,2022,2023,2024}
DATASET=${DATASET:-gowalla}
# Own folders: runs/<ds>/mr_off_w1000_K3_d256 already holds the notebook-protocol
# replication on Gowalla (different protocol), which the clobber guard protects.
OFF="runs/$DATASET/mr_off_converged_w1000"
MR="runs/$DATASET/emb_mr_converged_w1000"
python -m src.train -m dataset=$DATASET arm=mr_off seed=$SEEDS \
  epochs=2000 arm.warmup_epochs=1000 warmstart.save=true \
  "paths.run_dir=$OFF/seed\${seed}" "$@"
python -m src.train -m dataset=$DATASET arm=emb_mr seed=$SEEDS \
  epochs=2000 arm.warmup_epochs=1000 \
  "warmstart.load=$OFF/seed\${seed}/warmstart_ep1000.pt" \
  "paths.run_dir=$MR/seed\${seed}" "$@"
python -m analysis.summarize runs --dataset "$DATASET" \
  --a emb_mr_converged_w1000 --b mr_off_converged_w1000 --metric recall@20
