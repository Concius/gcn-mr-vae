#!/usr/bin/env bash
# The September gate (Aug 2 plan): Gowalla, budget-matched, validation-selected.
# Two arms that differ in exactly one config value (arm.lambda_manifold):
#   mr_off : 100 BPR epochs, then 900 BPR epochs   (control)
#   emb_mr : 100 BPR epochs, then 900 BPR+MR epochs
# Run inside tmux:  tmux new -s gate; bash scripts/run_gate.sh; Ctrl-b d
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-2020,2021,2022,2023,2024}
DATASET=${DATASET:-gowalla}
python -m src.train -m dataset=$DATASET arm=mr_off,emb_mr seed=$SEEDS "$@"
python -m analysis.summarize runs --a emb_mr --b mr_off --metric recall@20
