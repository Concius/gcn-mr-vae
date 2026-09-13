#!/usr/bin/env bash
# The corrected gate: the protocol Tabela 7 actually specifies (W=100, E=1000),
# with the budget confound removed. Two arms differing in ONE config value.
#
#   mr_off : 100 BPR epochs, then 900 BPR epochs   (control)
#   emb_mr : 100 BPR epochs, then 900 BPR+MR epochs
#
# No warm-start files are needed. With the same seed both arms have the same
# initialisation and the same negative-sampling stream, and epochs 0-99 are
# BPR-only for both, so the warm-up is bit-identical by construction. This is
# verified on real data by tests/test_smoke.py::test_mr_off_and_emb_mr_share_warmup.
#
# ~25 min (mr_off) + ~37 min (emb_mr) per seed on an RTX 5060 Ti => ~5 h for 5 seeds.
# Run inside tmux:  tmux new -s gate; bash scripts/run_gate.sh; Ctrl-b d
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-2020,2021,2022,2023,2024}
DATASET=${DATASET:-gowalla}
python -m src.train -m dataset=$DATASET arm=mr_off,emb_mr seed=$SEEDS "$@"
python -m analysis.summarize runs --a emb_mr --b mr_off --metric recall@20
