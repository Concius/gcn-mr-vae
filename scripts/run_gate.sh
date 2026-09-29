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
# Measured on Gowalla, RTX 5060 Ti: ~1.76 s/epoch (mr_off), ~2.3 s/epoch (emb_mr),
# about 5.5 h for the full gate (5 seeds x 2 arms).
#
# Relative cost per training epoch, measured on CPU at d=64 (every batch
# re-propagates the whole graph, so cost ~ batches x edges):
#   gowalla 1.0x   yelp2018 1.95x   amazon-book 8.6x
# GPU scaling may be somewhat more favourable than CPU for the larger graphs,
# but this is unmeasured. Time the first ~20 epochs before walking away.
#
# The dataset must already be downloaded -- training does NOT fetch it:
#   python -m src.data.download yelp2018 amazon-book
#
# Usage:  DATASET=yelp2018 bash scripts/run_gate.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-2020,2021,2022,2023,2024}
DATASET=${DATASET:-gowalla}
if [ ! -f "data/$DATASET/train.txt" ] || [ ! -f "data/$DATASET/test.txt" ]; then
  echo "ERROR: data/$DATASET/{train,test}.txt not found."
  echo "       Run: python -m src.data.download $DATASET"
  exit 1
fi
python -m src.train -m dataset=$DATASET arm=mr_off,emb_mr seed=$SEEDS "$@"
# Summarise only the dataset just trained. Bare arm names are unambiguous there
# as long as it holds only the gate runs; if it also holds sweeps, this prints a
# clear "ambiguous" message listing the exact tags to pass instead.
python -m analysis.summarize runs --dataset "$DATASET" --a emb_mr --b mr_off --metric recall@20
