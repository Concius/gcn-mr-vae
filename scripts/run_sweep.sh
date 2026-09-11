#!/usr/bin/env bash
# Full corrected sweep (October track): 3 datasets x arms x 10 seeds.
# Amazon-Book last; consider eval.test_each_eval=false there to halve eval time.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-2020,2021,2022,2023,2024,2025,2026,2027,2028,2029}
ARMS=${ARMS:-mr_off,emb_mr}
for ds in gowalla yelp2018 amazon-book; do
  python -m src.train -m dataset=$ds arm=$ARMS seed=$SEEDS "$@"
done
python -m analysis.summarize runs
