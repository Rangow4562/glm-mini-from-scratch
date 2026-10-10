#!/usr/bin/env bash
# Research suite: data diversity, ordering/curriculum, vision comparison, full-model pilot.
# Results land under artifacts/experiments and artifacts/receipts.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${1:-11,22}
STEPS=${2:-60}

echo "== data diversity"
./run_gpu.sh experiments/pretraining_data_diversity.py \
  --output artifacts/experiments/pretraining-data-diversity \
  --steps "$STEPS" --batch-size 16 --seeds "$SEEDS"
./run_gpu.sh experiments/analyze_pretraining_data_diversity.py \
  --results artifacts/experiments/pretraining-data-diversity/results.json \
  --output artifacts/experiments/pretraining-data-diversity/analysis.json

echo "== ordering and curriculum"
./run_gpu.sh experiments/pretraining_curriculum_order.py \
  --output artifacts/experiments/pretraining-ordering \
  --steps $((STEPS * 2)) --batch-size 16 --seeds "$SEEDS"
./run_gpu.sh experiments/analyze_pretraining_curriculum_order.py \
  --results artifacts/experiments/pretraining-ordering/results.json \
  --output artifacts/experiments/pretraining-ordering/analysis.json

echo "== vision architecture comparison"
./run_gpu.sh scripts/train_vision.py --architecture both --device auto --seed 42 \
  --steps 120 --batch-size 20 --eval-examples 200 --learning-rate 0.002 \
  --output artifacts/receipts/vision-miniature-rgb.json

echo "== done"
