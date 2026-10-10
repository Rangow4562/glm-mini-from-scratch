#!/usr/bin/env bash
# Full recursive self-improvement run: pretrain (skipped if present) then RSI ladder.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-runs/self-improve-001}
CHECKPOINT=${2:-runs/pretrain-001/checkpoint-0100}
if [ ! -d "$CHECKPOINT" ]; then
  echo "checkpoint $CHECKPOINT missing; pretraining first" >&2
  ./run_gpu.sh scripts/train_pretrain.py \
    --output runs/pretrain-001 --steps 150 --checkpoints 50,100,150 \
    --batch-size 16 --sequence-length 128 --seed 42
fi
./run_gpu.sh scripts/train_self_improve.py \
  --initial-checkpoint "$CHECKPOINT" \
  --output "$RUN" \
  --cycles 3 --proposals-per-task 16 --temperature 0.8 \
  --fine-tune-steps 24 --fine-tune-batch-size 8 --learning-rate 1e-4 \
  --families-ladder increment,double --per-family 4 --eval-per-family 2 --seed 73
