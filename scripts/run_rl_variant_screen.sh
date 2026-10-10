#!/usr/bin/env bash
# RL design-choice screen: every arm gets the same completion budget on the same
# frozen dev tasks; only one design choice changes at a time. Then a paired
# exact McNemar analysis with Holm correction ranks the arms.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-runs/rl-screen}
CKPT=${2:-runs/pretrain-001/checkpoint-0100}
FAMILIES=increment,double,square,even,reverse
BUDGET=128
train_and_eval () {
  name=$1; groups=$2; gsize=$3; temp=$4; reward=$5
  echo "== arm $name (groups=$groups group=$gsize temp=$temp reward=$reward)"
  ./run_gpu.sh scripts/train_rl.py \
    --initial-checkpoint "$CKPT" --output "$OUT/$name" \
    --groups "$groups" --checkpoints "$groups" \
    --group-size "$gsize" --temperature "$temp" --reward-mode "$reward" \
    --families "$FAMILIES" --tasks-per-family 4 --seed 31415
  ./run_gpu.sh scripts/evaluate.py \
    --checkpoint "$OUT/$name/checkpoint-$(printf '%04d' "$groups")" \
    --split dev --per-family 4 --families "$FAMILIES" \
    --output "$OUT/$name/dev-eval.json"
}
train_and_eval baseline       16  8 0.35 binary
train_and_eval partial-reward 16  8 0.35 case-fraction
train_and_eval temp-020       32  4 0.20 binary
train_and_eval temp-080       16  8 0.80 binary
train_and_eval group-16        8 16 0.35 binary
./run_gpu.sh scripts/analyze_rl_variants.py --screen "$OUT" --output "$OUT/analysis.json"
