#!/usr/bin/env python3
"""Recursive self-improvement: the model trains on its own verified programs."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm_mini import ByteTokenizer, GLMMiniFromScratch
from glm_mini.runtime import load_checkpoint, resolve_device
from glm_mini.self_improve import SelfImproveConfig, run_self_improvement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--proposals-per-task", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--fine-tune-steps", type=int, default=24)
    parser.add_argument("--fine-tune-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-pass-rate", type=float, default=1.0)
    parser.add_argument("--families-ladder", default="increment,double,even")
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--eval-per-family", type=int, default=2)
    parser.add_argument("--expand-on-master", action="store_true",
                       help="unlock the next family rung only after each cycle")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ladder = tuple(value.strip() for value in args.families_ladder.split(",") if value.strip())
    if not ladder:
        raise ValueError("families ladder must not be empty")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    tokenizer = ByteTokenizer()
    model = load_checkpoint(args.initial_checkpoint, device)
    config = SelfImproveConfig(
        cycles=args.cycles,
        proposals_per_task=args.proposals_per_task,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        fine_tune_steps=args.fine_tune_steps,
        fine_tune_batch_size=args.fine_tune_batch_size,
        learning_rate=args.learning_rate,
        target_pass_rate=args.target_pass_rate,
        families_ladder=ladder,
        per_family=args.per_family,
        eval_per_family=args.eval_per_family,
        expand_on_master=args.expand_on_master,
        seed=args.seed,
    )
    started = time.perf_counter()
    receipt = run_self_improvement(model, tokenizer, config, args.output, device)
    final = receipt["cycles"][-1]
    print(json.dumps({
        "status": receipt["status"],
        "cycles": len(receipt["cycles"]),
        "verified_memory": receipt["verified_memory_final_size"],
        "final_pass_rate": final["pass_rate_after"],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "wall_seconds": round(time.perf_counter() - started, 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
