#!/usr/bin/env python3
"""Matched miniature study: example ordering and curriculum on identical compute.

Three arms per seed:

- ``interleaved``: every batch draws from all 88 structures.
- ``blocked``: the exact same examples, sorted so one structure trains at a time.
- ``curriculum``: the first third of examples uses only 8 structures, the rest all 88.

All arms train on the same number of examples with the same model and optimizer;
only ordering or early composition changes. Evaluation uses 32 held-out structures.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm_mini import ByteTokenizer, GLMMiniFromScratch, ModelConfig
from experiments.pretraining_data_diversity import OPS, PHRASES, all_structures, evaluate, render_example


def fixed_examples(pool: list[tuple[str, ...]], count: int, *, seed: int) -> list[tuple[str, tuple[str, ...]]]:
    return [render_example(pool[index % len(pool)], index=index, seed=seed) for index in range(count)]


def blocked_variant(examples: list[tuple[str, tuple[str, ...]]]) -> list[str]:
    return [text for text, _ in sorted(examples, key=lambda item: item[1])]


def interleaved_variant(examples: list[tuple[str, tuple[str, ...]]]) -> list[str]:
    return [text for text, _ in examples]


def curriculum_variant(examples: list[tuple[str, tuple[str, ...]]], pool: list[tuple[str, ...]], *, seed: int) -> list[str]:
    warmup = len(examples) // 3
    warmup_pool = pool[:8]
    warmup_texts = [
        render_example(warmup_pool[index % len(warmup_pool)], index=500_000 + index, seed=seed)[0]
        for index in range(warmup)
    ]
    rest_texts = [text for text, _ in examples][warmup:]
    return warmup_texts + rest_texts


def make_batch(tokenizer: ByteTokenizer, texts: list[str], *, step: int, batch_size: int,
               sequence_length: int, device: torch.device):
    rows = []
    for offset in range(batch_size):
        text = texts[(step * batch_size + offset) % len(texts)]
        ids = tokenizer.encode(text, bos=True, eos=True)
        if len(ids) > sequence_length + 1:
            raise ValueError(f"example is {len(ids)} tokens, limit is {sequence_length + 1}")
        ids += [tokenizer.pad_id] * (sequence_length + 1 - len(ids))
        rows.append(ids)
    values = torch.tensor(rows, dtype=torch.long, device=device)
    labels = values[:, 1:].clone()
    labels[labels == tokenizer.pad_id] = -100
    return values[:, :-1], labels, int((labels != -100).sum())


def train_arm(*, texts: list[str], seed: int, steps: int, batch_size: int, learning_rate: float,
              sequence_length: int, device: torch.device, eval_pool: list) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    config = ModelConfig(
        dim=48, layers=3, heads=3, expert_hidden=96, experts=4, top_k=1, streams=2,
        sparse_window=24, sparse_stride=24, max_sequence_length=sequence_length,
    )
    model = GLMMiniFromScratch(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    tokenizer = ByteTokenizer()
    curve, tokens_seen = [], 0
    started = time.perf_counter()
    before = evaluate(model, tokenizer, eval_pool, seed=seed, device=device)
    model.train()
    for step in range(steps):
        inputs, labels, tokens = make_batch(
            tokenizer, texts, step=step, batch_size=batch_size,
            sequence_length=sequence_length, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, usage = model(inputs)
        language_loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100,
        )
        balance = ((usage.mean(dim=0) - 1.0 / config.experts) ** 2).mean() * config.experts
        loss = language_loss + 0.01 * balance
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_seen += tokens
        if step == 0 or (step + 1) % 10 == 0:
            curve.append({"step": step + 1, "training_loss": float(language_loss), "tokens_seen": tokens_seen})
    after = evaluate(model, tokenizer, eval_pool, seed=seed, device=device)
    return {
        "seed": seed, "steps": steps, "batch_size": batch_size, "examples": len(texts),
        "tokens_seen": tokens_seen, "elapsed_seconds": round(time.perf_counter() - started, 3),
        "parameter_counts": model.parameter_counts(), "before": before, "after": after,
        "training_curve": curve,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device_name = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device
    device = torch.device(device_name)
    structures = all_structures()
    train_pool, eval_pool = structures[:88], structures[88:120]
    seeds = [int(value) for value in args.seeds.split(",")]
    rows = []
    started = time.perf_counter()
    for seed in seeds:
        examples = fixed_examples(train_pool, args.steps * args.batch_size, seed=seed)
        arms = {
            "interleaved": interleaved_variant(examples),
            "blocked": blocked_variant(examples),
            "curriculum": curriculum_variant(examples, train_pool, seed=seed),
        }
        for arm, texts in arms.items():
            row = train_arm(
                texts=texts, seed=seed, steps=args.steps, batch_size=args.batch_size,
                learning_rate=args.learning_rate, sequence_length=args.sequence_length,
                device=device, eval_pool=eval_pool,
            )
            row["arm"] = arm
            rows.append(row)
            print(json.dumps({
                "arm": arm, "seed": seed,
                "accuracy": row["after"]["target_byte_accuracy"],
                "exact": row["after"]["target_exact_accuracy"], "seconds": row["elapsed_seconds"],
            }), flush=True)
    receipt = {
        "schema_version": "1.0", "status": "complete",
        "research_question": "With examples and compute fixed, does example ordering or an 8-then-88 curriculum change held-out learning?",
        "independent_variable": "example ordering / early curriculum composition",
        "controlled": ["examples within arm", "model", "initialization within seed", "optimizer", "steps", "batch size", "evaluation set"],
        "structural_holdout": [list(value) for value in eval_pool],
        "device": device.type, "steps": args.steps, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "seeds": seeds,
        "elapsed_seconds": round(time.perf_counter() - started, 3), "runs": rows,
    }
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
