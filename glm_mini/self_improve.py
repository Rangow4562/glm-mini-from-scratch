"""Recursive self-improvement: propose, verify, distill, retrain, repeat.

The model improves on its own outputs. Every cycle the current policy samples
completions for a task pool, the executable verifier keeps only completions
that pass every hidden test, and the surviving (prompt, completion) pairs join
a cumulative, deduplicated memory. The model is fine-tuned on that memory,
re-evaluated on frozen tasks, and — when it masters the current rung — the
next, harder family rung is unlocked. The executable verifier makes the
self-generated signal honest: the model can only learn from programs that
actually run and actually pass.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .evaluator import evaluate_source
from .model import GLMMiniFromScratch
from .runtime import generate_group, save_checkpoint
from .tasks import CodingTask, frozen_tasks
from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class SelfImproveConfig:
    cycles: int = 3
    proposals_per_task: int = 16
    max_new_tokens: int = 64
    temperature: float = 0.8
    fine_tune_steps: int = 24
    fine_tune_batch_size: int = 8
    learning_rate: float = 1e-4
    target_pass_rate: float = 1.0
    families_ladder: tuple[str, ...] = ("increment", "double", "even")
    per_family: int = 4
    eval_per_family: int = 2
    expand_on_master: bool = False
    seed: int = 73


@dataclass
class VerifiedMemory:
    """Cumulative, deduplicated memory of the model's own verified programs."""

    pairs: list[tuple[CodingTask, str]] = field(default_factory=list)
    sources: set[tuple[str, str]] = field(default_factory=set)

    def add(self, task: CodingTask, completion: str) -> bool:
        key = (task.task_id, completion)
        if key in self.sources:
            return False
        self.sources.add(key)
        self.pairs.append((task, completion))
        return True

    def __len__(self) -> int:
        return len(self.pairs)


def distill_batch(
    tokenizer: ByteTokenizer,
    pairs: list[tuple[CodingTask, str]],
    *,
    step: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """One padded next-byte batch drawn round-robin from the verified memory."""
    sequence_length = 192
    rows = []
    for offset in range(batch_size):
        task, completion = pairs[(step * batch_size + offset) % len(pairs)]
        ids = tokenizer.encode(task.prompt, bos=True) + tokenizer.encode(completion, eos=True)
        if len(ids) > sequence_length + 1:
            raise ValueError(f"distilled example exceeds context: {len(ids)}")
        ids += [tokenizer.pad_id] * (sequence_length + 1 - len(ids))
        rows.append(ids)
    values = torch.tensor(rows, dtype=torch.long, device=device)
    inputs, labels = values[:, :-1], values[:, 1:]
    labels = labels.masked_fill(labels == tokenizer.pad_id, -100)
    return inputs, labels, int((labels != -100).sum().item())


def propose(
    model: GLMMiniFromScratch,
    tokenizer: ByteTokenizer,
    tasks: list[CodingTask],
    *,
    proposals_per_task: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> tuple[list[tuple[CodingTask, str]], list[dict[str, Any]]]:
    """Sample completions and keep exactly the ones that pass every hidden test."""
    accepted: list[tuple[CodingTask, str]] = []
    details: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        generations = generate_group(
            model, tokenizer, task, group_size=proposals_per_task,
            max_new_tokens=max_new_tokens, temperature=temperature,
            sample=True, seed=seed + task_index * 104_729,
        )
        passed = 0
        for proposal_index, generation in enumerate(generations):
            evaluation = evaluate_source(task, task.prompt + generation["completion"])
            if evaluation.passed:
                accepted.append((task, generation["completion"]))
                passed += 1
        details.append({
            "task_id": task.task_id,
            "family": task.family,
            "proposals": proposals_per_task,
            "passed": passed,
            "valid": sum(
                evaluate_source(task, task.prompt + g["completion"]).status != "invalid"
                for g in generations
            ),
        })
    return accepted, details


def evaluate_pool(
    model: GLMMiniFromScratch,
    tokenizer: ByteTokenizer,
    tasks: list[CodingTask],
    *,
    seed: int,
) -> dict[str, Any]:
    """Greedy pass@1 over a frozen evaluation pool."""
    episodes = []
    for index, task in enumerate(tasks):
        generated = generate_group(
            model, tokenizer, task, group_size=1, max_new_tokens=64,
            temperature=1.0, sample=False, seed=seed + index,
        )[0]
        evaluation = evaluate_source(task, task.prompt + generated["completion"])
        episodes.append({
            "task_id": task.task_id,
            "family": task.family,
            "completion": generated["completion"],
            "passed": evaluation.passed,
            "status": evaluation.status,
            "tests_passed": evaluation.tests_passed,
        })
    solved = sum(episode["passed"] for episode in episodes)
    return {
        "tasks": len(episodes),
        "solved": solved,
        "pass_rate": solved / len(episodes),
        "episodes": episodes,
    }


def fine_tune(
    model: GLMMiniFromScratch,
    tokenizer: ByteTokenizer,
    memory: VerifiedMemory,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Brief supervised fine-tune on the model's own verified programs."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    model.train()
    curve = []
    tokens_seen = 0
    for step in range(steps):
        inputs, labels, tokens = distill_batch(
            tokenizer, memory.pairs, step=step, batch_size=batch_size, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, usage = model(inputs)
        language_loss = F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size), labels.reshape(-1), ignore_index=-100
        )
        balance = ((usage.mean(dim=0) - 1.0 / model.config.experts) ** 2).mean() * model.config.experts
        loss = language_loss + 0.01 * balance
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite self-improvement loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        tokens_seen += tokens
        curve.append({"step": step + 1, "loss": float(language_loss.detach()), "tokens": tokens})
    return {"steps": steps, "tokens_seen": tokens_seen, "curve": curve}


def run_self_improvement(
    model: GLMMiniFromScratch,
    tokenizer: ByteTokenizer,
    config: SelfImproveConfig,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    """The recursive loop: propose -> verify -> distill -> fine-tune -> evaluate."""
    output.mkdir(parents=True)
    memory = VerifiedMemory()
    cycles: list[dict[str, Any]] = []
    initial = save_checkpoint(model, output / "checkpoint-cycle-00", {
        "stage": "self_improvement_start", "seed": config.seed,
    })
    started = time.perf_counter()
    for cycle in range(1, config.cycles + 1):
        cycle_started = time.perf_counter()
        rung = (cycle - 1) if config.expand_on_master else (config.cycles - 1)
        rung = min(rung, len(config.families_ladder) - 1)
        ladder = set(config.families_ladder[: rung + 1])
        train_pool = [t for t in frozen_tasks("rl", per_family=config.per_family) if t.family in ladder]
        eval_pool = [t for t in frozen_tasks("dev", per_family=config.eval_per_family) if t.family in ladder]
        before = evaluate_pool(model, tokenizer, eval_pool, seed=config.seed + cycle * 7919)

        accepted, proposal_details = propose(
            model, tokenizer, train_pool,
            proposals_per_task=config.proposals_per_task,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            seed=config.seed * 31 + cycle * 1009,
        )
        added = sum(1 for task, completion in accepted if memory.add(task, completion))

        training = None
        if added > 0:
            training = fine_tune(
                model, tokenizer, memory,
                steps=config.fine_tune_steps, batch_size=config.fine_tune_batch_size,
                learning_rate=config.learning_rate, seed=config.seed + cycle, device=device,
            )
        after = evaluate_pool(model, tokenizer, eval_pool, seed=config.seed + cycle * 7919)

        checkpoint = save_checkpoint(model, output / f"checkpoint-cycle-{cycle:02d}", {
            "stage": "self_improvement", "cycle": cycle, "verified_pairs": len(memory),
        })
        cycles.append({
            "cycle": cycle,
            "families": sorted(ladder),
            "verified_pairs_total": len(memory),
            "verified_pairs_added": added,
            "proposal_details": proposal_details,
            "pass_rate_before": before["pass_rate"],
            "pass_rate_after": after["pass_rate"],
            "fine_tune": training,
            "evaluation": after,
            "checkpoint": checkpoint,
            "seconds": round(time.perf_counter() - cycle_started, 3),
        })
        print(json_line(cycles[-1]), flush=True)
        if after["pass_rate"] >= config.target_pass_rate:
            break
    receipt = {
        "schema_version": "1.0",
        "status": "complete",
        "stage": "recursive_self_improvement",
        "config": {
            "cycles": config.cycles,
            "proposals_per_task": config.proposals_per_task,
            "temperature": config.temperature,
            "fine_tune_steps": config.fine_tune_steps,
            "fine_tune_batch_size": config.fine_tune_batch_size,
            "learning_rate": config.learning_rate,
            "target_pass_rate": config.target_pass_rate,
            "families_ladder": list(config.families_ladder),
            "expand_on_master": config.expand_on_master,
            "seed": config.seed,
        },
        "initial_checkpoint": initial,
        "verified_memory_final_size": len(memory),
        "cycles": cycles,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "device": device.type,
    }
    (output / "training-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def json_line(cycle: dict[str, Any]) -> str:
    """Compact one-line progress record for stdout."""
    return json.dumps({
        key: cycle[key]
        for key in ("cycle", "families", "verified_pairs_total", "verified_pairs_added",
                    "pass_rate_before", "pass_rate_after", "seconds")
    }, sort_keys=True)
