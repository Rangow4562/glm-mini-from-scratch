#!/usr/bin/env python3
"""Paired exact McNemar analysis of RL screen arms against the baseline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def exact_mcnemar(arm_solved: set[str], baseline_solved: set[str]) -> tuple[float, int, int]:
    only_arm = arm_solved - baseline_solved
    only_baseline = baseline_solved - arm_solved
    n = len(only_arm) + len(only_baseline)
    if n == 0:
        return 1.0, len(only_arm), len(only_baseline)
    tail = sum(math.comb(n, k) for k in range(min(len(only_arm), len(only_baseline)) + 1)) / 2**n
    return min(1.0, 2 * tail), len(only_arm), len(only_baseline)


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (len(p_values) - rank)))
        adjusted[index] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True, help="rl-screen directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline_path = args.screen / "baseline" / "dev-eval.json"
    baseline = json.loads(baseline_path.read_text())
    baseline_solved = {row["task_id"] for row in baseline["episodes"] if row["evaluation"]["passed"]}
    arms = {}
    for arm_dir in sorted(args.screen.iterdir()):
        payload_path = arm_dir / "dev-eval.json"
        if not payload_path.exists() or arm_dir.name == "baseline":
            continue
        payload = json.loads(payload_path.read_text())
        solved = {row["task_id"] for row in payload["episodes"] if row["evaluation"]["passed"]}
        p_value, gained, lost = exact_mcnemar(solved, baseline_solved)
        arms[arm_dir.name] = {
            "solved": len(solved),
            "tasks": payload["summary"]["tasks_total"],
            "solved_vs_baseline": f"+{gained}/-{lost}",
            "exact_mcnemar_p": p_value,
        }
    names = sorted(arms)
    adjusted = holm([arms[name]["exact_mcnemar_p"] for name in names])
    for name, p_value in zip(names, adjusted):
        arms[name]["holm_adjusted_p"] = p_value
    summary = {
        "schema_version": "1.0",
        "baseline": {"solved": len(baseline_solved), "tasks": baseline["summary"]["tasks_total"]},
        "arms": arms,
        "note": "Screen selects a candidate; it does not establish a result. Confirm on a held-out split with fresh seeds.",
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
