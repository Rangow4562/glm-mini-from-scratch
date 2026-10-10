#!/usr/bin/env python3
"""Paired analysis of the ordering and curriculum study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def exact_sign_test(differences: list[float]) -> float:
    wins = sum(d > 0 for d in differences)
    losses = sum(d < 0 for d in differences)
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_comparison(rows: list[dict], left: str, right: str) -> dict:
    left_rows = {row["seed"]: row for row in rows if row["arm"] == left}
    right_rows = {row["seed"]: row for row in rows if row["arm"] == right}
    seeds = sorted(set(left_rows) & set(right_rows))
    differences = [
        left_rows[seed]["after"]["target_byte_accuracy"] - right_rows[seed]["after"]["target_byte_accuracy"]
        for seed in seeds
    ]
    return {
        "arm_left": left,
        "arm_right": right,
        "seeds": seeds,
        "left_accuracy": [left_rows[seed]["after"]["target_byte_accuracy"] for seed in seeds],
        "right_accuracy": [right_rows[seed]["after"]["target_byte_accuracy"] for seed in seeds],
        "mean_difference": sum(differences) / len(differences),
        "differences": differences,
        "exact_sign_test_p": exact_sign_test(differences),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.results.read_text())
    rows = receipt["runs"]
    arms = sorted({row["arm"] for row in rows})
    comparisons = [paired_comparison(rows, arm, "interleaved") for arm in arms if arm != "interleaved"]
    for comparison in comparisons:
        print(json.dumps({
            "pair": f"{comparison['arm_left']} vs {comparison['arm_right']}",
            "mean_difference": round(comparison["mean_difference"], 4),
            "p": round(comparison["exact_sign_test_p"], 4),
        }))
    if args.output:
        summary = {
            "schema_version": "1.0",
            "source": str(args.results),
            "comparisons": comparisons,
            "note": "Seed-level exact sign test against the interleaved arm. Seed counts are small; treat p-values as descriptive.",
        }
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
