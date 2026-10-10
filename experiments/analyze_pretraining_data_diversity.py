#!/usr/bin/env python3
"""Paired analysis of the pretraining data diversity study."""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path


def exact_sign_test(differences: list[float]) -> float:
    """Two-sided exact binomial sign test on nonzero paired differences."""
    wins = sum(d > 0 for d in differences)
    losses = sum(d < 0 for d in differences)
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, p_values[index] * (len(p_values) - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_comparison(rows: list[dict], left: int, right: int) -> dict:
    left_rows = {row["seed"]: row for row in rows if row["diversity"] == left}
    right_rows = {row["seed"]: row for row in rows if row["diversity"] == right}
    seeds = sorted(set(left_rows) & set(right_rows))
    differences = [
        left_rows[seed]["after"]["target_byte_accuracy"] - right_rows[seed]["after"]["target_byte_accuracy"]
        for seed in seeds
    ]
    return {
        "diversity_left": left,
        "diversity_right": right,
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
    diversities = sorted({row["diversity"] for row in rows})
    comparisons = [
        paired_comparison(rows, left, right)
        for left, right in combinations(diversities, 2)
        if left < right
    ]
    adjusted = holm([comparison["exact_sign_test_p"] for comparison in comparisons])
    for comparison, p_value in zip(comparisons, adjusted):
        comparison["holm_adjusted_p"] = p_value
    summary = {
        "schema_version": "1.0",
        "source": str(args.results),
        "comparisons": comparisons,
        "note": "Seed-level exact sign test. Seed counts are small; treat p-values as descriptive.",
    }
    for comparison in comparisons:
        print(json.dumps({
            "pair": f"{comparison['diversity_left']} vs {comparison['diversity_right']}",
            "mean_difference": round(comparison["mean_difference"], 4),
            "p": round(comparison["exact_sign_test_p"], 4),
            "holm_p": round(comparison["holm_adjusted_p"], 4),
        }))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
