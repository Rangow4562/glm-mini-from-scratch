#!/usr/bin/env python3
"""Generate SVG charts from training receipts and evaluation payloads."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WIDTH, HEIGHT = 960, 560
PLOT_LEFT, PLOT_TOP, PLOT_WIDTH, PLOT_HEIGHT = 90, 80, 790, 360
INK = "#f4f0e8"
MUTED = "#aaa69d"
GRID = "#353530"
BACKGROUND = "#171714"
ACCENTS = ("#d97757", "#7aa2c9", "#9eb67b", "#e0b45c")


def svg_document(title: str, subtitle: str, body: list[str]) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect width="100%" height="100%" fill="{BACKGROUND}"/>',
        f'<text x="{PLOT_LEFT}" y="40" fill="{INK}" font-family="Arial" font-size="25" font-weight="700">{title}</text>',
        f'<text x="{PLOT_LEFT}" y="64" fill="{MUTED}" font-family="Arial" font-size="14">{subtitle}</text>',
    ]
    parts.extend(body)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def line_chart(values: list[float], *, labels: list[str], y_max: float, y_label: str,
              title: str, subtitle: str) -> str:
    body = []
    ticks = 5
    for tick in range(ticks + 1):
        value = y_max * tick / ticks
        y = PLOT_TOP + PLOT_HEIGHT * (1 - tick / ticks)
        body.append(f'<line x1="{PLOT_LEFT}" y1="{y:.1f}" x2="{PLOT_LEFT + PLOT_WIDTH}" y2="{y:.1f}" stroke="{GRID}"/>')
        body.append(f'<text x="{PLOT_LEFT - 12}" y="{y + 5:.1f}" fill="{MUTED}" text-anchor="end" font-family="Arial" font-size="13">{value:g}</text>')
    points = []
    label_every = max(1, len(values) // 8)
    for index, value in enumerate(values):
        x = PLOT_LEFT + PLOT_WIDTH * index / max(1, len(values) - 1)
        y = PLOT_TOP + PLOT_HEIGHT * (1 - value / y_max)
        points.append(f"{x:.1f},{y:.1f}")
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{ACCENTS[0]}"/>')
        if index % label_every == 0 or index == len(values) - 1:
            label = labels[index] if index < len(labels) else str(index + 1)
            body.append(f'<text x="{x:.1f}" y="{PLOT_TOP + PLOT_HEIGHT + 24}" fill="{MUTED}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>')
    body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{ACCENTS[0]}" stroke-width="3"/>')
    body.append(f'<text x="22" y="{PLOT_TOP + PLOT_HEIGHT / 2:.0f}" fill="{MUTED}" font-family="Arial" font-size="14" transform="rotate(-90 22 {PLOT_TOP + PLOT_HEIGHT / 2:.0f})">{y_label}</text>')
    return svg_document(title, subtitle, body)


def grouped_bars(groups: list[tuple[str, list[tuple[str, float]]]], *, y_max: float,
                 y_label: str, title: str, subtitle: str) -> str:
    body = []
    ticks = 5
    for tick in range(ticks + 1):
        value = y_max * tick / ticks
        y = PLOT_TOP + PLOT_HEIGHT * (1 - tick / ticks)
        body.append(f'<line x1="{PLOT_LEFT}" y1="{y:.1f}" x2="{PLOT_LEFT + PLOT_WIDTH}" y2="{y:.1f}" stroke="{GRID}"/>')
        body.append(f'<text x="{PLOT_LEFT - 12}" y="{y + 5:.1f}" fill="{MUTED}" text-anchor="end" font-family="Arial" font-size="13">{value:g}</text>')
    group_width = PLOT_WIDTH / max(1, len(groups))
    for group_index, (group_label, entries) in enumerate(groups):
        bar_width = min(120, (group_width - 40) / max(1, len(entries)))
        for entry_index, (entry_label, value) in enumerate(entries):
            x = PLOT_LEFT + group_width * group_index + (group_width - bar_width * len(entries)) / 2 + bar_width * entry_index
            y = PLOT_TOP + PLOT_HEIGHT * (1 - value / y_max)
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 8:.1f}" height="{PLOT_TOP + PLOT_HEIGHT - y:.1f}" rx="6" fill="{ACCENTS[entry_index % len(ACCENTS)]}"/>')
            body.append(f'<text x="{x + (bar_width - 8) / 2:.1f}" y="{y - 10:.1f}" fill="{INK}" text-anchor="middle" font-family="Arial" font-size="16" font-weight="700">{value:g}</text>')
            body.append(f'<text x="{x + (bar_width - 8) / 2:.1f}" y="{PLOT_TOP + PLOT_HEIGHT + 24}" fill="{MUTED}" text-anchor="middle" font-family="Arial" font-size="13">{entry_label}</text>')
        body.append(f'<text x="{PLOT_LEFT + group_width * group_index + group_width / 2:.1f}" y="{PLOT_TOP + PLOT_HEIGHT + 46}" fill="{INK}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700">{group_label}</text>')
    body.append(f'<text x="22" y="{PLOT_TOP + PLOT_HEIGHT / 2:.0f}" fill="{MUTED}" font-family="Arial" font-size="14" transform="rotate(-90 22 {PLOT_TOP + PLOT_HEIGHT / 2:.0f})">{y_label}</text>')
    return svg_document(title, subtitle, body)


def smooth(values: list[float], window: int = 5) -> list[float]:
    return [sum(values[max(0, index - window + 1): index + 1]) / len(values[max(0, index - window + 1): index + 1])
            for index in range(len(values))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/charts"))
    parser.add_argument("--pretrain-receipt", type=Path)
    parser.add_argument("--self-improve-receipt", type=Path)
    parser.add_argument("--eval-before", type=Path)
    parser.add_argument("--eval-after", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    written = []

    if args.pretrain_receipt and args.pretrain_receipt.exists():
        receipt = json.loads(args.pretrain_receipt.read_text())
        losses = [row["loss"] for row in receipt["training_curve"]]
        steps = [row["step"] for row in receipt["training_curve"]]
        chart = line_chart(
            smooth(losses), labels=[str(step) for step in steps],
            y_max=max(losses) * 1.1, y_label="Training loss (next byte)",
            title="Pretraining loss curve", subtitle="Next-byte cross-entropy on the synthetic coding corpus",
        )
        destination = args.output / "pretrain-loss.svg"
        destination.write_text(chart)
        written.append(destination)

    if args.self_improve_receipt and args.self_improve_receipt.exists():
        receipt = json.loads(args.self_improve_receipt.read_text())
        cycles = receipt["cycles"]
        before = [cycle["pass_rate_before"] * 100 for cycle in cycles]
        after = [cycle["pass_rate_after"] * 100 for cycle in cycles]
        labels = [f"cycle {cycle['cycle']}" for cycle in cycles]
        pairs = [(label, value) for label, value in zip(labels, after)]
        chart = grouped_bars(
            [(label, [(f"{value:.0f}%", value) for value in [after[index]]]) for index, label in enumerate(labels)],
            y_max=100, y_label="Greedy pass@1 (%)",
            title="Recursive self-improvement ladder",
            subtitle="Each cycle trains only on completions that pass the executable verifier",
        )
        body = chart
        destination = args.output / "self-improvement-ladder.svg"
        destination.write_text(body)
        written.append(destination)

    if args.eval_before and args.eval_after and args.eval_before.exists() and args.eval_after.exists():
        payload_before = json.loads(args.eval_before.read_text())
        payload_after = json.loads(args.eval_after.read_text())
        groups = []
        for label, payload in (("before", payload_before), ("after", payload_after)):
            by_family = payload["by_family"]
            entries = [(family, row["solved"] / row["tasks"] * 100) for family, row in sorted(by_family.items())]
            groups.append((label, entries))
        chart = grouped_bars(
            groups, y_max=100, y_label="Tasks solved (%)",
            title="Frozen confirm split, before vs after",
            subtitle="Greedy pass@1 per family on held-out tasks",
        )
        destination = args.output / "before-after-families.svg"
        destination.write_text(chart)
        written.append(destination)

    print(json.dumps({"written": [str(path) for path in written]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
