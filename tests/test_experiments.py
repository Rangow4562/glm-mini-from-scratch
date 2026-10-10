"""Experiment modules load cleanly and expose the expected research surface."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments.analyze_pretraining_curriculum_order import exact_sign_test as order_sign_test
from experiments.analyze_pretraining_data_diversity import exact_sign_test as diversity_sign_test
from experiments.analyze_pretraining_data_diversity import holm


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diversity_experiment_surface():
    module = load("pretraining_data_diversity")
    structures = module.all_structures()
    assert len(structures) == 120
    assert all(len(set(structure)) >= 2 for structure in structures)
    assert len(structures[:88]) == 88 and len(structures[88:120]) == 32


def test_render_example_is_deterministic():
    module = load("pretraining_data_diversity")
    structure = module.all_structures()[0]
    first = module.render_example(structure, index=5, seed=42)
    second = module.render_example(structure, index=5, seed=42)
    assert first == second


def test_curriculum_experiment_variants():
    module = load("pretraining_curriculum_order")
    pool = module.all_structures()
    examples = module.fixed_examples(pool[:88], 60, seed=42)
    interleaved = module.interleaved_variant(examples)
    blocked = module.blocked_variant(examples)
    curriculum = module.curriculum_variant(examples, pool[:88], seed=42)
    assert sorted(interleaved) == sorted(blocked)  # identical multiset of examples
    assert len(curriculum) == len(interleaved)
    # Blocked groups identical structures together.
    assert blocked != interleaved


def test_sign_test_known_values():
    assert diversity_sign_test([0.5, 0.5, 0.5]) < 0.3
    assert order_sign_test([0.1, -0.1, 0.05]) == 1.0
    assert diversity_sign_test([0.0, 0.0, 0.0]) == 1.0


def test_holm_correction_monotonic():
    adjusted = holm([0.01, 0.04, 0.03])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    assert all(p >= raw for p, raw in zip(adjusted, [0.01, 0.04, 0.03]))


def test_pilot_weight_loading_skips_mismatched_embedding(tmp_path):
    from glm_mini import GLMMiniFromScratch, ModelConfig
    from glm_mini.runtime import save_checkpoint
    from glm_mini.vision import MiniVisionConfig, VisionLanguageModel

    torch.manual_seed(0)
    coding = GLMMiniFromScratch(ModelConfig(vocab_size=260, dim=32, layers=2, heads=4))
    save_checkpoint(coding, tmp_path / "checkpoint", {"stage": "test"})
    wrapper = VisionLanguageModel(
        GLMMiniFromScratch(ModelConfig(vocab_size=263, dim=32, layers=2, heads=4)),
        MiniVisionConfig(),
    )
    spec = importlib.util.spec_from_file_location("pilot", REPO / "experiments" / "full_25m_vision_digit_pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    report = pilot.load_coding_weights(wrapper, tmp_path / "checkpoint")
    assert "embedding.weight" in report["skipped_tensors"]
    assert "output.weight" in report["skipped_tensors"]
    assert report["copied_tensors"] > 0


def test_pilot_trainable_surface():
    from glm_mini import GLMMiniFromScratch, MiniVisionConfig, ModelConfig, VisionLanguageModel

    wrapper = VisionLanguageModel(
        GLMMiniFromScratch(ModelConfig(vocab_size=263, dim=32, layers=4, heads=4)),
        MiniVisionConfig(),
    )
    spec = importlib.util.spec_from_file_location("pilot", REPO / "experiments" / "full_25m_vision_digit_pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    report = pilot.restrict_trainable_surface(wrapper)
    total = sum(p.numel() for p in wrapper.parameters())
    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    assert report["total_parameters"] == total
    assert report["trainable_parameters"] == trainable
    assert trainable < total


def test_make_charts_writes_svg(tmp_path):
    spec = importlib.util.spec_from_file_location("make_charts", REPO / "scripts" / "make_charts.py")
    charts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(charts)
    chart = charts.line_chart([1.0, 0.5, 0.2], labels=["0", "10", "20"], y_max=1.1,
                              y_label="loss", title="t", subtitle="s")
    assert chart.startswith("<svg") and chart.rstrip().endswith("</svg>")
