"""Recursive self-improvement invariants: memory dedupe, distillation, one full cycle."""
import pytest
import torch

from glm_mini import ByteTokenizer, ModelConfig
from glm_mini.model import GLMMiniFromScratch
from glm_mini.self_improve import SelfImproveConfig, VerifiedMemory, distill_batch, run_self_improvement
from glm_mini.tasks import frozen_tasks


def tiny_model() -> GLMMiniFromScratch:
    torch.manual_seed(11)
    return GLMMiniFromScratch(ModelConfig(
        dim=32, heads=4, layers=2, expert_hidden=48, experts=4, top_k=2, streams=2,
    )).eval()


def test_verified_memory_deduplicates():
    memory = VerifiedMemory()
    task = frozen_tasks("rl", per_family=1)[0]
    assert memory.add(task, "\n    return x + 1\n") is True
    assert memory.add(task, "\n    return x + 1\n") is False
    assert memory.add(task, "\n    return x + 2\n") is True
    assert len(memory) == 2


def test_distill_batch_shapes_and_masking():
    tokenizer = ByteTokenizer()
    tasks = frozen_tasks("rl", per_family=1)
    pairs = [(task, task.reference_completion) for task in tasks]
    inputs, labels, tokens = distill_batch(tokenizer, pairs, step=0, batch_size=8, device=torch.device("cpu"))
    assert inputs.shape == (8, 192)
    assert labels.shape == (8, 192)
    assert tokens > 0
    # Prompt and answer positions are unmasked; padding is -100.
    assert (labels[:, :80] != -100).all()
    assert (labels == -100).any()


def test_distill_batch_rejects_oversize():
    tokenizer = ByteTokenizer()
    task = frozen_tasks("rl", per_family=1)[0]
    huge = "\n    return " + "x" * 300 + "\n"

    with pytest.raises(ValueError):
        distill_batch(tokenizer, [(task, huge)], step=0, batch_size=1, device=torch.device("cpu"))


def test_one_improvement_cycle_runs(tmp_path):
    model = tiny_model()
    tokenizer = ByteTokenizer()
    config = SelfImproveConfig(
        cycles=1, proposals_per_task=2, max_new_tokens=32, temperature=1.0,
        fine_tune_steps=2, fine_tune_batch_size=2, learning_rate=1e-4,
        families_ladder=("increment",), per_family=1, eval_per_family=1, seed=5,
    )
    receipt = run_self_improvement(model, tokenizer, config, tmp_path / "run", torch.device("cpu"))
    assert receipt["status"] == "complete"
    assert len(receipt["cycles"]) == 1
    cycle = receipt["cycles"][0]
    assert cycle["families"] == ["increment"]
    assert 0.0 <= cycle["pass_rate_before"] <= 1.0
    assert 0.0 <= cycle["pass_rate_after"] <= 1.0
    assert cycle["fine_tune"] is None or cycle["fine_tune"]["steps"] == 2
    assert (tmp_path / "run" / "training-receipt.json").exists()
    assert (tmp_path / "run" / "checkpoint-cycle-00" / "model.pt").exists()
    assert (tmp_path / "run" / "checkpoint-cycle-01" / "model.pt").exists()
