"""Checkpoint behavior and generation semantics."""
import pytest
import torch

from glm_mini import ByteTokenizer, ModelConfig
from glm_mini.model import GLMMiniFromScratch
from glm_mini.runtime import generate_group, load_checkpoint, save_checkpoint
from glm_mini.tasks import frozen_tasks


def tiny_model() -> GLMMiniFromScratch:
    torch.manual_seed(7)
    return GLMMiniFromScratch(ModelConfig(
        dim=32, heads=4, layers=2, expert_hidden=64,
        experts=4, top_k=2, streams=2,
    )).eval()


def test_checkpoint_round_trip(tmp_path):
    model = tiny_model()
    inputs = torch.randint(4, 260, (1, 16))
    with torch.no_grad():
        before, _ = model(inputs)
    first = save_checkpoint(model, tmp_path / "checkpoint", {"stage": "test"})
    assert len(first["weights_sha256"]) == 64
    with pytest.raises(FileExistsError):
        save_checkpoint(model, tmp_path / "checkpoint", {"stage": "test"})
    loaded = load_checkpoint(tmp_path / "checkpoint", torch.device("cpu")).eval()
    with torch.no_grad():
        after, _ = loaded(inputs)
    assert torch.allclose(before, after)


def test_generation_masks_control_tokens_and_is_deterministic():
    model = tiny_model()
    tokenizer = ByteTokenizer()
    task = frozen_tasks("rl", per_family=1)[0]
    group = generate_group(
        model, tokenizer, task, group_size=3, max_new_tokens=12,
        temperature=1.0, sample=True, seed=99,
    )
    assert len(group) == 3
    for row in group:
        assert len(row["token_ids"]) == row["tokens"]
        assert tokenizer.decode(row["token_ids"]) == row["completion"]
        for special in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.sep_id):
            assert special not in row["token_ids"]
    repeat = generate_group(
        model, tokenizer, task, group_size=3, max_new_tokens=12,
        temperature=1.0, sample=True, seed=99,
    )
    assert [row["token_ids"] for row in group] == [row["token_ids"] for row in repeat]


def test_generation_validates_arguments():
    model = tiny_model()
    tokenizer = ByteTokenizer()
    task = frozen_tasks("rl", per_family=1)[0]
    with pytest.raises(ValueError):
        generate_group(model, tokenizer, task, group_size=1, max_new_tokens=4,
                       temperature=0.0, sample=False, seed=1)
    with pytest.raises(ValueError):
        generate_group(model, tokenizer, task, group_size=1,
                       max_new_tokens=model.config.max_sequence_length,
                       temperature=1.0, sample=False, seed=1)
