"""Tokenizer round-trips, verifier behavior, and frozen-split determinism."""
import pytest

from glm_mini.evaluator import evaluate_source
from glm_mini.tasks import FAMILY_BY_NAME, frozen_tasks, pretraining_text
from glm_mini.tokenizer import ByteTokenizer

SPLITS = ("dev", "rl", "confirm")


def test_tokenizer_round_trip_including_unicode():
    tokenizer = ByteTokenizer()
    text = "héllo αβλ 😀\n    return x + 1\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_token_layout():
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("A", bos=True, eos=True)
    assert ids == [tokenizer.bos_id, tokenizer.byte_offset + 65, tokenizer.eos_id]


def test_reference_completions_pass_on_every_split():
    for split in SPLITS:
        for task in frozen_tasks(split):
            evaluation = evaluate_source(task, task.reference_source)
            assert evaluation.passed, (task.task_id, evaluation.message)


def test_stub_completion_fails_but_stays_valid():
    task = frozen_tasks("dev")[0]
    evaluation = evaluate_source(task, task.prompt + "\n    pass\n")
    assert not evaluation.passed
    assert evaluation.status == "failed"


def test_verifier_rejects_disallowed_sources():
    task = frozen_tasks("dev")[0]
    argument = FAMILY_BY_NAME[task.family].argument
    sources = [
        task.prompt + "\n    import os\n",
        task.prompt + "\n    return __name__\n",
        task.prompt + "\n    for _ in range(3):\n        pass\n",
        task.prompt + "\n    return x.upper()\n",
        task.prompt + f"\n    return {task.entry_point}({argument})\n",
    ]
    for source in sources:
        evaluation = evaluate_source(task, source)
        assert evaluation.status == "invalid", (task.task_id, source, evaluation.message)


def test_frozen_splits_use_disjoint_entry_points():
    by_split = {split: {task.entry_point for task in frozen_tasks(split)} for split in SPLITS}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            assert by_split[left].isdisjoint(by_split[right])


def test_task_generation_is_deterministic():
    tasks = frozen_tasks("dev")
    again = frozen_tasks("dev")
    assert [task.entry_point for task in tasks] == [task.entry_point for task in again]
    assert pretraining_text(3) == pretraining_text(3)
    assert pretraining_text(3) != pretraining_text(4)


def test_unknown_split_rejected():
    with pytest.raises(ValueError):
        frozen_tasks("train")
