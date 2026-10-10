"""Deterministic checkpoints, device resolution, and batched autoregressive generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .evaluator import completion_is_parseable
from .model import GLMMiniFromScratch, ModelConfig
from .tasks import CodingTask
from .tokenizer import ByteTokenizer


def resolve_device(specification: str) -> torch.device:
    """Map auto|cuda|mps|cpu onto the best available backend."""
    if specification == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        name = specification
    return torch.device(name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    model: GLMMiniFromScratch, path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Write model.pt, config.json and metadata.json; refuse to overwrite."""
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    weights = path / "model.pt"
    torch.save(model.state_dict(), weights)
    (path / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2, sort_keys=True) + "\n")
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {"path": str(path), "weights_sha256": sha256_file(weights)}


def load_checkpoint(path: Path, device: torch.device) -> GLMMiniFromScratch:
    config = ModelConfig(**json.loads((path / "config.json").read_text()))
    model = GLMMiniFromScratch(config).to(device)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
    return model


@torch.no_grad()
def generate_group(
    model: GLMMiniFromScratch,
    tokenizer: ByteTokenizer,
    task: CodingTask,
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    sample: bool,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample one rollout group from identical prompts, stopping row by row.

    A row finishes when it emits EOS or its completion would pass the AST gate.
    Finished rows keep receiving EOS so the batch stays rectangular.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    prompt = tokenizer.encode(task.prompt, bos=True)
    if len(prompt) + max_new_tokens > model.config.max_sequence_length:
        raise ValueError("prompt plus generation allowance exceeds the context")
    rows = torch.tensor([prompt] * group_size, dtype=torch.long, device=next(model.parameters()).device)
    completions: list[list[int]] = [[] for _ in range(group_size)]
    finished = [False] * group_size
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.eval()
    for _ in range(max_new_tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=rows.is_cuda):
            logits, _ = model(rows)
        next_logits = logits[:, -1, :].float() / temperature
        next_logits[:, [tokenizer.pad_id, tokenizer.bos_id, tokenizer.sep_id]] = -torch.inf
        next_tokens = (
            torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1).squeeze(-1)
            if sample else next_logits.argmax(dim=-1)
        )
        appended = []
        for index, token in enumerate(next_tokens.tolist()):
            if finished[index]:
                token = tokenizer.eos_id
            else:
                completions[index].append(token)
                if token == tokenizer.eos_id or completion_is_parseable(task, tokenizer.decode(completions[index])):
                    finished[index] = True
            appended.append(token)
        rows = torch.cat([rows, torch.tensor(appended, device=rows.device)[:, None]], dim=1)
        if all(finished):
            break
    results = []
    for tokens in completions:
        results.append({
            "completion": tokenizer.decode(tokens),
            "token_ids": tokens,
            "tokens": len(tokens),
            "hit_token_cap": len(tokens) == max_new_tokens and (not tokens or tokens[-1] != tokenizer.eos_id),
        })
    return results
