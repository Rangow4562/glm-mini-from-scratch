#!/usr/bin/env python3
"""Bounded pilot: attach the miniature vision tower to the full 12-layer coding model.

The complete language model executes; training touches only the vision encoder,
the final LM block, the final norm, and the tied embedding/output matrix.
Coding weights are loaded from an existing checkpoint where shapes allow. No
weights are ever saved.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from glm_mini import (  # noqa: E402
    ByteTokenizer,
    GLMMiniFromScratch,
    MiniVisionConfig,
    ModelConfig,
    VisionLanguageModel,
    answer_only_labels,
)
from glm_mini.runtime import resolve_device  # noqa: E402

_spec = importlib.util.spec_from_file_location("train_vision", REPO / "scripts" / "train_vision.py")
train_vision = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_vision)


def load_coding_weights(wrapper: VisionLanguageModel, checkpoint: Path) -> dict:
    """Copy every matching tensor; the 263-row tied embedding stays trainable."""
    source = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
    target = wrapper.language_model.state_dict()
    copied = {}
    skipped = []
    for key, value in source.items():
        if key in target and target[key].shape == value.shape:
            copied[key] = value
        else:
            skipped.append(key)
    wrapper.language_model.load_state_dict(copied, strict=False)
    return {"copied_tensors": len(copied), "skipped_tensors": sorted(skipped)}


def restrict_trainable_surface(wrapper: VisionLanguageModel) -> dict:
    for parameter in wrapper.parameters():
        parameter.requires_grad = False
    surfaces = [wrapper.vision_encoder, wrapper.language_model.layers[-1],
                wrapper.language_model.final_norm, wrapper.language_model.embedding]
    for surface in surfaces:
        for parameter in surface.parameters():
            parameter.requires_grad = True
    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapper.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total,
            "trainable_surfaces": ["vision_encoder", "last_block", "final_norm", "tied_embedding"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-seconds", type=int, default=540)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    tokenizer = ByteTokenizer()

    language_config = ModelConfig(vocab_size=263, max_sequence_length=96)
    vision_config = MiniVisionConfig(
        image_size=32, patch_size=4, hidden_size=24, depth=2, heads=4,
        intermediate_size=48, spatial_merge_size=2, projection_intermediate_size=64,
    )
    language_model = GLMMiniFromScratch(language_config)
    wrapper = VisionLanguageModel(language_model, vision_config).to(device)
    load_report = load_coding_weights(wrapper, args.checkpoint)
    surface_report = restrict_trainable_surface(wrapper)
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    evaluation_seed = 10_000 + args.seed
    before = train_vision.held_out_metrics(
        wrapper, tokenizer, examples=args.eval_examples, seed=evaluation_seed, device=device,
    )
    started = time.perf_counter()
    curve = []
    stopped_early = False
    wrapper.train()
    for step in range(args.steps):
        if time.perf_counter() - started > args.max_seconds:
            stopped_early = True
            break
        labels = (torch.arange(args.batch_size) + step) % 10
        images = train_vision.generated_rgb_digit_images(labels, seed=args.seed + step).to(device)
        tokens, prompt_length = train_vision.text_batch(tokenizer, labels, device)
        targets = answer_only_labels(
            tokens, prompt_length=prompt_length, visual_token_count=wrapper.visual_token_count,
            boundary_token_count=wrapper.inserted_token_count - wrapper.visual_token_count,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, usage = wrapper(images, tokens[:, :-1])
        language_loss = F.cross_entropy(
            logits.reshape(-1, language_config.vocab_size), targets.reshape(-1), ignore_index=-100,
        )
        balance = ((usage.mean(dim=0) - 1.0 / language_config.experts).square().mean() * language_config.experts)
        loss = language_loss + 0.01 * balance
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite pilot loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        curve.append({"step": step + 1, "loss": float(language_loss.detach())})
    if device.type == "cuda":
        torch.cuda.synchronize()
    after = train_vision.held_out_metrics(
        wrapper, tokenizer, examples=args.eval_examples, seed=evaluation_seed, device=device,
    )
    receipt = {
        "schema_version": "1.0",
        "status": "complete",
        "stage": "full_25m_vision_digit_pilot",
        "checkpoint_saved": False,
        "source_checkpoint": str(args.checkpoint),
        "weight_loading": load_report,
        "trainable_surface": surface_report,
        "language_model": language_config.to_dict(),
        "vision_model": asdict(vision_config),
        "config": {
            "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
            "eval_examples": args.eval_examples, "learning_rate": args.learning_rate,
            "max_seconds": args.max_seconds, "stopped_early": stopped_early,
        },
        "environment": {
            "device": device.type, "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        },
        "before": before,
        "after": after,
        "training_curve": curve,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({
        "before": {"accuracy": before["accuracy"], "loss": before["loss"]},
        "after": {"accuracy": after["accuracy"], "loss": after["loss"]},
        "elapsed_seconds": receipt["elapsed_seconds"], "stopped_early": stopped_early,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
