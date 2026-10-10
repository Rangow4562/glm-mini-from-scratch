# MULTIMODAL.md — the image path, end to end

The language model is byte-level and autoregressive. Multimodality enters without changing it: images are converted into ordinary-looking embeddings and spliced into the token sequence. This document explains the whole path, what is faithful to the released GLM-5.3-Flash topology, and where the teaching version cuts corners.

## The pipeline

```text
32 x 32 RGB image
    -> 4 x 4 patch embedding (Conv2d): 8 x 8 = 64 patch tokens
    -> + learned 2D position table
    -> two bidirectional vision transformer blocks (q/k RMSNorm, SwiGLU)
    -> post-encoder RMSNorm
    -> learned 2 x 2 spatial merge: 4 x 4 = 16 visual tokens
    -> gated multimodal projector: vision width -> LM width (192)
    -> <image_start> <image> x16 <image_end> spliced after BOS
    -> existing GLMMiniFromScratch.forward_embeddings path
    -> autoregressive next-byte prediction
```

## Why placeholder replacement

`build_multimodal_input_ids` inserts `[image_start, placeholder × 16, image_end]` after the BOS token. `embed_multimodal` then runs the tower and scatters the 16 encoder outputs into exactly those placeholder rows via `masked_scatter`. The language model never needs to know images exist — it sees a normal embedding tensor. This mirrors the released model's approach: dedicated vision tokens inside the LM sequence, not a separate decoder head.

## The gated projector

The projector is GLM-style:

```text
h = LayerNorm(W h)          # align modalities
h = GELU(h)
out = down( silu(gate(h)) * up(h) )   # gated SwiGLU back to LM width
```

The gate lets the model scale image information per dimension — the same trick the LM's own MoE experts use internally.

## The 2×2 spatial merger

The merger reshapes 64 patch tokens back into an 8×8 grid, groups each 2×2 neighborhood, and applies one linear map from `4 × vision_dim` to LM width. Rearrange-plus-linear is mathematically identical to a stride-2 convolution and sidesteps a Conv2d backward quirk on some PyTorch builds. It reduces 64 tokens to 16 — four times fewer positions for the LM to attend over.

## What is faithful vs simplified

| Released GLM-5.3-Flash vision | This miniature |
|---|---|
| 448×448 input, temporal patching | fixed 32×32, single frame |
| 24 blocks, width 1024, 16 heads | 2 blocks, width 24, 4 heads |
| 2D rotary positions | learned position table |
| 2×2 learned downsampler | same idea, same 2×2 |
| gated projector to 4096-wide LM | same structure to 192-wide LM |
| variable-resolution token packing | none |

The `DirectPatchVisionLanguageModel` baseline (raw patches straight to LM tokens, no tower) exists only as a labeled control — it is *not* GLM-faithful, and it sometimes wins at tiny scale (see `VISION_REPORT.md`).

## Training details

- `answer_only_labels` masks everything except the answer bytes, so loss counts only the digit the image depicts — the model cannot cheat off the prompt.
- The experiment's images are generated, not downloaded: seven-segment digits with random shifts, colors, intensity, and noise. Train and held-out sets use different rendering seeds but the same digit classes.
- The full-model pilot (`experiments/full_25m_vision_digit_pilot.py`) attaches the tower to the complete 12-layer LM from a coding checkpoint. Only the vision tower, final block, final norm, and tied embedding train; no weights are saved.

## File map

- `glm_mini/vision.py` — config, patch embedding, ViT blocks, merger, projector, wrapper, baseline, labels.
- `scripts/train_vision.py` — generated-digit training and the faithful-vs-baseline comparison.
- `experiments/full_25m_vision_digit_pilot.py` — bounded full-LM integration pilot.
- `tests/test_vision.py` — shapes, ordering, gradient flow, overfit smoke.

## Honest limitations

Synthetic digits, one fixed resolution, no video, no variable-resolution packing, tiny data. This proves the wiring works and that the LM can condition on image tokens — nothing more.
