# VISION_REPORT — the miniature GLM-style image path, measured

All numbers below come from real runs of this repository on CPU (torch 2.10.0). Raw receipts: `artifacts/receipts/vision-miniature-rgb.json` and `artifacts/receipts/full-25m-vision-digit-pilot.json`. Architecture details: `MULTIMODAL.md`. No dataset downloads; images are generated seven-segment digits with random shifts, colors, and noise.

## Experiment 1 — faithful tower vs direct-patch baseline

`scripts/train_vision.py --architecture both --device cpu --seed 42 --steps 120 --batch-size 40 --eval-examples 200 --learning-rate 0.002`

Both models share the same tiny language model and answer-only loss; the held-out rendering seed (10,042) is untouched by training and identical before/after.

| Architecture | Parameters | Held-out before | Held-out after | Loss | Wall time |
|---|---:|---:|---:|---|---:|
| GLM-faithful tower (2 ViT blocks → 2×2 merge → gated projector) | 60,924 | 0/200 | **109/200 (54.5%)** | 5.58 → 1.22 | 6.0 s |
| Direct-patch baseline (patches straight to LM tokens) | 42,500 | 0/200 | **187/200 (93.5%)** | 5.58 → 0.89 | 3.1 s |

The clearly labeled, *not* GLM-faithful baseline wins this tiny short-run comparison, as it did upstream. This does not show the production topology is worse — it shows that added structure did not pay off under a 120-step budget at teaching scale, and the repository refuses to pretend otherwise.

## Experiment 2 — full 25.7M language-model integration pilot

`experiments/full_25m_vision_digit_pilot.py --checkpoint runs/pretrain-001/checkpoint-0100 --device cpu --steps 40 --batch-size 10 --eval-examples 100 --learning-rate 0.001 --seed 73`

The complete 12-layer hybrid-attention MoE language model was rebuilt at vocab 263 and loaded from the step-100 coding checkpoint. Shape-matched tensors were copied; the tied embedding/output matrix was not (260 → 263 rows) and was trained from its saved initialization instead.

| Metric | Value |
|---|---|
| Tensors copied from the coding checkpoint | 301 |
| Tensors skipped (shape mismatch) | `embedding.weight`, `output.weight` |
| Trainable surface | 2,298,272 / 25,838,712 parameters |
| Trainable surfaces | vision tower, last LM block, final norm, tied embedding |
| Held-out accuracy | 0/100 → **45/100** |
| Held-out cross-entropy | 5.70 → **1.43** |
| Steps / early stop | 40 / no |
| Weights saved | none, by design |

This proves the integration wiring end to end: image tokens flow through every LM layer and the frozen coding layers still support answer-only next-byte learning. It does not establish general vision ability.

## Component audit vs the released model

| Released GLM-5.3-Flash vision | This miniature |
|---|---|
| 448×448 RGB, temporal patching | fixed 32×32 RGB, single frame |
| 24 vision blocks, width 1024, 16 heads | 2 blocks, width 24, 4 heads |
| 2D rotary positions | learned position table |
| 2×2 learned downsampler | 2×2 rearrange + linear (equivalent) |
| Gated projector to the 4096-wide LM | same structure to the 192-wide LM |
| Variable-resolution token packing | none |

## Limitations

Synthetic digits only, one resolution, one seed per comparison, tiny budgets, and a benchmark where the simpler baseline wins. The value of these experiments is the working, measured wiring — not a claim about vision quality.
