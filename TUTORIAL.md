# The Beginner Tutorial: Build, Train, and Self-Improve

This is the guided lesson. Each step runs on CPU, macOS, or an NVIDIA GPU. Everything is deterministic — run it twice, get identical receipts.

## 0. Setup

```bash
cd glm-mini-from-scratch
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q              # 37 tests should pass
```

## 1. Tokens are bytes

```python
from glm_mini import ByteTokenizer

tokenizer = ByteTokenizer()
ids = tokenizer.encode("def f(x):\n    return x + 1\n", bos=True, eos=True)
print(len(ids), tokenizer.decode(ids) == "def f(x):\n    return x + 1\n")
```

Every UTF-8 byte becomes one token (plus special ids 0–3). No learned vocabulary, no `<unk>` — any text round-trips exactly. The price: sequences are long, so the model has a 192-token context. The model's whole job is predicting the **next byte**.

## 2. The model in five sentences

`glm_mini/model.py` defines the whole architecture:

1. Bytes are embedded and copied into **four residual streams**.
2. Each layer has a **hybrid attention** block: three layers of linear attention (ELU+1 features, prefix sums — O(length)), then one layer of gathered sparse attention (local window + strided anchors, true softmax).
3. Each layer also has a **mixture of experts**: a router picks top-2 of 8 SwiGLU experts per token; a shared expert is always on.
4. Streams are mixed into each block by learned weights and the block's output is routed back by learned weights.
5. A final RMSNorm plus the **tied** embedding matrix turns streams back into byte logits.

```python
import torch
from glm_mini import GLMMiniFromScratch, ModelConfig

model = GLMMiniFromScratch(ModelConfig())
print(model.parameter_counts())
# {'total': 25730592, 'active_per_token_estimate': 9805344}
```

Why fewer "active" parameters than total? Every token touches the shared expert plus 2 of 8 routed experts — that is what "mixture of experts" buys you.

## 3. The tasks and the verifier

`glm_mini/tasks.py` procedurally generates tiny Python problems from eight families (`increment`, `double`, `square`, …). Task names are random, but everything derives from integer seeds — `frozen_tasks("confirm")` always returns the same held-out tasks.

`glm_mini/evaluator.py` is fail-closed: candidate code may only define the requested function, may not import, loop, recurse, touch attributes, or call anything outside `sum, len, abs, min, max, bool, int, str`. Then hidden tests run in a restricted namespace.

```python
from glm_mini.evaluator import evaluate_source
from glm_mini.tasks import frozen_tasks

task = frozen_tasks("dev")[0]
print(evaluate_source(task, task.reference_source).status)   # passed
print(evaluate_source(task, task.prompt + "\n    pass\n").status)  # failed
```

## 4. Pretrain

```bash
run_cpu.bat scripts\train_pretrain.py --output runs\tutorial --steps 30 --checkpoints 30 --batch-size 8 --seed 42
```

Watch the printed loss fall. The script writes `training-receipt.json` next to the checkpoints — config, curve, SHA-256 of every weight file. Loss around 5.5 is uniform randomness; below 3 means the model is really learning byte-level Python.

## 5. Post-train with RLOO

Reinforcement learning with leave-one-out baselines, using executable tests as reward:

```bash
run_cpu.bat scripts\train_rl.py --initial-checkpoint runs\tutorial\checkpoint-0030 --output runs\tutorial-rl --groups 8 --checkpoints 8 --group-size 4 --families increment --tasks-per-family 2 --reward-mode binary --seed 31415
```

Per group: sample 4 completions, score them with the verifier, advantage = own reward minus the mean of the others. Groups where every reward is equal skip the update (nothing to learn from a tie).

## 6. Recursive self-improvement

Now the flagship loop — the model trains on **its own passing programs**:

```bash
run_cpu.bat scripts\train_self_improve.py --initial-checkpoint runs\tutorial\checkpoint-0030 --output runs\tutorial-rsi --cycles 2 --proposals-per-task 8 --fine-tune-steps 12 --families-ladder increment --seed 73
```

Each cycle: propose → verify → distill into cumulative memory → fine-tune → evaluate. With `--expand-on-master`, the family ladder unlocks rung by rung. Open `runs\tutorial-rsi\training-receipt.json` and compare `pass_rate_before` to `pass_rate_after` per cycle.

## 7. Look at the pictures

```bash
run_cpu.bat scripts\make_charts.py --output artifacts\charts --pretrain-receipt runs\tutorial\training-receipt.json
```

## 8. Optional: vision

```bash
run_cpu.bat scripts\train_vision.py --smoke-test --architecture both --device auto
```

A tiny GLM-style vision tower turns a 32×32 RGB digit into 16 tokens that flow through the same byte LM. See `MULTIMODAL.md`.

## 9. The slides

```bash
python slides/serve_slides.py --port 8765
```

## What to read next

- `REPORT.md` — the full experimental method and honest limitations.
- `EXAMPLES.md` — real generations before and after training.
- `VISION_REPORT.md`, `MULTIMODAL.md` — the image path.
- `glm_mini/model.py` — 257 lines; read it with this tutorial open.
