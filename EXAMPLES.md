# EXAMPLES — real generations before and after recursive self-improvement

Every completion below is a real greedy output of checkpoints in this repository, transcribed from the evaluation payloads (`runs/probe-all-families.json`, `runs/confirm-after.json`). No cherry-picking: these are the tasks named in `REPORT.md`.

## The failing family, repaired by its own verified outputs

Task `confirm-even-000` — held out from all training and self-improvement:

```text
# Complete this Python function.
# Return whether x is even.
def even_zfsvekw(x):
```

**Before** (pretraining checkpoint-0150) — invalid code, a syntax error:

```python
    return x if x = 0 else 0
```

**After** (RSI checkpoint-cycle-03, trained on 4 self-verified programs):

```python
    return x % 2 == 0
```

The entire repair dataset was 4 completions the model itself wrote and the executable verifier itself approved.

## The price: interference on previously-passing families

Same checkpoints, same greedy decoding, task `confirm-square-000`:

**Before** — correct:

```python
    return x * x
```

**After** — invalid; the even-pattern leaked into everything:

```python
    return x % 2 ==== 0
```

Task `confirm-reverse-000`:

**Before** — correct:

```python
    return text[::-1]
```

**After** — degenerate repetition:

```python
    return text[:::::::::::::::::::::::::::::::
```

## A family that survived, then recovered

Task `confirm-list_sum-000` was broken at RSI cycle 1 and passed again at cycle 3 as the verified memory grew:

**Before** — correct:

```python
    return sum(values)
```

**Cycle 3** — correct again:

```python
    return sum(values)
```

## Scoreboard (greedy pass@1, one unseen task per family)

| Checkpoint | even | square | absolute | nonnegative | reverse | list_sum | Total |
|---|---|---|---|---|---|---|---|
| pretraining step-150 | 0 | 1 | 1 | 1 | 1 | 1 | 5/6 |
| RSI cycle 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1/6 |
| RSI cycle 3 (final) | 1 | 0 | 0 | 1 | 0 | 1 | 3/6 |

The lesson is the whole point of this repository: closed-loop self-improvement works on the target behavior and costs you elsewhere unless you engineer against interference. See `REPORT.md` for the measured ladder.
