# Two-mode acceleration benchmark — 2026-11-05

At the tested settings, `instant` generated responses 3.53× faster on GSM8K
and 3.93× faster on MBPP than the integrated Fast-dLLM v1 dual-cache baseline,
with lower task accuracy.

## Run setup

- SafeDiff commit: `dd10902b12fe2c89e02d9d766a7c2ab4f458aed5`, before the
  three-policy adapter and vanilla `llada` mode were added. Evaluation covered
  `instant` and `fastdllm`; `SAFE` remains unevaluated on LLaDA-8B.
- Model: `GSAI-ML/LLaDA-8B-Instruct`, checkpoint revision
  `08b83a6feb34df1a6011b80c3c00c7563e963b07`. Both modes used the same checkpoint,
  tokenizer, exact rendered prompt tokens, and shared extended model.
- Each benchmark used one NVIDIA A100-SXM4-40GB host for both modes; the two
  benchmarks used separate hosts. Batch size 1, temperature 0, seed 42,
  bfloat16, PyTorch 2.4.0/CUDA 12.4, Transformers 4.49.0, and SDPA.
- `instant`: contiguous ALL8, K=8, block length 32, threshold 0.9, BERM mu 16,
  at most 64 forwards per group. `fastdllm`: the pinned official v1 dual-cache
  decoder, block length 32, threshold 0.9, `steps=gen_length`, no BERM.
  The baseline runs the upstream decoder through SafeDiff's shared extended
  model implementation.
- GSM8K: all 1,319 official test questions, five training examples per prompt,
  fixed few-shot sampling seed 1234, generation canvas 256 tokens. MBPP: the
  original standard 500-question test split (task IDs 11–510), three examples
  (IDs 2, 3, 4), all three public tests in each prompt, canvas 1,024 tokens.
  Both used the Instruct chat template.

## Results

| Benchmark | Mode | Primary score | GSM flexible diagnostic | Generation minutes | Actual NFE |
|---|---|---:|---:|---:|---:|
| GSM8K | `instant` | 316/1319 — 23.96% | 727/1319 — 55.12% | 27.68 | 29,981 |
| GSM8K | `fastdllm` | 481/1319 — 36.47% | 1040/1319 — 78.85% | 97.59 | 113,293 |
| MBPP | `instant` | 73/500 — 14.60% | — | 21.39 | 18,392 |
| MBPP | `fastdllm` | 181/500 — 36.20% | — | 84.06 | 59,725 |

GSM8K uses strict `#### ` answer extraction for the primary score and flexible
numeric extraction as a diagnostic. The `instant` accuracy gaps are −12.51
percentage points under strict extraction and −23.73 points under flexible
extraction. Scores follow the frozen prompting and extraction protocol above.

MBPP measures one generated candidate per task passing all three original
assertions, with the supplied setup code. Its primary accuracy gap is −21.60
points. Canonical reference solutions passed 500/500 in the same scorer.
The original prompt for `mbpp_493` required 4,332 input tokens; adding the
1,024-token canvas exceeded the 4,096-token context. Both modes retained the
original prompt and recorded an input-overflow failure. The score denominator
is **500**; timing covers **499 actual generation pairs**.

All actual generations finished with zero residual MASK tokens and contained
an EOS/EOT, so raw and MASK-adjusted scores coincide. Execution failures were
retained, including two `instant` and five `fastdllm` MBPP workers that exited
without a verdict (return code −9); those candidates counted as failures.

## Timing and coverage

Speed ratios are total `fastdllm` generation time divided by total `instant`
generation time on identical, same-host pairs: 1,319 pairs for GSM8K and 499
for MBPP. CUDA-synchronized generation time includes each decoder's observers
and diagnostics; it excludes loading, warmup, tokenization, text decoding,
scoring, and result-file I/O. NFE is the number of model forward calls.

Coverage is a single seed, fixed presets and output budgets, and same-host
pairs on A100-SXM4-40GB. The `FAST`/`NORMAL`/`SAFE` API mapping was added after
this run. Safety evaluation and a full SAFE/LLaDA-8B benchmark remain pending.
The shared model includes position/cache extensions; numerical equivalence
to the original LLaDA model on a real checkpoint remains unverified.

## Evidence identity

Values above were transcribed from the archived `final-report/summary.json`
(created `2026-11-05T03:36:00.137197+00:00`), SHA256
`bfb2e04ae8690174938149a6520416902a46acb7fe107d03ed94064d3ea37209`.
The full report, per-question outputs, frozen inputs, scorer acceptance records,
and original archives were retained in the campaign delivery package.

| Final archive | SHA256 |
|---|---|
| `gsm-final-20261105T032434.tar.gz` | `dea827142a64b2bbd749d2ad1130798d62c643a5a55c02bd60b45fa4edba7c8b` |
| `mbpp_retry-final-20261105T032115.tar.gz` | `4b96afa31a5645db0da9f1ced1ac451afac4727dbe62c8670f2d2a718ea5ce02` |
