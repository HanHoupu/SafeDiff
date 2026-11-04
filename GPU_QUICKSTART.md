# SafeDiff — GPU quick start (Week 10)

This branch is for the Week 10 TA sync. The repository already contains the LLaDA inference path, hidden-state tail detector, trajectory analysis, and an exploratory risk controller. The immediate goal is to run a reproducible GPU smoke test and then a larger safety evaluation.

## 1. GPU sanity check

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

## 2. Install

Use the repository requirements on a CUDA-enabled machine:

```bash
git clone https://github.com/jeyadev-needhi/SafeDiff.git
cd SafeDiff
pip install -r requirements_no_av.txt
```

If the environment already has a compatible CUDA/PyTorch stack, avoid reinstalling PyTorch unnecessarily.

## 3. Model

The current inference script accepts a Hugging Face/local LLaDA model path directly. For the existing prototype, use the same LLaDA checkpoint used for the earlier 20-example experiment so that the new measurements remain comparable.

Example:

```bash
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
```

The first run will download model weights, so allow enough disk space and GPU memory.

## 4. Build the Week 10 evaluation set

```bash
python analysis/build_safety_eval.py --output analysis/datasets/safety_eval_v1.json
```

This creates a local, balanced evaluation set from:
- WildJailbreak: adversarial harmful + adversarial benign
- XSTest: safe + unsafe contrast pairs
- CARES-18K: medical safety prompts with graded harmfulness

The downloaded prompts are intentionally **not committed to GitHub**. Only the reproducible builder script is tracked.

## 5. First GPU smoke test

Start with a tiny run before launching the larger evaluation:

```bash
python models/jailbreakbench_llada.py \
  --model_path "$MODEL_PATH" \
  --attack_prompt analysis/datasets/wildjailbreak_20_refined_v3.json \
  --output_json results/gpu_smoke_test.json \
  --steps 16 \
  --gen_length 64 \
  --block_length 32 \
  --ref_tail_len 128 \
  --debug_print
```

The script auto-selects the GPU with the most free memory when CUDA is available.

## 6. Week 10 experiment order

1. GPU smoke test: verify model loading + hidden-state extraction.
2. Baseline timing: vanilla LLaDA.
3. Fast/cache timing: same prompts, same generation length.
4. Safety signal: `sp_hid_tail` at 128 tokens.
5. Risk-conditioned policy: low-risk → aggressive configuration; high-risk → conservative configuration.
6. Compare **latency + quality + safety**, not latency alone.

Do not call the exploratory 0.045201 threshold a validated threshold. It was selected on the same 20 examples and must be recalibrated on a larger held-out set.
