import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3Guard-Gen-0.6B"
INPUT = "analysis/datasets/safediff_eval_smoke.json"
OUTPUT = "analysis/datasets/safediff_eval_qwenguard.json"

rows = json.loads(Path(INPUT).read_text())

print("Loading Qwen3Guard-0.6B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

def classify(text):
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    inputs = tokenizer(
        [prompt],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

    generated = out[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    m = re.search(
        r"Safety:\s*(Safe|Unsafe|Controversial)",
        answer,
        re.I
    )

    label = m.group(1).capitalize() if m else "Unknown"
    return label, answer.strip()

counts = {
    "Safe": 0,
    "Controversial": 0,
    "Unsafe": 0,
    "Unknown": 0
}

for i, row in enumerate(rows):
    label, raw = classify(row["text"])

    # Medical prompts are NOT automatically malicious.
    # We retain the dataset/domain information for adaptive routing.
    if row["risk_tier"] == "MEDIUM":
        policy = "NORMAL"
    elif label == "Unsafe":
        policy = "SAFE"
    elif label == "Controversial":
        policy = "NORMAL"
    elif label == "Safe":
        policy = "FAST"
    else:
        policy = "SAFE"

    row["qwen_guard"] = label
    row["qwen_guard_raw"] = raw
    row["adaptive_policy"] = policy

    counts[label] += 1

    print(
        f"[{i+1:02d}/{len(rows)}] "
        f"dataset={row['risk_tier']:6s} "
        f"guard={label:13s} "
        f"policy={policy}"
    )

Path(OUTPUT).write_text(
    json.dumps(rows, indent=2, ensure_ascii=False)
)

print("\n" + "="*70)
print("QWEN3GUARD RESULTS")
print("="*70)

for k, v in counts.items():
    print(f"{k:15s}: {v}")

print("\nAdaptive routing:")
for p in ["FAST", "NORMAL", "SAFE"]:
    print(
        f"{p:10s}: "
        f"{sum(r['adaptive_policy'] == p for r in rows)}"
    )

print(f"\nSaved: {OUTPUT}")
