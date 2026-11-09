#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3Guard-Gen-0.6B"

print("=" * 70)
print("SafeDiff Adaptive Acceleration - Interactive Prototype")
print("=" * 70)
print(f"Loading {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()

def classify(prompt):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return response


def get_policy(result):
    r = result.lower()

    if "unsafe" in r:
        return "HIGH", "SAFE", 64

    if "controversial" in r:
        return "MEDIUM", "NORMAL", 32

    if "safe" in r:
        return "LOW", "FAST", 16

    return "UNKNOWN", "NORMAL", 32


while True:

    print()
    prompt = input("Enter prompt (or 'exit'): ").strip()

    if prompt.lower() == "exit":
        break

    if not prompt:
        continue

    print()
    print("Analyzing risk...")

    raw = classify(prompt)

    risk, policy, steps = get_policy(raw)

    baseline = 64
    reduction = 100.0 * (1.0 - steps / baseline)

    print()
    print("=" * 70)
    print("SAFEDIFF ADAPTIVE CONTROLLER")
    print("=" * 70)

    print(f"Prompt              : {prompt}")
    print()
    print(f"Safety model        : Qwen3Guard-0.6B")
    print(f"Guard output        : {raw}")
    print(f"Risk level          : {risk}")
    print()
    print(f"Selected policy     : {policy}")
    print(f"Adaptive steps      : {steps}")
    print(f"Baseline steps      : {baseline}")
    print(f"Allocated reduction : {reduction:.1f}%")
    print()
    print("LLaDA generation    : NOT RUN")
    print("Reason              : LLaDA compatibility patch pending")
    print("=" * 70)
