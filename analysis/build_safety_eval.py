#!/usr/bin/env python3
"""Build a local, balanced Week 10 safety evaluation set.

No dataset examples are written to the repository. The script downloads the
public research datasets locally and writes a JSON file for experiments.

This is an evaluation manifest, not a training set. We deliberately do not
train a safety classifier here: doing so would change the current
training-free claim of SafeDiff. A learned probe can be a separate ablation.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


SEED = 42


def take_rows(rows, n, rng):
    rows = list(rows)
    if len(rows) < n:
        raise RuntimeError(f"Requested {n} rows but only {len(rows)} are available.")
    return rng.sample(rows, n)


def normalize_wild(row):
    label = str(row.get("label", "")).lower()
    data_type = str(row.get("data_type", "")).lower()
    prompt = row.get("prompt") or row.get("adversarial") or row.get("vanilla") or ""
    text = prompt.strip()

    if "benign" in label or "benign" in data_type:
        y = 0
        subset = "adversarial_benign"
    elif "harmful" in label or "harmful" in data_type:
        y = 1
        subset = "adversarial_harmful"
    else:
        return None

    return {
        "source": "WildJailbreak",
        "subset": subset,
        "label": y,
        "risk_level": y,
        "prompt": text,
    }


def load_wild(rng):
    # Official WildJailbreak evaluation configuration.
    try:
        ds = load_dataset("allenai/wildjailbreak", "eval", split="train")
    except Exception:
        # Small public mirror useful for a quick smoke run.
        ds = load_dataset("walledai/WildJailbreak", split="train")

    harmful = []
    benign = []
    for row in ds:
        item = normalize_wild(row)
        if item is None or not item["prompt"]:
            continue
        (harmful if item["label"] == 1 else benign).append(item)

    return take_rows(harmful, 20, rng) + take_rows(benign, 20, rng)


def load_xstest(rng):
    ds = load_dataset("Paul/XSTest", split="train")

    safe = []
    unsafe = []
    for row in ds:
        prompt = str(row.get("prompt", "")).strip()
        label = str(row.get("label", "")).lower()
        if not prompt:
            continue
        item = {
            "source": "XSTest",
            "subset": str(row.get("type", "")),
            "label": 0 if label == "safe" else 1,
            "risk_level": 0 if label == "safe" else 1,
            "prompt": prompt,
        }
        (safe if item["label"] == 0 else unsafe).append(item)

    return take_rows(safe, 10, rng) + take_rows(unsafe, 10, rng)


def load_cares(rng):
    ds = load_dataset("HFXM/CARES-18K", split="train")

    safe = []
    risky = []
    for row in ds:
        prompt = str(row.get("prompt", "")).strip()
        level = int(row.get("harmful_level", 0))
        if not prompt:
            continue

        item = {
            "source": "CARES-18K",
            "subset": str(row.get("method", "")),
            "label": 0 if level == 0 else 1,
            "risk_level": level,
            "prompt": prompt,
            "medical_principle": row.get("principle_index"),
        }

        if level == 0:
            safe.append(item)
        elif level >= 2:
            # Keep the quick experiment focused on clearly risky medical inputs.
            risky.append(item)

    return take_rows(safe, 10, rng) + take_rows(risky, 10, rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="analysis/datasets/safety_eval_v1.json",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)

    records = []
    records.extend(load_wild(rng))
    records.extend(load_xstest(rng))
    records.extend(load_cares(rng))

    rng.shuffle(records)

    for i, record in enumerate(records):
        record["id"] = i

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print("SafeDiff Week 10 evaluation set")
    print("=" * 72)
    print(f"Total examples: {len(records)}")
    for source in sorted({r["source"] for r in records}):
        subset = [r for r in records if r["source"] == source]
        print(
            f"{source:16s} n={len(subset):2d} "
            f"safe={sum(r['label'] == 0 for r in subset):2d} "
            f"risky={sum(r['label'] == 1 for r in subset):2d}"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
