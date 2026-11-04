#!/usr/bin/env python3
"""
Prepare a small, reproducible evaluation set for SafeDiff.

This is an EVALUATION-only script: it does not train or fine-tune LLaDA.
It creates balanced harmful/benign subsets from public safety benchmarks.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def take_balanced(rows, n_per_label, seed=42):
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        label = row["label"]
        groups.setdefault(label, []).append(row)
    out = []
    for label in sorted(groups):
        items = groups[label][:]
        rng.shuffle(items)
        out.extend(items[:n_per_label])
    rng.shuffle(out)
    return out


def load_wildjailbreak(n_per_label):
    # Compact HF mirror used for a quick evaluation smoke test.
    ds = load_dataset("walledai/WildJailbreak", split="train")
    rows = []
    for r in ds:
        label = str(r.get("label", "")).lower()
        prompt = r.get("prompt", "")
        if not prompt:
            continue
        if "benign" in label:
            y = "benign"
        elif "harmful" in label:
            y = "harmful"
        else:
            continue
        rows.append({"text": prompt, "label": y, "source": "WildJailbreak"})
    return take_balanced(rows, n_per_label)


def load_xstest():
    # XSTest is primarily a safe/unsafe contrast set for false-refusal evaluation.
    ds = load_dataset("Paul/XSTest", split="train")
    rows = []
    for r in ds:
        prompt = r.get("prompt") or r.get("text") or ""
        label = str(r.get("label", "")).lower()
        if not prompt:
            continue
        if label in {"safe", "benign"}:
            y = "benign"
        elif label in {"unsafe", "harmful"}:
            y = "harmful"
        else:
            continue
        rows.append({"text": prompt, "label": y, "source": "XSTest"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wild-n", type=int, default=10)
    ap.add_argument("--xstest-n", type=int, default=10)
    ap.add_argument("--out", default="analysis/datasets/safediff_eval_smoke.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    rows.extend(load_wildjailbreak(args.wild_n))
    xstest = load_xstest()
    if xstest:
        rows.extend(take_balanced(xstest, args.xstest_n, args.seed))

    # Stable IDs and explicit split metadata.
    for i, r in enumerate(rows):
        r["id"] = i

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(rows)} examples -> {out}")
    for source in sorted(set(r["source"] for r in rows)):
        subset = [r for r in rows if r["source"] == source]
        print(f"{source}: {len(subset)}")
        for label in ("harmful", "benign"):
            print(f"  {label}: {sum(r['label'] == label for r in subset)}")


if __name__ == "__main__":
    main()
