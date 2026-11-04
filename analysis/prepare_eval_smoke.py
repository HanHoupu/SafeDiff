#!/usr/bin/env python3
"""
Build a small, labelled evaluation set for the SafeDiff adaptive-acceleration prototype.

Evaluation only: no model is trained or fine-tuned.

Risk tiers:
  LOW    = ordinary safe prompts
  MEDIUM = benign medical prompts (benign, but higher consequence if mishandled)
  HIGH   = unsafe/adversarial prompts
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def sample_rows(rows, n, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_xstest(n, seed):
    ds = load_dataset("Paul/XSTest", split="train")
    safe, unsafe = [], []

    for r in ds:
        text = (r.get("prompt") or "").strip()
        label = str(r.get("label", "")).lower()
        if not text:
            continue
        if label == "safe":
            safe.append({
                "text": text,
                "source": "XSTest",
                "source_label": "safe",
                "risk_tier": "LOW",
            })
        elif label == "unsafe":
            unsafe.append({
                "text": text,
                "source": "XSTest",
                "source_label": "unsafe",
                "risk_tier": "HIGH",
            })

    return sample_rows(safe, n, seed), sample_rows(unsafe, n, seed + 1)


def load_telemedicine(n, seed):
    ds = load_dataset(
        "alpha-max/telemedicine_adverserial_dataset",
        split="train",
    )
    safe, adversarial = [], []

    for r in ds:
        text = (r.get("prompt") or "").strip()
        if not text:
            continue
        try:
            label = int(r.get("label"))
        except (TypeError, ValueError):
            continue

        rec = {
            "text": text,
            "source": "Telemedicine-Adversarial",
            "source_label": str(label),
            "category": r.get("category"),
        }

        if label == 0:
            # Benign medical != malicious. We assign MEDIUM because
            # medical requests are higher-consequence workloads.
            rec["risk_tier"] = "MEDIUM"
            safe.append(rec)
        elif label == 1:
            rec["risk_tier"] = "HIGH"
            adversarial.append(rec)

    return (
        sample_rows(safe, n, seed + 2),
        sample_rows(adversarial, n, seed + 3),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10,
                    help="Examples per class/source; default 10.")
    ap.add_argument("--out", default="analysis/datasets/safediff_eval_smoke.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    x_safe, x_unsafe = load_xstest(args.n, args.seed)
    med_safe, med_adv = load_telemedicine(args.n, args.seed)

    rows = x_safe + med_safe + x_unsafe + med_adv
    random.Random(args.seed).shuffle(rows)

    for i, row in enumerate(rows):
        row["id"] = i

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                    encoding="utf-8")

    print("=" * 70)
    print("SafeDiff ADAPTIVE-ACCELERATION SMOKE SET")
    print("=" * 70)
    print(f"Total: {len(rows)}")
    for tier in ("LOW", "MEDIUM", "HIGH"):
        print(f"{tier:7s}: {sum(r['risk_tier'] == tier for r in rows)}")
    print()
    for source in sorted(set(r["source"] for r in rows)):
        print(f"{source}: {sum(r['source'] == source for r in rows)}")
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
