#!/usr/bin/env python3
"""
SafeDiff adaptive-acceleration prototype.

Initial routing prototype: use the labelled evaluation-set risk level as an
ORACLE routing signal to test the adaptive-allocation mechanism before
replacing the oracle with the trajectory-based risk signal.

LOW  -> FAST (16 steps)
MID  -> NORMAL (32 steps)
HIGH -> SAFE (64 steps)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def route(record):
    level = int(record.get("risk_level", record.get("label", 0)))
    if level <= 0:
        return "FAST", 16
    if level == 1:
        return "NORMAL", 32
    return "SAFE", 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="analysis/datasets/safety_eval_v1.json")
    ap.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--gen_length", type=int, default=64)
    ap.add_argument("--ref_tail_len", type=int, default=128)
    ap.add_argument("--out_dir", default="results/adaptive_prototype")
    args = ap.parse_args()

    records = json.loads(Path(args.input).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = {"FAST": [], "NORMAL": [], "SAFE": []}
    for r in records:
        policy, steps = route(r)
        x = dict(r)
        x["adaptive_policy"] = policy
        x["adaptive_steps"] = steps
        groups[policy].append(x)

    total_allocated = sum(
        len(groups[p]) * {"FAST": 16, "NORMAL": 32, "SAFE": 64}[p]
        for p in groups
    )
    uniform = len(records) * 64

    print("=" * 72)
    print("SAFEDIFF ADAPTIVE ACCELERATION PROTOTYPE")
    print("=" * 72)
    for p in ("FAST", "NORMAL", "SAFE"):
        print(f"{p:8s}: n={len(groups[p]):3d}, steps={route({'risk_level': {'FAST':0,'NORMAL':1,'SAFE':2}[p]})[1]}")
    print(f"Adaptive allocated steps: {total_allocated}")
    print(f"Uniform 64-step budget : {uniform}")
    print(f"Budget reduction       : {100*(1-total_allocated/uniform):.2f}%")

    timings = {}
    merged = []

    for policy in ("FAST", "NORMAL", "SAFE"):
        rows = groups[policy]
        if not rows:
            continue

        steps = {"FAST": 16, "NORMAL": 32, "SAFE": 64}[policy]
        inp = out_dir / f"{policy.lower()}_input.json"
        out = out_dir / f"{policy.lower()}_output.json"
        inp.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

        cmd = [
            sys.executable, "models/jailbreakbench_llada.py",
            "--model_path", args.model_path,
            "--attack_prompt", str(inp),
            "--output_json", str(out),
            "--steps", str(steps),
            "--gen_length", str(args.gen_length),
            "--block_length", "32",
            "--ref_tail_len", str(args.ref_tail_len),
        ]

        print(f"\n[{policy}] running {len(rows)} examples at {steps} steps...")
        start = time.perf_counter()
        subprocess.run(cmd, check=True)
        timings[policy] = time.perf_counter() - start

        result = json.loads(out.read_text())
        for x in result:
            x["adaptive_policy"] = policy
            x["adaptive_steps"] = steps
        merged.extend(result)
        print(f"[{policy}] wall time: {timings[policy]:.2f}s")

    merged_path = out_dir / "adaptive_results.json"
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    print("MEASURED ADAPTIVE RESULTS")
    print("=" * 72)
    print(f"Total wall time: {sum(timings.values()):.2f}s")
    for p in ("FAST", "NORMAL", "SAFE"):
        if p in timings:
            print(f"{p:8s}: {len(groups[p]):3d} examples, {timings[p]:.2f}s")
    print(f"Merged output: {merged_path}")
    print("\nNOTE: routing uses oracle dataset risk labels for this first prototype.")
    print("Next step: replace oracle routing with the SafeDiff trajectory signal.")


if __name__ == "__main__":
    main()
