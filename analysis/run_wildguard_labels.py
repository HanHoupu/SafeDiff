import json
from pathlib import Path

from wildguard import load_wildguard


INPUT = "analysis/datasets/safediff_eval_smoke.json"
OUTPUT = "analysis/datasets/safediff_eval_wildguard.json"


def main():
    rows = json.loads(Path(INPUT).read_text())

    items = [{"prompt": r["text"]} for r in rows]

    print(f"Loading WildGuard for {len(items)} prompts...")

    wg = load_wildguard(
        use_vllm=False,
        device="cuda",
        batch_size=8,
        ephemeral_model=True,
    )

    results = wg.classify(items)

    for row, result in zip(rows, results):
        row["wildguard_prompt_harmfulness"] = result.get(
            "prompt_harmfulness"
        )
        row["wildguard_parse_error"] = result.get(
            "is_parsing_error", False
        )

    Path(OUTPUT).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False)
    )

    print("\n=== SafeDiff WildGuard Results ===")

    for label in ["harmful", "unharmful"]:
        count = sum(
            r["wildguard_prompt_harmfulness"] == label
            for r in rows
        )
        print(f"{label}: {count}")

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
