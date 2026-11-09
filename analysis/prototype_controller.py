import json
import statistics

SCORE_FILE = "analysis/results/refined_tail_128.json"

# Exploratory threshold from our current 20-example experiment.
# DO NOT treat this as a final validated threshold.
THRESHOLD = 0.045201

with open(SCORE_FILE, encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print("RISK-ADAPTIVE FAST-dLLM PROTOTYPE")
print("=" * 80)
print(f"Threshold = {THRESHOLD}")
print()

low = []
high = []

for x in data:
    score = x.get("sp_hid_tail")

    if score is None:
        continue

    score = float(score)

    if score >= THRESHOLD:
        risk = "HIGH"
        high.append(score)
    else:
        risk = "LOW"
        low.append(score)

    # Prototype policy
    if risk == "LOW":
        policy = {
            "mode": "FAST",
            "steps": 16,
            "block_length": 32,
            "parallel": True,
            "cache": True,
        }
    else:
        policy = {
            "mode": "SAFE",
            "steps": 32,
            "block_length": 32,
            "parallel": True,
            "cache": True,
        }

    print(
        f"id={x['id']:2d} "
        f"label={x.get('label',''):22s} "
        f"score={score:.6f} "
        f"risk={risk:4s} "
        f"policy={policy['mode']:4s} "
        f"steps={policy['steps']}"
    )

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print("LOW risk :", len(low))
print("HIGH risk:", len(high))

if low:
    print("LOW mean :", statistics.mean(low))

if high:
    print("HIGH mean:", statistics.mean(high))

print()
print("NOTE:")
print("Threshold is exploratory and selected on the same 20 examples.")
print("This is a prototype controller, not a validated safety detector.")
