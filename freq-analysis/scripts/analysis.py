import os
import pandas as pd

# ─── Path Resolution ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

results_dir = os.path.join(PROJECT_DIR, "results")
processed_dir = os.path.join(PROJECT_DIR, "processed")

# ─── Step 1: Load Results ─────────────────────────────────
all_results = []

for file in sorted(os.listdir(results_dir)):
    if not file.endswith("_results.txt"):
        continue

    with open(os.path.join(results_dir, file)) as f:
        lines = f.readlines()
        scores = {}
        for line in lines:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                key = key.strip()
                if key not in ("Framework", "Total Words", "=" * 40):
                    try:
                        scores[key] = float(val.strip())
                    except ValueError:
                        continue
                elif key == "Framework":
                    scores["Framework"] = val.strip()
                elif key == "Total Words":
                    scores["Total Words"] = int(val.strip())
        all_results.append(scores)

df = pd.DataFrame(all_results)
df.set_index("Framework", inplace=True)

# Separate word counts from dimension scores
word_counts = df["Total Words"]
df = df.drop(columns=["Total Words"])

# ─── Step 2: Save Raw Results ─────────────────────────────
df.to_csv(os.path.join(SCRIPT_DIR, "final_results.csv"))
print("✅ Saved raw results to final_results.csv")

# ─── Step 3: Normalize by Document Length (per 1000 words) ─
df_normalised = df.div(word_counts, axis=0) * 1000
df_normalised.to_csv(os.path.join(SCRIPT_DIR, "final_results_normalised.csv"))
print("✅ Saved normalised results to final_results_normalised.csv")

# ─── Step 4: Print Comparison ─────────────────────────────
print(f"\n{'='*60}")
print("RAW COUNTS:")
print('='*60)
print(df)

print(f"\nDocument word counts:")
print(word_counts)

print(f"\n{'='*60}")
print("NORMALISED (per 1000 words):")
print('='*60)
print(df_normalised)

# ─── Step 5: Dimension Weights ────────────────────────────
dimension_variance = df_normalised.var()
print(f"\nDimension Variance:\n{dimension_variance}")

weights = dimension_variance / dimension_variance.sum()
print(f"\nDimension Weights:\n{weights}")

weights_df = weights.reset_index()
weights_df.columns = ["Dimension", "Weight"]
weights_df.to_csv(os.path.join(SCRIPT_DIR, "dimension_weights.csv"), index=False)
print("✅ Saved dimension weights to dimension_weights.csv")
