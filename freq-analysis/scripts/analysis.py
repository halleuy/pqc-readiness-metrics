import os
import pandas as pd

results_dir = "results/"
all_results = []

for file in os.listdir(results_dir):
    if file.endswith("_results.txt"):
        with open(results_dir + file) as f:
            lines = f.readlines()
            scores = {}
            for line in lines:
                if ":" in line:
                    key, val = line.strip().split(":", 1)
                    key = key.strip()
                    if key != "Framework":
                        scores[key] = float(val.split()[0])
                    else:
                        scores["Framework"] = val.strip()
            all_results.append(scores)

df = pd.DataFrame(all_results)
df.set_index("Framework", inplace=True)
df_normalised = df.div(df.sum(axis=1), axis=0)
df.to_csv("final_results.csv")
df_normalised.to_csv("final_results_normalised.csv")
print("Saved final results to final_results.csv")
print("Saved normalised results to final_results_normalised.csv")

dimension_variance = df_normalised.var()
print("\nDimension Variance:")
print(dimension_variance)

weights = dimension_variance / dimension_variance.sum()
print("\nDimension Weights:")
print(weights)