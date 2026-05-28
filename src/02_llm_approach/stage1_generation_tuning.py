#!/usr/bin/env python3
import subprocess
import itertools

temps = [0.0, 0.2, 0.4, 0.6]
aggs = ["max", "mean", "top2_mean"]

configs = list(itertools.product(temps, aggs))
total = len(configs)

print(f"Stage 1: Testing {total} configurations")
print("=" * 50)

for i, (temp, agg) in enumerate(configs, 1):
    suffix = f"T{temp}_{agg}"
    print(f"\n[{i}/{total}] Temp={temp}, Agg={agg}")
    
    cmd = [
        "python", "llm_3090_score.py",
        "--model_name", "Qwen/Qwen2.5-32B-Instruct",
        "--temperature", str(temp),
        "--aggregation", agg,
        "--chunk_size", "1200",
        "--max_chunks", "3",
        "--output_suffix", suffix
    ]
    
    subprocess.run(cmd)

print("\n" + "=" * 50)
print("Stage 1 Complete!")
