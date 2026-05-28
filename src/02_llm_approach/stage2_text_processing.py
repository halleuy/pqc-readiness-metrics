#!/usr/bin/env python3
import subprocess
import itertools

chunk_sizes = [1000, 1200, 2000, 3000]
max_chunks = [1, 2, 3, 5]

configs = list(itertools.product(chunk_sizes, max_chunks))
total = len(configs)

print(f"Stage 2: Testing {total} chunking configurations")
print("Winner config: 32B, temp=0.2, max aggregation")
print("=" * 60)

for i, (chunk_size, max_chunk) in enumerate(configs, 1):
    suffix = f"CS{chunk_size}_MC{max_chunk}"
    print(f"\n[{i}/{total}] ChunkSize={chunk_size}, MaxChunks={max_chunk}")
    
    cmd = [
        "python", "llm_3090_score.py",
        "--model_name", "Qwen/Qwen2.5-32B-Instruct",
        "--temperature", "0.2",
        "--aggregation", "max",
        "--chunk_size", str(chunk_size),
        "--max_chunks", str(max_chunk),
        "--output_suffix", suffix
    ]
    
    subprocess.run(cmd)

print("\n" + "=" * 60)
print("Stage 2 Complete! Analyzing results...")
print()

# Show results
import json
from pathlib import Path

results = []
for f in Path("outputs").glob("metrics_CS*.json"):
    with open(f) as file:
        data = json.load(file)
        config = f.stem.replace("metrics_", "")
        results.append((config, data["overall_mae"]))

results.sort(key=lambda x: x)

print("Results sorted by MAE:")
for config, mae in results:
    print(f"  {config:20s}: {mae:.3f}")

print()
print(f"Best: {results} with MAE={results:.3f}")
