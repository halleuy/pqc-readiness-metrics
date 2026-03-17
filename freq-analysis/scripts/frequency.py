from collections import Counter
import os
from keyword_map import DIMENSIONS

def count_frequencies(text):
    words = text.split()
    word_counts = Counter(words)

    results = {}
    for dim, keywords in DIMENSIONS.items():
        results[dim] = sum(word_counts[k] for k in keywords)

    return results

input_dir = "processed/"
output_dir = "results/"

for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        with open(input_dir + file) as f:
            text = f.read()

        scores = count_frequencies(text)

        output_file = output_dir + file.replace(".txt", "_results.txt")
        
        with open(output_file, "w") as out:
            out.write(f"Framework: {file}\n")
            out.write("=" * 40 + "\n")

            for dim, score in scores.items():
                out.write(f"{dim}: {score}\n")

        print(f"Saved results to {output_file}")