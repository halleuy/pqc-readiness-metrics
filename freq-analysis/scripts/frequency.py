import os
import re
from keyword_map import DIMENSIONS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

def count_frequencies(text):
    text_lower = text.lower()

    results = {}
    for dim, keywords in DIMENSIONS.items():
        dim_count = 0
        for keyword in keywords:
            matches = re.findall(
                r'\b' +re.escape(keyword.lower()) + r'\b',
                text_lower
            )
            dim_count += len(matches)
        results[dim] = dim_count

    return results

input_dir = os.path.join(PROJECT_DIR, "processed/")
output_dir = os.path.join(PROJECT_DIR, "results/")

def get_word_count(text):
    return len(text.split())

os.makedirs(output_dir, exist_ok=True)

for file in sorted(os.listdir(input_dir)):
    if not file.endswith(".txt"):
        continue
    if file[:3].isdigit() and len(file.split('.')) == 3:
        continue

    filepath = os.path.join(input_dir, file)

    with open(filepath) as f:
        text = f.read()

    scores = count_frequencies(text)
    word_count = get_word_count(text)

    output_file = os.path.join(output_dir, file.replace(".txt", "_results.txt"))

    with open(output_file, "w") as out:
        out.write(f"Framework: {file}\n")
        out.write(f"Total Words: {word_count}\n")
        out.write("=" * 40 + "\n")
        for dim, score in scores.items():
            out.write(f"{dim}: {score}\n")

        print(f"✅ {file}")
        print(f"   Words: {word_count}")
        print(f"   Scores: {scores}")
        print()

print("Frequency analysis completed. Results saved in 'results/' directory.")