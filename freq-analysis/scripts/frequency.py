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
for file in os.listdir(input_dir):
    if file.endswith(".txt"):
        with open(input_dir + file) as f:
            text = f.read()

        scores = count_frequencies(text)
        print(file, scores)