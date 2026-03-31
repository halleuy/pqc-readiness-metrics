import subprocess
import os
import sys

# ─── Path Resolution ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable  # Uses same Python you invoked this with

# ─── Pipeline Steps ───────────────────────────────────────
steps = [
    ("Step 1: Preprocessing PDFs to TXTs with numeric IDs...", "preprocess.py"),
    ("Step 2: Counting keyword frequencies...", "frequency.py"),
    ("Step 3: Analyzing results and calculating weights...", "analysis.py"),
]

for description, script in steps:
    print(f"\n{'='*60}")
    print(description)
    print('='*60)

    script_path = os.path.join(SCRIPT_DIR, script)

    # Check script exists before running
    if not os.path.exists(script_path):
        print(f"❌ ERROR: {script_path} not found!")
        sys.exit(1)

    result = subprocess.run([PYTHON, script_path])

    if result.returncode != 0:
        print(f"❌ ERROR: {script} failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"✅ {script} completed successfully")

print(f"\n{'='*60}")
print("🎉 Pipeline completed successfully!")
print(f"{'='*60}")
print(f"\nCheck your outputs:")
print(f"  📊 Results:    {os.path.join(SCRIPT_DIR, '..', 'results')}")
print(f"  📈 Scores:     {os.path.join(SCRIPT_DIR, 'final_results.csv')}")
print(f"  📈 Normalised: {os.path.join(SCRIPT_DIR, 'final_results_normalised.csv')}")
print(f"  ⚖️  Weights:   {os.path.join(SCRIPT_DIR, 'dimension_weights.csv')}")
