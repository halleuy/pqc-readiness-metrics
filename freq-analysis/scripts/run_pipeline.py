import subprocess

print("Step 1: Preprocessing PDFs to TXTs with numeric IDs...")
subprocess.run(["python", "preprocess.py"], check=True)

print("\nStep 2: Counting keyword frequencies...")
subprocess.run(["python", "frequency.py"], check=True)

print("\nStep 3: Analyzing results and calculating weights...")
subprocess.run(["python", "analysis.py"], check=True)

print("\nPipeline completed successfully. Check the results and weights in the output files.")