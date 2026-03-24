import pdfplumber
import os
import re
import csv
from framework_ids import generate_framework_id

input_dir = "../data/"
output_dir = "../processed/"
os.makedirs(output_dir, exist_ok=True)

existing_ids = set()
for file in os.listdir(input_dir):
    if file.endswith(".pdf"):
        fw_id = generate_framework_id(existing_ids)

mapping_file = "framework_mapping.csv"

with open(mapping_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Framework ID", "PDF_Title"])

    for file in os.listdir(input_dir):
        if file.endswith(".pdf"):
            fw_id = generate_framework_id(existing_ids)

        text = ""
        with pdfplumber.open(os.path.join(input_dir, file)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        
        clean = re.sub(r'[^a-z\s]', '', text.lower())

        txt_filename = f"{fw_id}.txt"
        with open(os.path.join(output_dir, txt_filename), "w") as f:
            f.write(clean)

        writer.writerow([fw_id, file])

        print(f"Processed {file} as {txt_filename} with ID {fw_id}.")

print(f"Framework mapping saved to {mapping_file}.")