import pdfplumber
import os
import re
import csv
from framework_ids import generate_framework_id

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

input_dir = os.path.join(PROJECT_DIR, "data")
output_dir = os.path.join(PROJECT_DIR, "processed")
mapping_file = os.path.join(PROJECT_DIR, "framework_mapping.csv")

os.makedirs(output_dir, exist_ok=True)

existing_ids = set()
existing_pdfs = set()

if os.path.exists(mapping_file):
    with open(mapping_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            existing_ids.add(row["Framework ID"])
            existing_pdfs.add(row["PDF_Title"])

new_mappings = []

for file in sorted(os.listdir(input_dir)):
    if not file.endswith(".pdf"):
        continue
    if file in existing_pdfs:
        print(f"⏭️  Skipping {file} — already processed")
        continue

    fw_id = generate_framework_id(existing_ids)

    text = ""
    filepath = os.path.join(input_dir, file)
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    clean = re.sub(r'[^a-z\s]', '', text.lower())

    txt_filename = f"{fw_id}.txt"
    with open(os.path.join(output_dir, txt_filename), "w") as f:
        f.write(clean)

    new_mappings.append([fw_id, file])
    print(f"✅ Processed {file} → {txt_filename} (ID: {fw_id})")

write_header = not os.path.exists(mapping_file) or os.path.getsize(mapping_file) == 0

with open(mapping_file, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(["Framework ID", "PDF_Title"])
    for row in new_mappings:
        writer.writerow(row)

if new_mappings:
    print(f"\n📝 Added {len(new_mappings)} new frameworks to {mapping_file}")
else:
    print(f"\n📝 No new frameworks to process")

print(f"📊 Total frameworks: {len(existing_ids)}")