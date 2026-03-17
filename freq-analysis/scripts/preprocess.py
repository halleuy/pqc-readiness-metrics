import pdfplumber
import os
import re

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

input_dir = "data/"
output_dir = "processed/"

for file in os.listdir(input_dir):
    if file.endswith(".pdf"):
        raw = extract_text(input_dir + file)
        clean = clean_text(raw)

        with open(output_dir + file.replace(".pdf", ".txt"), "w") as f:
            f.write(clean)

