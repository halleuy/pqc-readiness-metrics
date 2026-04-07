from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pandas as pd

DIMENSION_REFERENCES = {
    "crypto_assets": """
        A comprehensive cryptographic asset inventory that identifies and catalogs
        all cryptographic algorithms, keys, certificates, TLS configurations,
        VPN implementations, and data protection mechanisms currently in use.
        Covers both data at rest and data in transit. Maps cryptographic
        dependencies across systems and applications.
    """,

    "crypto_agility": """
        Detailed treatment of cryptographic agility including the ability to
        rapidly switch between cryptographic algorithms without significant
        system redesign. Covers algorithm substitution, key rotation mechanisms,
        hybrid cryptographic approaches combining classical and post-quantum
        algorithms, and backward compatibility during migration.
    """,

    "migration_planning": """
        A structured migration plan and roadmap for transitioning from classical
        to post-quantum cryptography. Includes phased implementation strategy,
        timeline, milestones, resource allocation, deployment priorities,
        testing procedures, and rollback mechanisms. Addresses organizational
        change management for PQC transition.
    """,

    "risk_management": """
        Comprehensive risk assessment addressing quantum computing threats to
        current cryptographic systems. Covers harvest now decrypt later attacks,
        vulnerability identification, threat modeling specific to quantum
        adversaries, risk prioritization, and mitigation strategies for
        cryptographic vulnerabilities.
    """,

    "standards_compliance": """
        Alignment with NIST post-quantum cryptography standards including
        CRYSTALS-Kyber, CRYSTALS-Dilithium, FALCON, and SPHINCS+. References
        regulatory requirements, compliance frameworks, industry guidelines,
        and government mandates for PQC transition.
    """
}

DIMENSIONS = list(DIMENSION_REFERENCES.keys())

def load_framework_mapping(mapping_path):
    """Load framework_mapping.csv and return {pdf_filename: framework_id} dict."""
    mapping_df = pd.read_csv(mapping_path)
    # Column names: "Framework ID" and "PDF_Title"
    mapping = {}
    for _, row in mapping_df.iterrows():
        pdf_title = row['PDF_Title'].strip()
        framework_id = int(row['Framework ID'])
        mapping[pdf_title] = framework_id
    return mapping

class NLPScorer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)

        self.ref_embeddings = {}
        for dim, text in DIMENSION_REFERENCES.items():
            self.ref_embeddings[dim] = self.model.encode(text)

        # Calibration parameters (loaded from calibration_params.json)
        self.cal_params = {}
        self.calibrated = False

    def score_document(self, text, chunk_size=512):
        """Returns raw cosine similarity scores per dimension."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        if not chunks:
            return {dim: 0.0 for dim in DIMENSIONS}

        chunk_embeddings = self.model.encode(chunks)

        scores = {}
        for dim, ref_embedding in self.ref_embeddings.items():
            similarities = cosine_similarity(
                [ref_embedding], chunk_embeddings
            )  # flatten to 1D

            max_sim = float(np.max(similarities))
            mean_sim = float(np.mean(similarities))
            top3_mean = float(np.mean(sorted(similarities, reverse=True)[:3]))

            # Combined score: top-3 mean captures depth better than max+min
            combined = 0.5 * top3_mean + 0.3 * max_sim + 0.2 * mean_sim
            scores[dim] = round(combined, 4)

        return scores

    def similarity_to_score(self, similarity, dimension):
        """Map raw similarity to 0-5 using calibrated linear model."""
        if self.calibrated and dimension in self.cal_params:
            p = self.cal_params[dimension]
            raw_score = p['slope'] * similarity + p['intercept']
            return int(np.clip(round(raw_score), 0, 5))
        else:
            # Fallback: default thresholds (rough, uncalibrated)
            thresholds = [0.15, 0.25, 0.35, 0.45, 0.55]
            for i, t in enumerate(thresholds):
                if similarity < t:
                    return i
            return 5

    def load_calibration(self, calibration_path):
        """Load learned linear regression parameters from JSON."""
        import json
        with open(calibration_path, 'r') as f:
            params = json.load(f)

        self.cal_params = params
        self.calibrated = True
        print(f"✅ Loaded calibration from {calibration_path}")
        for dim, p in params.items():
            print(f"   {dim}: score = {p['slope']:.2f} × raw + {p['intercept']:.2f}")
        print()

def load_text_for_nlp(pdf_path):
    """Extract text from PDF using pdfplumber."""
    import pdfplumber

    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    text = ' '.join(text.split())
    return text

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    data_dir = os.path.join(PROJECT_DIR, 'freq-analysis', 'data')
    calibration_path = os.path.join(SCRIPT_DIR, 'calibration_params.json')
    mapping_path = os.path.join(SCRIPT_DIR, 'framework_mapping.csv')

    # ---- Load framework mapping ----
    if os.path.exists(mapping_path):
        filename_to_id = load_framework_mapping(mapping_path)
        print(f"✅ Loaded framework mapping ({len(filename_to_id)} frameworks)\n")
    else:
        print(f"❌ framework_mapping.csv not found at: {mapping_path}")
        print("   Place it in the ml-model/ directory.")
        exit(1)

    scorer = NLPScorer()

    # ---- Load calibration if available ----
    if os.path.exists(calibration_path):
        scorer.load_calibration(calibration_path)
    else:
        print("⚠️  No calibration file found. Using default thresholds.")
        print("   Run calibrate.py first for accurate scores.\n")

    results = []
    for file in sorted(os.listdir(data_dir)):
        if not file.endswith('.pdf'):
            continue

        # Look up framework ID from mapping
        framework_id = filename_to_id.get(file)
        if framework_id is None:
            print(f"⚠️  Skipping {file} — not found in framework_mapping.csv")
            continue

        filepath = os.path.join(data_dir, file)
        text = load_text_for_nlp(filepath)

        # Raw similarity scores
        similarities = scorer.score_document(text)

        # Mapped to 0-5 (using calibrated or default thresholds)
        dimension_scores = {
            dim: scorer.similarity_to_score(sim, dim)
            for dim, sim in similarities.items()
        }

        total = sum(dimension_scores.values())

        results.append({
            "Framework_ID": framework_id,
            "Framework": file,
            # Raw scores (for calibration)
            **{f"{dim}_raw": sim for dim, sim in similarities.items()},
            # Mapped scores (0-5)
            **{f"{dim}_score": score for dim, score in dimension_scores.items()},
            "total_score": total
        })

        print(f"\n📄 [{framework_id:03d}] {file}")
        for dim in DIMENSIONS:
            print(f"    {dim}: raw={similarities[dim]:.4f} → score={dimension_scores[dim]}")
        print(f"    TOTAL: {total}/25")

    results_df = pd.DataFrame(results).sort_values('Framework_ID')

    # Save full results (raw + scored)
    results_df.to_csv(os.path.join(SCRIPT_DIR, 'nlp_scores.csv'), index=False)

    # Save just raw scores for calibration input
    raw_cols = ['Framework_ID', 'Framework'] + [f"{dim}_raw" for dim in DIMENSIONS]
    results_df[raw_cols].to_csv(
        os.path.join(SCRIPT_DIR, 'nlp_raw_scores.csv'), index=False
    )

    print(f"\n✅ Saved nlp_scores.csv and nlp_raw_scores.csv")
