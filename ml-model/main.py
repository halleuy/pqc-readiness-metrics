from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

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

class NLPScorer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)

        self.ref_embeddings = {}
        for dim, text in DIMENSION_REFERENCES.items():
            self.ref_embeddings[dim] = self.model.encode(text)

    def score_document(self, text, chunk_size=512):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)

        if not chunks:
            return {dim: 0.0 for dim in DIMENSION_REFERENCES}
        
        chunk_embeddings = self.model.encode(chunks)

        scores = {}
        for dim, ref_embedding in self.ref_embeddings.items():
            similarities = cosine_similarity(
                [ref_embedding], chunk_embeddings
            )

            max_sim = np.max(similarities)
            min_sim = np.min(similarities)

            combined = 0.6 * max_sim + 0.4 * min_sim
            scores[dim] = round(float(combined), 4)

        return scores
    
    def similarity_to_score(self, similarity):
        if similarity >= 0.8:
            return 5
        elif similarity >= 0.6:
            return 4
        elif similarity >= 0.4:
            return 3
        elif similarity >= 0.2:
            return 2
        else:
            return 1
        
def load_text_for_nlp(pdf_path):
    import pdfplumber

    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    text = ' '.join(text.split())
    return text

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    data_dir = os.path.join(PROJECT_DIR,'freq-analysis', 'data')

    scorer = NLPScorer()

    results = []
    for file in sorted(os.listdir(data_dir)):
        if not file.endswith('.txt'):
            continue
        
        filepath = os.path.join(data_dir, file)
        text = load_text_for_nlp(filepath)

        similarities = scorer.score_document(text)
        dimension_scores = {
            dim: scorer.similarity_to_score(sim)
            for dim, sim in similarities.items()
        }

        results.append({
            "Framework": file,
            **{f"{dim}_similarity": sim for dim, sim in similarities.items()},
            **{f"{dim}_score": score for dim, score in dimension_scores.items()}
        })

        print(f"\n📄 {file}")
        for dim in DIMENSION_REFERENCES:
            print(f"    {dim}: similarity={similarities[dim]:.4f} → score={dimension_scores[dim]}")

    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(SCRIPT_DIR, 'nlp_scores.csv'), index=False)
    print(f"\nSaved NLP scores to nlp_scores.csv")