import os
import csv
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'freq-analysis', 'data')
MAPPING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'framework_mapping.csv')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = 'all-MiniLM-L6-v2'
SENTENCES_PER_CHUNK = 5
RELEVANCE_THRESHOLD = 0.30

# =============================================================================
# DIMENSION DEFINITIONS — References + Keywords
# =============================================================================

DIMENSIONS = {
    'crypto_assets': {
        'high': (
            "Comprehensive cryptographic asset inventory methodology with automated discovery "
            "tools for certificates, keys, TLS configurations, VPN tunnels, PKI infrastructure, "
            "code signing, data at rest and data in transit encryption. Classification by algorithm "
            "type, key length, quantum vulnerability level, and lifecycle management. Includes "
            "specific inventory templates, asset registers, dependency mapping, and discovery procedures."
        ),
        'low': (
            "Briefly mentions encryption or cryptographic assets in passing. No inventory process, "
            "no discovery methodology, no classification system."
        ),
        # Keywords that indicate ACTIONABLE depth (not just topic mention)
        'keywords_specific': [
            'inventory', 'asset register', 'asset discovery', 'cryptographic inventory',
            'certificate management', 'key management', 'key lifecycle',
            'x.509', 'hsm', 'hardware security module', 'key store', 'keystore',
            'cryptographic module', 'pkcs', 'pki infrastructure',
            'data at rest', 'data in transit', 'data in motion',
            'tls 1.2', 'tls 1.3', 'ipsec', 'vpn tunnel',
            'algorithm type', 'key length', 'key size', 'bit key',
            'dependency map', 'crypto catalog', 'asset classification',
            'discovery tool', 'scanning', 'enumerat'
        ],
        'keywords_general': [
            'certificate', 'encryption', 'pki', 'tls', 'ssl', 'vpn',
            'key', 'cryptograph', 'cipher', 'encrypt', 'decrypt'
        ]
    },
    'crypto_agility': {
        'high': (
            "Detailed crypto-agility framework with modular architecture design patterns, algorithm "
            "switching mechanisms, hybrid classical-quantum key exchange approaches, key rotation "
            "procedures, backward compatibility strategies, cryptographic abstraction layers, API "
            "design for algorithm independence, and hot-swapping cryptographic algorithms."
        ),
        'low': (
            "Briefly mentions the concept of cryptographic agility or flexibility "
            "in cryptographic choices. No specific mechanisms or architecture patterns."
        ),
        'keywords_specific': [
            'crypto agility', 'cryptographic agility', 'crypto-agility',
            'algorithm switch', 'algorithm swap', 'hot-swap', 'hot swap',
            'hybrid mode', 'hybrid key', 'hybrid approach', 'hybrid scheme',
            'abstraction layer', 'cryptographic abstraction', 'crypto abstraction',
            'pluggable', 'modular crypto', 'modular design',
            'algorithm independence', 'algorithm negotiation',
            'backward compatibility', 'backward compatible', 'backwards compatible',
            'key rotation', 'key rollover',
            'fallback mechanism', 'graceful degradation',
            'dual mode', 'dual algorithm', 'composite key', 'composite signature'
        ],
        'keywords_general': [
            'agility', 'agile', 'modular', 'flexible', 'adaptable',
            'hybrid', 'interoperable', 'interoperability', 'wrapper'
        ]
    },
    'migration_planning': {
        'high': (
            "Comprehensive phased migration roadmap with specific milestones, timelines, resource "
            "allocation plans, rollback procedures, change management processes, testing and validation "
            "protocols, pilot deployment strategies, stakeholder coordination, and step-by-step "
            "implementation guidance for transitioning to post-quantum cryptography."
        ),
        'low': (
            "Mentions the need to eventually migrate or transition to post-quantum cryptography "
            "without any structured plan, timeline, or actionable steps."
        ),
        'keywords_specific': [
            'migration plan', 'migration roadmap', 'transition plan', 'transition roadmap',
            'phased approach', 'phased migration', 'phase 1', 'phase 2', 'phase 3',
            'milestone', 'timeline', 'target date', 'deadline', 'by 2025', 'by 2026',
            'by 2027', 'by 2028', 'by 2029', 'by 2030', 'by 2035',
            'pilot', 'proof of concept', 'poc', 'pilot deployment',
            'rollback', 'roll back', 'fallback plan', 'contingency',
            'change management', 'stakeholder',
            'resource allocation', 'budget', 'staffing', 'workforce',
            'testing protocol', 'validation', 'verification',
            'deployment plan', 'implementation plan', 'action plan',
            'prioriti', 'priority system', 'critical system'
        ],
        'keywords_general': [
            'migration', 'transition', 'roadmap', 'plan', 'phase',
            'implement', 'deploy', 'schedule', 'step'
        ]
    },
    'risk_management': {
        'high': (
            "Detailed quantum threat risk assessment framework including harvest-now-decrypt-later "
            "attack modeling, comprehensive risk matrices with likelihood and impact scoring, "
            "vulnerability assessment methodology, risk prioritization with quantitative scoring, "
            "business impact analysis, and threat timeline estimation."
        ),
        'low': (
            "Briefly acknowledges that quantum computing poses risks to current cryptography. "
            "No structured risk assessment or mitigation strategies."
        ),
        'keywords_specific': [
            'harvest now decrypt later', 'hndl', 'store now decrypt later',
            'risk assessment', 'risk analysis', 'risk evaluation',
            'risk matrix', 'risk score', 'risk rating', 'risk level',
            'threat model', 'threat assessment', 'threat analysis',
            'likelihood', 'probability', 'impact score', 'impact analysis',
            'business impact', 'business continuity',
            'vulnerability assessment', 'vulnerability scan',
            'risk prioriti', 'risk categori', 'risk classif',
            'quantitative risk', 'qualitative risk',
            'risk mitigation', 'risk treatment', 'risk response',
            'threat timeline', 'quantum timeline', 'cryptanalytically relevant',
            'mosca', 'mosca inequality', 'shelf life'
        ],
        'keywords_general': [
            'risk', 'threat', 'vulnerab', 'impact', 'assess',
            'mitigation', 'likelihood', 'exposure'
        ]
    },
    'standards_compliance': {
        'high': (
            "Specific detailed references to NIST FIPS 203 FIPS 204 FIPS 205 ML-KEM ML-DSA "
            "SLH-DSA CRYSTALS-Kyber CRYSTALS-Dilithium SPHINCS+ algorithms, CNSA 2.0 suite, "
            "ETSI quantum-safe guidelines, ISO 27001 integration, IETF post-quantum standards, "
            "regulatory compliance mapping, and certification processes."
        ),
        'low': (
            "Mentions standards or compliance in general terms without naming specific standards "
            "or algorithms."
        ),
        'keywords_specific': [
            'fips 203', 'fips 204', 'fips 205', 'fips 140',
            'ml-kem', 'ml-dsa', 'slh-dsa',
            'crystals-kyber', 'crystals kyber', 'kyber',
            'crystals-dilithium', 'crystals dilithium', 'dilithium',
            'sphincs', 'sphincs+', 'falcon',
            'frodokem', 'bike', 'hqc', 'classic mceliece',
            'cnsa 2.0', 'cnsa suite',
            'nist sp 800', 'nist special publication',
            'etsi', 'iso 27001', 'iso 27002', 'ietf',
            'rfc 8554', 'rfc 9180', 'x.509',
            'common criteria', 'fedramp', 'fedramp',
            'compliance framework', 'compliance requirement',
            'regulatory requirement', 'regulatory compliance',
            'certification', 'audit', 'accreditation',
            'nist post-quantum', 'nist pqc'
        ],
        'keywords_general': [
            'nist', 'fips', 'standard', 'compliance', 'regulation',
            'certif', 'guideline', 'requirement', 'framework'
        ]
    }
}

# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber (preferred) or PyPDF2 fallback."""
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except ImportError:
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except ImportError:
            print("  ERROR: Install pdfplumber or PyPDF2: pip install pdfplumber")
            return ""
    return text

# =============================================================================
# CHUNKING
# =============================================================================

def sentence_chunk(text, n_sentences=SENTENCES_PER_CHUNK):
    """Split text into overlapping chunks of N sentences."""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) == 0:
        return []

    chunks = []
    stride = max(1, n_sentences - 1)
    for i in range(0, len(sentences), stride):
        chunk = ' '.join(sentences[i:i + n_sentences])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

# =============================================================================
# KEYWORD COUNTING
# =============================================================================

def count_keywords(text, keyword_list):
    """Count total occurrences of keywords in text (case-insensitive)."""
    text_lower = text.lower()
    count = 0
    for kw in keyword_list:
        count += len(re.findall(re.escape(kw.lower()), text_lower))
    return count

def keyword_signals(full_text, dim_config):
    """
    Compute keyword-based signals for a dimension.
    Returns: keyword_specific (normalized), keyword_general (normalized), keyword_ratio
    """
    text_lower = full_text.lower()
    word_count = max(len(text_lower.split()), 1)

    # Count specific (actionable) keywords
    specific_count = count_keywords(full_text, dim_config['keywords_specific'])

    # Count general (topic) keywords
    general_count = count_keywords(full_text, dim_config['keywords_general'])

    # Normalize per 1000 words
    specific_density = (specific_count / word_count) * 1000
    general_density = (general_count / word_count) * 1000

    # Ratio: specific / (general + 1) — measures depth vs superficial mention
    keyword_ratio = specific_count / (general_count + 1)

    return {
        'kw_specific': round(specific_density, 4),
        'kw_general': round(general_density, 4),
        'kw_ratio': round(keyword_ratio, 4)
    }

# =============================================================================
# NLP SIGNAL COMPUTATION
# =============================================================================

def compute_nlp_signals(chunk_embeddings, high_emb, low_emb):
    """
    Compute NLP-based signals from chunk embeddings vs HIGH/LOW references.
    """
    n = len(chunk_embeddings)
    if n == 0:
        return {s: 0.0 for s in ['max_sim', 'top_k_mean', 'coverage_count',
                                   'depth', 'concentration', 'differential']}

    high_norm = high_emb / (np.linalg.norm(high_emb) + 1e-10)
    low_norm = low_emb / (np.linalg.norm(low_emb) + 1e-10)
    chunk_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-10)

    high_sims = chunk_norms @ high_norm
    low_sims = chunk_norms @ low_norm

    # Signal 1: MAX_SIM
    max_sim = float(np.max(high_sims))

    # Signal 2: TOP_K_MEAN — average of top 10% chunks (sustained quality)
    n_top = max(3, n // 10)
    top_sims = np.sort(high_sims)[-n_top:]
    top_k_mean = float(np.mean(top_sims))

    # Signal 3: COVERAGE_COUNT — absolute count of relevant chunks (not fraction!)
    # This rewards detailed documents instead of penalizing them
    relevant_mask = high_sims >= RELEVANCE_THRESHOLD
    coverage_count = float(np.sum(relevant_mask))
    # Log-scale to compress range (10 chunks vs 100 chunks)
    coverage_log = float(np.log1p(coverage_count))

    # Signal 4: DEPTH — mean sim of relevant chunks
    if np.any(relevant_mask):
        depth = float(np.mean(high_sims[relevant_mask]))
    else:
        depth = float(np.mean(high_sims))

    # Signal 5: CONCENTRATION — are top chunks clustered together?
    top_indices = np.argsort(high_sims)[-n_top:]
    if len(top_indices) > 1 and n > 1:
        index_spread = (float(np.max(top_indices)) - float(np.min(top_indices))) / n
        concentration = 1.0 - index_spread
        concentration = max(0.0, min(1.0, concentration))
    else:
        concentration = 0.5

    # Signal 6: DIFFERENTIAL — HIGH vs LOW reference matching
    if np.any(relevant_mask):
        diff_scores = high_sims[relevant_mask] - low_sims[relevant_mask]
    else:
        diff_scores = high_sims - low_sims
    raw_diff = float(np.mean(diff_scores))
    differential = (raw_diff + 0.3) / 0.6
    differential = max(0.0, min(1.0, differential))

    return {
        'max_sim': round(max_sim, 4),
        'top_k_mean': round(top_k_mean, 4),
        'coverage_log': round(coverage_log, 4),
        'depth': round(depth, 4),
        'concentration': round(concentration, 4),
        'differential': round(differential, 4)
    }

# =============================================================================
# COMBINED SCORING
# =============================================================================

def compute_combined(nlp_signals, kw_signals):
    """Combine NLP + keyword signals into a single raw score."""
    combined = (
        0.10 * nlp_signals['max_sim']
        + 0.10 * nlp_signals['top_k_mean']
        + 0.05 * min(nlp_signals['coverage_log'] / 5.0, 1.0)  # normalize log coverage
        + 0.10 * nlp_signals['depth']
        + 0.05 * nlp_signals['concentration']
        + 0.10 * nlp_signals['differential']
        + 0.25 * min(kw_signals['kw_specific'] / 10.0, 1.0)   # normalize density
        + 0.10 * min(kw_signals['kw_general'] / 30.0, 1.0)    # normalize density
        + 0.15 * min(kw_signals['kw_ratio'] / 3.0, 1.0)       # normalize ratio
    )
    return round(combined, 4)

def map_to_score(combined, floor=0.10, ceiling=0.65):
    """Linearly stretch combined score to 0-5 range."""
    normed = (combined - floor) / (ceiling - floor)
    score = normed * 5.0
    return round(max(0.0, min(5.0, score)), 2)

# =============================================================================
# MAPPING LOADER
# =============================================================================

def load_mapping(path):
    """Load framework_mapping.csv → dict {pdf_filename: framework_id}"""
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row.get('Framework ID', row.get('Framework_ID', '')).strip()
            pdf = row.get('PDF_Title', row.get('PDF_Filename', '')).strip()
            if fid and pdf:
                try:
                    fid = str(int(fid)).zfill(3)
                except ValueError:
                    pass
                mapping[pdf] = fid
    return mapping

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  PQC READINESS — HYBRID NLP + KEYWORD SCORER (v3)")
    print("=" * 70)

    # 1. Load model
    print("\n[1/5] Loading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    # 2. Encode references
    print("[2/5] Encoding HIGH / LOW reference descriptions...")
    ref_embs = {}
    for dim, config in DIMENSIONS.items():
        ref_embs[dim] = {
            'high': model.encode(config['high']),
            'low': model.encode(config['low'])
        }

    # 3. Load mapping
    mapping = load_mapping(MAPPING_FILE)

    # 4. Find PDFs
    print(f"[3/5] Scanning PDFs in: {PDF_DIR}")
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')])
    print(f"      Found {len(pdf_files)} PDFs\n")

    # 5. Score each framework
    print("[4/5] Scoring frameworks...\n")

    all_signals = ['max_sim', 'top_k_mean', 'coverage_log', 'depth',
                   'concentration', 'differential',
                   'kw_specific', 'kw_general', 'kw_ratio', 'combined']

    results = []
    details = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        fid = mapping.get(pdf_file, 'Unknown')
        short_name = pdf_file[:60]
        print(f"  [{fid:>3s}] {short_name}...")

        # Extract text
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text or len(full_text.strip()) < 100:
            print(f"       ⚠ Could not extract usable text — skipping")
            continue

        # Chunk and encode
        chunks = sentence_chunk(full_text)
        if len(chunks) < 2:
            print(f"       ⚠ Only {len(chunks)} chunk(s) — skipping")
            continue

        chunk_embs = model.encode(chunks, show_progress_bar=False, batch_size=64)
        word_count = len(full_text.split())
        print(f"       {len(chunks)} chunks, {word_count} words")

        row = {'Framework': pdf_file, 'Framework_ID': fid}
        drow = {'Framework': pdf_file, 'Framework_ID': fid}

        for dim, config in DIMENSIONS.items():
            # NLP signals
            nlp_sig = compute_nlp_signals(chunk_embs, ref_embs[dim]['high'], ref_embs[dim]['low'])

            # Keyword signals
            kw_sig = keyword_signals(full_text, config)

            # Combined
            combined = compute_combined(nlp_sig, kw_sig)

            row[f'{dim}_raw'] = combined
            row[f'{dim}_mapped'] = map_to_score(combined)

            # Store all individual signals for detailed output
            for sig_name, sig_val in nlp_sig.items():
                drow[f'{dim}_{sig_name}'] = sig_val
            for sig_name, sig_val in kw_sig.items():
                drow[f'{dim}_{sig_name}'] = sig_val
            drow[f'{dim}_combined'] = combined

        results.append(row)
        details.append(drow)

        # Print compact summary
        summary = "  ".join(
            f"{d[:5]}: {row[f'{d}_raw']:.3f}→{row[f'{d}_mapped']}"
            for d in DIMENSIONS
        )
        print(f"       {summary}")

    # 6. Save outputs
    print(f"\n[5/5] Saving outputs ({len(results)} frameworks)...")
    dim_list = list(DIMENSIONS.keys())

    # --- nlp_scores.csv ---
    fields = ['Framework', 'Framework_ID']
    for d in dim_list:
        fields += [f'{d}_raw', f'{d}_mapped']
    _write_csv(os.path.join(OUTPUT_DIR, 'nlp_scores.csv'), fields, results)

    # --- nlp_raw_scores.csv ---
    raw_fields = ['Framework', 'Framework_ID'] + [f'{d}_raw' for d in dim_list]
    _write_csv(os.path.join(OUTPUT_DIR, 'nlp_raw_scores.csv'), raw_fields, results)

    # --- nlp_detailed_features.csv ---
    det_fields = ['Framework', 'Framework_ID']
    for d in dim_list:
        for s in all_signals:
            det_fields.append(f'{d}_{s}')
    _write_csv(os.path.join(OUTPUT_DIR, 'nlp_detailed_features.csv'), det_fields, details)

    print(f"\n{'=' * 70}")
    print(f"  DONE — {len(results)} frameworks × {len(DIMENSIONS)} dimensions")
    print(f"{'=' * 70}")

def _write_csv(path, fields, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"  ✅ {os.path.basename(path)}")

if __name__ == '__main__':
    main()
