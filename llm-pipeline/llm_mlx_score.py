# =============================================================================
# llm_score_mlx.py — MLX-based PQC Readiness Scoring
# =============================================================================
# 
# Scores PQC readiness frameworks using local LLM via MLX on Apple Silicon.
# Optimized for M4 MacBook Air with 16GB RAM.
#
# Usage:  python llm_score_mlx.py
# =============================================================================

import os
import json
import csv
import re
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from mlx_lm import load, generate

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths (relative to llm-pipeline/)
BASE_DIR = Path(__file__).parent
FREQ_ANALYSIS_DIR = BASE_DIR.parent / "freq-analysis"
PROCESSED_DIR = FREQ_ANALYSIS_DIR / "processed"
LABELS_FILE = FREQ_ANALYSIS_DIR / "scripts" / "labels.csv"
MAPPING_FILE = FREQ_ANALYSIS_DIR / "framework_mapping.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# Model configuration
MODEL_NAME = "mlx-community/Qwen2.5-7B-Instruct-4bit"  # 4-bit quantized for efficiency
MAX_INPUT_WORDS = 3000  # Truncate documents to this length
MAX_TOKENS = 150        # Max tokens for LLM response
TEMPERATURE = 0.1       # Low temperature for consistency

# Dimensions to score
DIMENSIONS = ['crypto_assets', 'crypto_agility', 'migration_planning',
              'risk_management', 'standards_compliance']

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert evaluator of Post-Quantum Cryptography (PQC) readiness frameworks. Your task is to analyze technical documents and score them on five dimensions using a 0-5 scale."""

SCORING_RUBRIC = """
**Scoring Dimensions (0-5 scale):**

1. **crypto_assets**: Cryptographic asset inventory coverage
   - 0: No mention
   - 1: Brief mention only
   - 2: Acknowledges need, minimal detail
   - 3: Describes process with some specifics (certificates, keys, lifecycle)
   - 4: Comprehensive methodology with tools, templates, classifications
   - 5: Exhaustive with automated discovery, dependency mapping, full lifecycle

2. **crypto_agility**: Cryptographic agility and algorithm flexibility
   - 0: No mention
   - 1: Brief mention of flexibility
   - 2: Discusses concept without implementation
   - 3: Modular design patterns, algorithm switching mechanisms
   - 4: Detailed architecture, hybrid approaches, abstraction layers
   - 5: Comprehensive with hot-swapping, backward compatibility, examples

3. **migration_planning**: Post-quantum migration roadmap detail
   - 0: No migration discussion
   - 1: Mentions need to migrate
   - 2: General concepts, no plan
   - 3: Phased approach with some milestones
   - 4: Detailed roadmap with resources, testing, rollback procedures
   - 5: Comprehensive with specific dates, stakeholders, step-by-step guidance

4. **risk_management**: Quantum threat risk assessment thoroughness
   - 0: No risk discussion
   - 1: Acknowledges quantum threat
   - 2: Basic threat discussion
   - 3: Risk assessment with some quantification (HNDL, timelines)
   - 4: Comprehensive matrices, vulnerability assessments, business impact
   - 5: Detailed framework with quantitative scoring, threat modeling, prioritization

5. **standards_compliance**: PQC standards and algorithm specificity
   - 0: No standards mentioned
   - 1: Generic references
   - 2: Names standards bodies (NIST, ETSI) without specifics
   - 3: References specific standards (FIPS 203/204, CNSA 2.0)
   - 4: Detailed coverage with algorithm names (ML-KEM, ML-DSA)
   - 5: Exhaustive with certification, regulatory requirements, parameters
"""

def build_scoring_prompt(text_excerpt):
    """Create the scoring prompt for a framework."""
    prompt = f"""{SYSTEM_PROMPT}

{SCORING_RUBRIC}

**Document to Evaluate:**
{text_excerpt}

**Instructions:**
Based on the document above, assign integer scores (0-5) for each dimension.
Return ONLY a valid JSON object with this exact structure, no additional text:

{{"crypto_assets": <score>, "crypto_agility": <score>, "migration_planning": <score>, "risk_management": <score>, "standards_compliance": <score>}}

JSON response:"""
    
    return prompt

# =============================================================================
# DATA LOADING
# =============================================================================

def load_expert_scores(path):
    """Load expert scores from labels.csv."""
    scores = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = str(row['Framework_ID']).strip().zfill(3)
            scores[fid] = {dim: int(row[dim]) for dim in DIMENSIONS}
    return scores

def load_framework_mapping(path):
    """Load framework ID to filename mapping."""
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = str(row['Framework ID']).strip().zfill(3)
            filename = row['PDF_Title'].strip()
            mapping[fid] = filename
    return mapping

def load_processed_text(txt_path, max_words=MAX_INPUT_WORDS):
    """Load and truncate processed text file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Truncate to max_words
        words = text.split()[:max_words]
        truncated = ' '.join(words)
        
        return truncated
    except Exception as e:
        print(f"    ❌ Error loading {txt_path}: {e}")
        return None

# =============================================================================
# LLM SCORING
# =============================================================================

def parse_llm_response(response_text):
    """Extract JSON scores from LLM response."""
    try:
        # Find JSON object in response
        json_match = re.search(r'\{[^}]+\}', response_text)
        if not json_match:
            return None
        
        json_str = json_match.group(0)
        scores = json.loads(json_str)
        
        # Validate all dimensions present and in range
        for dim in DIMENSIONS:
            if dim not in scores:
                return None
            score = int(scores[dim])
            if not (0 <= score <= 5):
                return None
            scores[dim] = score
        
        return scores
    
    except (json.JSONDecodeError, ValueError, KeyError):
        return None

def score_framework_with_llm(model, tokenizer, text, max_retries=3):
    """Score a framework using the LLM."""
    prompt = build_scoring_prompt(text)
    
    for attempt in range(max_retries):
        try:
            # Generate response
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                verbose=False
            )
            
            # Parse scores
            scores = parse_llm_response(response)
            
            if scores:
                return scores, response
            else:
                print(f"      ⚠️  Attempt {attempt + 1}: Failed to parse valid JSON")
        
        except Exception as e:
            print(f"      ⚠️  Attempt {attempt + 1}: Error - {e}")
    
    # Fallback: return middle scores if all attempts fail
    print(f"      ❌ All attempts failed, using fallback scores (2)")
    fallback = {dim: 2 for dim in DIMENSIONS}
    return fallback, "FAILED"

# =============================================================================
# EVALUATION
# =============================================================================

def compute_metrics(llm_scores, expert_scores, dimension):
    """Compute MAE and other metrics for a dimension."""
    llm_vals = [llm_scores[fid][dimension] for fid in llm_scores.keys()]
    expert_vals = [expert_scores[fid][dimension] for fid in llm_scores.keys()]
    
    mae = np.mean(np.abs(np.array(llm_vals) - np.array(expert_vals)))
    exact_match = np.mean(np.array(llm_vals) == np.array(expert_vals))
    
    return {
        'mae': mae,
        'exact': exact_match,
        'llm_mean': np.mean(llm_vals),
        'expert_mean': np.mean(expert_vals)
    }

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("  PQC READINESS — MLX LLM SCORING PIPELINE")
    print("=" * 70)
    print(f"\n  Model: {MODEL_NAME}")
    print(f"  Input: {PROCESSED_DIR}")
    print(f"  Max input words: {MAX_INPUT_WORDS}")
    
    # Load model
    print(f"\n[1/5] Loading MLX model...")
    print(f"      (First run will download model — may take a few minutes)")
    
    try:
        model, tokenizer = load(MODEL_NAME)
        print(f"      ✅ Model loaded")
    except Exception as e:
        print(f"      ❌ Failed to load model: {e}")
        print(f"\n  Troubleshooting:")
        print(f"    1. Check internet connection for first-time download")
        print(f"    2. Try: mlx_lm.convert --hf-path Qwen/Qwen2.5-3B-Instruct")
        print(f"    3. Or use: mlx-community/Llama-3.2-3B-Instruct-4bit")
        return
    
    # Load data
    print(f"\n[2/5] Loading expert scores and mappings...")
    expert_scores = load_expert_scores(LABELS_FILE)
    mapping = load_framework_mapping(MAPPING_FILE)
    
    print(f"      Expert scores: {len(expert_scores)} frameworks")
    print(f"      Mapping: {len(mapping)} frameworks")
    
    # Score frameworks
    print(f"\n[3/5] Scoring frameworks with LLM...\n")
    
    results = []
    llm_scores = {}
    
    # Process each framework
    for fid in sorted(expert_scores.keys()):
        txt_file = f"{fid}.txt"
        txt_path = PROCESSED_DIR / txt_file
        
        if not txt_path.exists():
            print(f"  [{fid}] ⚠️  Text file not found — skipping")
            continue
        
        framework_name = mapping.get(fid, txt_file)
        print(f"  [{fid}] {framework_name[:50]}...")
        
        # Load text
        text = load_processed_text(txt_path)
        if not text or len(text) < 100:
            print(f"      ⚠️  Text too short — skipping")
            continue
        
        print(f"      Loaded {len(text.split())} words, generating scores...")
        
        # Get LLM scores
        scores, raw_response = score_framework_with_llm(model, tokenizer, text)
        llm_scores[fid] = scores
        
        # Build result row
        row = {
            'Framework_ID': fid,
            'Framework': framework_name
        }
        
        # Add scores and deltas
        for dim in DIMENSIONS:
            llm_score = scores[dim]
            expert_score = expert_scores[fid][dim]
            delta = abs(llm_score - expert_score)
            
            row[f'{dim}_llm'] = llm_score
            row[f'{dim}_expert'] = expert_score
            row[f'{dim}_delta'] = delta
        
        row['raw_response'] = raw_response[:500]  # Truncate for CSV
        results.append(row)
        
        # Print summary
        summary = " | ".join(
            f"{dim[:5]}: L={scores[dim]} E={expert_scores[fid][dim]} (Δ{abs(scores[dim] - expert_scores[fid][dim])})"
            for dim in DIMENSIONS
        )
        print(f"      {summary}")
    
    # Save results
    print(f"\n[4/5] Saving results...")
    
    # CSV output
    csv_path = OUTPUT_DIR / "llm_scores.csv"
    fieldnames = ['Framework_ID', 'Framework']
    for dim in DIMENSIONS:
        fieldnames += [f'{dim}_llm', f'{dim}_expert', f'{dim}_delta']
    fieldnames.append('raw_response')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"      ✅ {csv_path}")
    
    # JSON output (cleaner format)
    json_path = OUTPUT_DIR / "llm_scores.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"      ✅ {json_path}")
    
    # Compute and display metrics
    print(f"\n[5/5] Computing metrics...\n")
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    
    overall_mae = []
    
    for dim in DIMENSIONS:
        metrics = compute_metrics(llm_scores, expert_scores, dim)
        overall_mae.append(metrics['mae'])
        
        print(f"\n  {dim.replace('_', ' ').title()}")
        print(f"    MAE:         {metrics['mae']:.3f}")
        print(f"    Exact match: {metrics['exact']:.1%}")
        print(f"    LLM mean:    {metrics['llm_mean']:.2f}")
        print(f"    Expert mean: {metrics['expert_mean']:.2f}")
    
    final_mae = np.mean(overall_mae)
    
    print(f"\n{'=' * 70}")
    print(f"  Overall MAE:    {final_mae:.3f}  {'✅' if final_mae < 1.0 else '❌'}")
    print(f"  SVR Baseline:   0.797")
    
    if final_mae < 0.797:
        improvement = ((0.797 - final_mae) / 0.797) * 100
        print(f"  Improvement:    +{improvement:.1f}% 🎉")
    else:
        regression = ((final_mae - 0.797) / 0.797) * 100
        print(f"  Regression:     -{regression:.1f}%")
    
    print(f"  Frameworks:     {len(llm_scores)}")
    print("=" * 70)
    
    # Save summary
    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"LLM Scoring Summary\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Frameworks scored: {len(llm_scores)}\n\n")
        f.write(f"Overall MAE: {final_mae:.3f}\n\n")
        for dim in DIMENSIONS:
            metrics = compute_metrics(llm_scores, expert_scores, dim)
            f.write(f"{dim}: MAE={metrics['mae']:.3f}, Exact={metrics['exact']:.1%}\n")
    
    print(f"\n  💾 Summary saved to {summary_path}")

if __name__ == '__main__':
    main()
