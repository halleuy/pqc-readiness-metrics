# =============================================================================
# llm_score_cuda.py — CUDA-based PQC Readiness Scoring (RTX 3060)
# =============================================================================
# 
# Scores PQC readiness frameworks using local LLM via transformers + CUDA.
# Optimized for RTX 3060 with 8GB VRAM on Windows.
#
# Setup:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#   pip install transformers accelerate bitsandbytes scipy scikit-learn
#
# Environment:
#   Set HF_TOKEN environment variable (do NOT hardcode in script!)
#   Windows: setx HF_TOKEN "your_token_here"
#   Or create .env file
#
# Usage:  python llm_score_cuda.py
# =============================================================================

import os
import json
import csv
import re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Security: Load token from environment (REQUIRED)
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    print("❌ ERROR: HF_TOKEN environment variable not set!")
    print("   Run: setx HF_TOKEN \"your_token_here\"")
    print("   Then restart terminal and try again.")
    exit(1)

# Paths (relative to llm-pipeline/)
BASE_DIR = Path(__file__).parent
FREQ_ANALYSIS_DIR = BASE_DIR.parent / "freq-analysis"
PROCESSED_DIR = FREQ_ANALYSIS_DIR / "processed"
LABELS_FILE = FREQ_ANALYSIS_DIR / "scripts" / "labels.csv"
MAPPING_FILE = FREQ_ANALYSIS_DIR / "framework_mapping.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# Model configuration (optimized for 8GB VRAM)
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"  # or "mistralai/Mistral-7B-Instruct-v0.3"
USE_4BIT = True                          # MUST be True for 8GB VRAM
MAX_INPUT_TOKENS = 3000                  # Context window limit
MAX_NEW_TOKENS = 150                     # Response length
TEMPERATURE = 0.0                        # Low for consistency
TOP_P = 0.9

# Chunking configuration
USE_CHUNKING = True                      # Recommended for better coverage
CHUNK_SIZE = 1500                        # Words per chunk
MAX_CHUNKS = 2                           # Evaluate top N chunks per framework
AGGREGATION = 'max'                      # 'max', 'mean', or 'top2_mean'

# Retry configuration
MAX_RETRIES = 2

# Dimensions to score
DIMENSIONS = ['crypto_assets', 'crypto_agility', 'migration_planning',
              'risk_management', 'standards_compliance']

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert evaluator of Post-Quantum Cryptography (PQC) readiness frameworks. 
Your task is to analyze technical documents and assign scores (0-5) based ONLY on explicit evidence in the text.
Do not infer capabilities that are not clearly stated. When evidence is ambiguous, assign the lower score."""

SCORING_RUBRIC = """
**Scoring Scale (0-5):**

**crypto_assets** — Cryptographic Asset Inventory Coverage
- 0: No mention of cryptographic assets or inventory
- 1: Brief mention only (e.g., "identify crypto assets")
- 2: Acknowledges need, minimal detail
- 3: Describes process with specifics (certificates, keys, algorithms, lifecycle)
- 4: Comprehensive methodology with tools, templates, classifications, dependencies
- 5: Exhaustive with automated discovery, dependency mapping, full lifecycle management

**crypto_agility** — Cryptographic Agility and Algorithm Flexibility
- 0: No mention of agility or algorithm flexibility
- 1: Brief mention of flexibility or modularity
- 2: Discusses concept without implementation details
- 3: Modular design patterns, algorithm switching mechanisms described
- 4: Detailed architecture, hybrid approaches, abstraction layers, versioning
- 5: Comprehensive with hot-swapping, backward compatibility, concrete examples

**migration_planning** — Post-Quantum Migration Roadmap Detail
- 0: No migration discussion
- 1: Mentions need to migrate to PQC
- 2: General concepts, no concrete plan
- 3: Phased approach with some milestones or stages
- 4: Detailed roadmap with resources, testing phases, rollback procedures, timelines
- 5: Comprehensive step-by-step guidance with dates, stakeholders, contingencies

**risk_management** — Quantum Threat Risk Assessment Thoroughness
- 0: No risk or threat discussion
- 1: Acknowledges quantum threat exists
- 2: Basic threat discussion without quantification
- 3: Risk assessment with some quantification (HNDL, timelines, threat actors)
- 4: Comprehensive matrices, vulnerability assessments, business impact analysis
- 5: Detailed framework with quantitative scoring, threat modeling, prioritization matrix

**standards_compliance** — PQC Standards and Algorithm Specificity
- 0: No standards mentioned
- 1: Generic references to "standards" or "best practices"
- 2: Names standards bodies (NIST, ETSI, ISO) without specific documents
- 3: References specific standards (FIPS 203/204, CNSA 2.0, SP 800-208)
- 4: Detailed coverage with algorithm names (ML-KEM, ML-DSA, SLH-DSA)
- 5: Exhaustive with certification requirements, regulatory compliance, parameter sets
"""

def build_scoring_prompt(text_excerpt):
    """Create the scoring prompt for a framework."""
    prompt = f"""{SYSTEM_PROMPT}

{SCORING_RUBRIC}

**Document Excerpt:**
{text_excerpt}

**Task:**
Based ONLY on the evidence in the document above, assign integer scores (0-5) for each dimension.

Return ONLY a valid JSON object with this exact structure (no markdown, no additional text):

{{"crypto_assets": <int>, "crypto_agility": <int>, "migration_planning": <int>, "risk_management": <int>, "standards_compliance": <int>}}

JSON:"""
    
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

def load_and_chunk_text(txt_path, chunk_size=CHUNK_SIZE, max_chunks=MAX_CHUNKS):
    """Load text and split into chunks (words)."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if not text or len(text) < 100:
            return None
        
        words = text.split()
        
        if not USE_CHUNKING or len(words) <= chunk_size:
            # Return single chunk
            return [' '.join(words[:MAX_INPUT_TOKENS])]
        
        # Create overlapping chunks
        chunks = []
        step = int(chunk_size * 0.8)  # 20% overlap
        
        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            if len(chunks) >= max_chunks:
                break
        
        return chunks
    
    except Exception as e:
        print(f"    ❌ Error loading {txt_path}: {e}")
        return None

# =============================================================================
# LLM SETUP
# =============================================================================

def setup_model():
    """Load model with 4-bit quantization for 8GB VRAM."""
    print(f"  Model: {MODEL_NAME}")
    print(f"  Quantization: {'4-bit (required for 8GB VRAM)' if USE_4BIT else 'Full precision'}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU (WARNING: slow!)'}")
    
    if not torch.cuda.is_available():
        print("\n  ⚠️  WARNING: CUDA not available! Falling back to CPU.")
        print("     Install CUDA-enabled PyTorch:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu121")
        response = input("\n  Continue on CPU? (y/n): ")
        if response.lower() != 'y':
            exit(1)
    
    # 4-bit quantization config (REQUIRED for 8GB VRAM)
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.float16 if not USE_4BIT else None
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        print(f"  ✅ Model loaded successfully")
        print(f"     VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        return pipe, tokenizer
    
    except Exception as e:
        print(f"\n  ❌ Failed to load model: {e}")
        print(f"\n  Troubleshooting:")
        print(f"    1. Check HF_TOKEN is valid")
        print(f"    2. Install dependencies: pip install bitsandbytes accelerate")
        print(f"    3. Try smaller model: microsoft/Phi-3-mini-4k-instruct")
        print(f"    4. Check CUDA: python -c \"import torch; print(torch.cuda.is_available())\"")
        exit(1)

# =============================================================================
# LLM SCORING
# =============================================================================

def parse_llm_response(response_text):
    """Extract and validate JSON scores from LLM response."""
    try:
        # Clean response
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if '```json' in response_text:
            response_text = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if response_text:
                response_text = response_text.group(1)
        elif '```' in response_text:
            response_text = re.search(r'```\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if response_text:
                response_text = response_text.group(1)
        
        # Find JSON object
        json_match = re.search(r'\{[^{}]*\}', response_text)
        if not json_match:
            return None, "No JSON found"
        
        json_str = json_match.group(0)
        scores = json.loads(json_str)
        
        # Validate structure
        for dim in DIMENSIONS:
            if dim not in scores:
                return None, f"Missing dimension: {dim}"
            
            score = scores[dim]
            
            # Handle string numbers
            if isinstance(score, str):
                score = int(score)
            
            if not isinstance(score, int):
                return None, f"Invalid type for {dim}: {type(score)}"
            
            if not (0 <= score <= 5):
                return None, f"Out of range for {dim}: {score}"
            
            scores[dim] = score
        
        return scores, "OK"
    
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return None, f"Parse error: {str(e)}"

def score_chunk_with_llm(pipe, chunk, max_retries=MAX_RETRIES):
    """Score a single chunk using the LLM."""
    prompt = build_scoring_prompt(chunk)
    
    for attempt in range(max_retries):
        try:
            # Generate
            outputs = pipe(prompt, return_full_text=False)
            response = outputs['generated_text']
            
            # Parse
            scores, status = parse_llm_response(response)
            
            if scores:
                return scores, response, True
            else:
                if attempt < max_retries - 1:
                    print(f"        Retry {attempt + 1}: {status}")
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"        Retry {attempt + 1}: {str(e)[:50]}")
    
    return None, "", False

def aggregate_chunk_scores(chunk_scores, method='max'):
    """Aggregate scores from multiple chunks."""
    if not chunk_scores:
        return None
    
    if len(chunk_scores) == 1:
        return chunk_scores
    
    aggregated = {}
    
    for dim in DIMENSIONS:
        dim_scores = [cs[dim] for cs in chunk_scores]
        
        if method == 'max':
            aggregated[dim] = max(dim_scores)
        elif method == 'mean':
            aggregated[dim] = int(round(np.mean(dim_scores)))
        elif method == 'top2_mean':
            top2 = sorted(dim_scores, reverse=True)[:2]
            aggregated[dim] = int(round(np.mean(top2)))
        else:
            aggregated[dim] = max(dim_scores)  # Default to max
    
    return aggregated

def score_framework_with_llm(pipe, chunks):
    """Score a framework (potentially across multiple chunks)."""
    chunk_scores = []
    responses = []
    
    print(f"      Scoring {len(chunks)} chunk(s)...")
    
    for i, chunk in enumerate(chunks):
        print(f"        Chunk {i+1}/{len(chunks)}...", end=' ')
        
        scores, response, success = score_chunk_with_llm(pipe, chunk)
        
        if success:
            chunk_scores.append(scores)
            responses.append(response[:300])
            print("✓")
        else:
            print("✗ (failed)")
    
    if not chunk_scores:
        print(f"      ❌ All chunks failed")
        return None, "ALL_CHUNKS_FAILED"
    
    # Aggregate
    final_scores = aggregate_chunk_scores(chunk_scores, method=AGGREGATION)
    
    combined_response = f"[Aggregated from {len(chunk_scores)}/{len(chunks)} chunks] " + " | ".join(responses)
    
    return final_scores, combined_response

# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_metrics(llm_scores, expert_scores, dimension):
    """Compute comprehensive metrics for a dimension."""
    fids = list(llm_scores.keys())
    
    llm_vals = np.array([llm_scores[fid][dimension] for fid in fids])
    expert_vals = np.array([expert_scores[fid][dimension] for fid in fids])
    
    # Core metrics
    mae = np.mean(np.abs(llm_vals - expert_vals))
    rmse = np.sqrt(np.mean((llm_vals - expert_vals) ** 2))
    exact_match = np.mean(llm_vals == expert_vals)
    within_one = np.mean(np.abs(llm_vals - expert_vals) <= 1)
    
    # Correlation
    rho, p_value = spearmanr(llm_vals, expert_vals)
    
    # Cohen's kappa
    try:
        kappa = cohen_kappa_score(expert_vals, llm_vals)
    except:
        kappa = 0.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'exact': exact_match,
        'within_1': within_one,
        'rho': rho,
        'p_value': p_value,
        'kappa': kappa,
        'llm_mean': np.mean(llm_vals),
        'expert_mean': np.mean(expert_vals),
        'llm_std': np.std(llm_vals),
        'expert_std': np.std(expert_vals)
    }

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("  PQC READINESS — LLM SCORING PIPELINE (CUDA/Windows)")
    print("=" * 70)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    print(f"\n[1/5] Loading model (first run will download ~4GB)...")
    pipe, tokenizer = setup_model()
    
    # Load data
    print(f"\n[2/5] Loading expert scores and mappings...")
    expert_scores = load_expert_scores(LABELS_FILE)
    mapping = load_framework_mapping(MAPPING_FILE)
    
    print(f"      Expert scores: {len(expert_scores)} frameworks")
    print(f"      Mapping: {len(mapping)} frameworks")
    
    # Score frameworks
    print(f"\n[3/5] Scoring frameworks with LLM...")
    print(f"      Chunking: {'Enabled' if USE_CHUNKING else 'Disabled'}")
    print(f"      Aggregation: {AGGREGATION}\n")
    
    results = []
    llm_scores = {}
    failed_count = 0
    
    for fid in sorted(expert_scores.keys()):
        txt_file = f"{fid}.txt"
        txt_path = PROCESSED_DIR / txt_file
        
        if not txt_path.exists():
            print(f"  [{fid}] ⚠️  Text file not found — skipping")
            continue
        
        framework_name = mapping.get(fid, txt_file)
        print(f"  [{fid}] {framework_name[:55]}...")
        
        # Load and chunk
        chunks = load_and_chunk_text(txt_path)
        if not chunks:
            print(f"      ⚠️  Could not load text — skipping")
            failed_count += 1
            continue
        
        # Score
        scores, raw_response = score_framework_with_llm(pipe, chunks)
        
        if scores is None:
            failed_count += 1
            continue
        
        llm_scores[fid] = scores
        
        # Build result row
        row = {
            'Framework_ID': fid,
            'Framework': framework_name,
            'chunks_evaluated': len(chunks)
        }
        
        deltas = []
        for dim in DIMENSIONS:
            llm_score = scores[dim]
            expert_score = expert_scores[fid][dim]
            delta = abs(llm_score - expert_score)
            deltas.append(delta)
            
            row[f'{dim}_llm'] = llm_score
            row[f'{dim}_expert'] = expert_score
            row[f'{dim}_delta'] = delta
        
        row['mean_delta'] = np.mean(deltas)
        row['raw_response'] = raw_response[:500]
        results.append(row)
        
        # Print summary
        summary = " | ".join(
            f"{dim[:5]}: L{scores[dim]} E{expert_scores[fid][dim]} (Δ{abs(scores[dim] - expert_scores[fid][dim])})"
            for dim in DIMENSIONS
        )
        print(f"      {summary}")
        print(f"      Mean Δ: {np.mean(deltas):.2f}")
    
    if not llm_scores:
        print("\n  ❌ No frameworks successfully scored!")
        return
    
    # Save results
    print(f"\n[4/5] Saving results...")
    
    # CSV
    csv_path = OUTPUT_DIR / "llm_scores_cuda.csv"
    fieldnames = ['Framework_ID', 'Framework', 'chunks_evaluated']
    for dim in DIMENSIONS:
        fieldnames += [f'{dim}_llm', f'{dim}_expert', f'{dim}_delta']
    fieldnames += ['mean_delta', 'raw_response']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"      ✅ {csv_path}")
    
    # JSON
    json_path = OUTPUT_DIR / "llm_scores_cuda.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"      ✅ {json_path}")
    
    # Compute metrics
    print(f"\n[5/5] Computing metrics...\n")
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    
    all_metrics = {}
    overall_mae = []
    
    for dim in DIMENSIONS:
        metrics = compute_metrics(llm_scores, expert_scores, dim)
        all_metrics[dim] = metrics
        overall_mae.append(metrics['mae'])
        
        print(f"\n  {dim.replace('_', ' ').title()}")
        print(f"    MAE:         {metrics['mae']:.3f}")
        print(f"    RMSE:        {metrics['rmse']:.3f}")
        print(f"    Exact match: {metrics['exact']:.1%}")
        print(f"    Within ±1:   {metrics['within_1']:.1%}")
        print(f"    Spearman ρ:  {metrics['rho']:.3f} (p={metrics['p_value']:.4f})")
        print(f"    Cohen's κ:   {metrics['kappa']:.3f}")
        print(f"    LLM mean:    {metrics['llm_mean']:.2f} (σ={metrics['llm_std']:.2f})")
        print(f"    Expert mean: {metrics['expert_mean']:.2f} (σ={metrics['expert_std']:.2f})")
    
    final_mae = np.mean(overall_mae)
    
    print(f"\n{'=' * 70}")
    print(f"  Overall MAE:         {final_mae:.3f}  {'✅' if final_mae < 1.0 else '⚠️'}")
    print(f"  SVR Baseline:        0.781")
    
    if final_mae < 0.781:
        improvement = ((0.781 - final_mae) / 0.781) * 100
        print(f"  Improvement:         +{improvement:.1f}% 🎉")
    else:
        regression = ((final_mae - 0.781) / 0.781) * 100
        print(f"  Regression:          -{regression:.1f}%")
    
    print(f"\n  Frameworks scored:   {len(llm_scores)}/{len(expert_scores)}")
    print(f"  Failed:              {failed_count}")
    print(f"  Success rate:        {len(llm_scores)/(len(expert_scores))*100:.1f}%")
    print(f"  Aggregation method:  {AGGREGATION}")
    print("=" * 70)
    
    # Save detailed summary
    summary_path = OUTPUT_DIR / "summary_cuda.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PQC READINESS — LLM SCORING SUMMARY (CUDA)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Quantization: {'4-bit' if USE_4BIT else 'Full precision'}\n")
        f.write(f"Chunking: {'Enabled' if USE_CHUNKING else 'Disabled'}\n")
        f.write(f"Aggregation: {AGGREGATION}\n")
        f.write(f"Frameworks scored: {len(llm_scores)}/{len(expert_scores)}\n")
        f.write(f"Overall MAE: {final_mae:.3f}\n\n")
        
        for dim in DIMENSIONS:
            m = all_metrics[dim]
            f.write(f"{dim}:\n")
            f.write(f"  MAE: {m['mae']:.3f}\n")
            f.write(f"  RMSE: {m['rmse']:.3f}\n")
            f.write(f"  Exact: {m['exact']:.1%}\n")
            f.write(f"  Within ±1: {m['within_1']:.1%}\n")
            f.write(f"  Spearman ρ: {m['rho']:.3f} (p={m['p_value']:.4f})\n")
            f.write(f"  Cohen's κ: {m['kappa']:.3f}\n\n")
    
    print(f"\n  💾 Summary saved to {summary_path}")
    
    # GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n  🧹 GPU cache cleared")

if __name__ == '__main__':
    main()
