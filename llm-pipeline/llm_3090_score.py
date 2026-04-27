# =============================================================================
# llm_score_multigpu.py — Multi-GPU PQC Readiness Scoring (3x RTX 3090)
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
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print('❌ ERROR: HF_TOKEN environment variable not set!')
    exit(1)

BASE_DIR = Path(__file__).parent
FREQ_ANALYSIS_DIR = BASE_DIR.parent / "freq-analysis"
PROCESSED_DIR = FREQ_ANALYSIS_DIR / "processed"
LABELS_FILE = FREQ_ANALYSIS_DIR / "scripts" / "labels.csv"
MAPPING_FILE = FREQ_ANALYSIS_DIR / "framework_mapping.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------
# Model choices
# -------------------------
# Best practical choice:
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

# If you want to push harder later:
# MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

USE_4BIT = True
MAX_NEW_TOKENS = 120
TEMPERATURE = 0.0
TOP_P = 1.0

# Context / chunking
MAX_INPUT_TOKENS = 3000
USE_CHUNKING = True
CHUNK_SIZE = 1200
MAX_CHUNKS = 3
AGGREGATION = "max"

MAX_RETRIES = 2

DIMENSIONS = [
    "crypto_assets",
    "crypto_agility",
    "migration_planning",
    "risk_management",
    "standards_compliance"
]

# Reserve some VRAM headroom per GPU
GPU_MAX_MEMORY = {
    0: "22GiB",
    1: "22GiB",
    2: "22GiB",
    "cpu": "64GiB"
}

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
    return f"""{SYSTEM_PROMPT}

{SCORING_RUBRIC}

**Document Excerpt:**
{text_excerpt}

**Task:**
Based ONLY on the evidence in the document above, assign integer scores (0-5) for each dimension.

Return ONLY a valid JSON object with this exact structure (no markdown, no additional text):

{{"crypto_assets": <int>, "crypto_agility": <int>, "migration_planning": <int>, "risk_management": <int>, "standards_compliance": <int>}}

JSON:"""

# =============================================================================
# DATA LOADING
# =============================================================================

def load_expert_scores(path):
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = str(row["Framework_ID"]).strip().zfill(3)
            scores[fid] = {dim: int(row[dim]) for dim in DIMENSIONS}
    return scores

def load_framework_mapping(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = str(row["Framework ID"]).strip().zfill(3)
            mapping[fid] = row["PDF_Title"].strip()
    return mapping

def load_and_chunk_text(txt_path, chunk_size=CHUNK_SIZE, max_chunks=MAX_CHUNKS):
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text or len(text) < 100:
            return None

        words = text.split()

        if not USE_CHUNKING or len(words) <= chunk_size:
            return [" ".join(words[:chunk_size])]

        chunks = []
        step = int(chunk_size * 0.8)

        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            if len(chunks) >= max_chunks:
                break

        return chunks if chunks else None

    except Exception as e:
        print(f"    ❌ Error loading {txt_path}: {e}")
        return None

# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model():
    print(f"  Model: {MODEL_NAME}")
    print(f"  Quantization: {'4-bit' if USE_4BIT else 'Full precision'}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  GPU count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("❌ CUDA is required for this multi-GPU script.")
        exit(1)

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {props.name} | {props.total_memory / 1e9:.1f} GB")

    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            trust_remote_code=True,
            use_fast=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=GPU_MAX_MEMORY,
            low_cpu_mem_usage=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

        print("  ✅ Model loaded successfully")
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"    GPU {i}: allocated={alloc:.2f} GB | reserved={reserved:.2f} GB")

        return pipe, tokenizer

    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check HF_TOKEN")
        print("  2. Ensure bitsandbytes/accelerate are installed")
        print("  3. Reduce model size")
        print("  4. Reduce MAX_INPUT_TOKENS / CHUNK_SIZE")
        exit(1)

# =============================================================================
# LLM SCORING
# =============================================================================

def parse_llm_response(response_text):
    try:
        response_text = response_text.strip()

        if "```json" in response_text:
            m = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if m:
                response_text = m.group(1)
        elif "```" in response_text:
            m = re.search(r"```\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if m:
                response_text = m.group(1)

        json_match = re.search(r"\{[^{}]*\}", response_text)
        if not json_match:
            return None, "No JSON found"

        scores = json.loads(json_match.group(0))

        for dim in DIMENSIONS:
            if dim not in scores:
                return None, f"Missing dimension: {dim}"

            val = scores[dim]
            if isinstance(val, str):
                val = int(val)

            if not isinstance(val, int):
                return None, f"Invalid type for {dim}: {type(val)}"

            if not (0 <= val <= 5):
                return None, f"Out of range for {dim}: {val}"

            scores[dim] = val

        return scores, "OK"

    except Exception as e:
        return None, f"Parse error: {e}"

def generate_response(pipe, prompt):
    outputs = pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        return_full_text=False,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    return outputs["generated_text"]

def score_chunk_with_llm(pipe, chunk, max_retries=MAX_RETRIES):
    prompt = build_scoring_prompt(chunk)

    for attempt in range(max_retries):
        try:
            response = generate_response(pipe, prompt)
            scores, status = parse_llm_response(response)

            if scores is not None:
                return scores, response, True
            else:
                if attempt < max_retries - 1:
                    print(f"        Retry {attempt + 1}: {status}")

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"        Retry {attempt + 1}: {str(e)[:100]}")

    return None, "", False

def aggregate_chunk_scores(chunk_scores, method="max"):
    if not chunk_scores:
        return None

    if len(chunk_scores) == 1:
        return chunk_scores

    aggregated = {}
    for dim in DIMENSIONS:
        vals = [cs[dim] for cs in chunk_scores]

        if method == "max":
            aggregated[dim] = max(vals)
        elif method == "mean":
            aggregated[dim] = int(round(np.mean(vals)))
        elif method == "top2_mean":
            top2 = sorted(vals, reverse=True)[:2]
            aggregated[dim] = int(round(np.mean(top2)))
        else:
            aggregated[dim] = max(vals)

    return aggregated

def score_framework_with_llm(pipe, chunks):
    chunk_scores = []
    responses = []

    print(f"      Scoring {len(chunks)} chunk(s)...")

    for i, chunk in enumerate(chunks):
        print(f"        Chunk {i+1}/{len(chunks)}...", end=" ")

        scores, response, success = score_chunk_with_llm(pipe, chunk)

        if success:
            chunk_scores.append(scores)
            responses.append(response[:300])
            print("✓")
        else:
            print("✗")

    if not chunk_scores:
        print("      ❌ All chunks failed")
        return None, "ALL_CHUNKS_FAILED"

    final_scores = aggregate_chunk_scores(chunk_scores, method=AGGREGATION)
    combined_response = f"[Aggregated from {len(chunk_scores)}/{len(chunks)} chunks] " + " | ".join(responses)
    return final_scores, combined_response

# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(llm_scores, expert_scores, dimension):
    fids = list(llm_scores.keys())
    llm_vals = np.array([llm_scores[fid][dimension] for fid in fids])
    expert_vals = np.array([expert_scores[fid][dimension] for fid in fids])

    mae = np.mean(np.abs(llm_vals - expert_vals))
    rmse = np.sqrt(np.mean((llm_vals - expert_vals) ** 2))
    exact_match = np.mean(llm_vals == expert_vals)
    within_one = np.mean(np.abs(llm_vals - expert_vals) <= 1)

    rho, p_value = spearmanr(llm_vals, expert_vals)

    try:
        kappa = cohen_kappa_score(expert_vals, llm_vals)
    except Exception:
        kappa = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "exact": exact_match,
        "within_1": within_one,
        "rho": rho,
        "p_value": p_value,
        "kappa": kappa,
        "llm_mean": np.mean(llm_vals),
        "expert_mean": np.mean(expert_vals),
        "llm_std": np.std(llm_vals),
        "expert_std": np.std(expert_vals)
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 72)
    print("  PQC READINESS — LLM SCORING PIPELINE (MULTI-GPU 3x3090)")
    print("=" * 72)

    print("\n[1/5] Loading model...")
    pipe, tokenizer = setup_model()

    print("\n[2/5] Loading expert scores and mappings...")
    expert_scores = load_expert_scores(LABELS_FILE)
    mapping = load_framework_mapping(MAPPING_FILE)

    print(f"      Expert scores: {len(expert_scores)} frameworks")
    print(f"      Mapping: {len(mapping)} frameworks")

    print("\n[3/5] Scoring frameworks with LLM...")
    print(f"      Chunking: {'Enabled' if USE_CHUNKING else 'Disabled'}")
    print(f"      Aggregation: {AGGREGATION}\n")

    results = []
    llm_scores = {}
    failed_count = 0

    for fid in sorted(expert_scores.keys()):
        txt_path = PROCESSED_DIR / f"{fid}.txt"

        if not txt_path.exists():
            print(f"  [{fid}] ⚠️ Text file not found — skipping")
            continue

        framework_name = mapping.get(fid, f"{fid}.txt")
        print(f"  [{fid}] {framework_name[:55]}...")

        chunks = load_and_chunk_text(txt_path)
        if not chunks:
            print("      ⚠️ Could not load text — skipping")
            failed_count += 1
            continue

        scores, raw_response = score_framework_with_llm(pipe, chunks)
        if scores is None:
            failed_count += 1
            continue

        llm_scores[fid] = scores

        row = {
            "Framework_ID": fid,
            "Framework": framework_name,
            "chunks_evaluated": len(chunks)
        }

        deltas = []
        for dim in DIMENSIONS:
            llm_score = scores[dim]
            expert_score = expert_scores[fid][dim]
            delta = abs(llm_score - expert_score)
            deltas.append(delta)

            row[f"{dim}_llm"] = llm_score
            row[f"{dim}_expert"] = expert_score
            row[f"{dim}_delta"] = delta

        row["mean_delta"] = float(np.mean(deltas))
        row["raw_response"] = raw_response[:500]
        results.append(row)

        summary = " | ".join(
            f"{dim[:5]}: L{scores[dim]} E{expert_scores[fid][dim]} (Δ{abs(scores[dim] - expert_scores[fid][dim])})"
            for dim in DIMENSIONS
        )
        print(f"      {summary}")
        print(f"      Mean Δ: {np.mean(deltas):.2f}")

    if not llm_scores:
        print("\n❌ No frameworks successfully scored!")
        return

    print("\n[4/5] Saving results...")

    csv_path = OUTPUT_DIR / "llm_scores_multigpu.csv"
    fieldnames = ["Framework_ID", "Framework", "chunks_evaluated"]
    for dim in DIMENSIONS:
        fieldnames += [f"{dim}_llm", f"{dim}_expert", f"{dim}_delta"]
    fieldnames += ["mean_delta", "raw_response"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    json_path = OUTPUT_DIR / "llm_scores_multigpu.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"      ✅ {csv_path}")
    print(f"      ✅ {json_path}")

    print("\n[5/5] Computing metrics...\n")
    print("=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)

    all_metrics = {}
    overall_mae = []

    for dim in DIMENSIONS:
        metrics = compute_metrics(llm_scores, expert_scores, dim)
        all_metrics[dim] = metrics
        overall_mae.append(metrics["mae"])

        print(f"\n  {dim.replace('_', ' ').title()}")
        print(f"    MAE:         {metrics['mae']:.3f}")
        print(f"    RMSE:        {metrics['rmse']:.3f}")
        print(f"    Exact match: {metrics['exact']:.1%}")
        print(f"    Within ±1:   {metrics['within_1']:.1%}")
        print(f"    Spearman ρ:  {metrics['rho']:.3f} (p={metrics['p_value']:.4f})")
        print(f"    Cohen's κ:   {metrics['kappa']:.3f}")
        print(f"    LLM mean:    {metrics['llm_mean']:.2f} (σ={metrics['llm_std']:.2f})")
        print(f"    Expert mean: {metrics['expert_mean']:.2f} (σ={metrics['expert_std']:.2f})")

    final_mae = float(np.mean(overall_mae))

    print(f"\n{'=' * 72}")
    print(f"  Overall MAE:         {final_mae:.3f}")
    print(f"  Frameworks scored:   {len(llm_scores)}/{len(expert_scores)}")
    print(f"  Failed:              {failed_count}")
    print(f"  Success rate:        {len(llm_scores)/len(expert_scores)*100:.1f}%")
    print(f"  Aggregation method:  {AGGREGATION}")
    print("=" * 72)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
        print("\n  🧹 GPU cache cleared")

if __name__ == "__main__":
    main()
