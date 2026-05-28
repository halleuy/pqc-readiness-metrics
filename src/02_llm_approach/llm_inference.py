
import os
import json
import csv
import re
import gc
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr, bootstrap
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
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU PQC Readiness Scoring with configurable parameters")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="Model name or path")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--aggregation", type=str, default="max", choices=["max", "mean", "top2_mean"], help="Aggregation method for chunk scores")
    parser.add_argument("--chunk_size", type=int, default=1200, help="Chunk size for text splitting")
    parser.add_argument("--max_chunks", type=int, default=3, help="Max number of chunks to process per document")
    parser.add_argument("--use_cache", action="store_true", help="Enable caching of chunk scoring results")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output filenames")
    return parser.parse_args()

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

USE_4BIT = True
MAX_NEW_TOKENS = 120
DO_SAMPLE = False  # We'll drive sampling via temperature argument

MAX_INPUT_TOKENS = 3000
MAX_TOTAL_TOKENS = 4096
USE_CHUNKING = True
CHUNK_OVERLAP = 0.8

MAX_RETRIES = 2

GPU_MAX_MEMORY = {
    0: "22GiB",
    1: "22GiB",
    2: "22GiB",
    "cpu": "64GiB"
}

DIMENSIONS = [
    "crypto_assets",
    "crypto_agility",
    "migration_planning",
    "risk_management",
    "standards_compliance"
]

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = OUTPUT_DIR / f"run_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert evaluator of Post-Quantum Cryptography (PQC) readiness frameworks.
Your task is to analyze technical documents and assign scores (0-5)."""

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
  
**Example Responses:**  
  
Document: "Organizations must identify all cryptographic assets including certificates, keys, and algorithms."  
Response: {"crypto_assets": 2, "crypto_agility": 0, "migration_planning": 0, "risk_management": 0, "standards_compliance": 0}  
  
Document: "Implement ML-KEM-768 for key encapsulation following FIPS 203. Use modular architecture for algorithm switching."  
Response: {"crypto_assets": 0, "crypto_agility": 3, "migration_planning": 0, "risk_management": 0, "standards_compliance": 4}
"""

def build_scoring_prompt(text_excerpt):
    return f"""{SYSTEM_PROMPT}

{SCORING_RUBRIC}

**Document Excerpt:**
{text_excerpt}

**Task:**
Based ONLY on the evidence in the document above, assign integer scores (0-5) for each dimension.

Return ONLY a valid JSON object with this exact structure (no explanations, no markdown):

{{"crypto_assets": <int>, "crypto_agility": <int>, "migration_planning": <int>, "risk_management": <int>, "standards_compliance": <int>}}

JSON:"""

# =============================================================================
# CACHING
# =============================================================================

class ChunkCache:
    def __init__(self, cache_file, enabled):
        self.cache_file = cache_file
        self.enabled = enabled
        self.cache = self._load_cache()
    
    def _load_cache(self):
        if self.enabled and self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"Loaded cache with {len(cache)} entries")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        if self.enabled:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    def get_hash(self, text, model_name, temperature, aggregation, chunk_size, max_chunks):
        # Include key params to avoid contamination across runs
        key_source = f"{text}|{model_name}|{temperature}|{aggregation}|{chunk_size}|{max_chunks}"
        return hashlib.md5(key_source.encode('utf-8')).hexdigest()
    
    def get(self, chunk, model_name, temperature, aggregation, chunk_size, max_chunks):
        if not self.enabled:
            return None
        chunk_hash = self.get_hash(chunk, model_name, temperature, aggregation, chunk_size, max_chunks)
        return self.cache.get(chunk_hash)
    
    def set(self, chunk, scores, response, model_name, temperature, aggregation, chunk_size, max_chunks):
        if self.enabled:
            chunk_hash = self.get_hash(chunk, model_name, temperature, aggregation, chunk_size, max_chunks)
            self.cache[chunk_hash] = {
                'scores': scores,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()

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
    logger.info(f"Loaded expert scores for {len(scores)} frameworks")
    return scores

def load_framework_mapping(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = str(row["Framework ID"]).strip().zfill(3)
            mapping[fid] = row["PDF_Title"].strip()
    logger.info(f"Loaded mapping for {len(mapping)} frameworks")
    return mapping

def load_and_chunk_text(txt_path, tokenizer, chunk_size, max_chunks):
    # Adapted to use passed params instead of global
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text or len(text) < 100:
            logger.warning(f"Text too short: {txt_path}")
            return None

        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= MAX_INPUT_TOKENS:
            return [text]

        if len(tokens) > MAX_TOTAL_TOKENS:
            logger.warning(f"Text exceeds max tokens, truncating: {txt_path}")
            text = tokenizer.decode(tokens[:MAX_TOTAL_TOKENS], skip_special_tokens=True)

        if not USE_CHUNKING:
            truncated_tokens = tokenizer.encode(text, add_special_tokens=False)[:MAX_INPUT_TOKENS]
            return [tokenizer.decode(truncated_tokens, skip_special_tokens=True)]

        words = text.split()
        chunks = []
        step = int(chunk_size * CHUNK_OVERLAP)

        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)

            chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
            if len(chunk_tokens) > MAX_INPUT_TOKENS:
                chunk_text = tokenizer.decode(chunk_tokens[:MAX_INPUT_TOKENS], skip_special_tokens=True)

            if chunk_text.strip():
                chunks.append(chunk_text)

            if len(chunks) >= max_chunks:
                break

        logger.info(f"Created {len(chunks)} chunks for {txt_path.name}")
        return chunks if chunks else None

    except Exception as e:
        logger.error(f"Error loading {txt_path}: {e}")
        return None

# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model(model_name):
    logger.info(f"Setting up model: {model_name}")
    logger.info(f"Quantization: {'4-bit' if USE_4BIT else 'Full precision'}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        logger.error("CUDA is required for this multi-GPU script")
        exit(1)

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name} | {props.total_memory / 1e9:.1f} GB")

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
            model_name,
            token=HF_TOKEN,
            trust_remote_code=True,
            use_fast=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=False,
            token=HF_TOKEN,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=GPU_MAX_MEMORY,
            low_cpu_mem_usage=True
        )

        if hasattr(model, "generation_config"):
            model.generation_config.max_length = None
            model.generation_config.max_new_tokens = None

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id
        )

        logger.info("Model loaded successfully")

        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logger.info(f"GPU {i}: allocated={alloc:.2f} GB | reserved={reserved:.2f} GB")

        return pipe, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
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
            return None, "No JSON object found"

        scores = json.loads(json_match.group(0))

        for dim in DIMENSIONS:
            if dim not in scores:
                return None, f"Missing dimension: {dim}"

            val = scores[dim]

            if val is None or val == "null":
                return None, f"Null value for {dim}"

            try:
                if isinstance(val, str):
                    val = float(val)
                val = int(round(val))
            except (ValueError, TypeError):
                return None, f"Invalid type for {dim}: {type(val).__name__}"

            if not (0 <= val <= 5):
                return None, f"Score out of range for {dim}: {val}"

            scores[dim] = val

        return scores, "OK"

    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    except Exception as e:
        return None, f"Parse error: {e}"

def generate_response(pipe, prompt, temperature):
    try:
        outputs = pipe(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id
        )

        if isinstance(outputs, list):
            if len(outputs) > 0:
                return outputs[0]["generated_text"]
            else:
                raise ValueError("Pipeline returned empty list")
        elif isinstance(outputs, dict):
            return outputs.get("generated_text", "")
        else:
            raise ValueError(f"Unexpected pipeline output type: {type(outputs)}")

    except Exception as e:
        logger.error(f"Error in generate_response: {e}")
        raise

def score_chunk_with_llm(pipe, chunk, cache, model_name, temperature, aggregation, chunk_size, max_chunks, max_retries=MAX_RETRIES):
    cached = cache.get(chunk, model_name, temperature, aggregation, chunk_size, max_chunks)
    if cached:
        logger.debug("Cache hit")
        return cached['scores'], cached['response'], True

    prompt = build_scoring_prompt(chunk)

    for attempt in range(max_retries):
        try:
            response = generate_response(pipe, prompt, temperature)
            scores, status = parse_llm_response(response)

            if scores is not None:
                cache.set(chunk, scores, response, model_name, temperature, aggregation, chunk_size, max_chunks)
                return scores, response, True
            else:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries}: {status}")

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Retry {attempt + 1}/{max_retries}: {str(e)[:100]}")
            else:
                logger.error(f"All retries failed: {str(e)[:100]}")

    return None, "", False

def aggregate_chunk_scores(chunk_scores, method="max"):
    if not chunk_scores:
        return None

    if len(chunk_scores) == 1:
        return chunk_scores[0]

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

def score_framework_with_llm(pipe, chunks, cache, model_name, temperature, aggregation, chunk_size, max_chunks):
    chunk_scores = []
    responses = []

    logger.info(f"Scoring {len(chunks)} chunk(s)...")

    for i, chunk in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)}...")

        scores, response, success = score_chunk_with_llm(pipe, chunk, cache,
                                                        model_name,
                                                        temperature,
                                                        aggregation,
                                                        chunk_size,
                                                        max_chunks)

        if success:
            chunk_scores.append(scores)
            responses.append(response[:300])
            logger.info("  ✓ Success")
        else:
            logger.warning("  ✗ Failed")

    if not chunk_scores:
        logger.error("All chunks failed")
        return None, "ALL_CHUNKS_FAILED"

    final_scores = aggregate_chunk_scores(chunk_scores, method=aggregation)
    combined_response = f"[Aggregated from {len(chunk_scores)}/{len(chunks)} chunks using {aggregation}] " + " | ".join(responses)

    return final_scores, combined_response

# =============================================================================
# METRICS (unchanged)
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

    ci_low, ci_high = 0.0, 0.0
    try:
        def mae_stat(llm, exp):
            return np.mean(np.abs(llm - exp))

        rng = np.random.default_rng(42)
        res = bootstrap(
            (llm_vals, expert_vals),
            mae_stat,
            n_resamples=1000,
            confidence_level=0.95,
            random_state=rng,
            vectorized=False,
            method='percentile'
        )
        ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
    except Exception as e:
        logger.warning(f"Failed to compute CI: {e}")

    return {
        "mae": mae,
        "mae_ci_low": ci_low,
        "mae_ci_high": ci_high,
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
    args = parse_args()

    # Update globals from args
    global MODEL_NAME, TEMPERATURE, AGGREGATION, CHUNK_SIZE, MAX_CHUNKS, USE_CACHE
    MODEL_NAME = args.model_name
    TEMPERATURE = args.temperature
    AGGREGATION = args.aggregation
    CHUNK_SIZE = args.chunk_size
    MAX_CHUNKS = args.max_chunks
    USE_CACHE = args.use_cache

    cache_file = OUTPUT_DIR / f"chunk_cache_{args.output_suffix}.json" if args.output_suffix else OUTPUT_DIR / "chunk_cache.json"
    cache = ChunkCache(cache_file, USE_CACHE)

    logger.info("=" * 72)
    logger.info(f"  PQC READINESS — LLM SCORING PIPELINE (MULTI-GPU)")
    logger.info("=" * 72)
    logger.info(f"\nModel: {MODEL_NAME}")
    logger.info(f"Temperature: {TEMPERATURE}")
    logger.info(f"Aggregation: {AGGREGATION}")
    logger.info(f"Chunk size: {CHUNK_SIZE}")
    logger.info(f"Max chunks: {MAX_CHUNKS}")
    logger.info(f"Caching enabled: {USE_CACHE}\n")

    logger.info("\n[1/5] Loading model...")
    pipe, tokenizer = setup_model(MODEL_NAME)

    logger.info("\n[2/5] Loading expert scores and mappings...")
    expert_scores = load_expert_scores(LABELS_FILE)
    mapping = load_framework_mapping(MAPPING_FILE)

    logger.info("\n[3/5] Scoring frameworks with LLM...")
    logger.info(f"  Chunking: {'Enabled' if USE_CHUNKING else 'Disabled'}")
    logger.info(f"  Aggregation: {AGGREGATION}")
    logger.info(f"  Caching: {'Enabled' if USE_CACHE else 'Disabled'}\n")

    results = []
    llm_scores = {}
    failed_count = 0

    for idx, fid in enumerate(sorted(expert_scores.keys())):
        txt_path = PROCESSED_DIR / f"{fid}.txt"

        if not txt_path.exists():
            logger.warning(f"[{fid}] Text file not found")
            continue

        framework_name = mapping.get(fid, f"{fid}.txt")
        logger.info(f"[{fid}] {framework_name[:55]}...")

        chunks = load_and_chunk_text(txt_path, tokenizer, CHUNK_SIZE, MAX_CHUNKS)
        if not chunks:
            logger.warning("  Could not load text")
            failed_count += 1
            continue

        scores, raw_response = score_framework_with_llm(pipe, chunks, cache,
                                                        MODEL_NAME,
                                                        TEMPERATURE,
                                                        AGGREGATION,
                                                        CHUNK_SIZE,
                                                        MAX_CHUNKS)
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
        logger.info(f"  {summary}")
        logger.info(f"  Mean Δ: {np.mean(deltas):.2f}")

        if (idx + 1) % 10 == 0:
            logger.info("  🧹 Cleaning GPU cache...")
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
            gc.collect()

    if not llm_scores:
        logger.error("\nNo frameworks successfully scored!")
        return

    logger.info("\n[4/5] Saving results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = args.output_suffix if args.output_suffix else "default"
    csv_path = OUTPUT_DIR / f"llm_scores_{suffix}_{timestamp}.csv"
    json_path = OUTPUT_DIR / f"llm_scores_{suffix}_{timestamp}.json"

    fieldnames = ["Framework_ID", "Framework", "chunks_evaluated"]
    for dim in DIMENSIONS:
        fieldnames += [f"{dim}_llm", f"{dim}_expert", f"{dim}_delta"]
    fieldnames += ["mean_delta", "raw_response"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"  ✅ {csv_path}")
    logger.info(f"  ✅ {json_path}")

    logger.info("\n[5/5] Computing metrics...\n")
    logger.info("=" * 72)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 72)

    all_metrics = {}
    overall_mae = []

    for dim in DIMENSIONS:
        metrics = compute_metrics(llm_scores, expert_scores, dim)
        all_metrics[dim] = metrics
        overall_mae.append(metrics["mae"])

        logger.info(f"\n  {dim.replace('_', ' ').title()}")
        logger.info(f"    MAE:         {metrics['mae']:.3f} (95% CI: [{metrics['mae_ci_low']:.3f}, {metrics['mae_ci_high']:.3f}])")
        logger.info(f"    RMSE:        {metrics['rmse']:.3f}")
        logger.info(f"    Exact match: {metrics['exact']:.1%}")
        logger.info(f"    Within ±1:   {metrics['within_1']:.1%}")
        logger.info(f"    Spearman ρ:  {metrics['rho']:.3f} (p={metrics['p_value']:.4f})")
        logger.info(f"    Cohen's κ:   {metrics['kappa']:.3f}")
        logger.info(f"    LLM mean:    {metrics['llm_mean']:.2f} (σ={metrics['llm_std']:.2f})")
        logger.info(f"    Expert mean: {metrics['expert_mean']:.2f} (σ={metrics['expert_std']:.2f})")

    final_mae = float(np.mean(overall_mae))

    logger.info(f"\n{'=' * 72}")
    logger.info(f"  Overall MAE:         {final_mae:.3f}")
    logger.info(f"  Frameworks scored:   {len(llm_scores)}/{len(expert_scores)}")
    logger.info(f"  Failed:              {failed_count}")
    logger.info(f"  Success rate:        {len(llm_scores)/len(expert_scores)*100:.1f}%")
    logger.info(f"  Aggregation method:  {AGGREGATION}")
    logger.info("=" * 72)

    logger.info("\n" + "=" * 72)
    logger.info("  PHASE 1: DIAGNOSTIC ANALYSIS")
    logger.info("=" * 72)

    logger.info("\n[DIAGNOSTIC] Error Analysis by Dimension...")
    for dim in DIMENSIONS:
        errors = []
        underscores = []
        overscores = []

        for fid in llm_scores.keys():
            llm_val = llm_scores[fid][dim]
            expert_val = expert_scores[fid][dim]
            delta = llm_val - expert_val

            errors.append(abs(delta))
            if delta < 0:
                underscores.append(delta)
            elif delta > 0:
                overscores.append(delta)

        logger.info(f"\n  {dim}:")
        logger.info(f"    Mean error:      {np.mean(errors):.3f}")
        logger.info(f"    Underscoring:    {len(underscores)} cases (avg: {np.mean(underscores) if underscores else 0:.2f})")
        logger.info(f"    Overscoring:     {len(overscores)} cases (avg: {np.mean(overscores) if overscores else 0:.2f})")
        bias = np.mean([llm_scores[fid][dim] - expert_scores[fid][dim] for fid in llm_scores.keys()])
        logger.info(f"    Bias:            {bias:.3f}")

    logger.info("\n[DIAGNOSTIC] Worst-performing frameworks...")
    framework_errors = []
    for fid in llm_scores.keys():
        mean_error = np.mean([abs(llm_scores[fid][dim] - expert_scores[fid][dim]) for dim in DIMENSIONS])
        framework_errors.append((fid, mapping.get(fid, fid), mean_error))

    framework_errors.sort(key=lambda x: x, reverse=True)
    for fid, name, error in framework_errors[:5]:
        logger.info(f"  [{fid}] {name[:50]}: MAE={error:.2f}")

    metrics_path = OUTPUT_DIR / f"metrics_{args.output_suffix}_{timestamp}.json" if args.output_suffix else OUTPUT_DIR / f"metrics_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall_mae": final_mae,
            "success_rate": len(llm_scores)/len(expert_scores),
            "config": {
                "model": MODEL_NAME,
                "aggregation": AGGREGATION,
                "chunking": USE_CHUNKING,
                "chunk_size": CHUNK_SIZE,
                "max_chunks": MAX_CHUNKS,
                "temperature": TEMPERATURE,
                "quantization": "4bit" if USE_4BIT else "none"
            },
            "dimensions": all_metrics
        }, f, indent=2)

    logger.info(f"\n  ✅ Metrics saved to {metrics_path}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
        logger.info("\n  🧹 GPU cache cleared")

if __name__ == "__main__":
    main()
