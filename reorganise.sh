#!/bin/bash

echo "=== FrameRATER Project Reorganization ==="
echo "Creating new directory structure..."

# Create main directories
mkdir -p data/{raw,processed,ground_truth}
mkdir -p src/{01_traditional_ml/config,02_llm_approach,03_analysis,utils}
mkdir -p experiments/{01_ml_baseline/results,02_llm_optimization/results}
mkdir -p results/{final,paper_tables}
mkdir -p paper/figures
mkdir -p docs

echo "Moving data files..."
# Move raw PDFs with sequential naming
cp freq-analysis/data/*.pdf data/raw/
# Rename for clarity (optional - can provide Python script for this)

# Move processed texts
cp freq-analysis/processed/*.txt data/processed/

# Move ground truth
cp freq-analysis/framework_mapping.csv data/ground_truth/
cp freq-analysis/scripts/labels.csv data/ground_truth/expert_labels.csv

echo "Moving source code..."
# Traditional ML
cp freq-analysis/scripts/preprocess.py src/01_traditional_ml/
cp freq-analysis/scripts/frequency.py src/01_traditional_ml/feature_extraction.py
cp freq-analysis/scripts/analysis.py src/01_traditional_ml/train_models.py
cp freq-analysis/scripts/keyword_map.py src/01_traditional_ml/keyword_dictionaries.py
cp freq-analysis/scripts/dimension_weights.csv src/01_traditional_ml/config/
cp ml-model/svr_config.json src/01_traditional_ml/config/

# LLM approach
cp llm-pipeline/run_stage1.py src/02_llm_approach/stage1_generation_tuning.py
cp llm-pipeline/run_stage2.py src/02_llm_approach/stage2_text_processing.py
cp llm-pipeline/llm_3090_score.py src/02_llm_approach/llm_inference.py

# Analysis
cp ml-model/composite_score_calculator.py src/03_analysis/

echo "Moving results..."
# ML results
cp ml-model/nlp_scores.csv results/final/ml_predictions.csv
cp ml-model/PQC_Framework_Composite_Scores.csv results/final/framework_composite_scores.csv

# LLM final results (optimized configuration)
cp llm-pipeline/outputs/llm_scores_CS2000_MC2_20260520_043809.csv results/final/llm_predictions.csv

# Experiment results
cp -r llm-pipeline/outputs/llm_scores_test_*.csv experiments/02_llm_optimization/results/
cp -r llm-pipeline/outputs/llm_scores_T*.csv experiments/02_llm_optimization/results/
cp -r llm-pipeline/outputs/llm_scores_CS*.csv experiments/02_llm_optimization/results/

# Paper tables
cp freq-analysis/scripts/final_results.csv results/paper_tables/table5_source_coverage.csv

echo "Creating requirements.txt..."
cat freq-analysis/requirements.txt llm-pipeline/requirements.txt | sort | uniq > requirements.txt

echo "=== Reorganization Complete ==="
echo "Next steps:"
echo "1. Review the new structure"
echo "2. Run: python create_documentation.py"
echo "3. Test reproduction with: bash experiments/01_ml_baseline/run_ml_pipeline.sh"
