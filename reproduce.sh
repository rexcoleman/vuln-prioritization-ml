#!/usr/bin/env bash
# reproduce.sh — Full reproduction pipeline for vuln-prioritization-ml
#
# Recreates the conda environment, ingests data, builds features,
# trains all models across 5 seeds, runs diagnostics, and generates figures.
#
# Usage:
#   chmod +x reproduce.sh
#   ./reproduce.sh           # Full run
#   ./reproduce.sh --quick   # Smoke test (1% sample)
#
# Prerequisites:
#   - conda or miniconda installed
#   - Internet access (NVD API, EPSS API, ExploitDB)
#   - ~16 GB RAM recommended for full dataset
#
# Estimated time: ~2-4 hours (full), ~10 min (quick)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="vuln-prioritize"
SEEDS="42,123,456,789,1024"
SAMPLE_FLAG=""

# Parse arguments
if [[ "${1:-}" == "--quick" ]]; then
    SAMPLE_FLAG="--sample-frac 0.01"
    echo "=== QUICK MODE: using 1% sample ==="
fi

# Timestamp for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ============================================================
# Step 0: Environment setup
# ============================================================
log "Step 0/9: Setting up conda environment..."

# Initialize conda for this shell session
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not found. Install miniconda first."
    exit 1
fi

# Create or update the environment
if conda env list | grep -q "^${ENV_NAME} "; then
    log "Environment '${ENV_NAME}' exists. Updating..."
    conda env update -n "$ENV_NAME" -f environment.yml --prune
else
    log "Creating environment '${ENV_NAME}'..."
    conda env create -f environment.yml
fi

conda activate "$ENV_NAME"
log "Activated environment: $CONDA_DEFAULT_ENV"
log "Python: $(python --version)"

# ============================================================
# Step 1: Data ingestion
# ============================================================
log "Step 1/9: Ingesting data from NVD, EPSS, ExploitDB..."

python scripts/ingest_nvd.py
log "  NVD ingestion complete."

python scripts/ingest_epss.py
log "  EPSS ingestion complete."

python scripts/ingest_exploitdb.py
log "  ExploitDB ingestion complete."

# Verify data is ready
python scripts/check_data_ready.py
log "  Data readiness check passed."

# ============================================================
# Step 2: Feature engineering + temporal split
# ============================================================
log "Step 2/9: Building features and creating temporal split..."

python scripts/build_features.py $SAMPLE_FLAG
log "  Feature build complete. Outputs in data/processed/"

# ============================================================
# Step 3: Baseline evaluation (CVSS + EPSS thresholds)
# ============================================================
log "Step 3/9: Running baseline evaluations (CVSS/EPSS thresholds)..."

python scripts/train_baselines.py --data-dir data/
log "  Baselines complete. Outputs in outputs/baselines/"

# ============================================================
# Step 4: Model training (7 models x 5 seeds)
# ============================================================
log "Step 4/9: Training expanded model suite (7 models x 5 seeds)..."

python scripts/train_expanded_models.py --data-dir data/ --seeds "$SEEDS" $SAMPLE_FLAG
log "  Model training complete. Outputs in outputs/models/"

# ============================================================
# Step 5: Diagnostics (sanity baselines, learning curves, complexity)
# ============================================================
log "Step 5/9: Running diagnostics..."

log "  5a: Sanity baselines (dummy classifiers, shuffled labels)..."
python scripts/run_sanity_baselines.py --data-dir data/ --seeds "$SEEDS" $SAMPLE_FLAG

log "  5b: Learning curves (AUC vs training fraction)..."
python scripts/run_learning_curves.py --data-dir data/ --seeds "$SEEDS" $SAMPLE_FLAG

log "  5c: Complexity curves (HP sweeps)..."
python scripts/run_complexity_curves.py --data-dir data/ --seeds "$SEEDS" $SAMPLE_FLAG

log "  Diagnostics complete. Outputs in outputs/diagnostics/"

# ============================================================
# Step 6: Ablation + statistical tests
# ============================================================
log "Step 6/9: Running ablation and statistical tests..."

log "  6a: Feature group ablation..."
python scripts/run_ablation.py --data-dir data/ --seeds "$SEEDS" $SAMPLE_FLAG

log "  6b: Bootstrap CI + McNemar + DeLong tests..."
python scripts/run_statistical_tests.py --data-dir data/ --seeds "$SEEDS" $SAMPLE_FLAG

log "  Ablation and statistical tests complete."

# ============================================================
# Step 7: Explainability (SHAP)
# ============================================================
log "Step 7/9: Running SHAP explainability analysis..."

python scripts/run_explainability.py --data-dir data/
log "  SHAP analysis complete. Outputs in outputs/explainability/"

# ============================================================
# Step 8: Final evaluation on held-out test set
# ============================================================
log "Step 8/9: Running final evaluation on test set..."

python scripts/final_eval.py --data-dir data/
log "  Final evaluation complete. Outputs in outputs/final/"

# ============================================================
# Step 9: Generate figures
# ============================================================
log "Step 9/9: Generating report figures..."

python scripts/make_report_figures.py --project-dir . --output-dir figures/
log "  Figures generated in figures/"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
log "REPRODUCTION COMPLETE"
echo "============================================================"
echo ""
echo "Key outputs:"
echo "  data/processed/          — Feature matrices (train.parquet, test.parquet)"
echo "  data/splits/             — Split metadata (split_info.json)"
echo "  outputs/models/          — Per-seed and summary model results"
echo "  outputs/baselines/       — CVSS/EPSS threshold + sanity baselines"
echo "  outputs/diagnostics/     — Learning curves, complexity curves, ablation"
echo "  outputs/explainability/  — SHAP values and feature importance"
echo "  outputs/final/           — Final test-set evaluation"
echo "  figures/                 — Publication-ready PNG figures"
echo ""
echo "Running test suite..."
python -m pytest tests/ -v --tb=short
echo ""
echo "To re-run tests:  pytest tests/ -v"
echo "To verify:        python scripts/check_data_ready.py"

# --- Gate Validation (R50) ---
if [ -f "$HOME/ml-governance-templates/scripts/check_all_gates.sh" ]; then
    echo "--- Gate Validation (R50) ---"
    bash "$HOME/ml-governance-templates/scripts/check_all_gates.sh" .
else
    echo "WARN: govML not found — skipping gate validation (R50)"
fi
