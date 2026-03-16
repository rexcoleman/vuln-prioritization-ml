"""Shared fixtures for vuln-prioritization-ml test suite."""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = OUTPUTS_DIR / "models"
BASELINES_DIR = OUTPUTS_DIR / "baselines"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"

SEEDS = [42, 123, 456, 789, 1024]

FEATURE_GROUPS = {
    "epss_features": ["epss", "epss_percentile"],
    "cvss_features": ["cvss_v3", "cvss_v2", "cvss_score", "has_cvss_v3"],
    "temporal_features": ["pub_year", "pub_month", "pub_dayofweek", "cve_age_days"],
    "description_stats": ["desc_length", "desc_word_count"],
    "reference_features": ["ref_count", "has_exploit_ref", "has_patch_ref"],
    "vendor_features": ["vendor_cve_count"],
    "cwe_features": [
        "cwe_count", "has_cwe",
        "cwe_CWE-79", "cwe_CWE-89", "cwe_CWE-119", "cwe_CWE-20",
        "cwe_CWE-787", "cwe_CWE-200", "cwe_CWE-352", "cwe_CWE-22",
        "cwe_CWE-125", "cwe_CWE-862", "cwe_CWE-416", "cwe_CWE-264",
        "cwe_CWE-78", "cwe_CWE-476", "cwe_CWE-94", "cwe_CWE-284",
        "cwe_CWE-74", "cwe_CWE-287", "cwe_CWE-434", "cwe_CWE-120",
    ],
    "text_keywords": [
        "kw_remote_code_execution", "kw_sql_injection", "kw_buffer_overflow",
        "kw_xss", "kw_privilege_escalation", "kw_authentication_bypass",
        "kw_denial_of_service", "kw_information_disclosure", "kw_arbitrary_code",
        "kw_allows_attackers", "kw_crafted",
    ],
}

EXPECTED_NUM_FEATURES = 49
EXPECTED_MODELS = [
    "random_forest", "xgboost", "logistic_regression",
    "svm_rbf", "lightgbm", "knn", "mlp",
]


@pytest.fixture
def feature_cols():
    """Load the canonical feature column list."""
    path = DATA_DIR / "processed" / "feature_cols.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def metadata():
    """Load the processed data metadata."""
    path = DATA_DIR / "processed" / "metadata.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def expanded_seed42():
    """Load expanded model results for seed 42."""
    path = MODELS_DIR / "expanded_seed42.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def expanded_summary():
    """Load the expanded model summary (5 seeds)."""
    path = MODELS_DIR / "expanded_summary.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def sanity_stratified():
    """Load stratified dummy baseline results."""
    path = BASELINES_DIR / "sanity_stratified.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def sanity_most_frequent():
    """Load most-frequent dummy baseline results."""
    path = BASELINES_DIR / "sanity_most_frequent.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def sanity_shuffled():
    """Load shuffled-label baseline results."""
    path = BASELINES_DIR / "sanity_shuffled.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def baselines_seed42():
    """Load CVSS/EPSS threshold baselines for seed 42."""
    path = BASELINES_DIR / "baselines_seed42.json"
    with open(path) as f:
        return json.load(f)
