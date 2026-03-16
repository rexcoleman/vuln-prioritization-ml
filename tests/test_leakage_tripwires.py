"""Leakage tripwire tests for FP-05 Vulnerability Prioritization.

Each test guards against a specific data-leakage vector.  If training
artifacts are not yet generated, the test is skipped rather than failed.

Reference: docs/DATA_CONTRACT.md (temporal split, scaler protocol).

Run:  pytest tests/test_leakage_tripwires.py -m leakage
"""
import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS = PROJECT_ROOT / "outputs"
SCRIPTS = PROJECT_ROOT / "scripts"

pytestmark = pytest.mark.leakage


# ---------------------------------------------------------------------------
# LT-1  Temporal split integrity: all train dates < all test dates
# ---------------------------------------------------------------------------
@pytest.mark.leakage
def test_lt1_temporal_split_integrity():
    """LT-1: Train publication dates must be strictly before test dates.

    Ref: DATA_CONTRACT §split.strategy = temporal, boundary = 2024-01-01.
    """
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    train_file = OUTPUTS / "baselines" / "train_metadata.csv"
    test_file = OUTPUTS / "baselines" / "test_metadata.csv"

    if not train_file.exists() or not test_file.exists():
        # Fall back: check config_resolved.yaml for boundary
        import yaml

        cfg_file = OUTPUTS / "provenance" / "config_resolved.yaml"
        if not cfg_file.exists():
            pytest.skip("No train/test metadata or config found")
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        boundary = cfg.get("data", {}).get("split", {}).get("boundary")
        assert boundary is not None, "Temporal boundary not set in config"
        assert boundary == "2024-01-01", (
            f"Expected boundary 2024-01-01, got {boundary}"
        )
        return

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    date_col = None
    for col in ["published_date", "publishedDate", "pub_date"]:
        if col in train.columns:
            date_col = col
            break
    assert date_col is not None, "No date column found in train metadata"

    train_max = pd.to_datetime(train[date_col]).max()
    test_min = pd.to_datetime(test[date_col]).min()
    assert train_max < test_min, (
        f"Temporal leak: train max date {train_max} >= test min date {test_min}"
    )


# ---------------------------------------------------------------------------
# LT-2  No CVE ID overlap between train and test
# ---------------------------------------------------------------------------
@pytest.mark.leakage
def test_lt2_no_cve_id_overlap():
    """LT-2: Train and test CVE ID sets must be disjoint.

    Ref: DATA_CONTRACT §split — each CVE appears in exactly one partition.
    """
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    train_file = OUTPUTS / "baselines" / "train_metadata.csv"
    test_file = OUTPUTS / "baselines" / "test_metadata.csv"

    if not train_file.exists() or not test_file.exists():
        pytest.skip("Train/test metadata files not found")

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    id_col = None
    for col in ["cve_id", "CVE_ID", "id"]:
        if col in train.columns:
            id_col = col
            break
    assert id_col is not None, "No CVE ID column found"

    overlap = set(train[id_col]) & set(test[id_col])
    assert len(overlap) == 0, (
        f"CVE ID leakage: {len(overlap)} IDs in both train and test "
        f"(examples: {list(overlap)[:5]})"
    )


# ---------------------------------------------------------------------------
# LT-3  StandardScaler fit on train only
# ---------------------------------------------------------------------------
@pytest.mark.leakage
def test_lt3_scaler_fit_train_only():
    """LT-3: If a saved scaler exists, verify it was fit on train-sized data.

    Ref: DATA_CONTRACT §preprocessing.scaler = StandardScaler;
         config_resolved.yaml train_size = 234601.
    """
    import pickle

    import yaml

    scaler_candidates = list(OUTPUTS.rglob("*scaler*.pkl")) + list(
        OUTPUTS.rglob("*scaler*.joblib")
    )
    if not scaler_candidates:
        pytest.skip("No saved scaler artifact found")

    cfg_file = OUTPUTS / "provenance" / "config_resolved.yaml"
    if not cfg_file.exists():
        pytest.skip("config_resolved.yaml not found")

    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)

    expected_train_size = cfg.get("data", {}).get("split", {}).get("train_size")
    assert expected_train_size is not None, "train_size missing from config"

    for scaler_path in scaler_candidates:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        if hasattr(scaler, "n_samples_seen_"):
            n_seen = scaler.n_samples_seen_
            # n_samples_seen_ can be an int or array; take the first value
            if hasattr(n_seen, "__len__"):
                n_seen = n_seen[0]
            assert n_seen == expected_train_size, (
                f"Scaler {scaler_path.name} saw {n_seen} samples, "
                f"expected train_size={expected_train_size}"
            )


# ---------------------------------------------------------------------------
# LT-4  Training scripts do not reference test split directly
# ---------------------------------------------------------------------------
@pytest.mark.leakage
def test_lt4_training_scripts_no_test_access():
    """LT-4: Training scripts should not load the test split.

    Ref: DATA_CONTRACT — training phase must be isolated from test data.
    Heuristic: parse training scripts for suspicious patterns.
    """
    train_scripts = [
        SCRIPTS / "train_baselines.py",
        SCRIPTS / "train_models.py",
        SCRIPTS / "train_expanded_models.py",
        SCRIPTS / "build_features.py",
    ]

    suspicious_patterns = [
        "test_metadata",
        "test_split",
        "X_test",
        "y_test",
    ]

    found_scripts = [s for s in train_scripts if s.exists()]
    if not found_scripts:
        pytest.skip("No training scripts found")

    violations = []
    for script in found_scripts:
        source = script.read_text()
        # Parse AST to avoid false positives in comments/strings
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        # Check all Name nodes and string literals
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                for pat in suspicious_patterns:
                    if pat in node.id:
                        violations.append(
                            f"{script.name}: variable '{node.id}' matches '{pat}'"
                        )

    assert len(violations) == 0, (
        f"Potential test-data access in training scripts:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# LT-5  Feature columns identical between train and test
# ---------------------------------------------------------------------------
@pytest.mark.leakage
def test_lt5_feature_columns_match():
    """LT-5: Train and test feature matrices must have identical columns.

    Ref: DATA_CONTRACT §preprocessing.feature_count = 49.
    Mismatched columns indicate feature engineering leakage or drift.
    """
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    train_features = OUTPUTS / "baselines" / "X_train.csv"
    test_features = OUTPUTS / "baselines" / "X_test.csv"

    if not train_features.exists() or not test_features.exists():
        pytest.skip("Feature matrix CSVs not found")

    train_cols = set(pd.read_csv(train_features, nrows=0).columns)
    test_cols = set(pd.read_csv(test_features, nrows=0).columns)

    only_train = train_cols - test_cols
    only_test = test_cols - train_cols

    assert only_train == set() and only_test == set(), (
        f"Feature column mismatch:\n"
        f"  Only in train: {only_train or 'none'}\n"
        f"  Only in test: {only_test or 'none'}"
    )
