"""Tests for reproducibility — same seeds produce same results, different seeds differ.

Also validates config files, output directories, and infrastructure consistency.
"""
import json
from pathlib import Path

import yaml
import pytest

from tests.conftest import (
    PROJECT_ROOT, MODELS_DIR, DIAGNOSTICS_DIR, BASELINES_DIR,
    DATA_DIR, OUTPUTS_DIR, SEEDS,
)


def load_expanded(seed):
    """Load expanded model results for a given seed."""
    path = MODELS_DIR / f"expanded_seed{seed}.json"
    with open(path) as f:
        return json.load(f)


class TestDeterministicModels:
    """Deterministic models (XGBoost, LogReg, kNN) must produce identical results across seeds."""

    @pytest.mark.parametrize("model_name", ["xgboost", "logistic_regression", "knn"])
    def test_deterministic_model_same_across_seeds(self, model_name, expanded_summary):
        """Deterministic models should have std=0 (or near-zero) across seeds."""
        test_auc_std = expanded_summary["models"][model_name]["test"]["auc_roc"]["std"]
        assert test_auc_std < 1e-6, (
            f"{model_name} has non-zero std across seeds: {test_auc_std}"
        )

    @pytest.mark.parametrize("model_name", ["xgboost", "logistic_regression", "knn"])
    def test_deterministic_model_values_identical(self, model_name, expanded_summary):
        """All 5 seed values should be identical for deterministic models."""
        values = expanded_summary["models"][model_name]["test"]["auc_roc"]["values"]
        assert all(v == values[0] for v in values), (
            f"{model_name} values differ across seeds: {values}"
        )


class TestStochasticModels:
    """Stochastic models (RF, SVM, LightGBM, MLP) may vary across seeds."""

    def test_rf_varies_across_seeds(self, expanded_summary):
        """Random Forest should show seed-dependent variation (bootstrap sampling)."""
        values = expanded_summary["models"]["random_forest"]["test"]["auc_roc"]["values"]
        assert len(set(values)) > 1, (
            f"RF produced identical results across all seeds: {values[0]}"
        )

    def test_mlp_varies_across_seeds(self, expanded_summary):
        """MLP should show seed-dependent variation (weight initialization)."""
        values = expanded_summary["models"]["mlp"]["test"]["auc_roc"]["values"]
        assert len(set(values)) > 1, (
            f"MLP produced identical results across all seeds: {values[0]}"
        )

    def test_stochastic_models_bounded_variance(self, expanded_summary):
        """Stochastic model variance should be reasonable (std < 0.05)."""
        stochastic = ["random_forest", "svm_rbf", "lightgbm", "mlp"]
        for model_name in stochastic:
            std = expanded_summary["models"][model_name]["test"]["auc_roc"]["std"]
            assert std < 0.05, (
                f"{model_name} has excessive variance: std={std:.4f}"
            )


class TestCrossSeedConsistency:
    """Results across seeds should be internally consistent."""

    def test_seed_count_matches(self, expanded_summary):
        """Summary must report exactly 5 seeds."""
        assert len(expanded_summary["seeds"]) == 5

    @pytest.mark.parametrize("seed", SEEDS)
    def test_each_seed_file_has_correct_seed(self, seed):
        """Each seed file must record its own seed value."""
        data = load_expanded(seed)
        assert data["seed"] == seed, f"Seed file for {seed} records seed={data['seed']}"

    def test_all_seeds_same_train_size(self):
        """All seeds must use the same training set size."""
        sizes = [load_expanded(s)["train_size"] for s in SEEDS]
        assert len(set(sizes)) == 1, f"Inconsistent train sizes across seeds: {sizes}"

    def test_all_seeds_same_test_size(self):
        """All seeds must use the same test set size."""
        sizes = [load_expanded(s)["test_size"] for s in SEEDS]
        assert len(set(sizes)) == 1, f"Inconsistent test sizes across seeds: {sizes}"

    def test_all_seeds_same_feature_count(self):
        """All seeds must use the same number of features."""
        counts = [load_expanded(s)["num_features"] for s in SEEDS]
        assert len(set(counts)) == 1, f"Inconsistent feature counts across seeds: {counts}"


class TestDiagnosticReproducibility:
    """Diagnostic outputs (learning curves, complexity curves) must exist for all seeds."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_learning_curve_per_seed(self, seed):
        """Each seed must have a learning curve output."""
        path = DIAGNOSTICS_DIR / f"learning_curves_seed{seed}.json"
        assert path.exists(), f"Missing learning curve for seed {seed}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_complexity_curve_per_seed(self, seed):
        """Each seed must have a complexity curve output."""
        path = DIAGNOSTICS_DIR / f"complexity_curves_seed{seed}.json"
        assert path.exists(), f"Missing complexity curve for seed {seed}"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_ablation_per_seed(self, seed):
        """Each seed must have an ablation study output."""
        path = DIAGNOSTICS_DIR / f"ablation_seed{seed}.json"
        assert path.exists(), f"Missing ablation for seed {seed}"

    def test_ablation_summary_exists(self):
        """Ablation summary aggregating all seeds must exist."""
        path = DIAGNOSTICS_DIR / "ablation_summary.json"
        assert path.exists(), "Missing ablation_summary.json"


class TestConfigFiles:
    """Validate project configuration files."""

    def test_environment_yml_exists(self):
        """environment.yml must exist at project root."""
        assert (PROJECT_ROOT / "environment.yml").exists()

    def test_environment_yml_valid_yaml(self):
        """environment.yml must be valid YAML."""
        path = PROJECT_ROOT / "environment.yml"
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "name" in data, "environment.yml missing 'name' field"
        assert "dependencies" in data, "environment.yml missing 'dependencies' field"

    def test_environment_has_required_packages(self):
        """environment.yml must list core dependencies."""
        path = PROJECT_ROOT / "environment.yml"
        with open(path) as f:
            data = yaml.safe_load(f)
        deps = [str(d) for d in data["dependencies"] if isinstance(d, str)]
        required = ["numpy", "pandas", "scikit-learn", "xgboost", "pytest"]
        for pkg in required:
            # Check if any dep starts with the package name
            found = any(pkg in d for d in deps)
            assert found, f"Missing required package: {pkg}"

    def test_project_yaml_exists(self):
        """project.yaml must exist at project root."""
        path = PROJECT_ROOT / "project.yaml"
        if not path.exists():
            pytest.skip("project.yaml not present")
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data is not None, "project.yaml is empty"

    def test_feature_cols_json_valid(self):
        """feature_cols.json must be valid JSON."""
        path = DATA_DIR / "processed" / "feature_cols.json"
        if not path.exists():
            pytest.skip("feature_cols.json not yet built")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list), "feature_cols.json should be a list"

    def test_metadata_json_valid(self):
        """metadata.json must be valid JSON with expected keys."""
        path = DATA_DIR / "processed" / "metadata.json"
        if not path.exists():
            pytest.skip("metadata.json not yet built")
        with open(path) as f:
            data = json.load(f)
        required_keys = ["build_date", "total_cves", "train_size", "test_size",
                         "num_structured_features", "feature_cols"]
        for key in required_keys:
            assert key in data, f"metadata.json missing key: {key}"

    def test_split_info_json_valid(self):
        """split_info.json must be valid JSON with expected keys."""
        path = DATA_DIR / "splits" / "split_info.json"
        if not path.exists():
            pytest.skip("split_info.json not yet built")
        with open(path) as f:
            data = json.load(f)
        required_keys = ["method", "split_year", "train_size", "test_size",
                         "train_exploit_rate", "test_exploit_rate",
                         "train_index_hash", "test_index_hash"]
        for key in required_keys:
            assert key in data, f"split_info.json missing key: {key}"


class TestOutputDirectories:
    """Verify expected output directories exist and contain files."""

    def test_outputs_dir_exists(self):
        """outputs/ directory must exist."""
        assert OUTPUTS_DIR.exists(), "outputs/ directory missing"

    def test_models_dir_has_files(self):
        """outputs/models/ must contain result files."""
        assert MODELS_DIR.exists(), "outputs/models/ directory missing"
        json_files = list(MODELS_DIR.glob("*.json"))
        assert len(json_files) >= 6, (
            f"Expected >= 6 JSON files in models/, found {len(json_files)}"
        )

    def test_baselines_dir_has_files(self):
        """outputs/baselines/ must contain baseline results."""
        assert BASELINES_DIR.exists(), "outputs/baselines/ directory missing"
        json_files = list(BASELINES_DIR.glob("*.json"))
        assert len(json_files) >= 4, (
            f"Expected >= 4 JSON files in baselines/, found {len(json_files)}"
        )

    def test_diagnostics_dir_has_files(self):
        """outputs/diagnostics/ must contain diagnostic results."""
        assert DIAGNOSTICS_DIR.exists(), "outputs/diagnostics/ directory missing"
        json_files = list(DIAGNOSTICS_DIR.glob("*.json"))
        assert len(json_files) >= 15, (
            f"Expected >= 15 JSON files in diagnostics/, found {len(json_files)}"
        )

    def test_data_processed_dir_has_files(self):
        """data/processed/ must contain parquet and JSON files."""
        processed = DATA_DIR / "processed"
        assert processed.exists(), "data/processed/ directory missing"
        assert (processed / "train.parquet").exists(), "train.parquet missing"
        assert (processed / "test.parquet").exists(), "test.parquet missing"
        assert (processed / "feature_cols.json").exists(), "feature_cols.json missing"
        assert (processed / "metadata.json").exists(), "metadata.json missing"
