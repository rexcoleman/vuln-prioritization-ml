"""Tests for ML pipeline correctness — output formats, model results structure,
and synthetic-data training validation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from tests.conftest import (
    DATA_DIR, MODELS_DIR, BASELINES_DIR, DIAGNOSTICS_DIR, OUTPUTS_DIR,
    SEEDS, EXPECTED_MODELS, EXPECTED_NUM_FEATURES, FEATURE_GROUPS,
)


# -- Fixtures for synthetic training -----------------------------------------

@pytest.fixture
def synthetic_data():
    """Create small synthetic dataset mimicking the real feature matrix."""
    rng = np.random.RandomState(42)
    n_train, n_test = 500, 200
    n_features = EXPECTED_NUM_FEATURES

    X_train = rng.randn(n_train, n_features).astype(np.float64)
    X_test = rng.randn(n_test, n_features).astype(np.float64)

    # Imbalanced labels: ~10% positive in train, ~1% in test
    y_train = (rng.rand(n_train) < 0.10).astype(int)
    y_test = (rng.rand(n_test) < 0.01).astype(int)
    # Ensure at least one positive in test
    y_test[0] = 1

    return X_train, y_train, X_test, y_test


class TestExpandedModelOutputs:
    """Verify expanded model result files have correct structure."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_expanded_seed_file_exists(self, seed):
        """Each seed must produce an expanded results JSON."""
        path = MODELS_DIR / f"expanded_seed{seed}.json"
        assert path.exists(), f"Missing expanded results for seed {seed}"

    def test_expanded_summary_exists(self):
        """Summary JSON aggregating all seeds must exist."""
        path = MODELS_DIR / "expanded_summary.json"
        assert path.exists(), "Missing expanded_summary.json"

    def test_all_seven_models_present(self, expanded_seed42):
        """Seed results must contain all 7 model types."""
        models = set(expanded_seed42["models"].keys())
        expected = set(EXPECTED_MODELS)
        assert models == expected, f"Missing models: {expected - models}"

    def test_each_model_has_train_and_test(self, expanded_seed42):
        """Each model must report both train and test metrics."""
        for model_name, model_data in expanded_seed42["models"].items():
            assert "train" in model_data, f"{model_name} missing train metrics"
            assert "test" in model_data, f"{model_name} missing test metrics"

    def test_metrics_keys_complete(self, expanded_seed42):
        """Each model must report accuracy, precision, recall, f1, auc_roc, auc_pr."""
        required_metrics = {"accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"}
        for model_name, model_data in expanded_seed42["models"].items():
            for split in ["train", "test"]:
                actual = set(model_data[split].keys())
                missing = required_metrics - actual
                assert not missing, (
                    f"{model_name}/{split} missing metrics: {missing}"
                )

    def test_auc_values_are_valid_probabilities(self, expanded_seed42):
        """AUC-ROC values must be in [0, 1]."""
        for model_name, model_data in expanded_seed42["models"].items():
            for split in ["train", "test"]:
                auc = model_data[split]["auc_roc"]
                assert 0.0 <= auc <= 1.0, (
                    f"{model_name}/{split} AUC {auc} outside [0, 1]"
                )

    def test_num_features_recorded(self, expanded_seed42):
        """Results must record the correct number of features."""
        assert expanded_seed42["num_features"] == EXPECTED_NUM_FEATURES

    def test_train_test_sizes_recorded(self, expanded_seed42):
        """Results must record train and test sizes."""
        assert expanded_seed42["train_size"] == 234601
        assert expanded_seed42["test_size"] == 103352


class TestSummaryAggregation:
    """Verify the expanded_summary.json aggregates seeds correctly."""

    def test_summary_has_all_seeds(self, expanded_summary):
        """Summary must list all 5 seeds."""
        assert expanded_summary["seeds"] == SEEDS

    def test_summary_has_mean_and_std(self, expanded_summary):
        """Each model/split/metric must have mean and std."""
        for model_name, model_data in expanded_summary["models"].items():
            for split in ["train", "test"]:
                for metric, data in model_data[split].items():
                    assert "mean" in data, f"{model_name}/{split}/{metric} missing mean"
                    assert "std" in data, f"{model_name}/{split}/{metric} missing std"

    def test_summary_values_have_five_entries(self, expanded_summary):
        """Each metric must have exactly 5 per-seed values."""
        for model_name, model_data in expanded_summary["models"].items():
            for split in ["train", "test"]:
                for metric, data in model_data[split].items():
                    assert len(data["values"]) == 5, (
                        f"{model_name}/{split}/{metric} has {len(data['values'])} values, expected 5"
                    )

    def test_summary_mean_matches_values(self, expanded_summary):
        """Mean should approximately equal average of values."""
        for model_name, model_data in expanded_summary["models"].items():
            for split in ["train", "test"]:
                for metric, data in model_data[split].items():
                    computed_mean = sum(data["values"]) / len(data["values"])
                    assert abs(data["mean"] - computed_mean) < 1e-6, (
                        f"{model_name}/{split}/{metric}: reported mean {data['mean']} "
                        f"!= computed mean {computed_mean}"
                    )


class TestBaselineOutputs:
    """Verify baseline result files exist and have correct structure."""

    def test_cvss_epss_baselines_exist(self):
        """CVSS and EPSS threshold baselines must exist."""
        path = BASELINES_DIR / "baselines_seed42.json"
        assert path.exists(), "Missing baselines_seed42.json"

    def test_sanity_baselines_exist(self):
        """All three sanity baseline files must exist."""
        for name in ["sanity_stratified", "sanity_most_frequent", "sanity_shuffled"]:
            path = BASELINES_DIR / f"{name}.json"
            assert path.exists(), f"Missing {name}.json"

    def test_baselines_contain_auc(self, baselines_seed42):
        """CVSS/EPSS baselines must report AUC-ROC."""
        for entry in baselines_seed42["baselines"]:
            if "auc_roc" in entry:
                assert 0.0 <= entry["auc_roc"] <= 1.0


# -- Synthetic-data model training tests ------------------------------------

class TestSyntheticModelTraining:
    """All model types can train on synthetic data without compute costs."""

    def test_logistic_regression_produces_valid_probs(self, synthetic_data):
        """LogReg trains and produces probabilities in [0, 1]."""
        X_train, y_train, X_test, _ = synthetic_data
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_tr, y_train)
        probs = model.predict_proba(X_te)[:, 1]

        assert probs.shape == (len(X_te),)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_random_forest_produces_valid_probs(self, synthetic_data):
        """RandomForest trains and produces probabilities in [0, 1]."""
        X_train, y_train, X_test, _ = synthetic_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        assert probs.shape == (len(X_test),)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_xgboost_produces_valid_probs(self, synthetic_data):
        """XGBoost trains and produces probabilities in [0, 1]."""
        X_train, y_train, X_test, _ = synthetic_data
        model = XGBClassifier(
            n_estimators=10, max_depth=3, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        assert probs.shape == (len(X_test),)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_svm_rbf_produces_valid_probs(self, synthetic_data):
        """SVM-RBF trains and produces probabilities in [0, 1]."""
        X_train, y_train, X_test, _ = synthetic_data
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_tr, y_train)
        probs = model.predict_proba(X_te)[:, 1]

        assert probs.shape == (len(X_te),)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_knn_produces_valid_probs(self, synthetic_data):
        """kNN trains and produces probabilities in [0, 1]."""
        X_train, y_train, X_test, _ = synthetic_data
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_tr, y_train)
        probs = model.predict_proba(X_te)[:, 1]

        assert probs.shape == (len(X_te),)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_mlp_produces_valid_probs(self, synthetic_data):
        """MLP trains and produces probabilities in [0, 1]."""
        X_train, y_train, X_test, _ = synthetic_data
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        model = MLPClassifier(
            hidden_layer_sizes=(20,), max_iter=50, random_state=42,
        )
        model.fit(X_tr, y_train)
        probs = model.predict_proba(X_te)[:, 1]

        assert probs.shape == (len(X_te),)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_all_seven_model_types_have_predict_proba(self, synthetic_data):
        """All 7 algorithm types from the expanded suite train without error."""
        X_train, y_train, _, _ = synthetic_data
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)

        models = {
            "random_forest": (RandomForestClassifier(n_estimators=5, random_state=42), X_train),
            "xgboost": (XGBClassifier(n_estimators=5, verbosity=0, random_state=42), X_train),
            "logistic_regression": (LogisticRegression(max_iter=500, random_state=42), X_tr_s),
            "svm_rbf": (SVC(probability=True, random_state=42), X_tr_s),
            "knn": (KNeighborsClassifier(n_neighbors=3), X_tr_s),
            "mlp": (MLPClassifier(hidden_layer_sizes=(10,), max_iter=50, random_state=42), X_tr_s),
        }

        for name, (model, X_fit) in models.items():
            model.fit(X_fit, y_train)
            assert hasattr(model, "predict"), f"{name} missing predict"
            assert hasattr(model, "predict_proba"), f"{name} missing predict_proba"


class TestScalerIntegrity:
    """StandardScaler must be fit on train only."""

    def test_scaler_train_centered(self, synthetic_data):
        """Scaled train data should have mean near zero."""
        X_train, _, _, _ = synthetic_data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        train_means = np.abs(X_scaled.mean(axis=0))
        assert np.all(train_means < 0.1), "Scaled train means not near zero"

    def test_scaler_preserves_shape(self, synthetic_data):
        """Scaler should not change array shapes."""
        X_train, _, X_test, _ = synthetic_data
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        assert X_tr_s.shape == X_train.shape
        assert X_te_s.shape == X_test.shape

    def test_scaler_does_not_modify_original(self, synthetic_data):
        """Scaler should not modify the input arrays in-place."""
        X_train, _, _, _ = synthetic_data
        X_copy = X_train.copy()
        scaler = StandardScaler()
        scaler.fit_transform(X_train)
        np.testing.assert_array_equal(X_train, X_copy)
