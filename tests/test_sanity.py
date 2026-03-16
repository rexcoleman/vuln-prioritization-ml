"""Sanity tests — verify models beat dummy classifiers and produce valid outputs."""
import json
from pathlib import Path

import pytest

from tests.conftest import (
    MODELS_DIR, BASELINES_DIR, DIAGNOSTICS_DIR,
    SEEDS, EXPECTED_MODELS,
)


class TestModelBeatsDummy:
    """Every real model must beat dummy baselines on AUC-ROC."""

    def test_all_models_beat_stratified_dummy(self, expanded_summary, sanity_stratified):
        """All 7 models must have mean test AUC > stratified dummy mean AUC."""
        dummy_auc = sanity_stratified["summary"]["auc_mean"]
        for model_name in EXPECTED_MODELS:
            model_auc = expanded_summary["models"][model_name]["test"]["auc_roc"]["mean"]
            assert model_auc > dummy_auc, (
                f"{model_name} mean AUC {model_auc:.4f} <= stratified dummy {dummy_auc:.4f}"
            )

    def test_all_models_beat_most_frequent_dummy(self, expanded_summary, sanity_most_frequent):
        """All 7 models must have mean test AUC > most-frequent dummy (0.500)."""
        dummy_auc = sanity_most_frequent["summary"]["auc_mean"]
        for model_name in EXPECTED_MODELS:
            model_auc = expanded_summary["models"][model_name]["test"]["auc_roc"]["mean"]
            assert model_auc > dummy_auc, (
                f"{model_name} mean AUC {model_auc:.4f} <= most-frequent dummy {dummy_auc:.4f}"
            )

    def test_all_models_beat_shuffled_labels(self, expanded_summary, sanity_shuffled):
        """All 7 models must have mean test AUC > shuffled-label baseline."""
        dummy_auc = sanity_shuffled["summary"]["auc_mean"]
        for model_name in EXPECTED_MODELS:
            model_auc = expanded_summary["models"][model_name]["test"]["auc_roc"]["mean"]
            assert model_auc > dummy_auc, (
                f"{model_name} mean AUC {model_auc:.4f} <= shuffled dummy {dummy_auc:.4f}"
            )

    def test_best_model_exceeds_random_by_large_margin(self, expanded_summary):
        """Best model AUC must exceed 0.5 (random) by at least 15pp."""
        best_auc = max(
            expanded_summary["models"][m]["test"]["auc_roc"]["mean"]
            for m in EXPECTED_MODELS
        )
        assert best_auc > 0.65, (
            f"Best model AUC {best_auc:.4f} does not exceed 0.65 threshold"
        )


class TestModelBeatsBaseline:
    """Models must beat CVSS threshold baselines."""

    def test_logreg_beats_cvss(self, expanded_summary, baselines_seed42):
        """Logistic Regression AUC must beat best CVSS threshold."""
        # Best CVSS threshold AUC from baselines
        cvss_baselines = [
            b for b in baselines_seed42["baselines"]
            if b["method"].startswith("cvss_threshold")
        ]
        best_cvss_auc = max(b["auc_roc"] for b in cvss_baselines)
        logreg_auc = expanded_summary["models"]["logistic_regression"]["test"]["auc_roc"]["mean"]
        assert logreg_auc > best_cvss_auc, (
            f"LogReg AUC {logreg_auc:.4f} <= best CVSS {best_cvss_auc:.4f}"
        )

    def test_multiple_models_beat_cvss(self, expanded_summary, baselines_seed42):
        """At least 3 models must beat the best CVSS threshold."""
        cvss_baselines = [
            b for b in baselines_seed42["baselines"]
            if b["method"].startswith("cvss_threshold")
        ]
        best_cvss_auc = max(b["auc_roc"] for b in cvss_baselines)
        models_beating_cvss = sum(
            1 for m in EXPECTED_MODELS
            if expanded_summary["models"][m]["test"]["auc_roc"]["mean"] > best_cvss_auc
        )
        assert models_beating_cvss >= 3, (
            f"Only {models_beating_cvss} models beat CVSS (need >= 3)"
        )


class TestPredictionValidity:
    """Verify predictions are valid probabilities, not constant."""

    def test_auc_above_random(self, expanded_seed42):
        """All models must have test AUC > 0.5 (better than random)."""
        for model_name, model_data in expanded_seed42["models"].items():
            auc = model_data["test"]["auc_roc"]
            assert auc > 0.5, f"{model_name} test AUC {auc:.4f} <= 0.5 (random)"

    def test_train_auc_above_test_auc(self, expanded_seed42):
        """Train AUC should be >= test AUC (no inverted generalization)."""
        for model_name, model_data in expanded_seed42["models"].items():
            train_auc = model_data["train"]["auc_roc"]
            test_auc = model_data["test"]["auc_roc"]
            assert train_auc >= test_auc - 0.01, (
                f"{model_name}: train AUC {train_auc:.4f} < test AUC {test_auc:.4f}"
            )

    def test_precision_recall_non_negative(self, expanded_seed42):
        """Precision and recall must be in [0, 1]."""
        for model_name, model_data in expanded_seed42["models"].items():
            for split in ["train", "test"]:
                p = model_data[split]["precision"]
                r = model_data[split]["recall"]
                assert 0.0 <= p <= 1.0, f"{model_name}/{split} precision {p} invalid"
                assert 0.0 <= r <= 1.0, f"{model_name}/{split} recall {r} invalid"

    def test_accuracy_non_trivial(self, expanded_seed42):
        """Accuracy must be > 0 and <= 1."""
        for model_name, model_data in expanded_seed42["models"].items():
            for split in ["train", "test"]:
                acc = model_data[split]["accuracy"]
                assert 0.0 < acc <= 1.0, f"{model_name}/{split} accuracy {acc} invalid"


class TestSanityMargins:
    """Model AUC must exceed dummies by meaningful margins (not just barely)."""

    def test_model_beats_stratified_by_5pp(self, expanded_summary, sanity_stratified):
        """Best model AUC > stratified dummy AUC + 5pp."""
        dummy_auc = sanity_stratified["summary"]["auc_mean"]
        best_auc = max(
            expanded_summary["models"][m]["test"]["auc_roc"]["mean"]
            for m in EXPECTED_MODELS
        )
        assert best_auc > dummy_auc + 0.05, (
            f"Best model AUC {best_auc:.4f} not > stratified dummy {dummy_auc:.4f} + 5pp"
        )

    def test_model_beats_most_frequent_by_5pp(self, expanded_summary, sanity_most_frequent):
        """Best model AUC > most-frequent dummy AUC + 5pp."""
        dummy_auc = sanity_most_frequent["summary"]["auc_mean"]
        best_auc = max(
            expanded_summary["models"][m]["test"]["auc_roc"]["mean"]
            for m in EXPECTED_MODELS
        )
        assert best_auc > dummy_auc + 0.05, (
            f"Best model AUC {best_auc:.4f} not > most-frequent dummy {dummy_auc:.4f} + 5pp"
        )

    def test_shuffled_label_much_worse_than_real(self, expanded_summary, sanity_shuffled):
        """Shuffled-label AUC must be at least 10pp below the best real model."""
        dummy_auc = sanity_shuffled["summary"]["auc_mean"]
        best_auc = max(
            expanded_summary["models"][m]["test"]["auc_roc"]["mean"]
            for m in EXPECTED_MODELS
        )
        assert best_auc > dummy_auc + 0.10, (
            f"Best model AUC {best_auc:.4f} not > shuffled {dummy_auc:.4f} + 10pp"
        )

    def test_all_seeds_produce_auc_above_threshold(self, expanded_summary):
        """All seeds for the best model must produce AUC > 0.8 (no catastrophic failure)."""
        for model_name in EXPECTED_MODELS:
            values = expanded_summary["models"][model_name]["test"]["auc_roc"]["values"]
            for i, v in enumerate(values):
                assert v > 0.5, (
                    f"{model_name} seed index {i} has AUC {v:.4f} <= 0.5"
                )

    def test_logreg_all_seeds_above_08(self, expanded_summary):
        """LogReg (best model) must have AUC > 0.8 on every seed."""
        values = expanded_summary["models"]["logistic_regression"]["test"]["auc_roc"]["values"]
        for i, v in enumerate(values):
            assert v > 0.8, (
                f"LogReg seed index {i} has AUC {v:.4f} <= 0.8"
            )
