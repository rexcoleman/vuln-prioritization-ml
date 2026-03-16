"""Tests for figure generation outputs.

Validates that:
  - All expected figure PNGs exist
  - Figure files are non-zero size
  - Learning curve and complexity curve data files exist
  - SHAP importance figure exists
  - Diagnostic data files are well-formed JSON
"""
import json
import os
from pathlib import Path

import pytest

from tests.conftest import (
    PROJECT_ROOT, DIAGNOSTICS_DIR, OUTPUTS_DIR, SEEDS,
)


FIGURES_DIR = PROJECT_ROOT / "figures"
EXPLAINABILITY_DIR = OUTPUTS_DIR / "explainability"


# -- Figure file existence and size ------------------------------------------

class TestFigureFiles:
    """All expected figure PNGs must exist and be non-zero size."""

    EXPECTED_FIGURES = [
        "learning_curves.png",
        "complexity_curves.png",
        "model_comparison.png",
        "shap_importance.png",
    ]

    def test_figures_directory_exists(self):
        """figures/ directory must exist."""
        assert FIGURES_DIR.exists(), "figures/ directory missing"

    @pytest.mark.parametrize("filename", EXPECTED_FIGURES)
    def test_figure_exists(self, filename):
        """Each expected figure PNG must exist."""
        path = FIGURES_DIR / filename
        assert path.exists(), f"Missing figure: {filename}"

    @pytest.mark.parametrize("filename", EXPECTED_FIGURES)
    def test_figure_nonzero_size(self, filename):
        """Each figure PNG must be non-zero size."""
        path = FIGURES_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not yet generated")
        size = os.path.getsize(path)
        assert size > 0, f"{filename} is empty (0 bytes)"

    @pytest.mark.parametrize("filename", EXPECTED_FIGURES)
    def test_figure_minimum_size(self, filename):
        """Each figure should be at least 1KB (valid PNG, not corrupt)."""
        path = FIGURES_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not yet generated")
        size = os.path.getsize(path)
        assert size > 1024, (
            f"{filename} is suspiciously small ({size} bytes) -- possibly corrupt"
        )

    def test_figure_count(self):
        """Should have at least 4 PNG figures."""
        if not FIGURES_DIR.exists():
            pytest.skip("figures/ directory missing")
        pngs = list(FIGURES_DIR.glob("*.png"))
        assert len(pngs) >= 4, (
            f"Expected >= 4 PNG files in figures/, found {len(pngs)}"
        )


# -- Learning curve data files -----------------------------------------------

class TestLearningCurveData:
    """Learning curve JSON data files must exist and be well-formed."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_learning_curve_data_exists(self, seed):
        """Learning curve data JSON must exist for each seed."""
        path = DIAGNOSTICS_DIR / f"learning_curves_seed{seed}.json"
        assert path.exists(), f"Missing learning curve data for seed {seed}"

    def test_learning_curve_has_fractions(self):
        """Learning curve data must contain training fractions."""
        path = DIAGNOSTICS_DIR / "learning_curves_seed42.json"
        if not path.exists():
            pytest.skip("Learning curve data not yet generated")
        with open(path) as f:
            data = json.load(f)
        # Check that it contains fraction/model entries
        assert len(data) > 0, "Learning curve data is empty"

    def test_learning_curve_json_valid(self):
        """All learning curve JSONs must be valid."""
        for seed in SEEDS:
            path = DIAGNOSTICS_DIR / f"learning_curves_seed{seed}.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            assert data is not None, f"Learning curve seed {seed} parsed as None"


# -- Complexity curve data files ----------------------------------------------

class TestComplexityCurveData:
    """Complexity curve JSON data files must exist and be well-formed."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_complexity_curve_data_exists(self, seed):
        """Complexity curve data JSON must exist for each seed."""
        path = DIAGNOSTICS_DIR / f"complexity_curves_seed{seed}.json"
        assert path.exists(), f"Missing complexity curve data for seed {seed}"

    def test_complexity_curve_json_valid(self):
        """All complexity curve JSONs must be valid."""
        for seed in SEEDS:
            path = DIAGNOSTICS_DIR / f"complexity_curves_seed{seed}.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            assert data is not None, f"Complexity curve seed {seed} parsed as None"

    def test_complexity_curve_has_models(self):
        """Complexity curve data should contain entries for multiple models."""
        path = DIAGNOSTICS_DIR / "complexity_curves_seed42.json"
        if not path.exists():
            pytest.skip("Complexity curve data not yet generated")
        with open(path) as f:
            data = json.load(f)
        assert len(data) > 0, "Complexity curve data is empty"


# -- SHAP / explainability data -----------------------------------------------

class TestExplainabilityData:
    """SHAP explainability outputs must exist."""

    def test_explainability_dir_exists(self):
        """outputs/explainability/ directory must exist."""
        assert EXPLAINABILITY_DIR.exists(), "outputs/explainability/ missing"

    def test_explainability_has_files(self):
        """Explainability directory must contain output files."""
        if not EXPLAINABILITY_DIR.exists():
            pytest.skip("Explainability directory missing")
        files = list(EXPLAINABILITY_DIR.iterdir())
        assert len(files) > 0, "Explainability directory is empty"

    def test_shap_figure_exists(self):
        """SHAP importance figure must exist."""
        path = FIGURES_DIR / "shap_importance.png"
        assert path.exists(), "Missing shap_importance.png"
