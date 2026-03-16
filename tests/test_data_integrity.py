"""Data integrity tests for vulnerability prioritization project.

Tests cover two tiers:
  - Tier 1 (JSON metadata): Always runnable -- reads from feature_cols.json and metadata.json
  - Tier 2 (Parquet data): Requires data/processed/*.parquet (skipped if absent)

LT-1: No future data leakage (temporal split enforced)
LT-2: No label leakage (exploited status not in features)
LT-3: Join integrity (CVE IDs match across sources)
LT-4: Feature consistency and completeness
LT-5: Label distribution sanity
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import (
    DATA_DIR, EXPECTED_NUM_FEATURES, FEATURE_GROUPS,
)


PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"


# -- Fixtures (data-dependent, skippable) ------------------------------------

@pytest.fixture
def feature_cols():
    path = PROCESSED_DIR / "feature_cols.json"
    if not path.exists():
        pytest.skip("Feature cols not yet built")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def metadata():
    path = PROCESSED_DIR / "metadata.json"
    if not path.exists():
        pytest.skip("Metadata not yet built")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def train_df():
    path = PROCESSED_DIR / "train.parquet"
    if not path.exists():
        pytest.skip("Train data not yet built")
    return pd.read_parquet(path)


@pytest.fixture
def test_df():
    path = PROCESSED_DIR / "test.parquet"
    if not path.exists():
        pytest.skip("Test data not yet built")
    return pd.read_parquet(path)


@pytest.fixture
def split_info():
    path = SPLITS_DIR / "split_info.json"
    if not path.exists():
        pytest.skip("Split info not yet created")
    with open(path) as f:
        return json.load(f)


# -- Tier 1: Feature schema (JSON-only, always runnable) --------------------

class TestFeatureSchema:
    """Verify feature column schema matches expectations."""

    def test_feature_cols_file_exists(self):
        """feature_cols.json must exist in data/processed/."""
        assert (PROCESSED_DIR / "feature_cols.json").exists()

    def test_feature_count_matches_expected(self, feature_cols):
        """Must have exactly 49 structured features."""
        assert len(feature_cols) == EXPECTED_NUM_FEATURES, (
            f"Expected {EXPECTED_NUM_FEATURES} features, got {len(feature_cols)}"
        )

    def test_no_duplicate_features(self, feature_cols):
        """Feature names must be unique."""
        assert len(feature_cols) == len(set(feature_cols))

    def test_epss_features_present(self, feature_cols):
        """EPSS features (epss, epss_percentile) must be in feature set."""
        for feat in ["epss", "epss_percentile"]:
            assert feat in feature_cols, f"Missing EPSS feature: {feat}"

    def test_cvss_features_present(self, feature_cols):
        """CVSS features must be in feature set."""
        for feat in ["cvss_v3", "cvss_v2", "cvss_score", "has_cvss_v3"]:
            assert feat in feature_cols, f"Missing CVSS feature: {feat}"

    def test_keyword_features_present(self, feature_cols):
        """All 11 practitioner keyword features must be present."""
        for feat in FEATURE_GROUPS["text_keywords"]:
            assert feat in feature_cols, f"Missing keyword feature: {feat}"

    def test_feature_groups_cover_all_features(self, feature_cols):
        """All features must belong to at least one feature group."""
        all_grouped = set()
        for group_feats in FEATURE_GROUPS.values():
            all_grouped.update(group_feats)
        ungrouped = set(feature_cols) - all_grouped
        assert len(ungrouped) == 0, f"Features not in any group: {ungrouped}"

    def test_feature_groups_total_equals_feature_count(self, feature_cols):
        """Sum of feature group sizes must equal total features (no overlap)."""
        total = sum(len(v) for v in FEATURE_GROUPS.values())
        assert total == len(feature_cols)


# -- Tier 1: Metadata consistency --------------------------------------------

class TestMetadata:
    """Verify processed data metadata consistency."""

    def test_metadata_file_exists(self):
        """metadata.json must exist in data/processed/."""
        assert (PROCESSED_DIR / "metadata.json").exists()

    def test_total_cves_matches_train_plus_test(self, metadata):
        """total_cves should equal train_size + test_size."""
        assert metadata["total_cves"] == metadata["train_size"] + metadata["test_size"]

    def test_train_size_expected(self, metadata):
        """Training set should be 234,601 samples (pre-2024)."""
        assert metadata["train_size"] == 234601

    def test_test_size_expected(self, metadata):
        """Test set should be 103,352 samples (2024+)."""
        assert metadata["test_size"] == 103352

    def test_exploit_rate_train_reasonable(self, metadata):
        """Train exploit rate should be approximately 10.5%."""
        rate = metadata["exploit_rate_train"]
        assert 0.05 < rate < 0.20, f"Train exploit rate {rate} outside [5%, 20%]"

    def test_exploit_rate_test_low(self, metadata):
        """Test exploit rate should be very low (<1%) due to ground-truth lag."""
        rate = metadata["exploit_rate_test"]
        assert rate < 0.01, f"Test exploit rate {rate} not below 1%"

    def test_num_features_in_metadata(self, metadata, feature_cols):
        """Metadata feature count must match feature_cols.json."""
        assert metadata["num_structured_features"] == len(feature_cols)

    def test_feature_list_in_metadata_matches(self, metadata, feature_cols):
        """Feature column list in metadata must match feature_cols.json."""
        assert metadata["feature_cols"] == feature_cols


# -- Tier 2: Temporal split (requires parquet data) --------------------------

class TestTemporalLeakage:
    """No future data in training set."""

    def test_train_before_2024(self, train_df):
        """All training CVEs published before 2024."""
        assert train_df["pub_year"].max() < 2024, (
            f"Train contains CVEs from {train_df['pub_year'].max()}, expected < 2024"
        )

    def test_test_from_2024_onward(self, test_df):
        """All test CVEs published in 2024 or later."""
        assert test_df["pub_year"].min() >= 2024, (
            f"Test contains CVEs from {test_df['pub_year'].min()}, expected >= 2024"
        )

    def test_no_overlap(self, train_df, test_df):
        """No CVE appears in both train and test."""
        overlap = set(train_df["cve_id"]) & set(test_df["cve_id"])
        assert len(overlap) == 0, f"Found {len(overlap)} CVEs in both train and test"

    def test_test_dates_strictly_after_train(self, train_df, test_df):
        """Test set min year > train set max year (no temporal overlap)."""
        train_max_year = train_df["pub_year"].max()
        test_min_year = test_df["pub_year"].min()
        assert test_min_year > train_max_year, \
            f"Test min year {test_min_year} not strictly after train max year {train_max_year}"

    def test_split_year_is_2024(self, split_info):
        """Split year matches the documented design (ADR-0001)."""
        assert split_info["split_year"] == 2024, \
            f"Expected split year 2024, got {split_info['split_year']}"


# -- Tier 2: Label leakage --------------------------------------------------

class TestLabelLeakage:
    """Exploited label not leaked into features."""

    def test_exploited_not_in_features(self, feature_cols):
        """'exploited' column not in feature list."""
        assert "exploited" not in feature_cols

    def test_no_cve_id_in_features(self, feature_cols):
        """CVE ID not in feature list."""
        assert "cve_id" not in feature_cols

    def test_no_description_in_features(self, feature_cols):
        """Raw description text not in feature list."""
        assert "description" not in feature_cols

    def test_no_metadata_columns_in_features(self, feature_cols):
        """No metadata/identifier columns leaked into features."""
        forbidden = {"cve_id", "description", "published", "published_dt",
                     "cvss_v3_vector", "cvss_v3_severity", "cwe_primary",
                     "vendor", "product", "exploited"}
        leaked = set(feature_cols) & forbidden
        assert len(leaked) == 0, f"Metadata columns in features: {leaked}"


# -- Tier 2: Join integrity --------------------------------------------------

class TestJoinIntegrity:
    """Data joins are correct."""

    def test_no_null_labels(self, train_df, test_df):
        """Every CVE has an exploited label (0 or 1)."""
        assert train_df["exploited"].isna().sum() == 0
        assert test_df["exploited"].isna().sum() == 0

    def test_label_is_binary(self, train_df, test_df):
        """Label is strictly 0 or 1."""
        assert set(train_df["exploited"].unique()).issubset({0, 1})
        assert set(test_df["exploited"].unique()).issubset({0, 1})

    def test_both_classes_in_train(self, train_df):
        """Training data has both exploited and non-exploited CVEs."""
        assert train_df["exploited"].sum() > 0
        assert (train_df["exploited"] == 0).sum() > 0

    def test_both_classes_in_test(self, test_df):
        """Test data has both exploited and non-exploited CVEs."""
        assert test_df["exploited"].sum() > 0
        assert (test_df["exploited"] == 0).sum() > 0

    def test_cve_ids_are_unique(self, train_df, test_df):
        """No duplicate CVE IDs within train or test."""
        assert train_df["cve_id"].is_unique, \
            f"Duplicate CVEs in train: {train_df['cve_id'].duplicated().sum()}"
        assert test_df["cve_id"].is_unique, \
            f"Duplicate CVEs in test: {test_df['cve_id'].duplicated().sum()}"


# -- Tier 2: Label distribution ----------------------------------------------

class TestLabelDistribution:
    """Label distributions match expected values from FINDINGS.md."""

    def test_train_exploit_rate(self, train_df):
        """Train exploit rate is approximately 10.5%."""
        rate = train_df["exploited"].mean()
        assert 0.05 < rate < 0.20, \
            f"Train exploit rate {rate:.3f} outside expected range [0.05, 0.20]"

    def test_test_exploit_rate(self, test_df):
        """Test exploit rate is very low (<5%) due to ground truth lag."""
        rate = test_df["exploited"].mean()
        assert rate < 0.05, \
            f"Test exploit rate {rate:.3f} higher than expected (<0.05)"

    def test_split_sizes_match_metadata(self, train_df, test_df, metadata):
        """Parquet row counts match metadata."""
        assert len(train_df) == metadata["train_size"], \
            f"Train parquet has {len(train_df)} rows, metadata says {metadata['train_size']}"
        assert len(test_df) == metadata["test_size"], \
            f"Test parquet has {len(test_df)} rows, metadata says {metadata['test_size']}"

    def test_split_hashes_present(self, split_info):
        """Split info file contains valid SHA-256 hashes."""
        assert len(split_info["train_index_hash"]) == 64, "Invalid train hash"
        assert len(split_info["test_index_hash"]) == 64, "Invalid test hash"

    def test_no_all_null_features(self, train_df, feature_cols):
        """No feature column is entirely null."""
        for col in feature_cols:
            if col in train_df.columns:
                assert train_df[col].notna().any(), f"Feature {col} is all null"
