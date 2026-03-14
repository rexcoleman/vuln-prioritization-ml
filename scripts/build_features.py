#!/usr/bin/env python3
"""Build feature matrix from NVD + ExploitDB + EPSS data.

Joins all data sources, engineers features, and creates the train/test split.
Temporal split: train on pre-2024, test on 2024+ (prevents data leakage).

Usage:
    python scripts/build_features.py                    # Full feature build
    python scripts/build_features.py --sample-frac 0.01 # Smoke test on 1%
"""
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

NVD_DIR = Path("data/raw/nvd")
EXPLOITDB_DIR = Path("data/raw/exploitdb")
EPSS_DIR = Path("data/raw/epss")
OUTPUT_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")


def load_nvd_cves():
    """Load all NVD CVE batch files into a single DataFrame."""
    batch_files = sorted(NVD_DIR.glob("nvd_batch_*.json"))
    if not batch_files:
        raise FileNotFoundError("No NVD batch files found. Run ingest_nvd.py first.")

    all_cves = []
    for bf in batch_files:
        with open(bf) as f:
            batch = json.load(f)
            all_cves.extend(batch)

    print(f"Loaded {len(all_cves):,} CVEs from {len(batch_files)} NVD batch files")

    # Extract structured fields
    records = []
    for entry in all_cves:
        cve = entry.get("cve", {})
        cve_id = cve.get("id", "")

        # Description (English)
        descriptions = cve.get("descriptions", [])
        desc_en = ""
        for d in descriptions:
            if d.get("lang") == "en":
                desc_en = d.get("value", "")
                break

        # Published date
        published = cve.get("published", "")

        # CVSS v3.1 metrics
        cvss_v3 = None
        cvss_v3_vector = ""
        cvss_v3_severity = ""
        metrics = cve.get("metrics", {})
        for key in ["cvssMetricV31", "cvssMetricV30"]:
            if key in metrics and metrics[key]:
                m = metrics[key][0]
                cvss_data = m.get("cvssData", {})
                cvss_v3 = cvss_data.get("baseScore")
                cvss_v3_vector = cvss_data.get("vectorString", "")
                cvss_v3_severity = cvss_data.get("baseSeverity", "")
                break

        # CVSS v2 fallback
        cvss_v2 = None
        if "cvssMetricV2" in metrics and metrics["cvssMetricV2"]:
            cvss_v2 = metrics["cvssMetricV2"][0].get("cvssData", {}).get("baseScore")

        # CWE
        weaknesses = cve.get("weaknesses", [])
        cwe_ids = []
        for w in weaknesses:
            for desc in w.get("description", []):
                val = desc.get("value", "")
                if val.startswith("CWE-") and val != "CWE-noinfo":
                    cwe_ids.append(val)

        # Vendor / product from configurations
        vendor = ""
        product = ""
        configs = cve.get("configurations", [])
        if configs:
            for node in configs:
                for n in node.get("nodes", []):
                    for match in n.get("cpeMatch", []):
                        criteria = match.get("criteria", "")
                        parts = criteria.split(":")
                        if len(parts) >= 5:
                            vendor = parts[3]
                            product = parts[4]
                            break
                    if vendor:
                        break
                if vendor:
                    break

        # References count and types
        references = cve.get("references", [])
        ref_count = len(references)
        has_exploit_ref = any("exploit" in str(r.get("tags", [])).lower() for r in references)
        has_patch_ref = any("patch" in str(r.get("tags", [])).lower() for r in references)

        records.append({
            "cve_id": cve_id,
            "description": desc_en,
            "published": published,
            "cvss_v3": cvss_v3,
            "cvss_v3_vector": cvss_v3_vector,
            "cvss_v3_severity": cvss_v3_severity,
            "cvss_v2": cvss_v2,
            "cwe_primary": cwe_ids[0] if cwe_ids else "",
            "cwe_count": len(cwe_ids),
            "vendor": vendor,
            "product": product,
            "ref_count": ref_count,
            "has_exploit_ref": has_exploit_ref,
            "has_patch_ref": has_patch_ref,
        })

    df = pd.DataFrame(records)
    print(f"Structured fields extracted: {df.shape}")
    print(f"  CVEs with CVSS v3: {df['cvss_v3'].notna().sum():,}")
    print(f"  CVEs with CWE: {(df['cwe_primary'] != '').sum():,}")
    return df


def load_exploitdb_labels():
    """Load ExploitDB ground truth labels."""
    path = EXPLOITDB_DIR / "exploited_cves.json"
    with open(path) as f:
        exploited = set(json.load(f))
    print(f"ExploitDB ground truth: {len(exploited):,} exploited CVEs")
    return exploited


def load_epss_scores():
    """Load EPSS scores."""
    path = EPSS_DIR / "epss_scores.csv"
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    if "cve" in df.columns:
        df = df.rename(columns={"cve": "cve_id"})
    print(f"EPSS scores loaded: {len(df):,}")
    return df


def parse_cvss_vector(vector_string):
    """Parse CVSS v3 vector string into component features."""
    components = {}
    if not vector_string or not isinstance(vector_string, str):
        return components

    # CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H
    parts = vector_string.split("/")
    for part in parts:
        if ":" in part:
            key, val = part.split(":", 1)
            components[f"cvss_{key.lower()}"] = val
    return components


def engineer_features(nvd_df, exploited_cves, epss_df, sample_frac=1.0):
    """Engineer features from joined data."""
    df = nvd_df.copy()

    # Sample if requested (smoke testing)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"Sampled {len(df):,} CVEs ({sample_frac*100:.0f}%)")

    # --- Label ---
    df["exploited"] = df["cve_id"].isin(exploited_cves).astype(int)
    print(f"\nLabel distribution:")
    print(f"  Exploited: {df['exploited'].sum():,} ({df['exploited'].mean()*100:.1f}%)")
    print(f"  Not exploited: {(~df['exploited'].astype(bool)).sum():,}")

    # --- EPSS join ---
    epss_df_slim = epss_df[["cve_id", "epss", "percentile"]].copy() if "percentile" in epss_df.columns else epss_df[["cve_id", "epss"]].copy()
    df = df.merge(epss_df_slim, on="cve_id", how="left")
    df["epss"] = df["epss"].fillna(0.0).astype(float)
    if "percentile" in df.columns:
        df["epss_percentile"] = df["percentile"].fillna(0.0).astype(float)
        df = df.drop(columns=["percentile"])

    # --- Temporal features ---
    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    df["pub_year"] = df["published_dt"].dt.year
    df["pub_month"] = df["published_dt"].dt.month
    df["pub_dayofweek"] = df["published_dt"].dt.dayofweek
    df["cve_age_days"] = (pd.Timestamp.now(tz="UTC") - df["published_dt"]).dt.days

    # --- Description NLP features ---
    df["desc_length"] = df["description"].fillna("").str.len()
    df["desc_word_count"] = df["description"].fillna("").str.split().str.len()

    # Keyword flags (security-relevant terms from practitioner experience)
    keywords = {
        "remote_code_execution": r"remote\s+code\s+execution|rce",
        "sql_injection": r"sql\s+injection|sqli",
        "buffer_overflow": r"buffer\s+overflow|heap\s+overflow|stack\s+overflow",
        "xss": r"cross.site\s+scripting|xss",
        "privilege_escalation": r"privilege\s+escalation|privesc",
        "authentication_bypass": r"authentication\s+bypass|auth\s+bypass",
        "denial_of_service": r"denial\s+of\s+service|dos\b",
        "information_disclosure": r"information\s+disclosure|info\s+leak",
        "arbitrary_code": r"arbitrary\s+code",
        "allows_attackers": r"allows?\s+(remote\s+)?attackers?",
        "crafted": r"crafted\s+(request|packet|input|file|url)",
    }
    for name, pattern in keywords.items():
        df[f"kw_{name}"] = df["description"].fillna("").str.contains(
            pattern, case=False, regex=True
        ).astype(int)

    # --- CVSS vector components ---
    cvss_components = df["cvss_v3_vector"].apply(parse_cvss_vector)
    cvss_comp_df = pd.DataFrame(cvss_components.tolist(), index=df.index)
    df = pd.concat([df, cvss_comp_df], axis=1)

    # --- CVSS features ---
    df["cvss_score"] = df["cvss_v3"].fillna(df["cvss_v2"]).fillna(0.0)
    df["has_cvss_v3"] = df["cvss_v3"].notna().astype(int)

    # --- CWE features ---
    # Top CWEs as one-hot (top 20 by frequency)
    cwe_counts = df["cwe_primary"].value_counts()
    top_cwes = cwe_counts[cwe_counts.index != ""].head(20).index.tolist()
    for cwe in top_cwes:
        df[f"cwe_{cwe}"] = (df["cwe_primary"] == cwe).astype(int)
    df["has_cwe"] = (df["cwe_primary"] != "").astype(int)

    # --- Reference features ---
    df["has_exploit_ref"] = df["has_exploit_ref"].astype(int)
    df["has_patch_ref"] = df["has_patch_ref"].astype(int)

    # --- Vendor frequency (as proxy for vendor attack surface) ---
    vendor_freq = df["vendor"].value_counts()
    df["vendor_cve_count"] = df["vendor"].map(vendor_freq).fillna(0).astype(int)

    print(f"\nFeature matrix: {df.shape}")
    return df


def create_temporal_split(df, split_year=2024, seed=42):
    """Create temporal train/test split.

    Train: CVEs published before split_year
    Test: CVEs published in split_year or later

    This prevents data leakage from future information.
    """
    train = df[df["pub_year"] < split_year].copy()
    test = df[df["pub_year"] >= split_year].copy()

    print(f"\nTemporal split (year < {split_year} / >= {split_year}):")
    print(f"  Train: {len(train):,} ({train['exploited'].mean()*100:.1f}% exploited)")
    print(f"  Test:  {len(test):,} ({test['exploited'].mean()*100:.1f}% exploited)")

    # Save split indices for reproducibility
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    split_info = {
        "method": "temporal",
        "split_year": split_year,
        "seed": seed,
        "train_size": len(train),
        "test_size": len(test),
        "train_exploit_rate": float(train["exploited"].mean()),
        "test_exploit_rate": float(test["exploited"].mean()),
        "train_index_hash": hashlib.sha256(
            np.array(sorted(train.index), dtype=np.int64).tobytes()
        ).hexdigest(),
        "test_index_hash": hashlib.sha256(
            np.array(sorted(test.index), dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    with open(SPLITS_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    return train, test


def build_tfidf_features(train_desc, test_desc, max_features=500):
    """Build TF-IDF features from CVE descriptions."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
    )

    train_tfidf = tfidf.fit_transform(train_desc.fillna(""))
    test_tfidf = tfidf.transform(test_desc.fillna(""))

    feature_names = [f"tfidf_{n}" for n in tfidf.get_feature_names_out()]
    print(f"TF-IDF features: {len(feature_names)} (from {train_tfidf.shape[0]:,} train docs)")

    return train_tfidf, test_tfidf, feature_names, tfidf


def main():
    parser = argparse.ArgumentParser(description="Build feature matrix")
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Fraction of data to use (for smoke testing)")
    parser.add_argument("--tfidf-features", type=int, default=500,
                        help="Number of TF-IDF features")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data sources
    print("=" * 60)
    print("Loading data sources...")
    print("=" * 60)
    nvd_df = load_nvd_cves()
    exploited_cves = load_exploitdb_labels()
    epss_df = load_epss_scores()

    # Engineer features
    print("\n" + "=" * 60)
    print("Engineering features...")
    print("=" * 60)
    df = engineer_features(nvd_df, exploited_cves, epss_df, sample_frac=args.sample_frac)

    # Temporal split
    print("\n" + "=" * 60)
    print("Creating temporal split...")
    print("=" * 60)
    train_df, test_df = create_temporal_split(df, split_year=2024, seed=args.seed)

    # Identify feature columns (exclude metadata and label)
    exclude_cols = {
        "cve_id", "description", "published", "published_dt",
        "cvss_v3_vector", "cvss_v3_severity", "cwe_primary",
        "vendor", "product", "exploited",
    }
    # Also exclude any CVSS component string columns
    for col in df.columns:
        if col.startswith("cvss_") and df[col].dtype == object:
            exclude_cols.add(col)

    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, np.int32, np.float32, int, float, bool, np.bool_]]
    print(f"\nStructured features: {len(feature_cols)}")

    # Save processed data
    train_df.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    test_df.to_parquet(OUTPUT_DIR / "test.parquet", index=False)

    # Save feature column list
    with open(OUTPUT_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save metadata
    metadata = {
        "build_date": datetime.now().isoformat(),
        "sample_frac": args.sample_frac,
        "seed": args.seed,
        "total_cves": len(df),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "num_structured_features": len(feature_cols),
        "tfidf_features": args.tfidf_features,
        "exploit_rate_train": float(train_df["exploited"].mean()),
        "exploit_rate_test": float(test_df["exploited"].mean()),
        "feature_cols": feature_cols,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Feature build complete.")
    print(f"  Train: {OUTPUT_DIR / 'train.parquet'} ({len(train_df):,} rows)")
    print(f"  Test:  {OUTPUT_DIR / 'test.parquet'} ({len(test_df):,} rows)")
    print(f"  Features: {OUTPUT_DIR / 'feature_cols.json'} ({len(feature_cols)} structured)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
