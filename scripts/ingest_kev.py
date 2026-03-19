#!/usr/bin/env python3
"""Ingest CISA Known Exploited Vulnerabilities (KEV) catalog as second ground truth.

Downloads the KEV catalog JSON and creates a label file mapping CVE IDs to
known-exploited status. This addresses the "single ground truth source" criticism
by providing a second, independent exploit label alongside ExploitDB.

KEV source: https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json

Usage:
    python scripts/ingest_kev.py --data-dir data/
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def download_kev(output_dir):
    """Download CISA KEV catalog."""
    print(f"Downloading KEV catalog from {KEV_URL}...")
    raw_dir = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    response = urlopen(KEV_URL)
    data = json.loads(response.read().decode())

    kev_file = raw_dir / "cisa_kev.json"
    with open(kev_file, "w") as f:
        json.dump(data, f, indent=2)

    vulns = data.get("vulnerabilities", [])
    print(f"Downloaded {len(vulns)} KEV entries")
    print(f"Catalog version: {data.get('catalogVersion', 'unknown')}")
    print(f"Date released: {data.get('dateReleased', 'unknown')}")
    print(f"Saved: {kev_file}")

    return vulns


def build_kev_labels(vulns, data_dir):
    """Create KEV label mapping and join with existing dataset."""
    processed_dir = Path(data_dir) / "processed"

    # Build KEV CVE set
    kev_cves = set()
    for v in vulns:
        cve_id = v.get("cveID", "")
        if cve_id.startswith("CVE-"):
            kev_cves.add(cve_id)
    print(f"\nKEV CVE IDs: {len(kev_cves)}")

    # Load existing train/test data
    train = pd.read_parquet(processed_dir / "train.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")

    # Add KEV labels
    train["kev_exploited"] = train["cve_id"].isin(kev_cves).astype(int)
    test["kev_exploited"] = test["cve_id"].isin(kev_cves).astype(int)

    # Combined label: exploited by EITHER ExploitDB OR KEV
    train["either_exploited"] = ((train["exploited"] == 1) | (train["kev_exploited"] == 1)).astype(int)
    test["either_exploited"] = ((test["exploited"] == 1) | (test["kev_exploited"] == 1)).astype(int)

    # Stats
    print(f"\n--- Train set ({len(train):,} CVEs) ---")
    print(f"  ExploitDB exploited: {train['exploited'].sum():,} ({train['exploited'].mean()*100:.1f}%)")
    print(f"  KEV exploited:       {train['kev_exploited'].sum():,} ({train['kev_exploited'].mean()*100:.2f}%)")
    print(f"  Either exploited:    {train['either_exploited'].sum():,} ({train['either_exploited'].mean()*100:.1f}%)")
    overlap_train = ((train['exploited'] == 1) & (train['kev_exploited'] == 1)).sum()
    print(f"  Overlap (both):      {overlap_train:,}")
    kev_only_train = ((train['exploited'] == 0) & (train['kev_exploited'] == 1)).sum()
    print(f"  KEV-only (not in ExploitDB): {kev_only_train:,}")

    print(f"\n--- Test set ({len(test):,} CVEs) ---")
    print(f"  ExploitDB exploited: {test['exploited'].sum():,} ({test['exploited'].mean()*100:.2f}%)")
    print(f"  KEV exploited:       {test['kev_exploited'].sum():,} ({test['kev_exploited'].mean()*100:.2f}%)")
    print(f"  Either exploited:    {test['either_exploited'].sum():,} ({test['either_exploited'].mean()*100:.2f}%)")
    overlap_test = ((test['exploited'] == 1) & (test['kev_exploited'] == 1)).sum()
    print(f"  Overlap (both):      {overlap_test:,}")
    kev_only_test = ((test['exploited'] == 0) & (test['kev_exploited'] == 1)).sum()
    print(f"  KEV-only (not in ExploitDB): {kev_only_test:,}")

    # Save enriched datasets
    train.to_parquet(processed_dir / "train_kev.parquet", index=False)
    test.to_parquet(processed_dir / "test_kev.parquet", index=False)
    print(f"\nSaved: {processed_dir / 'train_kev.parquet'}")
    print(f"Saved: {processed_dir / 'test_kev.parquet'}")

    # Save KEV metadata
    kev_meta = {
        "date": datetime.now().isoformat(),
        "kev_url": KEV_URL,
        "kev_total": len(kev_cves),
        "train_kev_count": int(train["kev_exploited"].sum()),
        "test_kev_count": int(test["kev_exploited"].sum()),
        "train_either_count": int(train["either_exploited"].sum()),
        "test_either_count": int(test["either_exploited"].sum()),
        "train_overlap": int(overlap_train),
        "test_overlap": int(overlap_test),
        "train_kev_only": int(kev_only_train),
        "test_kev_only": int(kev_only_test),
    }
    meta_file = processed_dir / "kev_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(kev_meta, f, indent=2)
    print(f"Saved: {meta_file}")

    return train, test, kev_meta


def main():
    parser = argparse.ArgumentParser(description="Ingest CISA KEV catalog")
    parser.add_argument("--data-dir", type=str, default="data/")
    args = parser.parse_args()

    vulns = download_kev(args.data_dir)
    train, test, meta = build_kev_labels(vulns, args.data_dir)

    print("\n" + "=" * 60)
    print("KEV INGESTION COMPLETE")
    print("=" * 60)
    print(f"KEV adds {meta['train_kev_only']} train + {meta['test_kev_only']} test labels")
    print("not captured by ExploitDB — this is the dual ground truth value.")


if __name__ == "__main__":
    main()
