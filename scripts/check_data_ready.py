#!/usr/bin/env python3
"""Check data readiness for vulnerability prioritization project.

Verifies all data sources are present and have expected structure.

Usage:
    python scripts/check_data_ready.py
"""
import json
import sys
from pathlib import Path

import pandas as pd


def check_exploitdb():
    """Verify ExploitDB data."""
    path = Path("data/raw/exploitdb")
    checks = []

    csv_file = path / "cve_exploit_mappings.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        checks.append(f"  ExploitDB CVE mappings: {len(df):,} rows")
        checks.append(f"  Unique CVEs: {df['cve_id'].nunique():,}")
    else:
        print(f"  FAIL: {csv_file} not found")
        return False

    json_file = path / "exploited_cves.json"
    if json_file.exists():
        with open(json_file) as f:
            cves = json.load(f)
        checks.append(f"  Exploited CVE list: {len(cves):,}")
    else:
        print(f"  FAIL: {json_file} not found")
        return False

    for c in checks:
        print(c)
    return True


def check_epss():
    """Verify EPSS data."""
    path = Path("data/raw/epss")
    csv_file = path / "epss_scores.csv"

    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"  EPSS scores: {len(df):,} CVEs")
        print(f"  Columns: {list(df.columns)}")
        return True
    else:
        print(f"  FAIL: {csv_file} not found")
        return False


def check_nvd():
    """Verify NVD data (at least some batches downloaded)."""
    path = Path("data/raw/nvd")
    batch_files = sorted(path.glob("nvd_batch_*.json"))

    if not batch_files:
        print("  FAIL: No NVD batch files found")
        print("  Run: python scripts/ingest_nvd.py --start-year 2017")
        return False

    total_cves = 0
    for bf in batch_files:
        with open(bf) as f:
            data = json.load(f)
            total_cves += len(data)

    print(f"  NVD batches: {len(batch_files)} files")
    print(f"  Total CVEs: {total_cves:,}")

    meta_file = path / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"  Download date: {meta.get('download_date', 'unknown')}")

    return total_cves > 0


def main():
    print("=== Data Readiness Check ===\n")

    results = {}

    print("1. ExploitDB (ground truth labels):")
    results["exploitdb"] = check_exploitdb()
    print()

    print("2. EPSS (baseline comparison):")
    results["epss"] = check_epss()
    print()

    print("3. NVD (CVE features):")
    results["nvd"] = check_nvd()
    print()

    # Summary
    all_pass = all(results.values())
    print("=== Summary ===")
    for source, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {source}: {status}")

    if all_pass:
        print("\nAll data sources ready. Proceed to Phase 1.")
    else:
        failed = [s for s, p in results.items() if not p]
        print(f"\nMissing: {', '.join(failed)}. Fix before proceeding.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
