#!/usr/bin/env python3
"""Ingest EPSS (Exploit Prediction Scoring System) scores from First.org.

Downloads current EPSS scores for all CVEs. These serve as the baseline
comparison — our ML model needs to beat EPSS to be interesting.

Usage:
    python scripts/ingest_epss.py               # Download current scores
    python scripts/ingest_epss.py --check-only   # Verify API access
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import requests

# EPSS provides a daily CSV dump (gzipped)
EPSS_CSV_URL = "https://epss.cyentia.com/epss_scores-current.csv.gz"
EPSS_API_URL = "https://api.first.org/data/v1/epss"
OUTPUT_DIR = Path("data/raw/epss")


def check_access():
    """Verify EPSS data is accessible."""
    try:
        resp = requests.head(EPSS_CSV_URL, timeout=30, allow_redirects=True)
        # EPSS might return 403 on HEAD, try GET with range
        if resp.status_code != 200:
            resp = requests.get(EPSS_CSV_URL, timeout=30, stream=True)
            resp.raise_for_status()
            resp.close()
        print("EPSS CSV accessible")
        return True
    except requests.RequestException as e:
        print(f"WARNING: EPSS CSV not accessible ({e}), trying API...")
        try:
            resp = requests.get(EPSS_API_URL, params={"limit": 1}, timeout=30)
            resp.raise_for_status()
            print("EPSS API accessible")
            return True
        except requests.RequestException as e2:
            print(f"ERROR: EPSS not accessible: {e2}")
            return False


def ingest_epss():
    """Download EPSS scores."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading EPSS scores (CSV dump)...")
    try:
        resp = requests.get(EPSS_CSV_URL, timeout=120)
        resp.raise_for_status()

        # Decompress gzip
        raw_csv = gzip.decompress(resp.content).decode("utf-8")
        sha256 = hashlib.sha256(resp.content).hexdigest()

        # Parse CSV (first line is a comment with the model version/date)
        lines = raw_csv.strip().split("\n")
        header_line = None
        data_lines = []
        for line in lines:
            if line.startswith("#"):
                continue
            if header_line is None:
                header_line = line
            else:
                data_lines.append(line)

        # Save raw CSV
        csv_path = OUTPUT_DIR / "epss_scores.csv"
        with open(csv_path, "w") as f:
            f.write(header_line + "\n")
            for line in data_lines:
                f.write(line + "\n")

        print(f"Saved: {csv_path} ({len(data_lines):,} CVEs)")
        print(f"SHA-256: {sha256[:16]}...")

        # Parse and save summary
        reader = csv.DictReader(io.StringIO(header_line + "\n" + "\n".join(data_lines)))
        scores = list(reader)
        print(f"Parsed {len(scores):,} EPSS scores")

        if scores:
            # Show score distribution
            epss_values = [float(s.get("epss", 0)) for s in scores if s.get("epss")]
            print(f"EPSS score range: {min(epss_values):.6f} — {max(epss_values):.6f}")
            high_risk = sum(1 for v in epss_values if v > 0.1)
            print(f"CVEs with EPSS > 0.1 (high risk): {high_risk:,} ({high_risk/len(epss_values)*100:.1f}%)")

    except requests.RequestException as e:
        print(f"CSV download failed ({e}), falling back to API...")
        return ingest_epss_via_api()

    # Metadata
    metadata = {
        "download_date": datetime.now().isoformat(),
        "source": "EPSS (First.org) CSV dump",
        "url": EPSS_CSV_URL,
        "total_scores": len(data_lines),
        "sha256_gz": sha256,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return True


def ingest_epss_via_api():
    """Fallback: download EPSS scores via REST API (slower, paginated)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading EPSS scores via API (paginated, slower)...")
    all_scores = []
    offset = 0
    limit = 1000

    while True:
        resp = requests.get(EPSS_API_URL, params={"limit": limit, "offset": offset}, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        scores = data.get("data", [])
        if not scores:
            break

        all_scores.extend(scores)
        offset += limit
        print(f"  Downloaded {len(all_scores):,} scores...", end="\r")

    print(f"\nTotal: {len(all_scores):,} EPSS scores via API")

    # Save as CSV
    if all_scores:
        import pandas as pd
        df = pd.DataFrame(all_scores)
        df.to_csv(OUTPUT_DIR / "epss_scores.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'epss_scores.csv'}")

    metadata = {
        "download_date": datetime.now().isoformat(),
        "source": "EPSS API (First.org)",
        "total_scores": len(all_scores),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest EPSS scores")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        return 0 if check_access() else 1

    success = ingest_epss()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
