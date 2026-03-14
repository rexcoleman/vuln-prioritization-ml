#!/usr/bin/env python3
"""Ingest CVE data from NVD API 2.0.

Downloads all CVEs in paginated batches and saves as JSON files.
Rate limit: 50 requests per 30 seconds with API key, 5 without.

Usage:
    python scripts/ingest_nvd.py                    # Full download
    python scripts/ingest_nvd.py --check-only       # Just verify API access
    python scripts/ingest_nvd.py --start-year 2020  # Download from 2020 onward
    python scripts/ingest_nvd.py --resume            # Resume from last batch
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
OUTPUT_DIR = Path("data/raw/nvd")
RESULTS_PER_PAGE = 2000  # NVD max
RATE_LIMIT_DELAY = 0.65  # seconds between requests (with API key)
RATE_LIMIT_DELAY_NO_KEY = 6.5  # seconds between requests (without key)


def get_api_key():
    """Get NVD API key from environment."""
    key = os.environ.get("NVD_API_KEY")
    if key:
        print(f"Using NVD API key (rate limit: 50 req/30s)")
    else:
        print("WARNING: No NVD_API_KEY set. Rate limit: 5 req/30s (very slow).")
        print("Get a free key at: https://nvd.nist.gov/developers/request-an-api-key")
    return key


def check_api_access(api_key=None):
    """Verify NVD API is accessible."""
    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    params = {"resultsPerPage": 1, "startIndex": 0}

    try:
        resp = requests.get(NVD_API_BASE, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        total = data.get("totalResults", 0)
        print(f"NVD API accessible. Total CVEs available: {total:,}")
        return total
    except requests.RequestException as e:
        print(f"ERROR: NVD API not accessible: {e}")
        return None


def get_last_batch_index():
    """Find the highest batch file already downloaded for resume."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    batch_files = sorted(OUTPUT_DIR.glob("nvd_batch_*.json"))
    if not batch_files:
        return 0
    last = batch_files[-1].stem  # e.g., "nvd_batch_042"
    try:
        return (int(last.split("_")[-1]) + 1) * RESULTS_PER_PAGE
    except (ValueError, IndexError):
        return 0


def ingest_nvd(api_key=None, start_year=None, resume=False):
    """Download all CVEs from NVD API in paginated batches."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    delay = RATE_LIMIT_DELAY if api_key else RATE_LIMIT_DELAY_NO_KEY

    # Build base params
    params = {"resultsPerPage": RESULTS_PER_PAGE}
    if start_year:
        params["pubStartDate"] = f"{start_year}-01-01T00:00:00.000"
        params["pubEndDate"] = datetime.now().strftime("%Y-%m-%dT23:59:59.999")

    # Get total count
    total = check_api_access(api_key)
    if total is None:
        return False

    # Determine start index
    start_index = get_last_batch_index() if resume else 0
    if resume and start_index > 0:
        print(f"Resuming from index {start_index:,}")

    num_batches = (total - start_index + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
    all_cves = []
    batch_num = start_index // RESULTS_PER_PAGE

    print(f"Downloading {total - start_index:,} CVEs in {num_batches} batches...")
    print(f"Rate limit delay: {delay}s per request")
    estimated_time = num_batches * delay / 60
    print(f"Estimated time: {estimated_time:.0f} minutes")

    pbar = tqdm(total=total, initial=start_index, unit="CVEs")

    idx = start_index
    while idx < total:
        params["startIndex"] = idx

        try:
            resp = requests.get(NVD_API_BASE, params=params, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"\nERROR at index {idx}: {e}")
            print(f"Retrying in 30s...")
            time.sleep(30)
            continue

        vulnerabilities = data.get("vulnerabilities", [])
        if not vulnerabilities:
            break

        # Save batch
        batch_file = OUTPUT_DIR / f"nvd_batch_{batch_num:04d}.json"
        with open(batch_file, "w") as f:
            json.dump(vulnerabilities, f)

        batch_cve_count = len(vulnerabilities)
        all_cves.extend(vulnerabilities)
        pbar.update(batch_cve_count)

        idx += RESULTS_PER_PAGE
        batch_num += 1
        time.sleep(delay)

    pbar.close()

    # Save metadata
    metadata = {
        "download_date": datetime.now().isoformat(),
        "total_cves": len(all_cves),
        "total_batches": batch_num,
        "api_key_used": api_key is not None,
        "start_year": start_year,
        "source": "NVD API 2.0",
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. {len(all_cves):,} CVEs saved to {OUTPUT_DIR}/")
    print(f"Metadata: {OUTPUT_DIR}/metadata.json")
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest CVE data from NVD API")
    parser.add_argument("--check-only", action="store_true", help="Only verify API access")
    parser.add_argument("--start-year", type=int, help="Download CVEs from this year onward")
    parser.add_argument("--resume", action="store_true", help="Resume from last downloaded batch")
    args = parser.parse_args()

    api_key = get_api_key()

    if args.check_only:
        result = check_api_access(api_key)
        return 0 if result is not None else 1

    success = ingest_nvd(api_key=api_key, start_year=args.start_year, resume=args.resume)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
