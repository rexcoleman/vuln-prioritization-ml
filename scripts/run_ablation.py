#!/usr/bin/env python3
"""Feature group ablation study.

For each feature group, measures impact on XGBoost AUC when:
  1. That group is REMOVED (leave-one-group-out)
  2. ONLY that group is kept (single-group)

Feature groups:
  - text_keywords: kw_* features (CVE description keywords)
  - epss_features: epss, epss_percentile
  - cvss_features: cvss_v3, cvss_v2, cvss_score, has_cvss_v3
  - temporal_features: pub_year, pub_month, pub_dayofweek, cve_age_days
  - description_stats: desc_length, desc_word_count
  - cwe_features: cwe_*, has_cwe, cwe_count
  - reference_features: ref_count, has_exploit_ref, has_patch_ref
  - vendor_features: vendor_cve_count

Usage:
    python scripts/run_ablation.py --data-dir data/
    python scripts/run_ablation.py --data-dir data/ --seeds 42,123
    python scripts/run_ablation.py --data-dir data/ --sample-frac 0.01  # smoke test
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def load_data(data_dir, sample_frac=1.0, seed=42):
    """Load train/test data and feature columns."""
    processed = Path(data_dir) / "processed"
    train = pd.read_parquet(processed / "train.parquet")
    test = pd.read_parquet(processed / "test.parquet")

    with open(processed / "feature_cols.json") as f:
        feature_cols = json.load(f)

    if sample_frac < 1.0:
        train = train.sample(frac=sample_frac, random_state=seed)
        test = test.sample(frac=sample_frac, random_state=seed)

    return train, test, feature_cols


def define_feature_groups(feature_cols):
    """Assign each feature to a group based on naming patterns.

    Returns dict of {group_name: [feature_names]}.
    Every feature in feature_cols is assigned to exactly one group.
    """
    groups = {
        "text_keywords": [],
        "epss_features": [],
        "cvss_features": [],
        "temporal_features": [],
        "description_stats": [],
        "cwe_features": [],
        "reference_features": [],
        "vendor_features": [],
    }

    epss_names = {"epss", "epss_percentile"}
    cvss_names = {"cvss_v3", "cvss_v2", "cvss_score", "has_cvss_v3"}
    temporal_names = {"pub_year", "pub_month", "pub_dayofweek", "cve_age_days",
                      "days_since_published", "year", "month"}
    desc_names = {"desc_length", "desc_word_count"}
    ref_names = {"ref_count", "has_exploit_ref", "has_patch_ref", "reference_count"}
    vendor_names = {"vendor_cve_count", "has_vendor_advisory"}

    ungrouped = []

    for feat in feature_cols:
        if feat.startswith("kw_"):
            groups["text_keywords"].append(feat)
        elif feat in epss_names:
            groups["epss_features"].append(feat)
        elif feat in cvss_names:
            groups["cvss_features"].append(feat)
        elif feat in temporal_names:
            groups["temporal_features"].append(feat)
        elif feat in desc_names:
            groups["description_stats"].append(feat)
        elif feat.startswith("cwe_") or feat in ("has_cwe", "cwe_count"):
            groups["cwe_features"].append(feat)
        elif feat in ref_names:
            groups["reference_features"].append(feat)
        elif feat in vendor_names:
            groups["vendor_features"].append(feat)
        else:
            ungrouped.append(feat)

    # Put any ungrouped features into a catch-all
    if ungrouped:
        groups["other"] = ungrouped

    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}

    return groups


def train_xgboost_with_features(train_df, test_df, features, seed):
    """Train XGBoost on specified feature subset and return AUC.

    Returns (auc, n_features) or (None, n_features) if training fails.
    """
    if not features:
        return None, 0

    X_train = train_df[features].fillna(0).values
    y_train = train_df["exploited"].values
    X_test = test_df[features].fillna(0).values
    y_test = test_df["exploited"].values

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    model = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auc = None

    return auc, len(features)


def run_ablation_seed(train_df, test_df, feature_cols, groups, seed):
    """Run full ablation study for one seed.

    Returns dict with full_model, leave_one_out, and single_group results.
    """
    result = {"seed": seed}

    # ── Full model (all features) ────────────────────────────────────────
    print(f"    Full model ({len(feature_cols)} features)...", end=" ", flush=True)
    full_auc, _ = train_xgboost_with_features(
        train_df, test_df, feature_cols, seed
    )
    result["full_model"] = {"auc": full_auc, "n_features": len(feature_cols)}
    print(f"AUC={full_auc:.4f}" if full_auc else "AUC=N/A")

    # ── Leave-one-group-out ──────────────────────────────────────────────
    result["leave_one_out"] = {}
    print("    Leave-one-group-out:")
    for group_name, group_feats in sorted(groups.items()):
        remaining = [f for f in feature_cols if f not in group_feats]
        auc, n_feat = train_xgboost_with_features(
            train_df, test_df, remaining, seed
        )
        delta = (auc - full_auc) if (auc is not None and full_auc is not None) else None
        result["leave_one_out"][group_name] = {
            "auc": auc,
            "n_features": n_feat,
            "n_removed": len(group_feats),
            "delta_auc": float(delta) if delta is not None else None,
        }
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"      -{group_name:20s} ({len(group_feats):2d} feats removed): "
              f"AUC={auc_str}  delta={delta_str}")

    # ── Single-group (only this group) ───────────────────────────────────
    result["single_group"] = {}
    print("    Single-group (only this group):")
    for group_name, group_feats in sorted(groups.items()):
        auc, n_feat = train_xgboost_with_features(
            train_df, test_df, group_feats, seed
        )
        result["single_group"][group_name] = {
            "auc": auc,
            "n_features": n_feat,
        }
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"      only-{group_name:20s} ({len(group_feats):2d} feats): "
              f"AUC={auc_str}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feature group ablation study"
    )
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--seeds", type=str,
                        default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Subsample data before experiment (for smoke testing)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    output_dir = Path("outputs/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df, test_df, feature_cols = load_data(
        args.data_dir, sample_frac=args.sample_frac, seed=seeds[0]
    )
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Seeds: {seeds}\n")

    # Define and display feature groups
    groups = define_feature_groups(feature_cols)
    print("Feature groups:")
    for group_name, feats in sorted(groups.items()):
        print(f"  {group_name:20s}: {len(feats):2d} features  {feats[:3]}{'...' if len(feats) > 3 else ''}")
    print()

    all_seed_results = []

    for seed in seeds:
        print(f"=== Seed {seed} ===")
        result = run_ablation_seed(train_df, test_df, feature_cols, groups, seed)
        all_seed_results.append(result)

        # Save per-seed
        out_file = output_dir / f"ablation_seed{seed}.json"
        seed_summary = {
            "seed": seed,
            "sample_frac": args.sample_frac,
            "date": datetime.now().isoformat(),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "num_features": len(feature_cols),
            "feature_groups": {k: v for k, v in sorted(groups.items())},
            "results": result,
        }
        with open(out_file, "w") as f:
            json.dump(seed_summary, f, indent=2)
        print(f"  Saved: {out_file}\n")

    # ── Aggregate summary across seeds ───────────────────────────────────
    print("=" * 70)
    print("AGGREGATE RESULTS (mean +/- std across seeds)")
    print("=" * 70)

    # Full model
    full_aucs = [r["full_model"]["auc"] for r in all_seed_results
                 if r["full_model"]["auc"] is not None]
    if full_aucs:
        print(f"\nFull model: AUC = {np.mean(full_aucs):.4f} +/- {np.std(full_aucs):.4f}")

    # Leave-one-out
    print(f"\nLeave-one-group-out (negative delta = group was helpful):")
    print(f"  {'Group':20s}  {'Mean AUC':>10s}  {'Mean Delta':>12s}  {'Std Delta':>10s}  {'Importance':>10s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}")

    agg_loo = {}
    for group_name in sorted(groups.keys()):
        aucs = []
        deltas = []
        for r in all_seed_results:
            entry = r["leave_one_out"].get(group_name)
            if entry and entry["auc"] is not None and entry["delta_auc"] is not None:
                aucs.append(entry["auc"])
                deltas.append(entry["delta_auc"])
        if deltas:
            mean_auc = np.mean(aucs)
            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas)
            # Importance = magnitude of negative delta (higher = more important)
            importance = abs(mean_delta) if mean_delta < 0 else 0.0
            agg_loo[group_name] = {
                "mean_auc": float(mean_auc),
                "mean_delta": float(mean_delta),
                "std_delta": float(std_delta),
                "importance": float(importance),
                "n_features": len(groups[group_name]),
            }
            sign = "***" if importance > 0.01 else "**" if importance > 0.005 else "*" if importance > 0.001 else ""
            print(f"  {group_name:20s}  {mean_auc:10.4f}  {mean_delta:+12.4f}  {std_delta:10.4f}  {importance:10.4f} {sign}")

    # Single-group
    print(f"\nSingle-group (standalone predictive power):")
    print(f"  {'Group':20s}  {'Mean AUC':>10s}  {'Std AUC':>10s}  {'N Features':>10s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")

    agg_single = {}
    for group_name in sorted(groups.keys()):
        aucs = []
        for r in all_seed_results:
            entry = r["single_group"].get(group_name)
            if entry and entry["auc"] is not None:
                aucs.append(entry["auc"])
        if aucs:
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            agg_single[group_name] = {
                "mean_auc": float(mean_auc),
                "std_auc": float(std_auc),
                "n_features": len(groups[group_name]),
            }
            print(f"  {group_name:20s}  {mean_auc:10.4f}  {std_auc:10.4f}  {len(groups[group_name]):10d}")

    # Save aggregate
    agg_file = output_dir / "ablation_summary.json"
    agg_summary = {
        "date": datetime.now().isoformat(),
        "sample_frac": args.sample_frac,
        "seeds": seeds,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "num_features": len(feature_cols),
        "feature_groups": {k: v for k, v in sorted(groups.items())},
        "full_model_mean_auc": float(np.mean(full_aucs)) if full_aucs else None,
        "full_model_std_auc": float(np.std(full_aucs)) if full_aucs else None,
        "leave_one_out": agg_loo,
        "single_group": agg_single,
    }
    with open(agg_file, "w") as f:
        json.dump(agg_summary, f, indent=2)
    print(f"\nSaved aggregate: {agg_file}")


if __name__ == "__main__":
    main()
