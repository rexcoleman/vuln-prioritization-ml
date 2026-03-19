#!/usr/bin/env python3
"""Train models with CISA KEV labels as ground truth (dual ground truth experiment).

Compares 3 ground truth definitions:
  1. ExploitDB only (original)
  2. KEV only (CISA's authoritative list)
  3. Either (ExploitDB OR KEV)

Trains LogReg + XGBoost (tuned) across 5 seeds for each ground truth.
Also trains no-EPSS variants for the combined ground truth.

Usage:
    python scripts/train_kev_labels.py --data-dir data/
    python scripts/train_kev_labels.py --data-dir data/ --sample-frac 0.01
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
EPSS_FEATURES = {"epss", "epss_percentile"}
OUTPUT_DIR = Path("outputs/models")


def load_kev_data(data_dir, sample_frac=1.0, seed=42):
    processed = Path(data_dir) / "processed"
    train = pd.read_parquet(processed / "train_kev.parquet")
    test = pd.read_parquet(processed / "test_kev.parquet")

    with open(processed / "feature_cols.json") as f:
        feature_cols = json.load(f)

    if sample_frac < 1.0:
        train = train.sample(frac=sample_frac, random_state=seed)
        test = test.sample(frac=sample_frac, random_state=seed)

    return train, test, feature_cols


def eval_metrics(y_true, y_prob):
    metrics = {}
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = None
    try:
        metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auc_pr"] = None
    return metrics


def train_eval_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    return {
        "train": eval_metrics(y_train, train_prob),
        "test": eval_metrics(y_test, test_prob),
    }


def run_experiment(train_df, test_df, feature_cols, label_col, seeds, experiment_name):
    """Train LogReg + XGBoost (tuned) across seeds for a given label column."""
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name} (label={label_col})")
    print(f"{'='*60}")

    y_train = train_df[label_col].values
    y_test = test_df[label_col].values
    print(f"  Train positive rate: {y_train.mean()*100:.2f}% ({y_train.sum():,}/{len(y_train):,})")
    print(f"  Test positive rate:  {y_test.mean()*100:.2f}% ({y_test.sum():,}/{len(y_test):,})")

    if y_test.sum() == 0:
        print("  SKIP: No positive labels in test set")
        return None

    X_train_raw = train_df[feature_cols].fillna(0).values
    X_test_raw = test_df[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    spw = neg_count / max(pos_count, 1)

    all_results = []
    for seed in seeds:
        seed_result = {}

        # LogReg
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs")
        seed_result["logistic_regression"] = train_eval_model(lr, X_train_scaled, y_train, X_test_scaled, y_test)

        # XGBoost tuned (depth=3)
        xgb = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=seed, n_jobs=-1, verbosity=0,
        )
        seed_result["xgboost_tuned"] = train_eval_model(xgb, X_train_raw, y_train, X_test_raw, y_test)

        all_results.append({"seed": seed, "models": seed_result})
        lr_auc = seed_result["logistic_regression"]["test"]["auc_roc"] or 0
        xgb_auc = seed_result["xgboost_tuned"]["test"]["auc_roc"] or 0
        print(f"  Seed {seed}: LogReg={lr_auc:.4f}, XGB-tuned={xgb_auc:.4f}")

    # Aggregate
    summary = {}
    for model_name in ["logistic_regression", "xgboost_tuned"]:
        aucs = [r["models"][model_name]["test"]["auc_roc"] for r in all_results
                if r["models"][model_name]["test"]["auc_roc"] is not None]
        summary[model_name] = {
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "std_auc": float(np.std(aucs)) if aucs else None,
            "values": aucs,
        }
        if aucs:
            print(f"  {model_name}: AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    return {
        "experiment": experiment_name,
        "label_col": label_col,
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "train_positive_count": int(y_train.sum()),
        "test_positive_count": int(y_test.sum()),
        "seeds": seeds,
        "summary": summary,
        "per_seed": all_results,
    }


def run_no_epss_experiment(train_df, test_df, feature_cols, label_col, seeds, experiment_name):
    """Same but with EPSS features removed."""
    features_no_epss = [f for f in feature_cols if f not in EPSS_FEATURES]
    return run_experiment(train_df, test_df, features_no_epss, label_col, seeds, experiment_name)


def main():
    parser = argparse.ArgumentParser(description="Train with CISA KEV ground truth")
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--sample-frac", type=float, default=1.0)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df, feature_cols = load_kev_data(args.data_dir, args.sample_frac, seeds[0])

    results = {}

    # Experiment 1: ExploitDB labels (original — for comparison)
    results["exploitdb"] = run_experiment(
        train_df, test_df, feature_cols, "exploited", seeds, "ExploitDB ground truth"
    )

    # Experiment 2: KEV labels only
    results["kev"] = run_experiment(
        train_df, test_df, feature_cols, "kev_exploited", seeds, "KEV ground truth"
    )

    # Experiment 3: Either label (ExploitDB OR KEV)
    results["either"] = run_experiment(
        train_df, test_df, feature_cols, "either_exploited", seeds, "Either ground truth (ExploitDB OR KEV)"
    )

    # Experiment 4: Either label, NO EPSS features
    results["either_no_epss"] = run_no_epss_experiment(
        train_df, test_df, feature_cols, "either_exploited", seeds,
        "Either ground truth, NO EPSS"
    )

    # Experiment 5: KEV labels, NO EPSS features
    results["kev_no_epss"] = run_no_epss_experiment(
        train_df, test_df, feature_cols, "kev_exploited", seeds,
        "KEV ground truth, NO EPSS"
    )

    # Save all results
    output = {
        "date": datetime.now().isoformat(),
        "seeds": seeds,
        "sample_frac": args.sample_frac,
        "num_features": len(feature_cols),
        "num_features_no_epss": len(feature_cols) - len(EPSS_FEATURES),
        "experiments": {k: v for k, v in results.items() if v is not None},
    }
    out_file = OUTPUT_DIR / "kev_ground_truth_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 80)
    print("GROUND TRUTH COMPARISON (LogReg, test AUC, mean +/- std)")
    print("=" * 80)
    print(f"{'Experiment':<40} {'LogReg AUC':>15} {'XGB-Tuned AUC':>15}")
    print("-" * 80)
    for name, result in results.items():
        if result is None:
            continue
        lr = result["summary"]["logistic_regression"]
        xgb = result["summary"]["xgboost_tuned"]
        lr_str = f"{lr['mean_auc']:.4f}+/-{lr['std_auc']:.4f}" if lr["mean_auc"] else "N/A"
        xgb_str = f"{xgb['mean_auc']:.4f}+/-{xgb['std_auc']:.4f}" if xgb["mean_auc"] else "N/A"
        print(f"{name:<40} {lr_str:>15} {xgb_str:>15}")
    print("=" * 80)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
