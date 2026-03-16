#!/usr/bin/env python3
"""Train expanded model suite for vulnerability exploitability prediction.

Models (7 total):
  Existing: Random Forest, XGBoost, Logistic Regression
  New: SVM-RBF, LightGBM, kNN, MLP

Trains all models across multiple seeds and computes mean +/- std summary.

Usage:
    python scripts/train_expanded_models.py --data-dir data/
    python scripts/train_expanded_models.py --data-dir data/ --seeds 42,123,456,789,1024
    python scripts/train_expanded_models.py --data-dir data/ --sample-frac 0.01  # smoke test
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("WARNING: lightgbm not installed. Install with: pip install lightgbm")
    print("         LightGBM model will be skipped.\n")

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
SVM_MAX_TRAIN = 50_000  # Subsample SVM training if dataset exceeds this

OUTPUT_DIR = Path("outputs/models")


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

    X_train = train[feature_cols].fillna(0).values
    y_train = train["exploited"].values
    X_test = test[feature_cols].fillna(0).values
    y_test = test["exploited"].values

    return X_train, y_train, X_test, y_test, feature_cols


def evaluate(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = None
        metrics["auc_pr"] = None
    return metrics


def train_and_eval(model, X_train, y_train, X_test, y_test):
    """Fit model and return train + test metrics."""
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    train_metrics = evaluate(y_train, train_pred, train_prob)

    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_pred, test_prob)

    return train_metrics, test_metrics


def subsample_for_svm(X, y, seed):
    """Stratified subsample for SVM if dataset is too large."""
    if len(X) <= SVM_MAX_TRAIN:
        return X, y, False
    from sklearn.model_selection import StratifiedShuffleSplit
    n_samples = SVM_MAX_TRAIN
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=n_samples, random_state=seed
    )
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx], True


def run_seed(X_train, y_train, X_test, y_test, seed):
    """Train all 7 models for one seed. Returns dict of model results."""
    results = {}

    # Class imbalance weight for XGBoost / LightGBM
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    # Pre-compute scaled data for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 1. Random Forest ---
    print("  Training Random Forest...", end=" ", flush=True)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    train_m, test_m = train_and_eval(rf, X_train, y_train, X_test, y_test)
    results["random_forest"] = {"train": train_m, "test": test_m}
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}, F1={test_m['f1']:.4f}")

    # --- 2. XGBoost ---
    print("  Training XGBoost...", end=" ", flush=True)
    xgb = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    train_m, test_m = train_and_eval(xgb, X_train, y_train, X_test, y_test)
    results["xgboost"] = {"train": train_m, "test": test_m}
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}, F1={test_m['f1']:.4f}")

    # --- 3. Logistic Regression (scaled) ---
    print("  Training Logistic Regression...", end=" ", flush=True)
    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=seed, solver="lbfgs",
    )
    train_m, test_m = train_and_eval(
        lr, X_train_scaled, y_train, X_test_scaled, y_test
    )
    results["logistic_regression"] = {"train": train_m, "test": test_m}
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}, F1={test_m['f1']:.4f}")

    # --- 4. SVM-RBF (scaled, subsampled if needed) ---
    print("  Training SVM-RBF...", end=" ", flush=True)
    X_svm_train, y_svm_train, svm_subsampled = subsample_for_svm(
        X_train_scaled, y_train, seed
    )
    if svm_subsampled:
        print(f"(subsampled to {len(X_svm_train):,}) ", end="", flush=True)
    svm = SVC(
        kernel="rbf", C=1.0, gamma="scale", probability=True,
        class_weight="balanced", random_state=seed,
    )
    svm.fit(X_svm_train, y_svm_train)
    # Train metrics on SVM training subset
    svm_train_pred = svm.predict(X_svm_train)
    svm_train_prob = svm.predict_proba(X_svm_train)[:, 1]
    train_m = evaluate(y_svm_train, svm_train_pred, svm_train_prob)
    # Test metrics on full test set
    svm_test_pred = svm.predict(X_test_scaled)
    svm_test_prob = svm.predict_proba(X_test_scaled)[:, 1]
    test_m = evaluate(y_test, svm_test_pred, svm_test_prob)
    results["svm_rbf"] = {
        "train": train_m, "test": test_m,
        "subsampled": svm_subsampled,
        "train_size_actual": int(len(X_svm_train)),
    }
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}, F1={test_m['f1']:.4f}")

    # --- 5. LightGBM ---
    if HAS_LIGHTGBM:
        print("  Training LightGBM...", end=" ", flush=True)
        lgbm = lgb.LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            is_unbalance=True, random_state=seed, n_jobs=-1,
            verbose=-1,
        )
        train_m, test_m = train_and_eval(lgbm, X_train, y_train, X_test, y_test)
        results["lightgbm"] = {"train": train_m, "test": test_m}
        print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}, F1={test_m['f1']:.4f}")
    else:
        print("  Skipping LightGBM (not installed)")
        results["lightgbm"] = {"error": "lightgbm not installed"}

    # --- 6. kNN (scaled) ---
    print("  Training kNN...", end=" ", flush=True)
    knn = KNeighborsClassifier(
        n_neighbors=11, weights="uniform", metric="euclidean", n_jobs=-1,
    )
    train_m, test_m = train_and_eval(
        knn, X_train_scaled, y_train, X_test_scaled, y_test
    )
    results["knn"] = {"train": train_m, "test": test_m}
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}, F1={test_m['f1']:.4f}")

    # --- 7. MLP (scaled) ---
    print("  Training MLP...", end=" ", flush=True)
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,), activation="relu", solver="adam",
        learning_rate_init=0.001, max_iter=200, early_stopping=True,
        validation_fraction=0.1, random_state=seed,
    )
    train_m, test_m = train_and_eval(
        mlp, X_train_scaled, y_train, X_test_scaled, y_test
    )
    results["mlp"] = {"train": train_m, "test": test_m}
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}, F1={test_m['f1']:.4f}")

    return results


def compute_summary(all_seed_results):
    """Compute mean +/- std across seeds for each model and metric."""
    # Collect all model names from first seed
    model_names = [
        k for k in all_seed_results[0]["models"]
        if "error" not in all_seed_results[0]["models"][k]
    ]
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]

    summary = {}
    for model_name in model_names:
        model_summary = {}
        for split in ["train", "test"]:
            split_summary = {}
            for metric in metrics:
                values = []
                for seed_result in all_seed_results:
                    val = seed_result["models"].get(model_name, {}).get(split, {}).get(metric)
                    if val is not None:
                        values.append(val)
                if values:
                    split_summary[metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "values": values,
                    }
            model_summary[split] = split_summary
        summary[model_name] = model_summary

    return summary


def print_comparison_table(summary):
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 90)
    print("EXPANDED MODEL COMPARISON (test set, mean +/- std across seeds)")
    print("=" * 90)

    header = f"{'Model':<22} {'AUC-ROC':>14} {'F1':>14} {'Precision':>14} {'Recall':>14}"
    print(header)
    print("-" * 90)

    # Sort by test AUC descending
    sorted_models = sorted(
        summary.keys(),
        key=lambda m: summary[m].get("test", {}).get("auc_roc", {}).get("mean", 0),
        reverse=True,
    )

    for model_name in sorted_models:
        test = summary[model_name].get("test", {})
        cols = []
        for metric in ["auc_roc", "f1", "precision", "recall"]:
            if metric in test:
                mean = test[metric]["mean"]
                std = test[metric]["std"]
                cols.append(f"{mean:.4f}+/-{std:.4f}")
            else:
                cols.append("N/A")
        print(f"{model_name:<22} {cols[0]:>14} {cols[1]:>14} {cols[2]:>14} {cols[3]:>14}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Train expanded model suite (7 models x N seeds)"
    )
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument(
        "--seeds", type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--sample-frac", type=float, default=1.0,
        help="Subsample data before training (for smoke testing)",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_train, y_train, X_test, y_test, feature_cols = load_data(
        args.data_dir, sample_frac=args.sample_frac, seed=seeds[0]
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Train exploit rate: {y_train.mean()*100:.1f}%")
    print(f"Test exploit rate: {y_test.mean()*100:.1f}%")
    print(f"Seeds: {seeds}\n")

    all_seed_results = []

    for seed in seeds:
        print(f"=== Seed {seed} ===")
        model_results = run_seed(X_train, y_train, X_test, y_test, seed)

        seed_summary = {
            "seed": seed,
            "sample_frac": args.sample_frac,
            "date": datetime.now().isoformat(),
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "num_features": len(feature_cols),
            "models": model_results,
        }
        all_seed_results.append(seed_summary)

        # Save per-seed results
        out_file = OUTPUT_DIR / f"expanded_seed{seed}.json"
        with open(out_file, "w") as f:
            json.dump(seed_summary, f, indent=2)
        print(f"  Saved: {out_file}\n")

    # Compute and save cross-seed summary
    summary = compute_summary(all_seed_results)

    summary_file = OUTPUT_DIR / "expanded_summary.json"
    summary_data = {
        "date": datetime.now().isoformat(),
        "seeds": seeds,
        "sample_frac": args.sample_frac,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "num_features": len(feature_cols),
        "models": summary,
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved summary: {summary_file}")

    # Print comparison table
    print_comparison_table(summary)


if __name__ == "__main__":
    main()
