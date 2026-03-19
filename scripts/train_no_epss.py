#!/usr/bin/env python3
"""Train all 7 algorithms WITHOUT EPSS features — circularity fix.

Addresses reviewer criticism: "You're showing an ML model trained on EPSS learns EPSS."
By removing epss and epss_percentile, we answer: "What does ML add beyond EPSS?"

Usage:
    python scripts/train_no_epss.py --data-dir data/
    python scripts/train_no_epss.py --data-dir data/ --sample-frac 0.01  # smoke test
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
    print("WARNING: lightgbm not installed, skipping LightGBM.\n")

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
EPSS_FEATURES = {"epss", "epss_percentile"}
SVM_MAX_TRAIN = 50_000
OUTPUT_DIR = Path("outputs/models")


def load_data_no_epss(data_dir, sample_frac=1.0, seed=42):
    """Load train/test data with EPSS features removed."""
    processed = Path(data_dir) / "processed"
    train = pd.read_parquet(processed / "train.parquet")
    test = pd.read_parquet(processed / "test.parquet")

    with open(processed / "feature_cols.json") as f:
        all_features = json.load(f)

    # Remove EPSS features
    features_no_epss = [f for f in all_features if f not in EPSS_FEATURES]
    removed = [f for f in all_features if f in EPSS_FEATURES]
    print(f"Removed EPSS features: {removed}")
    print(f"Remaining features: {len(features_no_epss)} (was {len(all_features)})")

    if sample_frac < 1.0:
        train = train.sample(frac=sample_frac, random_state=seed)
        test = test.sample(frac=sample_frac, random_state=seed)

    X_train = train[features_no_epss].fillna(0).values
    y_train = train["exploited"].values
    X_test = test[features_no_epss].fillna(0).values
    y_test = test["exploited"].values

    return X_train, y_train, X_test, y_test, features_no_epss


def evaluate(y_true, y_pred, y_prob):
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
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    train_metrics = evaluate(y_train, train_pred, train_prob)
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_pred, test_prob)
    return train_metrics, test_metrics


def run_seed(X_train, y_train, X_test, y_test, seed):
    results = {}
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Random Forest
    print("  Training Random Forest...", end=" ", flush=True)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=5,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    train_m, test_m = train_and_eval(rf, X_train, y_train, X_test, y_test)
    results["random_forest"] = {"train": train_m, "test": test_m}
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}")

    # 2. XGBoost (default HP)
    print("  Training XGBoost (default)...", end=" ", flush=True)
    xgb = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    train_m, test_m = train_and_eval(xgb, X_train, y_train, X_test, y_test)
    results["xgboost_default"] = {"train": train_m, "test": test_m}
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}")

    # 3. XGBoost (tuned depth=3)
    print("  Training XGBoost (depth=3)...", end=" ", flush=True)
    xgb_tuned = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    train_m, test_m = train_and_eval(xgb_tuned, X_train, y_train, X_test, y_test)
    results["xgboost_tuned"] = {"train": train_m, "test": test_m}
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}")

    # 4. Logistic Regression
    print("  Training Logistic Regression...", end=" ", flush=True)
    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=seed, solver="lbfgs",
    )
    train_m, test_m = train_and_eval(lr, X_train_scaled, y_train, X_test_scaled, y_test)
    results["logistic_regression"] = {"train": train_m, "test": test_m}
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}")

    # 5. SVM-RBF
    print("  Training SVM-RBF...", end=" ", flush=True)
    if len(X_train_scaled) > SVM_MAX_TRAIN:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=SVM_MAX_TRAIN, random_state=seed)
        idx, _ = next(sss.split(X_train_scaled, y_train))
        X_svm, y_svm = X_train_scaled[idx], y_train[idx]
        print(f"(subsampled to {len(X_svm):,}) ", end="", flush=True)
    else:
        X_svm, y_svm = X_train_scaled, y_train
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
              class_weight="balanced", random_state=seed)
    svm.fit(X_svm, y_svm)
    svm_train_pred = svm.predict(X_svm)
    svm_train_prob = svm.predict_proba(X_svm)[:, 1]
    train_m = evaluate(y_svm, svm_train_pred, svm_train_prob)
    svm_test_pred = svm.predict(X_test_scaled)
    svm_test_prob = svm.predict_proba(X_test_scaled)[:, 1]
    test_m = evaluate(y_test, svm_test_pred, svm_test_prob)
    results["svm_rbf"] = {"train": train_m, "test": test_m}
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}")

    # 6. LightGBM
    if HAS_LIGHTGBM:
        print("  Training LightGBM...", end=" ", flush=True)
        lgbm = lgb.LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            is_unbalance=True, random_state=seed, n_jobs=-1, verbose=-1,
        )
        train_m, test_m = train_and_eval(lgbm, X_train, y_train, X_test, y_test)
        results["lightgbm"] = {"train": train_m, "test": test_m}
        print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}")

    # 7. kNN
    print("  Training kNN...", end=" ", flush=True)
    knn = KNeighborsClassifier(n_neighbors=11, weights="uniform", metric="euclidean", n_jobs=-1)
    train_m, test_m = train_and_eval(knn, X_train_scaled, y_train, X_test_scaled, y_test)
    results["knn"] = {"train": train_m, "test": test_m}
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}")

    # 8. MLP
    print("  Training MLP...", end=" ", flush=True)
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,), activation="relu", solver="adam",
        learning_rate_init=0.001, max_iter=200, early_stopping=True,
        validation_fraction=0.1, random_state=seed,
    )
    train_m, test_m = train_and_eval(mlp, X_train_scaled, y_train, X_test_scaled, y_test)
    results["mlp"] = {"train": train_m, "test": test_m}
    print(f"AUC={test_m.get('auc_roc', 'N/A'):.4f}")

    return results


def compute_summary(all_seed_results):
    model_names = [
        k for k in all_seed_results[0]["models"]
        if "error" not in all_seed_results[0]["models"].get(k, {})
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


def main():
    parser = argparse.ArgumentParser(description="Train 7 models WITHOUT EPSS features")
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--sample-frac", type=float, default=1.0)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NO-EPSS EXPERIMENT: Training all models without EPSS features")
    print("=" * 70)

    X_train, y_train, X_test, y_test, features = load_data_no_epss(
        args.data_dir, sample_frac=args.sample_frac, seed=seeds[0]
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Features: {len(features)} (EPSS removed)")
    print(f"Seeds: {seeds}\n")

    all_seed_results = []
    for seed in seeds:
        print(f"=== Seed {seed} ===")
        model_results = run_seed(X_train, y_train, X_test, y_test, seed)
        seed_data = {
            "seed": seed,
            "sample_frac": args.sample_frac,
            "date": datetime.now().isoformat(),
            "experiment": "no_epss",
            "epss_features_removed": list(EPSS_FEATURES),
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "num_features": len(features),
            "features": features,
            "models": model_results,
        }
        all_seed_results.append(seed_data)

        out_file = OUTPUT_DIR / f"no_epss_seed{seed}.json"
        with open(out_file, "w") as f:
            json.dump(seed_data, f, indent=2)
        print(f"  Saved: {out_file}\n")

    # Aggregate summary
    summary = compute_summary(all_seed_results)
    summary_file = OUTPUT_DIR / "no_epss_summary.json"
    summary_data = {
        "date": datetime.now().isoformat(),
        "experiment": "no_epss",
        "epss_features_removed": list(EPSS_FEATURES),
        "seeds": seeds,
        "sample_frac": args.sample_frac,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "num_features": len(features),
        "features": features,
        "models": summary,
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 90)
    print("NO-EPSS MODEL COMPARISON (test set, mean +/- std across seeds)")
    print("=" * 90)
    header = f"{'Model':<25} {'AUC-ROC':>14} {'F1':>14} {'Precision':>14} {'Recall':>14}"
    print(header)
    print("-" * 90)

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
        print(f"{model_name:<25} {cols[0]:>14} {cols[1]:>14} {cols[2]:>14} {cols[3]:>14}")

    print("=" * 90)
    print(f"\nSaved: {summary_file}")


if __name__ == "__main__":
    main()
