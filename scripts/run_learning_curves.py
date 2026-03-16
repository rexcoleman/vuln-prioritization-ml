#!/usr/bin/env python3
"""Generate learning curve data for all models.

Trains RF, XGBoost, LogReg, SVM-RBF, LightGBM, kNN, and MLP on increasing
fractions of training data to diagnose underfitting vs overfitting.

Usage:
    python scripts/run_learning_curves.py --data-dir data/
    python scripts/run_learning_curves.py --data-dir data/ --seeds 42,123
    python scripts/run_learning_curves.py --data-dir data/ --sample-frac 0.01  # smoke test
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
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
    print("WARNING: lightgbm not installed. LightGBM will be skipped.")

SVM_MAX_TRAIN = 50_000

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


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
    """Compute AUC and F1."""
    metrics = {"f1": float(f1_score(y_true, y_pred, zero_division=0))}
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = None
    return metrics


def build_models(seed, scale_pos_weight=1.0):
    """Build model dict with consistent hyperparameters matching train_models.py."""
    models = {
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=seed, n_jobs=-1, verbosity=0,
        ),
        "logreg": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=seed, solver="lbfgs",
        ),
        "svm_rbf": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=seed,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=11, weights="uniform", metric="euclidean", n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(100,), activation="relu", solver="adam",
            learning_rate_init=0.001, max_iter=200, early_stopping=True,
            validation_fraction=0.1, random_state=seed,
        ),
    }
    if HAS_LIGHTGBM:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            is_unbalance=True, random_state=seed, n_jobs=-1, verbose=-1,
        )
    return models

# Models that require StandardScaler
SCALED_MODELS = {"logreg", "svm_rbf", "knn", "mlp"}


def subsample_stratified(X, y, fraction, seed):
    """Stratified subsample of training data."""
    if fraction >= 1.0:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


def subsample_for_svm(X, y, seed):
    """Stratified subsample for SVM if dataset is too large."""
    if len(X) <= SVM_MAX_TRAIN:
        return X, y, False
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=SVM_MAX_TRAIN, random_state=seed
    )
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx], True


def run_learning_curves(X_train, y_train, X_test, y_test, seed):
    """Run learning curve experiment for one seed."""
    results = []
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    for frac in FRACTIONS:
        print(f"  Fraction {frac:.2f}...", end=" ", flush=True)
        X_sub, y_sub = subsample_stratified(X_train, y_train, frac, seed)

        models = build_models(seed, scale_pos_weight)
        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        X_test_scaled = scaler.transform(X_test)

        model_results = {}
        for name, model in models.items():
            # Use scaled data for models that need it
            if name in SCALED_MODELS:
                X_tr, y_tr = X_sub_scaled, y_sub
                X_te = X_test_scaled
                # SVM subsample for large datasets
                if name == "svm_rbf":
                    X_tr, y_tr, _ = subsample_for_svm(X_tr, y_tr, seed)
                model.fit(X_tr, y_tr)
                train_pred = model.predict(X_tr)
                train_prob = model.predict_proba(X_tr)[:, 1]
                val_pred = model.predict(X_te)
                val_prob = model.predict_proba(X_te)[:, 1]
                train_metrics = evaluate(y_tr, train_pred, train_prob)
            else:
                model.fit(X_sub, y_sub)
                train_pred = model.predict(X_sub)
                train_prob = model.predict_proba(X_sub)[:, 1]
                val_pred = model.predict(X_test)
                val_prob = model.predict_proba(X_test)[:, 1]
                train_metrics = evaluate(y_sub, train_pred, train_prob)

            val_metrics = evaluate(y_test, val_pred, val_prob)

            model_results[name] = {
                "train_auc": train_metrics["auc_roc"],
                "val_auc": val_metrics["auc_roc"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
            }

        entry = {
            "fraction": frac,
            "seed": seed,
            "train_size": int(len(X_sub)),
            "models": model_results,
        }
        results.append(entry)
        print(f"n={len(X_sub):,}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate learning curve data")
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Subsample data before experiment (for smoke testing)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    output_dir = Path("outputs/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    # Load once with first seed (data is the same, only sampling changes)
    X_train, y_train, X_test, y_test, feature_cols = load_data(
        args.data_dir, sample_frac=args.sample_frac, seed=seeds[0]
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train exploit rate: {y_train.mean()*100:.1f}%")
    print(f"Fractions: {FRACTIONS}")
    print(f"Seeds: {seeds}\n")

    for seed in seeds:
        print(f"=== Seed {seed} ===")
        results = run_learning_curves(X_train, y_train, X_test, y_test, seed)

        out_file = output_dir / f"learning_curves_seed{seed}.json"
        summary = {
            "seed": seed,
            "sample_frac": args.sample_frac,
            "date": datetime.now().isoformat(),
            "fractions": FRACTIONS,
            "train_size_full": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "num_features": len(feature_cols),
            "results": results,
        }
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {out_file}\n")

    print("Learning curves complete.")


if __name__ == "__main__":
    main()
