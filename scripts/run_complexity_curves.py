#!/usr/bin/env python3
"""Generate model complexity curve data.

Sweeps one hyperparameter per model (others at default) to diagnose
bias-variance tradeoff:
  - RF: n_estimators in [10, 50, 100, 200, 500]
  - XGBoost: max_depth in [2, 3, 5, 7, 10, 15]
  - LogReg: C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
  - SVM-RBF: C in [0.01, 0.1, 1.0, 10.0, 100.0]
  - LightGBM: max_depth in [2, 3, 5, 7, 10, 15]
  - kNN: n_neighbors in [1, 3, 5, 11, 21, 51]
  - MLP: hidden_layer_sizes in [(50,), (100,), (200,), (100,50), (200,100)]

Usage:
    python scripts/run_complexity_curves.py --data-dir data/
    python scripts/run_complexity_curves.py --data-dir data/ --seeds 42,123
    python scripts/run_complexity_curves.py --data-dir data/ --sample-frac 0.01  # smoke test
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

# Hyperparameter sweeps
HP_SWEEPS = {
    "rf": {
        "param_name": "n_estimators",
        "values": [10, 50, 100, 200, 500],
    },
    "xgboost": {
        "param_name": "max_depth",
        "values": [2, 3, 5, 7, 10, 15],
    },
    "logreg": {
        "param_name": "C",
        "values": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "svm_rbf": {
        "param_name": "C",
        "values": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "knn": {
        "param_name": "n_neighbors",
        "values": [1, 3, 5, 11, 21, 51],
    },
    "mlp": {
        "param_name": "hidden_layer_sizes",
        "values": [(50,), (100,), (200,), (100, 50), (200, 100)],
    },
}
if HAS_LIGHTGBM:
    HP_SWEEPS["lightgbm"] = {
        "param_name": "max_depth",
        "values": [2, 3, 5, 7, 10, 15],
    }

# Models that require StandardScaler
SCALED_MODELS = {"logreg", "svm_rbf", "knn", "mlp"}


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


def build_rf(hp_value, seed, **kwargs):
    """Build RF with n_estimators sweep, other params at default."""
    return RandomForestClassifier(
        n_estimators=hp_value, max_depth=20, min_samples_leaf=5,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )


def build_xgboost(hp_value, seed, scale_pos_weight=1.0):
    """Build XGBoost with max_depth sweep, other params at default."""
    return XGBClassifier(
        n_estimators=200, max_depth=hp_value, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=seed, n_jobs=-1, verbosity=0,
    )


def build_logreg(hp_value, seed, **kwargs):
    """Build LogReg with C sweep, other params at default."""
    return LogisticRegression(
        C=hp_value, max_iter=1000, class_weight="balanced",
        random_state=seed, solver="lbfgs",
    )


def build_svm_rbf(hp_value, seed, **kwargs):
    """Build SVM-RBF with C sweep, other params at default."""
    return SVC(
        kernel="rbf", C=hp_value, gamma="scale", probability=True,
        class_weight="balanced", random_state=seed,
    )


def build_lightgbm(hp_value, seed, **kwargs):
    """Build LightGBM with max_depth sweep, other params at default."""
    return lgb.LGBMClassifier(
        n_estimators=200, max_depth=hp_value, learning_rate=0.1,
        is_unbalance=True, random_state=seed, n_jobs=-1, verbose=-1,
    )


def build_knn(hp_value, seed, **kwargs):
    """Build kNN with n_neighbors sweep, other params at default."""
    return KNeighborsClassifier(
        n_neighbors=hp_value, weights="uniform", metric="euclidean", n_jobs=-1,
    )


def build_mlp(hp_value, seed, **kwargs):
    """Build MLP with hidden_layer_sizes sweep, other params at default."""
    return MLPClassifier(
        hidden_layer_sizes=hp_value, activation="relu", solver="adam",
        learning_rate_init=0.001, max_iter=200, early_stopping=True,
        validation_fraction=0.1, random_state=seed,
    )


MODEL_BUILDERS = {
    "rf": build_rf,
    "xgboost": build_xgboost,
    "logreg": build_logreg,
    "svm_rbf": build_svm_rbf,
    "knn": build_knn,
    "mlp": build_mlp,
}
if HAS_LIGHTGBM:
    MODEL_BUILDERS["lightgbm"] = build_lightgbm


def subsample_for_svm(X, y, seed):
    """Stratified subsample for SVM if dataset is too large."""
    if len(X) <= SVM_MAX_TRAIN:
        return X, y, False
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=SVM_MAX_TRAIN, random_state=seed
    )
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx], True


def run_complexity_curves(X_train, y_train, X_test, y_test, seed):
    """Run complexity curve experiment for one seed."""
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    # Pre-compute scaled data for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Pre-compute SVM subsample (scaled)
    X_svm_train, y_svm_train, svm_subsampled = subsample_for_svm(
        X_train_scaled, y_train, seed
    )
    if svm_subsampled:
        print(f"  (SVM subsampled to {len(X_svm_train):,} rows)")

    all_results = {}

    for model_name, sweep in HP_SWEEPS.items():
        param_name = sweep["param_name"]
        values = sweep["values"]
        print(f"  {model_name} ({param_name}):", end=" ", flush=True)

        model_results = []
        for val in values:
            builder = MODEL_BUILDERS[model_name]
            extra_kwargs = {"scale_pos_weight": scale_pos_weight} if model_name == "xgboost" else {}
            model = builder(val, seed, **extra_kwargs)

            if model_name in SCALED_MODELS:
                if model_name == "svm_rbf":
                    # Use subsampled data for SVM training
                    model.fit(X_svm_train, y_svm_train)
                    train_pred = model.predict(X_svm_train)
                    train_prob = model.predict_proba(X_svm_train)[:, 1]
                    train_metrics = evaluate(y_svm_train, train_pred, train_prob)
                else:
                    model.fit(X_train_scaled, y_train)
                    train_pred = model.predict(X_train_scaled)
                    train_prob = model.predict_proba(X_train_scaled)[:, 1]
                    train_metrics = evaluate(y_train, train_pred, train_prob)
                val_pred = model.predict(X_test_scaled)
                val_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                train_prob = model.predict_proba(X_train)[:, 1]
                train_metrics = evaluate(y_train, train_pred, train_prob)
                val_pred = model.predict(X_test)
                val_prob = model.predict_proba(X_test)[:, 1]

            val_metrics = evaluate(y_test, val_pred, val_prob)

            # Convert tuples to strings for JSON serialization
            param_value = str(val) if isinstance(val, tuple) else val
            model_results.append({
                "param_value": param_value,
                "train_auc": train_metrics["auc_roc"],
                "val_auc": val_metrics["auc_roc"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
            })
            print(f"{val}", end=" ", flush=True)

        all_results[model_name] = {
            "param_name": param_name,
            "results": model_results,
        }
        print()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Generate complexity curve data")
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Subsample data before experiment (for smoke testing)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    output_dir = Path("outputs/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_train, y_train, X_test, y_test, feature_cols = load_data(
        args.data_dir, sample_frac=args.sample_frac, seed=seeds[0]
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train exploit rate: {y_train.mean()*100:.1f}%")
    print(f"Seeds: {seeds}\n")

    for seed in seeds:
        print(f"=== Seed {seed} ===")
        results = run_complexity_curves(X_train, y_train, X_test, y_test, seed)

        out_file = output_dir / f"complexity_curves_seed{seed}.json"
        summary = {
            "seed": seed,
            "sample_frac": args.sample_frac,
            "date": datetime.now().isoformat(),
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "num_features": len(feature_cols),
            "sweeps": results,
        }
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {out_file}\n")

    print("Complexity curves complete.")


if __name__ == "__main__":
    main()
