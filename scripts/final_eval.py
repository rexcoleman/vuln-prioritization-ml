# TEST-ACCESS BARRIER: This is the only script that evaluates on test data.
# All other scripts use train/val only.
"""Final evaluation script — loads trained models and evaluates on held-out test set.

Usage:
    python scripts/final_eval.py --data-dir data/ [--models-dir outputs/models/]

Produces:
    outputs/final/final_results.json   — all model metrics on the test set
    stdout                             — formatted results table
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

try:
    import joblib
except ImportError:
    joblib = None


def load_test_data(data_dir: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load test split from parquet."""
    test_path = Path(data_dir) / "processed" / "test.parquet"
    if not test_path.exists():
        print(f"ERROR: Test data not found at {test_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(test_path)
    if "label" not in df.columns:
        print("ERROR: 'label' column not found in test data", file=sys.stderr)
        sys.exit(1)

    y = df["label"]
    X = df.drop(columns=["label"])
    print(f"Loaded test data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class distribution: {y.value_counts().to_dict()}")
    return X, y


def load_models(models_dir: str) -> dict:
    """Load all joblib models from the models directory."""
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"ERROR: Models directory not found at {models_path}", file=sys.stderr)
        sys.exit(1)

    if joblib is None:
        print("ERROR: joblib is required. pip install joblib", file=sys.stderr)
        sys.exit(1)

    models = {}
    for model_file in sorted(models_path.glob("*.joblib")):
        name = model_file.stem
        models[name] = joblib.load(model_file)
        print(f"  Loaded model: {name}")

    if not models:
        print(f"WARNING: No .joblib models found in {models_path}", file=sys.stderr)

    return models


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate a single model on the test set. Called exactly ONCE per model."""
    y_pred = model.predict(X)

    # Probability scores (if available)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        metrics["auc_roc"] = float(roc_auc_score(y, y_prob))
        metrics["auc_pr"] = float(average_precision_score(y, y_prob))

    return metrics


def print_results_table(results: dict):
    """Print formatted table of results."""
    print("\n" + "=" * 80)
    print("FINAL TEST SET RESULTS")
    print("=" * 80)

    header = f"{'Model':<30} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Acc':>8}"
    print(header)
    print("-" * 80)

    for name, metrics in sorted(results.items(), key=lambda x: x[1].get("auc_roc", 0), reverse=True):
        auc_roc = f"{metrics['auc_roc']:.3f}" if "auc_roc" in metrics else "N/A"
        auc_pr = f"{metrics['auc_pr']:.3f}" if "auc_pr" in metrics else "N/A"
        row = (
            f"{name:<30} "
            f"{auc_roc:>8} "
            f"{auc_pr:>8} "
            f"{metrics['f1']:.3f:>8} "
            f"{metrics['precision']:.3f:>8} "
            f"{metrics['recall']:.3f:>8} "
            f"{metrics['accuracy']:.3f:>8}"
        )
        print(row)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Final evaluation on held-out test set (TEST-ACCESS BARRIER)"
    )
    parser.add_argument(
        "--data-dir", default="data/",
        help="Root data directory containing processed/test.parquet"
    )
    parser.add_argument(
        "--models-dir", default="outputs/models/",
        help="Directory containing trained .joblib model files"
    )
    parser.add_argument(
        "--output-dir", default="outputs/final/",
        help="Directory for final results JSON"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TEST-ACCESS BARRIER: Final evaluation on held-out test set")
    print("This script should be run ONCE for final reporting only.")
    print("=" * 60)

    # Load data and models
    print("\nLoading test data...")
    X_test, y_test = load_test_data(args.data_dir)

    print("\nLoading trained models...")
    models = load_models(args.models_dir)

    if not models:
        print("No models to evaluate. Exiting.")
        sys.exit(1)

    # Evaluate each model exactly once
    print(f"\nEvaluating {len(models)} models on test set...")
    results = {}
    for name, model in models.items():
        print(f"  Evaluating: {name}")
        results[name] = evaluate_model(model, X_test, y_test)

    # Save results
    output_path = Path(args.output_dir) / "final_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print table
    print_results_table(results)


if __name__ == "__main__":
    main()
