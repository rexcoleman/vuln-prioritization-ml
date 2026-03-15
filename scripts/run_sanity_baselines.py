#!/usr/bin/env python3
"""Establish chance-level floor with sanity baselines.

Three baselines that any real model must beat:
  1. DummyClassifier(strategy="stratified") — random predictions matching class dist
  2. DummyClassifier(strategy="most_frequent") — always predict majority class
  3. Shuffled-label: shuffle y_train, train LogReg, evaluate — tests for label leakage

Usage:
    python scripts/run_sanity_baselines.py --data-dir data/
    python scripts/run_sanity_baselines.py --data-dir data/ --seeds 42,123
    python scripts/run_sanity_baselines.py --data-dir data/ --sample-frac 0.01  # smoke test
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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

    X_train = train[feature_cols].fillna(0).values
    y_train = train["exploited"].values
    X_test = test[feature_cols].fillna(0).values
    y_test = test["exploited"].values

    return X_train, y_train, X_test, y_test, feature_cols


def evaluate(y_true, y_pred, y_prob=None):
    """Compute accuracy, F1, and AUC."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc_roc"] = None
    else:
        metrics["auc_roc"] = None
    return metrics


def run_stratified(X_train, y_train, X_test, y_test, seed):
    """DummyClassifier with stratified strategy."""
    model = DummyClassifier(strategy="stratified", random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return evaluate(y_test, y_pred, y_prob)


def run_most_frequent(X_train, y_train, X_test, y_test, seed):
    """DummyClassifier with most_frequent strategy."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # most_frequent has no meaningful probability, but sklearn gives one
    y_prob = model.predict_proba(X_test)[:, 1]
    return evaluate(y_test, y_pred, y_prob)


def run_shuffled_label(X_train, y_train, X_test, y_test, seed):
    """Train LogReg on shuffled labels — tests for label leakage."""
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=seed, solver="lbfgs",
    )
    model.fit(X_train_scaled, y_shuffled)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    return evaluate(y_test, y_pred, y_prob)


BASELINE_RUNNERS = {
    "stratified": run_stratified,
    "most_frequent": run_most_frequent,
    "shuffled": run_shuffled_label,
}


def main():
    parser = argparse.ArgumentParser(description="Run sanity baselines")
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Subsample data before experiment (for smoke testing)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    output_dir = Path("outputs/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_train, y_train, X_test, y_test, feature_cols = load_data(
        args.data_dir, sample_frac=args.sample_frac, seed=seeds[0]
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train exploit rate: {y_train.mean()*100:.1f}%")
    print(f"Seeds: {seeds}\n")

    for baseline_name, runner in BASELINE_RUNNERS.items():
        print(f"=== {baseline_name} baseline ===")
        all_seed_results = []

        for seed in seeds:
            metrics = runner(X_train, y_train, X_test, y_test, seed)
            metrics["seed"] = seed
            all_seed_results.append(metrics)
            print(f"  seed {seed}: AUC={metrics['auc_roc']}, "
                  f"F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

        # Compute summary stats across seeds
        aucs = [r["auc_roc"] for r in all_seed_results if r["auc_roc"] is not None]
        f1s = [r["f1"] for r in all_seed_results]
        accs = [r["accuracy"] for r in all_seed_results]

        summary = {
            "baseline": baseline_name,
            "date": datetime.now().isoformat(),
            "sample_frac": args.sample_frac,
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "seeds": seeds,
            "per_seed": all_seed_results,
            "summary": {
                "auc_mean": float(np.mean(aucs)) if aucs else None,
                "auc_std": float(np.std(aucs)) if aucs else None,
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
            },
        }

        out_file = output_dir / f"sanity_{baseline_name}.json"
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {out_file}")

        print(f"  Mean AUC: {summary['summary']['auc_mean']}, "
              f"Mean F1: {summary['summary']['f1_mean']:.4f}\n")

    print("Sanity baselines complete.")


if __name__ == "__main__":
    main()
