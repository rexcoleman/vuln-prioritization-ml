#!/usr/bin/env python3
"""Bootstrap CI + McNemar + DeLong-style tests for model comparisons.

Computes:
  1. Bootstrap 95% CI on AUC for each model x seed
  2. McNemar test (chi-squared) for each model pair
  3. Bootstrap AUC difference test (DeLong alternative) for each model pair

Usage:
    python scripts/run_statistical_tests.py --data-dir data/
    python scripts/run_statistical_tests.py --data-dir data/ --seeds 42,123
    python scripts/run_statistical_tests.py --data-dir data/ --sample-frac 0.01  # smoke test
    python scripts/run_statistical_tests.py --data-dir data/ --n-bootstrap 2000
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
# Model pairs to compare (focus on best model vs each alternative)
COMPARISON_PAIRS = [
    ("xgboost", "rf"),
    ("xgboost", "logreg"),
    ("xgboost", "svm"),
]
if HAS_LGBM:
    COMPARISON_PAIRS.append(("xgboost", "lgbm"))


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
        "svm": SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=seed,
        ),
    }
    if HAS_LGBM:
        models["lgbm"] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    return models


def get_predictions(X_train, y_train, X_test, y_test, seed):
    """Train all models and return predictions dict."""
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    models = build_models(seed, scale_pos_weight)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    predictions = {}
    for name, model in models.items():
        print(f"    Training {name}...", end=" ", flush=True)
        if name in ("logreg", "svm"):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        print(f"AUC={auc:.4f}")
        predictions[name] = {
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    return predictions


# ── Bootstrap CI ─────────────────────────────────────────────────────────────


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000, seed=42, alpha=0.05):
    """Compute bootstrap confidence interval for AUC-ROC.

    Returns (point_estimate, lower, upper, bootstrap_aucs).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point_auc = float(roc_auc_score(y_true, y_prob))

    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_prob[idx]
        # Need both classes in bootstrap sample
        if len(np.unique(y_t)) < 2:
            continue
        aucs.append(roc_auc_score(y_t, y_p))

    aucs = np.array(aucs)
    lower = float(np.percentile(aucs, 100 * alpha / 2))
    upper = float(np.percentile(aucs, 100 * (1 - alpha / 2)))

    return point_auc, lower, upper, aucs


# ── McNemar Test ─────────────────────────────────────────────────────────────


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar test comparing two classifiers' binary predictions.

    Constructs 2x2 contingency table:
        - b: A correct, B incorrect
        - c: A incorrect, B correct
    Tests H0: b == c (models make the same errors).

    Returns (chi2, p_value, b, c).
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b = A right, B wrong; c = A wrong, B right
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    # McNemar with continuity correction
    if b + c == 0:
        return 0.0, 1.0, b, c

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(stats.chi2.sf(chi2, df=1))
    return float(chi2), p_value, b, c


# ── Bootstrap AUC Difference (DeLong alternative) ───────────────────────────


def bootstrap_auc_difference(y_true, y_prob_a, y_prob_b,
                              n_bootstrap=1000, seed=42):
    """Bootstrap test for AUC difference between two models.

    If 95% CI of (AUC_A - AUC_B) excludes 0, models are significantly different.

    Returns (mean_diff, lower, upper, p_value, diffs).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_pa = y_prob_a[idx]
        y_pb = y_prob_b[idx]
        if len(np.unique(y_t)) < 2:
            continue
        auc_a = roc_auc_score(y_t, y_pa)
        auc_b = roc_auc_score(y_t, y_pb)
        diffs.append(auc_a - auc_b)

    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    lower = float(np.percentile(diffs, 2.5))
    upper = float(np.percentile(diffs, 97.5))

    # Two-sided p-value: proportion of bootstrap diffs on opposite side of 0
    if mean_diff >= 0:
        p_value = float(2 * np.mean(diffs <= 0))
    else:
        p_value = float(2 * np.mean(diffs >= 0))
    p_value = min(p_value, 1.0)

    return mean_diff, lower, upper, p_value, diffs


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap CI + McNemar + DeLong tests for model comparisons"
    )
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--seeds", type=str,
                        default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
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
    print(f"Seeds: {seeds}")
    print(f"Bootstrap samples: {args.n_bootstrap}\n")

    all_results = {
        "date": datetime.now().isoformat(),
        "sample_frac": args.sample_frac,
        "n_bootstrap": args.n_bootstrap,
        "seeds": seeds,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "num_features": len(feature_cols),
        "bootstrap_ci": {},
        "mcnemar": {},
        "auc_difference": {},
    }

    for seed in seeds:
        print(f"=== Seed {seed} ===")
        print("  Training models to get predictions...")
        preds = get_predictions(X_train, y_train, X_test, y_test, seed)
        model_names = sorted(preds.keys())

        # ── Bootstrap CI per model ───────────────────────────────────────
        seed_key = str(seed)
        all_results["bootstrap_ci"][seed_key] = {}

        print(f"  Computing bootstrap CIs ({args.n_bootstrap} samples)...")
        for name in model_names:
            y_prob = preds[name]["y_prob"]
            point, lower, upper, _ = bootstrap_auc_ci(
                y_test, y_prob, n_bootstrap=args.n_bootstrap, seed=seed
            )
            all_results["bootstrap_ci"][seed_key][name] = {
                "auc_point": point,
                "ci_lower": lower,
                "ci_upper": upper,
            }
            print(f"    {name:12s}: AUC={point:.4f}  95% CI=[{lower:.4f}, {upper:.4f}]")

        # ── McNemar + AUC difference for model pairs ─────────────────────
        all_results["mcnemar"][seed_key] = {}
        all_results["auc_difference"][seed_key] = {}

        print("  Pairwise comparisons:")
        for name_a, name_b in COMPARISON_PAIRS:
            if name_a not in preds or name_b not in preds:
                continue

            pair_key = f"{name_a}_vs_{name_b}"

            # McNemar
            chi2, p_mc, b, c = mcnemar_test(
                y_test, preds[name_a]["y_pred"], preds[name_b]["y_pred"]
            )
            all_results["mcnemar"][seed_key][pair_key] = {
                "chi2": chi2,
                "p_value": p_mc,
                "a_right_b_wrong": b,
                "a_wrong_b_right": c,
                "significant_005": p_mc < 0.05,
            }

            # Bootstrap AUC difference
            mean_d, lower_d, upper_d, p_ad, _ = bootstrap_auc_difference(
                y_test,
                preds[name_a]["y_prob"],
                preds[name_b]["y_prob"],
                n_bootstrap=args.n_bootstrap,
                seed=seed,
            )
            sig_str = "***" if p_ad < 0.001 else "**" if p_ad < 0.01 else "*" if p_ad < 0.05 else "ns"
            excludes_zero = (lower_d > 0) or (upper_d < 0)
            all_results["auc_difference"][seed_key][pair_key] = {
                "mean_diff": mean_d,
                "ci_lower": lower_d,
                "ci_upper": upper_d,
                "p_value": p_ad,
                "excludes_zero": excludes_zero,
                "significant_005": p_ad < 0.05,
            }

            print(f"    {pair_key:25s}  McNemar chi2={chi2:8.2f} p={p_mc:.4f}  "
                  f"AUC diff={mean_d:+.4f} [{lower_d:+.4f},{upper_d:+.4f}] {sig_str}")

        print()

    # ── Aggregate across seeds ───────────────────────────────────────────
    print("=" * 70)
    print("AGGREGATE RESULTS (mean across seeds)")
    print("=" * 70)

    # Aggregate bootstrap CI
    print("\nBootstrap 95% CI on AUC:")
    print(f"  {'Model':12s}  {'Mean AUC':>10s}  {'Mean CI Low':>12s}  {'Mean CI High':>12s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*12}")

    agg_ci = {}
    # Get all model names across seeds
    all_model_names = set()
    for seed_key in all_results["bootstrap_ci"]:
        all_model_names.update(all_results["bootstrap_ci"][seed_key].keys())

    for name in sorted(all_model_names):
        aucs, lows, highs = [], [], []
        for seed_key in all_results["bootstrap_ci"]:
            if name in all_results["bootstrap_ci"][seed_key]:
                entry = all_results["bootstrap_ci"][seed_key][name]
                aucs.append(entry["auc_point"])
                lows.append(entry["ci_lower"])
                highs.append(entry["ci_upper"])
        if aucs:
            mean_auc = np.mean(aucs)
            mean_low = np.mean(lows)
            mean_high = np.mean(highs)
            agg_ci[name] = {
                "mean_auc": float(mean_auc),
                "mean_ci_lower": float(mean_low),
                "mean_ci_upper": float(mean_high),
            }
            print(f"  {name:12s}  {mean_auc:10.4f}  {mean_low:12.4f}  {mean_high:12.4f}")

    # Aggregate pairwise tests
    print(f"\nPairwise AUC Differences:")
    print(f"  {'Pair':25s}  {'Mean Diff':>10s}  {'CI Low':>8s}  {'CI High':>8s}  "
          f"{'Mean p':>8s}  {'Sig/Total':>10s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

    agg_diff = {}
    all_pair_keys = set()
    for seed_key in all_results["auc_difference"]:
        all_pair_keys.update(all_results["auc_difference"][seed_key].keys())

    for pair_key in sorted(all_pair_keys):
        diffs, p_vals, sig_count = [], [], 0
        for seed_key in all_results["auc_difference"]:
            if pair_key in all_results["auc_difference"][seed_key]:
                entry = all_results["auc_difference"][seed_key][pair_key]
                diffs.append(entry["mean_diff"])
                p_vals.append(entry["p_value"])
                if entry["significant_005"]:
                    sig_count += 1
        if diffs:
            mean_d = np.mean(diffs)
            low_d = np.percentile(diffs, 2.5) if len(diffs) > 1 else diffs[0]
            high_d = np.percentile(diffs, 97.5) if len(diffs) > 1 else diffs[0]
            mean_p = np.mean(p_vals)
            agg_diff[pair_key] = {
                "mean_diff": float(mean_d),
                "ci_lower": float(low_d),
                "ci_upper": float(high_d),
                "mean_p_value": float(mean_p),
                "significant_count": sig_count,
                "total_seeds": len(diffs),
            }
            print(f"  {pair_key:25s}  {mean_d:+10.4f}  {low_d:+8.4f}  {high_d:+8.4f}  "
                  f"{mean_p:8.4f}  {sig_count}/{len(diffs)}")

    all_results["aggregate"] = {
        "bootstrap_ci": agg_ci,
        "auc_difference": agg_diff,
    }

    # Save
    out_file = output_dir / "statistical_tests.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
