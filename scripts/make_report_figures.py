#!/usr/bin/env python3
"""Generate ALL report figures from raw output data (no hardcoded values).

Reads JSON outputs from diagnostics, models, baselines, and explainability
scripts and produces publication-ready matplotlib figures with mean +/- std
across seeds.

Figures produced:
  - figures/learning_curves.png      — train/val AUC vs training fraction
  - figures/complexity_curves.png    — train/val AUC vs HP value (3 subplots)
  - figures/model_comparison.png     — bar chart with error bars across seeds
  - figures/shap_importance.png      — top-20 SHAP features (mean across seeds)

Usage:
    python scripts/make_report_figures.py --project-dir .
    python scripts/make_report_figures.py --project-dir . --output-dir figures/
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Consistent styling
COLORS = {
    "rf": "#2196F3",
    "xgboost": "#4CAF50",
    "logreg": "#FF9800",
    "random_forest": "#2196F3",
    "logistic_regression": "#FF9800",
}
MODEL_LABELS = {
    "rf": "Random Forest",
    "xgboost": "XGBoost",
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}


def set_style():
    """Set clean matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })


def load_json_files(directory, pattern):
    """Load all JSON files matching pattern, return list of parsed dicts."""
    files = sorted(Path(directory).glob(pattern))
    data = []
    for f in files:
        with open(f) as fh:
            data.append(json.load(fh))
    return data


def make_learning_curves(project_dir, output_dir):
    """Generate learning_curves.png from diagnostics data."""
    diag_dir = Path(project_dir) / "outputs" / "diagnostics"
    files = load_json_files(diag_dir, "learning_curves_seed*.json")
    if not files:
        print("  SKIP: No learning curve data found.")
        return

    print(f"  Found {len(files)} seed files for learning curves.")

    # Collect per-model, per-fraction data across seeds
    model_names = list(files[0]["results"][0]["models"].keys())
    fractions = files[0]["fractions"]

    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4.5),
                             sharey=True)
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        train_aucs = {f: [] for f in fractions}
        val_aucs = {f: [] for f in fractions}

        for seed_data in files:
            for entry in seed_data["results"]:
                frac = entry["fraction"]
                m = entry["models"].get(model_name, {})
                if m.get("train_auc") is not None:
                    train_aucs[frac].append(m["train_auc"])
                if m.get("val_auc") is not None:
                    val_aucs[frac].append(m["val_auc"])

        fracs_arr = np.array(fractions)
        train_mean = np.array([np.mean(train_aucs[f]) if train_aucs[f] else np.nan for f in fractions])
        train_std = np.array([np.std(train_aucs[f]) if train_aucs[f] else 0 for f in fractions])
        val_mean = np.array([np.mean(val_aucs[f]) if val_aucs[f] else np.nan for f in fractions])
        val_std = np.array([np.std(val_aucs[f]) if val_aucs[f] else 0 for f in fractions])

        color = COLORS.get(model_name, "#666666")
        ax.plot(fracs_arr, train_mean, "o-", color=color, label="Train", alpha=0.8)
        ax.fill_between(fracs_arr, train_mean - train_std, train_mean + train_std,
                         color=color, alpha=0.15)
        ax.plot(fracs_arr, val_mean, "s--", color=color, label="Validation",
                alpha=0.8, markerfacecolor="white", markeredgecolor=color)
        ax.fill_between(fracs_arr, val_mean - val_std, val_mean + val_std,
                         color=color, alpha=0.1)

        ax.set_title(MODEL_LABELS.get(model_name, model_name))
        ax.set_xlabel("Training Fraction")
        if ax == axes[0]:
            ax.set_ylabel("AUC-ROC")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(0.05, 1.05)

    fig.suptitle("Learning Curves: AUC-ROC vs Training Data Size", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = Path(output_dir) / "learning_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_complexity_curves(project_dir, output_dir):
    """Generate complexity_curves.png from diagnostics data."""
    diag_dir = Path(project_dir) / "outputs" / "diagnostics"
    files = load_json_files(diag_dir, "complexity_curves_seed*.json")
    if not files:
        print("  SKIP: No complexity curve data found.")
        return

    print(f"  Found {len(files)} seed files for complexity curves.")

    model_names = list(files[0]["sweeps"].keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4.5))
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        sweep_info = files[0]["sweeps"][model_name]
        param_name = sweep_info["param_name"]
        param_values = [r["param_value"] for r in sweep_info["results"]]

        train_aucs = {v: [] for v in param_values}
        val_aucs = {v: [] for v in param_values}

        for seed_data in files:
            for entry in seed_data["sweeps"][model_name]["results"]:
                v = entry["param_value"]
                if entry.get("train_auc") is not None:
                    train_aucs[v].append(entry["train_auc"])
                if entry.get("val_auc") is not None:
                    val_aucs[v].append(entry["val_auc"])

        x = np.arange(len(param_values))
        train_mean = np.array([np.mean(train_aucs[v]) if train_aucs[v] else np.nan for v in param_values])
        train_std = np.array([np.std(train_aucs[v]) if train_aucs[v] else 0 for v in param_values])
        val_mean = np.array([np.mean(val_aucs[v]) if val_aucs[v] else np.nan for v in param_values])
        val_std = np.array([np.std(val_aucs[v]) if val_aucs[v] else 0 for v in param_values])

        color = COLORS.get(model_name, "#666666")
        ax.plot(x, train_mean, "o-", color=color, label="Train", alpha=0.8)
        ax.fill_between(x, train_mean - train_std, train_mean + train_std,
                         color=color, alpha=0.15)
        ax.plot(x, val_mean, "s--", color=color, label="Validation",
                alpha=0.8, markerfacecolor="white", markeredgecolor=color)
        ax.fill_between(x, val_mean - val_std, val_mean + val_std,
                         color=color, alpha=0.1)

        ax.set_title(f"{MODEL_LABELS.get(model_name, model_name)}")
        ax.set_xlabel(param_name)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in param_values], rotation=45)
        if ax == axes[0]:
            ax.set_ylabel("AUC-ROC")
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Complexity Curves: AUC-ROC vs Hyperparameter Value", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = Path(output_dir) / "complexity_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_model_comparison(project_dir, output_dir):
    """Generate model_comparison.png bar chart with error bars across seeds."""
    model_dir = Path(project_dir) / "outputs" / "models"
    files = load_json_files(model_dir, "models_seed*.json")
    if not files:
        print("  SKIP: No model result files found.")
        return

    print(f"  Found {len(files)} seed files for model comparison.")

    # Collect metrics per model across seeds
    all_models = set()
    for f in files:
        all_models.update(f["models"].keys())
    all_models = sorted(all_models)

    metrics_to_plot = ["auc_roc", "f1", "auc_pr"]
    metric_labels = {"auc_roc": "AUC-ROC", "f1": "F1 Score", "auc_pr": "AUC-PR"}

    fig, axes = plt.subplots(1, len(metrics_to_plot),
                             figsize=(4.5 * len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    x = np.arange(len(all_models))
    width = 0.6

    for ax, metric in zip(axes, metrics_to_plot):
        means = []
        stds = []
        colors = []
        for model_name in all_models:
            vals = []
            for f in files:
                v = f["models"].get(model_name, {}).get(metric)
                if v is not None:
                    vals.append(v)
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if len(vals) > 1 else 0)
            colors.append(COLORS.get(model_name, "#666666"))

        bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                      color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

        # Value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                        f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(metric_labels.get(metric, metric))
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in all_models],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, min(1.0, max(means) * 1.3 + 0.05) if means else 1.0)

    fig.suptitle(f"Model Comparison ({len(files)} seeds)", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = Path(output_dir) / "model_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_shap_importance(project_dir, output_dir):
    """Generate shap_importance.png from explainability data (mean across seeds)."""
    expl_dir = Path(project_dir) / "outputs" / "explainability"
    files = sorted(expl_dir.glob("feature_importance_seed*.csv"))
    if not files:
        print("  SKIP: No SHAP importance CSVs found.")
        return

    print(f"  Found {len(files)} seed files for SHAP importance.")

    import pandas as pd

    # Load all seed importance files and average
    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        all_dfs.append(df)

    # Merge on feature name, compute mean SHAP
    merged = all_dfs[0][["feature"]].copy()
    for i, df in enumerate(all_dfs):
        merged = merged.merge(
            df[["feature", "mean_abs_shap"]].rename(
                columns={"mean_abs_shap": f"shap_{i}"}
            ),
            on="feature", how="outer",
        )

    shap_cols = [c for c in merged.columns if c.startswith("shap_")]
    merged["mean_shap"] = merged[shap_cols].mean(axis=1)
    merged["std_shap"] = merged[shap_cols].std(axis=1).fillna(0)
    merged = merged.sort_values("mean_shap", ascending=False)

    top_n = 20
    top = merged.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, top["mean_shap"].values[::-1],
            xerr=top["std_shap"].values[::-1],
            capsize=3, color="#5C6BC0", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"].values[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Features by SHAP Importance ({len(files)} seeds)")
    plt.tight_layout()

    out_path = Path(output_dir) / "shap_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate report figures from raw data")
    parser.add_argument("--project-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="figures/")
    args = parser.parse_args()

    set_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Generating Report Figures ===\n")

    print("[1/4] Learning curves...")
    make_learning_curves(args.project_dir, output_dir)

    print("\n[2/4] Complexity curves...")
    make_complexity_curves(args.project_dir, output_dir)

    print("\n[3/4] Model comparison...")
    make_model_comparison(args.project_dir, output_dir)

    print("\n[4/4] SHAP importance...")
    make_shap_importance(args.project_dir, output_dir)

    print("\n=== All figures complete ===")


if __name__ == "__main__":
    main()
