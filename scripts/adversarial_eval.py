#!/usr/bin/env python3
"""Adversarial evaluation of the vulnerability prioritization model.

Tests whether an attacker can craft CVE descriptions that evade the model.
Applies feature controllability analysis (from FP-01 methodology):
  - Attacker-controllable: CVE description text, reference links
  - Defender-observable only: CVSS score, CWE, vendor, temporal features, EPSS
  - Environment: publication date, NVD processing metadata

Attack methods:
  - synonym_swap: Replace security terms with benign synonyms
  - field_injection: Add misleading terms to description
  - noise_perturbation: Random character perturbations

Usage:
    python scripts/adversarial_eval.py --seed 42
    python scripts/adversarial_eval.py --seed 42 --sample-frac 0.01  # smoke test
"""
import argparse
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

OUTPUT_DIR = Path("outputs/adversarial")
FIGURES_DIR = Path("outputs/figures")
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("outputs/models")


# --- Feature Controllability Matrix (from FP-01 methodology, ADR-0006) ---
ATTACKER_CONTROLLABLE = [
    # Text features (attacker writes the advisory / CVE description)
    "desc_length", "desc_word_count",
    # Keyword features (attacker controls what words appear)
    "kw_remote_code_execution", "kw_sql_injection", "kw_buffer_overflow",
    "kw_xss", "kw_privilege_escalation", "kw_authentication_bypass",
    "kw_denial_of_service", "kw_information_disclosure", "kw_arbitrary_code",
    "kw_allows_attackers", "kw_crafted",
    # Reference features (attacker controls what links are added)
    "ref_count", "has_exploit_ref",
]

DEFENDER_OBSERVABLE_ONLY = [
    # CVSS (assigned by NVD analysts, not attacker)
    "cvss_score", "has_cvss_v3",
    # CWE (assigned by NVD)
    "has_cwe", "cwe_count",
    # EPSS (computed by First.org model)
    "epss",
    # Temporal (objective fact)
    "pub_year", "pub_month", "pub_dayofweek", "cve_age_days",
    # Vendor metadata (from CPE, assigned by NVD)
    "vendor_cve_count",
    # Patch status (observable by defender)
    "has_patch_ref",
]


# --- Attack Methods ---

SYNONYM_MAP = {
    "remote code execution": "functionality issue",
    "buffer overflow": "memory handling concern",
    "sql injection": "input processing behavior",
    "cross-site scripting": "content display variation",
    "privilege escalation": "access level change",
    "denial of service": "availability impact",
    "authentication bypass": "login flow behavior",
    "arbitrary code": "unintended execution",
    "allows attackers": "enables users",
    "crafted request": "specific input",
    "crafted packet": "specific network data",
    "crafted file": "specific document",
}


def synonym_swap_attack(descriptions, seed=42):
    """Replace security terms with benign synonyms in CVE descriptions."""
    rng = np.random.RandomState(seed)
    perturbed = []
    changes = 0

    for desc in descriptions:
        new_desc = str(desc) if pd.notna(desc) else ""
        for attack_term, benign_term in SYNONYM_MAP.items():
            if attack_term.lower() in new_desc.lower():
                new_desc = re.sub(
                    re.escape(attack_term), benign_term, new_desc, flags=re.IGNORECASE
                )
                changes += 1
        perturbed.append(new_desc)

    print(f"  Synonym swap: {changes} substitutions across {len(descriptions)} descriptions")
    return perturbed


def field_injection_attack(descriptions, seed=42):
    """Inject misleading benign terms into CVE descriptions."""
    rng = np.random.RandomState(seed)
    benign_injections = [
        " This is a minor documentation update.",
        " No security impact expected.",
        " Routine maintenance change.",
        " Configuration improvement only.",
        " Performance optimization update.",
    ]

    perturbed = []
    for desc in descriptions:
        new_desc = str(desc) if pd.notna(desc) else ""
        injection = rng.choice(benign_injections)
        new_desc = new_desc + injection
        perturbed.append(new_desc)

    print(f"  Field injection: {len(descriptions)} descriptions modified")
    return perturbed


def noise_perturbation_attack(descriptions, noise_rate=0.05, seed=42):
    """Random character perturbations in CVE descriptions."""
    rng = np.random.RandomState(seed)
    perturbed = []
    total_changes = 0

    for desc in descriptions:
        chars = list(str(desc) if pd.notna(desc) else "")
        n_changes = max(1, int(len(chars) * noise_rate))
        for _ in range(n_changes):
            if chars:
                idx = rng.randint(0, len(chars))
                chars[idx] = chr(rng.randint(97, 122))  # random lowercase letter
                total_changes += 1
        perturbed.append("".join(chars))

    print(f"  Noise perturbation: {total_changes} character changes ({noise_rate*100:.0f}% rate)")
    return perturbed


def rebuild_features_from_perturbed_descriptions(test_df, perturbed_descs, feature_cols):
    """Rebuild feature matrix with perturbed descriptions.

    Only attacker-controllable features change. Defender-observable features stay the same.
    This is the feature controllability principle from FP-01.
    """
    perturbed_df = test_df.copy()
    perturbed_df["description"] = perturbed_descs

    # Recompute text-derived features
    perturbed_df["desc_length"] = perturbed_df["description"].fillna("").str.len()
    perturbed_df["desc_word_count"] = perturbed_df["description"].fillna("").str.split().str.len()

    # Recompute keyword features (inlined to avoid circular import)
    keywords = {
        "remote_code_execution": r"remote\s+code\s+execution|rce",
        "sql_injection": r"sql\s+injection|sqli",
        "buffer_overflow": r"buffer\s+overflow|heap\s+overflow|stack\s+overflow",
        "xss": r"cross.site\s+scripting|xss",
        "privilege_escalation": r"privilege\s+escalation|privesc",
        "authentication_bypass": r"authentication\s+bypass|auth\s+bypass",
        "denial_of_service": r"denial\s+of\s+service|dos\b",
        "information_disclosure": r"information\s+disclosure|info\s+leak",
        "arbitrary_code": r"arbitrary\s+code",
        "allows_attackers": r"allows?\s+(remote\s+)?attackers?",
        "crafted": r"crafted\s+(request|packet|input|file|url)",
    }
    for name, pattern in keywords.items():
        col = f"kw_{name}"
        if col in feature_cols:
            perturbed_df[col] = perturbed_df["description"].fillna("").str.contains(
                pattern, case=False, regex=True
            ).astype(int)

    X_perturbed = perturbed_df[feature_cols].fillna(0).values
    return X_perturbed


def evaluate_attack(model, X_original, X_perturbed, y_true, attack_name):
    """Evaluate attack effectiveness."""
    # Original predictions
    y_pred_orig = model.predict(X_original)
    y_prob_orig = model.predict_proba(X_original)[:, 1]

    # Perturbed predictions
    y_pred_pert = model.predict(X_perturbed)
    y_prob_pert = model.predict_proba(X_perturbed)[:, 1]

    # Evasion: originally correctly classified as exploited, now misclassified
    correctly_detected = (y_true == 1) & (y_pred_orig == 1)
    evaded = correctly_detected & (y_pred_pert == 0)
    evasion_rate = evaded.sum() / max(correctly_detected.sum(), 1)

    # Overall metrics change
    f1_orig = f1_score(y_true, y_pred_orig, zero_division=0)
    f1_pert = f1_score(y_true, y_pred_pert, zero_division=0)

    results = {
        "attack": attack_name,
        "original_f1": float(f1_orig),
        "perturbed_f1": float(f1_pert),
        "f1_drop": float(f1_orig - f1_pert),
        "evasion_rate": float(evasion_rate),
        "correctly_detected_count": int(correctly_detected.sum()),
        "evaded_count": int(evaded.sum()),
        "mean_prob_change": float(np.abs(y_prob_orig - y_prob_pert).mean()),
    }

    print(f"  {attack_name}: evasion={evasion_rate*100:.1f}%, F1 drop={f1_orig-f1_pert:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Adversarial evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model_file = MODEL_DIR / f"best_model_seed{args.seed}.pkl"
    with open(model_file, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    test = pd.read_parquet(DATA_DIR / "test.parquet")
    if args.sample_frac < 1.0:
        test = test.sample(frac=args.sample_frac, random_state=args.seed)

    X_test = test[feature_cols].fillna(0).values
    y_test = test["exploited"].values
    descriptions = test["description"].tolist()

    print(f"Test set: {len(test):,} CVEs ({y_test.mean()*100:.1f}% exploited)")
    print(f"\n=== Feature Controllability Matrix ===")
    print(f"Attacker-controllable features: {len(ATTACKER_CONTROLLABLE)}")
    print(f"Defender-observable-only features: {len(DEFENDER_OBSERVABLE_ONLY)}")

    # Run attacks
    all_results = []

    print(f"\n=== Attack: Synonym Swap ===")
    pert_descs = synonym_swap_attack(descriptions, seed=args.seed)
    X_pert = rebuild_features_from_perturbed_descriptions(test, pert_descs, feature_cols)
    all_results.append(evaluate_attack(model, X_test, X_pert, y_test, "synonym_swap"))

    print(f"\n=== Attack: Field Injection ===")
    pert_descs = field_injection_attack(descriptions, seed=args.seed)
    X_pert = rebuild_features_from_perturbed_descriptions(test, pert_descs, feature_cols)
    all_results.append(evaluate_attack(model, X_test, X_pert, y_test, "field_injection"))

    print(f"\n=== Attack: Noise Perturbation ===")
    pert_descs = noise_perturbation_attack(descriptions, noise_rate=0.05, seed=args.seed)
    X_pert = rebuild_features_from_perturbed_descriptions(test, pert_descs, feature_cols)
    all_results.append(evaluate_attack(model, X_test, X_pert, y_test, "noise_perturbation"))

    # Defense analysis: what if we only use defender-observable features?
    print(f"\n=== Defense: Defender-Observable-Only Model ===")
    defender_cols = [c for c in feature_cols if c in DEFENDER_OBSERVABLE_ONLY or c.startswith("cwe_")]
    print(f"  Defender-only features: {len(defender_cols)}")
    # Note: this requires retraining — log as future work

    # Save
    summary = {
        "seed": args.seed,
        "date": datetime.now().isoformat(),
        "test_size": len(test),
        "controllable_features": ATTACKER_CONTROLLABLE,
        "observable_features": DEFENDER_OBSERVABLE_ONLY,
        "attacks": all_results,
    }

    out_file = OUTPUT_DIR / f"adversarial_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
