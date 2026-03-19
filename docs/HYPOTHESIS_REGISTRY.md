# Hypothesis Registry

Pre-registered hypotheses for the vulnerability prioritization ML project.

> **Note:** These hypotheses were pre-registered retroactively on 2026-03-16,
> after experiments were complete. They are documented here for methodological
> transparency, not to claim prospective registration.

## Hypotheses

| ID  | Statement | Falsification Criterion | Metric | Resolution | Evidence | lock_commit |
|-----|-----------|------------------------|--------|------------|----------|-------------|
| H-1 | ML models trained on NVD+ExploitDB features outperform CVSS threshold baselines for exploit prediction | ML test AUC <= best CVSS threshold AUC (0.662) | AUC-ROC on temporal test set | **SUPPORTED** | LogReg AUC 0.903 vs CVSS best 0.662 (+24.1pp). 7 algorithms tested, all 7 exceed CVSS. [DEMONSTRATED: 5 seeds] | `PLACEHOLDER` |
| H-2 | Temporal train/test splits produce lower apparent performance than random splits due to ground-truth lag in ExploitDB labels | Temporal test exploit rate >= train exploit rate | Exploit rate comparison | **SUPPORTED** | Train exploit rate 10.5% vs test exploit rate 0.3% (35x drop). 2024+ CVEs too new for ExploitDB entries. F1 depressed across all models. | `PLACEHOLDER` |
| H-3 | EPSS percentile is the single strongest predictor of real-world exploitation | Another feature has higher mean absolute SHAP value than epss_percentile | SHAP rank on LogReg (scaled) | **SUPPORTED** | EPSS percentile mean SHAP 1.096, nearly 2x the next feature (has_exploit_ref 0.573). Ablation confirms: removing EPSS drops AUC by 15.5pp; EPSS alone achieves AUC 0.901. [DEMONSTRATED: 5 seeds] | `PLACEHOLDER` |
| H-4 | Hyperparameter tuning significantly improves tree-based model performance (>2pp AUC) | Tuned AUC <= default AUC + 0.02 | AUC delta (complexity sweep) | **SUPPORTED** | XGBoost max_depth=3 AUC 0.912 vs default max_depth=8 AUC 0.825 (+8.7pp). RF 500 trees AUC 0.877 vs 200 trees AUC 0.871. Zero variance across 5 seeds for XGBoost. [DEMONSTRATED: 5 seeds] | `PLACEHOLDER` |
| H-5 | Simple linear models (LogReg) outperform complex models (RF, XGBoost, MLP) at default hyperparameters | LogReg test AUC < best tree/ensemble test AUC (default HP) | Test AUC comparison | **SUPPORTED** | LogReg 0.903 > LightGBM 0.883 > RF 0.871 > XGBoost 0.825 > SVM 0.797 > MLP 0.762 > kNN 0.663. LogReg wins because regularized linear model cannot overfit; signal concentrates in few features. [DEMONSTRATED: 5 seeds] | `PLACEHOLDER` |
| H-6 | The model is robust to adversarial text manipulation because top features are defender-observable | Adversarial text attacks achieve >5% evasion rate | Evasion rate across 3 attack types | **SUPPORTED** | 0% evasion across synonym swap, field injection, and noise perturbation. Top features (EPSS, exploit refs, CVSS, vendor history) are not attacker-controllable. [SUGGESTED: single seed] | `PLACEHOLDER` |
| H-7 | Feature groups that are attacker-controllable (description text, keywords) provide less predictive signal than system-controlled features (EPSS, CVSS) | Attacker-controllable group ablation delta > system-controlled group delta | Leave-one-out AUC delta | **SUPPORTED** | EPSS removal: -15.5pp. CVSS removal: -2.9pp. Keywords removal: -2.9pp. Description removal: +2.4pp (harmful). System features dominate; text features are marginal or harmful. [DEMONSTRATED: 5 seeds] | `PLACEHOLDER` |

## Resolution Key

- **SUPPORTED**: The falsification criterion was NOT met; the hypothesis holds.
- **REFUTED**: The falsification criterion WAS met; the hypothesis is rejected.
- **INCONCLUSIVE**: Evidence is ambiguous or insufficient to determine resolution.

## Cross-References

- Expanded model results: `outputs/models/expanded_summary.json` (7 algorithms x 5 seeds)
- Sanity baselines: `outputs/baselines/sanity_*.json` (5 seeds)
- SHAP analysis: `outputs/explainability/feature_importance_seed42.csv`
- Ablation study: `outputs/diagnostics/ablation_summary.json`
- Complexity sweeps: `outputs/diagnostics/complexity_curves_seed*.json`
- Learning curves: `outputs/diagnostics/learning_curves_seed*.json`
- FINDINGS report: `FINDINGS.md`
