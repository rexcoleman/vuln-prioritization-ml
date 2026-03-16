# Hypothesis Registry

Pre-registered hypotheses for the vulnerability prioritization ML project.

> **Note:** These hypotheses were pre-registered retroactively on 2026-03-16,
> after experiments were complete. They are documented here for methodological
> transparency, not to claim prospective registration.

## Hypotheses

| ID  | Statement | Falsification Criterion | Metric | Resolution | Evidence |
|-----|-----------|------------------------|--------|------------|----------|
| H-1 | ML models outperform CVSS threshold baselines for exploit prediction | ML AUC <= CVSS-high threshold AUC | AUC on temporal test set | **SUPPORTED** | XGBoost AUC 0.739 vs CVSS 0.499 (+24pp) |
| H-2 | Temporal splits produce lower performance than random splits (due to ground-truth lag) | Temporal AUC >= random split AUC | AUC comparison | **SUPPORTED** | Temporal evaluation reveals label lag that random splits hide |
| H-3 | EPSS percentile is the strongest single predictor of exploitation | EPSS SHAP rank > #1 | SHAP importance rank | **SUPPORTED** | EPSS percentile: SHAP 1.096 (rank #1, scaled data) |
| H-4 | Hyperparameter tuning significantly improves XGBoost performance | Tuned AUC <= default AUC + 0.02 | AUC delta | **SUPPORTED** | XGBoost depth=3: AUC 0.912 vs default depth=8: AUC 0.825 (+8.7pp) |

## Resolution Key

- **SUPPORTED**: The falsification criterion was NOT met; the hypothesis holds.
- **REFUTED**: The falsification criterion WAS met; the hypothesis is rejected.
- **INCONCLUSIVE**: Evidence is ambiguous or insufficient to determine resolution.

## Cross-References

- Experimental results: `outputs/final/final_results.json`
- SHAP analysis: `outputs/explainability/`
- FINDINGS report: `FINDINGS.md`
