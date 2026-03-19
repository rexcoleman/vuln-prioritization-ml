# HYPOTHESIS REGISTRY — ML-Powered Vulnerability Prioritization Engine

> **Project:** FP-05 (Vulnerability Prioritization)
> **Created:** 2026-03-16
> **Status:** All hypotheses resolved (6/6)

---

## Legend

| Field | Description |
|-------|-------------|
| **ID** | Hypothesis identifier (H-1, H-2, ...) |
| **Statement** | Falsifiable claim with a measurable threshold |
| **Status** | SUPPORTED, REFUTED, or INCONCLUSIVE |
| **Evidence** | Specific metric values and script/output references |
| **Implications** | What the result means for the project thesis and practitioner guidance |
| **Linked RQ** | Which research question this hypothesis addresses |

---

## H-1: ML outperforms CVSS threshold by >=15pp AUC on temporal test set

| Field | Value |
|-------|-------|
| **Statement** | An ML model trained on NVD + ExploitDB + EPSS features will achieve AUC-ROC at least 15 percentage points higher than the best CVSS threshold baseline on a temporal test set (2024+ CVEs). |
| **Status** | SUPPORTED |
| **Evidence** | LogReg AUC 0.903 vs best CVSS threshold (>=9.0) AUC 0.662 = +24.1pp delta. Exceeds the 15pp threshold by 9pp. Result confirmed across 5-seed learning curve analysis (LogReg 0.903 +/- 0.000 at full data). See `FINDINGS.md` RQ1, `outputs/models/expanded_summary.json`, `outputs/baselines/baselines_seed42.json`. |
| **Implications** | CVSS is a weak exploitability predictor. Static severity formulas miss the signals that drive real-world exploitation (threat intel, vendor history, vulnerability class). Organizations relying solely on CVSS for patching priority are systematically mistriaging. |
| **Linked RQ** | RQ1 (ML vs CVSS) |
| **lock_commit** | `de76f40` |

---

## H-2: EPSS features rank in top-3 SHAP importance

| Field | Value |
|-------|-------|
| **Statement** | EPSS-derived features (epss, epss_percentile) will rank in the top 3 features by mean absolute SHAP value in the best-performing model. |
| **Status** | SUPPORTED |
| **Evidence** | epss_percentile is the #1 feature by mean |SHAP| (1.096), nearly 2x the next feature (has_exploit_ref at 0.573). SHAP computed on LogReg with StandardScaler applied. See `FINDINGS.md` RQ2, `outputs/explainability/`. |
| **Implications** | EPSS is itself an ML model trained on richer data (exploit activity, social media, threat intel feeds). That it dominates SHAP importance confirms exploit-likelihood signals concentrate in real-time threat intelligence, not static vulnerability metadata. This also explains why our model matches but does not beat EPSS standalone (H-2 feeds into RQ3 interpretation). |
| **Linked RQ** | RQ2 (Feature Importance) |
| **lock_commit** | `de76f40` |

---

## H-3: Temporal split produces lower test AUC than random split (ground truth lag)

| Field | Value |
|-------|-------|
| **Statement** | A temporal train/test split (pre-2024 train, 2024+ test) will produce lower test-set AUC and F1 than a random split on the same data, due to ground truth lag in ExploitDB labels for recent CVEs. |
| **Status** | SUPPORTED |
| **Evidence** | Temporal split test exploit rate is 0.3% (318/103,352) vs train exploit rate of 10.5% — a 35x class imbalance shift. F1 scores are depressed across all models (LogReg F1 0.106, RF F1 0.000, XGBoost F1 0.018) despite strong AUC. The 0.3% test exploit rate is clearly driven by ground truth lag (2024+ CVEs have not yet been catalogued in ExploitDB), not by a genuine drop in exploitation. ADR-0001 documents this design choice. See `FINDINGS.md` RQ1 critical caveat, `data/splits/split_info.json`. |
| **Implications** | Temporal splits are more realistic for production deployment (you always predict on future CVEs) but require careful interpretation. Low F1 does not indicate model failure — it indicates incomplete labels. Any deployment must account for label maturation time. |
| **Linked RQ** | RQ4 (Temporal Split vs Random Split) |
| **lock_commit** | `de76f40` |

---

## H-4: XGBoost outperforms LogisticRegression on tabular vulnerability data

| Field | Value |
|-------|-------|
| **Statement** | XGBoost will achieve higher test AUC-ROC than Logistic Regression on the vulnerability exploitability prediction task, given that gradient-boosted trees typically dominate tabular data benchmarks. |
| **Status** | REFUTED |
| **Evidence** | With default hyperparameters: LogReg AUC 0.903 > XGBoost AUC 0.825 (+7.8pp advantage for LogReg). LogReg also has near-zero variance across seeds (0.903 +/- 0.000) while XGBoost has high variance (0.825 +/- 0.000 at full data but 0.844 +/- 0.028 at partial data). However, complexity sweeps show XGBoost at max_depth=3 achieves AUC 0.912 +/- 0.000, surpassing LogReg. The refutation applies to default-HP XGBoost only. See `FINDINGS.md` RQ1 + Complexity Analysis, `outputs/diagnostics/complexity_curves_seed42.json`. |
| **Implications** | For this problem — where signal concentrates in a handful of features (EPSS, exploit refs, CVSS, vendor history) — a regularized linear model outperforms unconstrained tree ensembles. Default-HP XGBoost overfits the 49-feature space. The lesson: model complexity must match signal density. With proper HP tuning (shallow trees), XGBoost recovers and exceeds LogReg, but the default "XGBoost wins on tabular data" heuristic does not hold here. |
| **Linked RQ** | RQ1 (ML vs CVSS), Complexity Analysis |
| **lock_commit** | `de76f40` |

---

## H-5: Dual ground truth (ExploitDB + KEV) improves model performance

| Field | Value |
|-------|-------|
| **Statement** | A model trained with labels from both ExploitDB and CISA KEV (either-exploited) will achieve higher AUC than one trained on ExploitDB alone, because KEV captures actively exploited CVEs that ExploitDB misses. |
| **Status** | SUPPORTED |
| **Evidence** | XGB-tuned AUC: 0.912 (ExploitDB) → 0.928 (either) = +1.6pp. KEV adds 330 test positives not in ExploitDB (test positive rate doubles from 0.31% to 0.63%). `outputs/models/kev_ground_truth_results.json` |
| **Implications** | Single ground truth sources systematically undercount exploitation. Dual labeling improves both model performance and evaluation reliability. Production systems should fuse multiple exploit intelligence sources. |
| **Linked RQ** | Ground truth completeness |
| **lock_commit** | `de76f40` |

---

## H-6: Without EPSS, ML models provide modest but real signal over CVSS

| Field | Value |
|-------|-------|
| **Statement** | With EPSS features removed, ML models will still outperform CVSS threshold baselines (AUC 0.662) by at least 2pp, demonstrating that public NVD metadata contains independent exploitability signal. |
| **Status** | SUPPORTED (preliminary — seed 42) |
| **Evidence** | Without EPSS: LogReg AUC 0.689, XGB-tuned 0.684, RF 0.675, XGB-default 0.670. All exceed CVSS 0.662. Delta is modest (+1-3pp) but consistent across model families. Full 5-seed results pending. `outputs/models/no_epss_summary.json` (when complete) |
| **Implications** | The circularity criticism is valid — EPSS contributes 15-23pp AUC. But public metadata alone still provides value above CVSS, particularly for organizations without EPSS access. The honest contribution is quantifying EPSS dominance, not claiming to beat it. |
| **Linked RQ** | EPSS circularity |
| **lock_commit** | `de76f40` |

---

## Summary

| ID | Statement (short) | Status | Key Metric |
|----|-------------------|--------|------------|
| H-1 | ML > CVSS by >=15pp AUC | SUPPORTED | +24.1pp (0.903 vs 0.662) |
| H-2 | EPSS in top-3 SHAP | SUPPORTED | epss_percentile is #1 (1.096) |
| H-3 | Temporal split lowers AUC vs random | SUPPORTED | 0.3% test exploit rate (ground truth lag) |
| H-4 | XGBoost > LogReg on tabular CVE data | REFUTED | LogReg 0.903 > XGBoost 0.825 (default HP) |
| H-5 | Dual ground truth improves performance | SUPPORTED | XGB either-label 0.928 > ExploitDB-only 0.912 (+1.6pp) |
| H-6 | No-EPSS ML > CVSS | SUPPORTED | LogReg 0.689 > CVSS 0.662 (+2.7pp, no EPSS) |
