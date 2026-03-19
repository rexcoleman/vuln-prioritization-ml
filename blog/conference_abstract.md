# Conference Abstract — BSides / DEF CON AI Village

> **Title:** Why CVSS Gets It Wrong: ML-Powered Vulnerability Prioritization with Explainable Features and Adversarial Robustness
> **Speaker:** Rex Coleman
> **Track:** AI Security / ML for Cybersecurity
> **Length:** 20-30 minutes

## Abstract

CVSS is the industry standard for vulnerability scoring, but it was designed to measure severity — not exploitability. We demonstrate that CVSS predicts real-world exploitation with an AUC of only 0.66 (barely better than random), while ML models trained on public data achieve 0.90-0.93.

We train 7 algorithms across 5 seeds on 338,000 CVEs from the National Vulnerability Database, using dual ground truth labels from ExploitDB (25K exploits) and CISA's Known Exploited Vulnerabilities catalog (1,545 actively exploited CVEs). Using temporal train/test splits to prevent data leakage, we compare ML predictions against CVSS thresholds and EPSS (the Exploit Prediction Scoring System).

Key findings:
1. **ML crushes CVSS** (+24 AUC points) — but EPSS is doing the heavy lifting. Removing EPSS features drops all models to ~0.68 AUC, quantifying EPSS's 15-23pp contribution.
2. **Dual ground truth improves results.** Combining ExploitDB + CISA KEV labels pushes tuned XGBoost to AUC 0.928 — the strongest result — because KEV captures actively exploited CVEs that ExploitDB misses.
3. **SHAP explainability** reveals EPSS percentile dominates at 2x the next feature, followed by exploit references, CVSS score, and vendor history.
4. **Feature controllability analysis** shows 0% adversarial evasion because top features are defender-observable — validated across two security domains (IDS + CVE).
5. **Honest negative results:** Without EPSS, public metadata provides only modest signal over CVSS (~0.68 AUC). The contribution is quantifying this gap, not claiming to beat EPSS.

We release the full pipeline as open source with govML governance (7 algorithms × 5 seeds, documented decisions, SHAP visualizations, 167 passing tests).

## Bio (100 words)

Rex Coleman is building at the intersection of AI security and ML systems engineering. He spent over a decade at FireEye and Mandiant in data analytics and enterprise sales, working with security teams across Fortune 500 organizations. He is completing his MS in Computer Science at Georgia Tech (Machine Learning specialization), where he researches AI security — adversarial evaluation of ML systems, agent exploitation, and ML governance tooling. He is the creator of govML, an open-source governance framework for ML research projects. CFA charterholder.

## Why This Talk Matters

Every SOC in the world prioritizes vulnerabilities using CVSS. This research provides concrete, reproducible evidence that CVSS is a poor exploitability predictor and demonstrates a transparent, explainable alternative. The adversarial robustness analysis (using a novel feature controllability methodology) is directly relevant to the AI security community — it answers "can an attacker game this model?" with a clear "no, and here's why."

## Outline

1. **The Problem** (3 min): CVSS scores severity ≠ exploitability. Why every SOC is mistriaging.
2. **The Data** (3 min): NVD + ExploitDB + CISA KEV. Temporal split methodology. Dual ground truth.
3. **The Results** (8 min): AUC comparison table. SHAP feature importance. The EPSS circularity problem — what ML actually adds beyond EPSS.
4. **Feature Controllability** (5 min): Which features can an attacker manipulate? Why the model is robust. Cross-domain validation (IDS + CVE).
5. **Implications** (3 min): What this means for vulnerability management. Dual ground truth. When to use ML vs EPSS alone.
6. **Demo / Q&A** (5 min): GitHub repo walkthrough. govML governance.
