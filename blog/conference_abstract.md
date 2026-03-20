# Conference Abstract — BSides / DEF CON AI Village

> **Title:** Why CVSS Gets It Wrong: ML-Powered Vulnerability Prioritization with Explainable Features and Adversarial Robustness
> **Speaker:** Rex Coleman
> **Track:** AI Security / ML for Cybersecurity
> **Length:** 20-30 minutes

## Abstract

CVSS is the industry standard for vulnerability scoring, but it was designed to measure severity — not exploitability. We demonstrate that CVSS predicts real-world exploitation with an AUC of only 0.66 (barely better than random), while ML models trained on public data achieve 0.93.

We train 7 algorithms across 5 seeds on 338,000 CVEs from the National Vulnerability Database, using dual ground truth from ExploitDB and CISA KEV with temporal train/test splits. ML outperforms CVSS by 24 AUC points, but EPSS does the heavy lifting — removing EPSS features drops all models to 0.68 AUC. Dual ground truth (ExploitDB + KEV) pushes tuned XGBoost to 0.928. SHAP analysis reveals EPSS percentile dominates at 2x the next feature. Feature controllability analysis shows 0% adversarial evasion because top features are defender-observable. The honest negative result: without EPSS, public metadata provides only modest signal over CVSS.

Attendees will leave with a quantified comparison of CVSS vs ML vs EPSS for exploitation prediction, SHAP-based feature importance they can apply to their own vulnerability management programs, and access to the full open-source pipeline with 167 passing tests.

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
