# I trained ML models on 338K CVEs to predict exploitability — Logistic Regression hits 0.903 AUC, beating CVSS by 24 percentage points

I built an ML pipeline that ingests CVEs from NVD, ExploitDB, EPSS, and CISA KEV, engineers 49 features, and predicts real-world exploitability. Logistic Regression achieved 0.903 AUC-ROC on a temporal train/test split. Best CVSS threshold (>=9.0) only manages 0.662 AUC — barely better than a coin flip at predicting what actually gets exploited.

I tested 7 algorithms with 5 seeds, temporal splits, SHAP explainability, feature group ablation, and adversarial evaluation. The honest finding: EPSS (0.912 AUC) slightly beats my model (0.903), and when I remove EPSS features, every model collapses to ~0.68 AUC. The model is largely delegating to EPSS. But the value is real — organizations without EPSS access can build a model that beats CVSS using only public NVD data, and dual ground truth (ExploitDB + CISA KEV) pushes tuned XGBoost to 0.928 AUC.

Key findings:

- **Simpler models win** — Logistic Regression beats XGBoost and Random Forest because the signal is linear (vendor size + CVE age predict exploitation without complex interactions)
- **EPSS percentile is the #1 SHAP feature at 1.096 mean |SHAP|** — nearly 2x the next feature, confirming exploit-likelihood signals concentrate in threat intel
- **Vendor CVE count validates the deployment-ubiquity thesis** — Microsoft, Linux kernel, Chrome get exploited disproportionately because attackers target what's widely deployed
- **Ground truth lag is the biggest threat** — 2024+ test set has 0.3% exploit rate vs 10.5% in training, not because recent CVEs are safer but because ExploitDB hasn't caught up
- **Dual ground truth matters** — adding CISA KEV as a second label source improved best model from 0.912 to 0.928 AUC

Methodology: 338K CVEs, temporal pre-2024/2024+ split, 7 algorithms (LogReg, XGBoost, Random Forest, SVM, kNN, MLP, GradientBoosting), 5 seeds, SHAP analysis, feature controllability analysis. 167 passing tests.

Repo: [github.com/rexcoleman/vuln-prioritization-ml](https://github.com/rexcoleman/vuln-prioritization-ml)

Pipeline is open source. Happy to answer questions about the methodology or EPSS circularity analysis.
