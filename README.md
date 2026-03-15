# ML-Driven Vulnerability Prioritization

**Predict which CVEs get exploited using ML on 338K vulnerabilities from NVD, ExploitDB, and EPSS. Replaces CVSS-based triage with data-driven prioritization.**

## Key Results

| Metric | Value |
|--------|-------|
| Logistic Regression AUC | 0.903 (+24pp vs CVSS threshold) |
| EPSS AUC | 0.912 (slightly better, but opaque) |
| #1 Predictor | Vendor history (35x more important than next feature) |
| Features | 49 features from 3 data sources |
| Data | 337,953 CVEs (NVD) + 24,936 exploits (ExploitDB) + 320,502 EPSS scores |

**Core insight:** Vendor exploit history dominates all other predictors by 35x. SHAP analysis reveals that CVSS severity — the industry-standard prioritization metric — ranks below vendor history, reference count, and CWE category.

## Quick Start

```bash
git clone https://github.com/rexcoleman/vuln-prioritization-ml.git
cd vuln-prioritization-ml
conda env create -f environment.yml
conda activate vuln-prioritize
python scripts/run_pipeline.py --seed 42
```

## Architecture

```
scripts/
  ingest_nvd.py          # NVD API ingestion (338K CVEs)
  ingest_exploitdb.py    # ExploitDB label matching
  ingest_epss.py         # EPSS score ingestion
  build_features.py      # 49 features from 3 data sources
  train_baselines.py     # LR, RF, XGBoost vs CVSS threshold
  train_models.py        # Full model training + evaluation
  run_explainability.py  # SHAP analysis
  adversarial_eval.py    # Feature controllability analysis
```

## Methodology

This project validates the **adversarial controllability analysis** methodology (2nd domain after IDS). Features are classified by who controls them:
- **System-controlled:** vendor history, CWE category, reference count — adversaries cannot manipulate
- **Developer-controlled:** CVSS scores, description keywords — potentially gameable

See [FINDINGS.md](FINDINGS.md) for detailed results and [ADRs](docs/decisions/) for architectural decision records.

## Governed by [govML](https://github.com/rexcoleman/govML)

Built with reproducibility and decision traceability enforced across the entire pipeline.

## License

[MIT](LICENSE)
