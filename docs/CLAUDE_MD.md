# ML-Powered Vulnerability Prioritization Engine — Claude Code Context

> **govML v2.4** | Profile: security-ml (21 templates)

## Project Purpose

Build an ML model that predicts real-world exploitability of CVEs better than CVSS, using NVD + ExploitDB + EPSS data, with SHAP explainability and adversarial robustness evaluation.

- **Context:** Self-directed research (FP-05 in frontier project pipeline)
- **Profile:** security-ml
- **Python:** 3.11 | **Env:** vuln-prioritize
- **Brand pillar:** AI Security Architecture (40% pillar)
- **Blog title (working):** "Why CVSS Gets It Wrong: ML-Powered Vulnerability Prioritization with Explainable Features"

## Authority Hierarchy

When requirements conflict, higher tiers win. Always.

| Tier | Source | Path |
|------|--------|------|
| 1 (highest) | Project Brief — thesis, RQs, scope | `docs/PROJECT_BRIEF.md` |
| 2 | — | No external FAQ (self-directed) |
| 3 | Adversarial evaluation methodology | `docs/ADVERSARIAL_EVALUATION.md` |
| Contracts | Governance docs | `docs/*.md` |

## Current Phase

**Phase:** 0 — Environment & Data Acquisition

### Phase Commands

```bash
# Phase 0 verification
bash scripts/verify_env.sh
python scripts/check_data_ready.py
git remote -v

# Smoke test (use on every script before full runs)
python scripts/train_baselines.py --sample-frac 0.01 --seed 42

# Full experiment sweep (after Phase 2)
python scripts/train_models.py --seed 42
python scripts/train_models.py --seed 123
python scripts/train_models.py --seed 456
```

### Phase Progression

| Phase | Name | Status |
|-------|------|--------|
| 0 | Environment & Data Acquisition | **CURRENT** |
| 1 | Data Ingestion & Feature Engineering | Not started |
| 2 | Baselines & ML Models | Not started |
| 3 | Explainability & Adversarial | Not started |
| 4 | Findings & Publication Draft | Not started |

> Update the Status column as you progress. Do not advance to Phase N+1 until Phase N gate passes.

## Research Questions (from PROJECT_BRIEF)

| # | Question | Key |
|---|----------|-----|
| RQ1 | Can ML features predict exploitability better than CVSS base score? | AUC comparison |
| RQ2 | Which features matter most for real-world exploitability? | SHAP importance |
| RQ3 | How does the model compare to EPSS? | Agreement/disagreement analysis |
| RQ4 | Can adversarial perturbation of CVE descriptions evade the model? | Evasion rate measurement |

## Data Sources

| Source | Download Method | Local Path |
|--------|----------------|------------|
| NVD (NIST) | API (rate-limited, needs NVD_API_KEY) | `data/raw/nvd/` |
| ExploitDB | `git clone` from GitLab | `data/raw/exploitdb/` |
| EPSS (First.org) | API (REST, no auth) | `data/raw/epss/` |
| GitHub Advisory | API (needs GitHub token) | `data/raw/github_advisory/` |

## Key Files

| File | Purpose |
|------|---------|
| `project.yaml` | Structured config driving experiments |
| `docs/PROJECT_BRIEF.md` | **READ FIRST** — thesis, RQs, scope, architecture |
| `docs/PUBLICATION_PIPELINE.md` | Blog post governance + distribution checklist |
| `docs/DECISION_LOG.md` | All tradeoff decisions (mandatory at every phase gate) |
| `docs/ADVERSARIAL_EVALUATION.md` | Attack methodology for RQ4 |
| `docs/HYPOTHESIS_CONTRACT.md` | Hypotheses (fill before Phase 2 experiments) |

## AI Division of Labor

### Permitted

- Code generation, refactoring, debugging
- Data ingestion scripts (NVD API pagination, ExploitDB parsing, EPSS download)
- Feature engineering code
- Test generation and execution
- SHAP visualization code
- Adversarial perturbation code
- Governance audit (cross-reference docs against code)

### Prohibited

- Report prose / blog post authorship (human writes all narrative)
- Hypothesis formulation (human intellectual contribution)
- Feature importance interpretation or results discussion
- Deciding which features "matter" for exploitability (that's the research contribution)
- Modifying PROJECT_BRIEF thesis or research questions

### Anti-Ghostwriting Rule

AI generates code that produces metrics, charts, and SHAP plots. The **interpretation of why certain features predict exploitability** is the human's research contribution and the blog post's value.

## Conventions

- **Seeds:** [42, 123, 456] — never invent new ones
- **Smoke test first:** Always `--sample-frac 0.01` before full runs (WIN-011)
- **Temporal split:** Train on pre-2024 CVEs, test on 2024+ (prevents data leakage from future information)
- **Outputs:** All to `outputs/` matching project.yaml structure
- **Decisions:** Log in DECISION_LOG at every phase gate (mandatory per v2.4)
- **Changes:** CONTRACT_CHANGE protocol: Decision Log → Change → Changelog → commit
- **Tests:** Run `pytest` before committing
