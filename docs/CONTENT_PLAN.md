# CONTENT PLAN — FP-05 ML-Powered Vulnerability Prioritization

> **Project:** FP-05 (Vulnerability Prioritization)
> **Created:** 2026-03-19
> **Status:** DRAFT — blog drafted (updated with dual GT + EPSS circularity), conference abstract updated, 0 published

---

## Content Assets

| ID | Type | Title | Status | Target | Path |
|----|------|-------|--------|--------|------|
| C-01 | Blog post | ML-Powered Vulnerability Prioritization: Why CVSS Isn't Enough | DRAFTED | rexcoleman.dev | `blog/draft.md` |
| C-02 | LinkedIn post | Short-form summary for LinkedIn feed | DRAFTED | LinkedIn | `blog/linkedin_post.md` |
| C-03 | Substack intro | Newsletter introduction / cross-post | DRAFTED | Substack | `blog/substack_intro.md` |
| C-04 | Conference abstract | BSides / ISSA submission | DRAFTED | BSides 2026 | `blog/conference_abstract.md` |
| C-05 | TIL: CVSS coin flip | "CVSS predicts severity, not exploitability" — AUC 0.662 | PLANNED | dev.to / Hashnode | — |
| C-06 | TIL: EPSS dominates | "The best predictor of exploitation is another ML model" — SHAP analysis | PLANNED | dev.to / Hashnode | — |
| C-07 | TIL: Features that hurt | "More features made my model worse" — ablation story | PLANNED | dev.to / Hashnode | — |
| C-08 | TIL: LogReg beats XGBoost | "The simplest model won" — overfitting lesson | PLANNED | dev.to / Hashnode | — |
| C-09 | TIL: Ground truth lag | "Your labels are lying to you" — temporal split deep dive | PLANNED | dev.to / Hashnode | — |
| C-10 | TIL: Feature controllability | "Attackers can't fool this model" — architectural defense | PLANNED | dev.to / Hashnode | — |
| C-11 | TIL: One HP change | "One hyperparameter recovered 8.7pp AUC" — XGBoost depth=3 | PLANNED | dev.to / Hashnode | — |
| C-12 | TIL: EPSS circularity | "Remove one feature and your model falls apart" — EPSS dominance | PLANNED | dev.to / Hashnode | — |
| C-13 | TIL: Dual ground truth | "Two label sources > one: ExploitDB + CISA KEV" | PLANNED | dev.to / Hashnode | — |
| C-14 | TIL: KEV prediction | "Predicting CISA's mandatory-patch list from public data" | PLANNED | dev.to / Hashnode | — |
| C-15 | Thread | Twitter/X thread: 7 models, 338K CVEs, dual ground truth | PLANNED | Twitter/X | — |
| C-16 | Talk slides | Conference talk deck (20 min) | PLANNED | BSides / local meetup | — |
| C-17 | GitHub README showcase | Project card for profile README | PLANNED | GitHub | — |

---

## Publication Sequence

1. **Blog post (C-01)** — canonical long-form, establishes the narrative
2. **LinkedIn (C-02) + Substack (C-03)** — same day as blog, cross-promotion
3. **TIL series (C-05 through C-11)** — one per week over 7 weeks, each links back to blog
4. **Conference abstract (C-04)** — submit to next BSides CFP window
5. **Thread (C-12)** — after blog has 48hr engagement data
6. **Talk slides (C-13)** — only if conference acceptance

---

## Figures Available

| Figure | Path | Best Use |
|--------|------|----------|
| Model comparison bar chart | `blog/images/model_comparison.png` | C-01, C-08, C-12 |
| SHAP bar chart (top 20) | `blog/images/shap_bar_top20_seed42.png` | C-01, C-06, C-04 |
| SHAP summary beeswarm | `blog/images/shap_summary_seed42.png` | C-01 (deep dive section) |
| Learning curves | `blog/images/learning_curves.png` | C-01, C-09 |
| Complexity curves | `blog/images/complexity_curves.png` | C-01, C-11 |
| SHAP importance (report) | `blog/images/shap_importance.png` | C-04, C-13 |

---

## Content Pillar Mapping

| Pillar | Assets |
|--------|--------|
| AI Security Architecture (40%) | C-01, C-04, C-10, C-13 |
| ML Systems Governance (35%) | C-05, C-06, C-07, C-08, C-09, C-11 |
| Builder-in-Public (25%) | C-02, C-03, C-12, C-14 |

---

## Key Messages (reusable across all assets)

1. **CVSS is broken for prioritization.** AUC 0.662 — barely better than random at predicting exploitation.
2. **EPSS is the dominant signal — and that's the honest finding.** EPSS contributes 15-23pp AUC. Without it, models drop to ~0.68. The model is largely delegating to EPSS.
3. **Dual ground truth matters.** ExploitDB + CISA KEV combined pushes XGB-tuned to 0.928 AUC — best result.
4. **Public metadata alone still beats CVSS.** Without EPSS, ~0.68 AUC from CVSS, CWE, vendor history. Modest but real.
5. **Feature controllability is an architectural defense.** 0% adversarial evasion because top features are defender-observable.
6. **Negative results as contributions.** EPSS circularity, feature groups that hurt, kNN failure — these constrain the solution space.

---

## Audience Segments

| Segment | Hook | Assets |
|---------|------|--------|
| Security practitioners / CISOs | "CVSS is failing your patch prioritization" | C-01, C-02, C-04, C-05 |
| ML engineers / data scientists | "LogReg beat XGBoost on 338K rows" | C-01, C-07, C-08, C-11 |
| CS students / ML beginners | "What I learned training 7 models on real CVE data" | C-09, C-10, TIL series |
| Conference reviewers | Claim-based title, statistical rigor, reproducibility | C-04, C-13 |
