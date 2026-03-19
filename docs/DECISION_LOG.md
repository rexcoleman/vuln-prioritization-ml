# DECISION LOG — ML-Powered Vulnerability Prioritization Engine

<!-- version: 2.0 -->
<!-- created: 2026-03-14 -->

> Decisions logged at every phase gate per govML v2.4 (mandatory).
> Format: ADR (Architecture Decision Record)

---

## ADR-0001: Temporal split instead of random split
- **Date:** 2026-03-14 | **Phase:** 0
- **Context:** Need train/test split strategy. Random split would allow the model to see 2024+ CVEs during training, which leaks future information (exploit status, EPSS scores, reference patterns).
- **Decision:** Train on pre-2024 CVEs, test on 2024+ CVEs.
- **Consequences:** Test set may have different distribution than train (new CWE types, new vendors). This is realistic — in production, the model always predicts on future CVEs.
- **Contracts affected:** DATA_CONTRACT (split methodology), EXPERIMENT_CONTRACT (evaluation protocol)

## ADR-0002: ExploitDB CSV index instead of full git clone
- **Date:** 2026-03-14 | **Phase:** 0
- **Context:** ExploitDB full clone is ~1GB (includes exploit code). We only need CVE-to-exploit mappings for ground truth labels.
- **Decision:** Download only the CSV index file (files_exploits.csv) from GitLab raw URL. Parse CVE references from the "codes" column.
- **Consequences:** No exploit code available locally — but we don't need it for classification. Saves ~1GB disk on a VM with 4GB free.
- **Contracts affected:** DATA_CONTRACT (data source), project.yaml (download_method)

## ADR-0003: NVD 2017+ instead of full history
- **Date:** 2026-03-14 | **Phase:** 0
- **Context:** Full NVD is 338K CVEs. Without API key, download takes ~4 hours. Pre-2017 CVEs mostly have CVSS v2 only (different scoring system).
- **Decision:** Download CVEs from 2017 onward (~200K). Covers CVSS v3 era.
- **Consequences:** Missing older CVEs. Some ExploitDB matches will be for CVEs we don't have. Acceptable — older CVEs have sparse descriptions and different scoring methodology.
- **Contracts affected:** DATA_CONTRACT (data scope)

## ADR-0004: No NVD API key for initial ingestion
- **Date:** 2026-03-14 | **Phase:** 0
- **Context:** NVD offers free API keys (50 req/30s vs 5 req/30s without).
- **Decision:** Proceed without key for initial download (background process, ~3 hours). Register for key before re-ingestion.
- **Consequences:** Slow initial download. Acceptable for one-time overnight ingestion. Resume capability in ingest_nvd.py.
- **Contracts affected:** ENVIRONMENT_CONTRACT

## ADR-0005: TF-IDF over BERT for initial NLP features
- **Date:** 2026-03-14 | **Phase:** 0
- **Context:** CVE descriptions are the richest text feature. Options: TF-IDF (fast, interpretable), BERT (slow, GPU-preferred, less interpretable).
- **Decision:** TF-IDF with bigrams as primary. BERT as stretch goal only.
- **Consequences:** May miss semantic relationships. But TF-IDF is interpretable (SHAP shows which words predict exploitability), fast (no GPU), and sufficient for RQ1-RQ2.
- **Contracts affected:** build_features.py, EXPERIMENT_CONTRACT

## ADR-0006: Practitioner keyword features as explicit feature engineering
- **Date:** 2026-03-14 | **Phase:** 0
- **Context:** 15 years of Mandiant vulnerability triage suggests certain description patterns are strong exploitability signals ("remote code execution", "allows attackers", "crafted request").
- **Decision:** Encode 11 practitioner-knowledge keyword features as binary flags alongside TF-IDF. This makes the "practitioner vs formula" thesis explicit in features.
- **Consequences:** If these dominate SHAP importance → validates that domain knowledge outperforms static formulas. If TF-IDF dominates → model learns patterns practitioners don't explicitly articulate. Either outcome is interesting for the blog post.
- **Contracts affected:** build_features.py, PUBLICATION_PIPELINE (key talking point)

## ADR-0007: Structured features only (no TF-IDF in v1)
- **Date:** 2026-03-14 | **Phase:** 1
- **Context:** build_tfidf_features() was written but structured features alone achieved AUC 0.903. Adding 500 TF-IDF dimensions would increase training time and reduce interpretability.
- **Decision:** Ship v1 with 49 structured features only. TF-IDF is stretch goal.
- **Consequences:** May miss semantic patterns in descriptions. But SHAP on structured features is cleaner and the blog story (practitioner keywords vs non-obvious features) is stronger without TF-IDF noise.
- **Contracts affected:** EXPERIMENT_CONTRACT, FINDINGS.md (documented as limitation)

## ADR-0008: class_weight="balanced" for all models
- **Date:** 2026-03-14 | **Phase:** 2
- **Context:** Training set is 10.5% exploited (imbalanced). Test set is 0.3% (extreme). Without class weighting, models predict majority class.
- **Decision:** Use class_weight="balanced" (sklearn) and scale_pos_weight (XGBoost) to handle imbalance.
- **Consequences:** Improves recall at cost of precision. Appropriate for security triage where missing an exploitable CVE is worse than false alarms.
- **Contracts affected:** train_models.py, METRICS_CONTRACT

## ADR-0009: LogisticRegression as best model (simplicity over complexity)
- **Date:** 2026-03-14 | **Phase:** 2
- **Context:** LogReg AUC (0.903) > RF (0.864) > XGBoost (0.825). Simpler model won. LogReg also enables linear SHAP (exact, not approximate) for better explainability.
- **Decision:** Use LogReg as primary model for SHAP analysis and adversarial eval.
- **Consequences:** Linear model is more interpretable and faster. Fits the "explainable security" blog angle. RF/XGBoost results still reported for completeness.
- **Contracts affected:** run_explainability.py, adversarial_eval.py, FINDINGS.md

## ADR-0010: Feature controllability matrix reused from FP-01
- **Date:** 2026-03-14 | **Phase:** 3
- **Context:** FP-01 established the attacker-controllable vs defender-observable split for IDS features. Same methodology applies to CVE features — some fields (description, references) are attacker-influenced, others (CVSS, EPSS, CWE) are not.
- **Decision:** Reuse FP-01 methodology. Define 15 attacker-controllable and 11 defender-observable features. Test robustness by perturbing only controllable features.
- **Consequences:** Cross-project validation of feature controllability. 0% evasion because model relies on uncontrollable features. This is the strongest finding — publishable methodology.
- **Contracts affected:** adversarial_eval.py, ADVERSARIAL_EVALUATION.md, FINDINGS.md

## ADR-0011: Anti-ghostwriting rule lifted for FINDINGS interpretation
- **Date:** 2026-03-14 | **Phase:** 4
- **Context:** CLAUDE_MD §AI Division of Labor prohibits AI from writing "interpretation, analysis, and written discussion." User explicitly overrode this rule for FP-05 FINDINGS.md interpretation sections. The interpretations are grounded in the user's documented profile (15yr FireEye/Mandiant vulnerability triage) and the project's data.
- **Decision:** AI writes interpretation draft. User reviews, edits, and takes ownership before publication. The published blog post remains the user's voice.
- **Consequences:** Faster iteration on FINDINGS.md. Risk: interpretation may lack nuance that only practitioner experience provides. Mitigation: user reviews before any publication.
- **Contracts affected:** CLAUDE_MD (AI Division of Labor override), FINDINGS.md, PUBLICATION_PIPELINE

## ADR-0012: Add CISA KEV as second ground truth source
- **Date:** 2026-03-19 | **Phase:** 4
- **Context:** Reviewer criticism: "Single ground truth source (ExploitDB)." ExploitDB tracks exploit code availability but misses actively exploited CVEs without public PoCs. CISA KEV is a government-curated list of 1,545 vulnerabilities confirmed as actively exploited in the wild.
- **Decision:** Download CISA KEV catalog. Create dual labels: ExploitDB-only, KEV-only, Either (ExploitDB OR KEV). Retrain best models (LogReg, XGB-tuned) on all three label sets.
- **Consequences:** Test positive count nearly doubles (318 → 648). XGB-tuned with either-label achieves AUC 0.928 (best result). KEV-only is harder (only 1,179 train positives) but XGB still achieves 0.875. Dual labeling partially addresses ground truth lag.
- **Contracts affected:** DATA_CONTRACT (new data source), FINDINGS.md, HYPOTHESIS_REGISTRY (H-5)

## ADR-0013: EPSS circularity analysis — retrain without EPSS features
- **Date:** 2026-03-19 | **Phase:** 4
- **Context:** Reviewer criticism: "You're showing an ML model trained on EPSS learns EPSS. This is circular." EPSS percentile is the #1 SHAP feature at 2x the next feature. The existing ablation showed removing EPSS drops XGBoost AUC by 15.5pp, but we need the full picture: all 7 models × 5 seeds without EPSS.
- **Decision:** Retrain all models with EPSS features (epss, epss_percentile) completely removed. 47 features remain. Include both default and tuned XGBoost variants.
- **Consequences:** Confirms circularity — all models drop to ~0.68 AUC without EPSS. Reframes the contribution: EPSS dominance quantified at 15-23pp across model families. Public metadata alone provides modest but real signal above CVSS (0.662). Honest negative result foregrounded.
- **Contracts affected:** FINDINGS.md (EPSS Circularity Analysis section), HYPOTHESIS_REGISTRY (H-6), Blog Post Angle (reframed)
