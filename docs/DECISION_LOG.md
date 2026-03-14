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
