# PUBLICATION PIPELINE

<!-- version: 1.0 -->
<!-- created: 2026-03-14 -->
<!-- last_validated_against: adversarial-ids-ml (FP-01) -->

> **Authority Hierarchy**
>
> | Priority | Document | Role |
> |----------|----------|------|
> | Tier 1 | `docs/PROJECT_BRIEF.md` | Primary spec — highest authority |
> | Tier 2 | `null # No external FAQ` | Clarifications — cannot override Tier 1 |
> | Tier 3 | `docs/ADVERSARIAL_EVALUATION.md` | Advisory only — non-binding if inconsistent with Tier 1/2 |
> | Contract | This document | Implementation detail — subordinate to all tiers above |
>
> **Conflict rule:** When a higher-tier document and this contract disagree, the higher tier wins.
> Update this contract via `CONTRACT_CHANGE` or align implementation to the higher tier.

### Companion Contracts

**Upstream (this contract depends on):**
- See [PUBLICATION_BRIEF](PUBLICATION_BRIEF.tmpl.md) for audience, narrative constraints, and message governance
- See [PROJECT_BRIEF](PROJECT_BRIEF.tmpl.md) for thesis statement and research questions
- See [HYPOTHESIS_CONTRACT](../core/HYPOTHESIS_CONTRACT.tmpl.md) for hypothesis resolution (source of key claims)
- See [FIGURES_TABLES_CONTRACT](../core/FIGURES_TABLES_CONTRACT.tmpl.md) for artifact inventory
- See [DECISION_LOG](../management/DECISION_LOG.tmpl.md) for architectural decisions to highlight

**Downstream (depends on this contract):**
- See [ACADEMIC_INTEGRITY_FIREWALL](ACADEMIC_INTEGRITY_FIREWALL.tmpl.md) for content reuse boundaries

## Customization Guide

Fill in all `{{PLACEHOLDER}}` values before use. Delete this section when customization is complete.

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `Adversarial ML on Network Intrusion Detection` | Project name | Adversarial ML on IDS |
| `{{BLOG_TITLE}}` | Working title for blog post | Adversarial ML on Network Intrusion Detection |
| `{{CONTENT_PILLAR}}` | Which brand pillar (AI Security Architecture / ML Systems Governance / Builder-in-Public) | AI Security Architecture |
| `{{TARGET_AUDIENCE}}` | Primary audience for this post | P1: AI security hiring managers |
| `{{CANONICAL_URL}}` | Blog URL where this will live | https://rexcoleman.dev/posts/adversarial-ids/ |
| `{{REPO_URL}}` | GitHub repository URL | https://github.com/rexcoleman/adversarial-ids-ml |
| `docs/PROJECT_BRIEF.md` | Tier 1 authority document | Project requirements spec (or null for self-directed) |
| `null # No external FAQ` | Tier 2 authority document | FAQ or null |
| `docs/ADVERSARIAL_EVALUATION.md` | Tier 3 authority document | Advisory or null |

---

## 1) Target Venue

- [ ] Blog (canonical home — Hugo site)
- [ ] Conference CFP (specify: _____________)
- [ ] Academic workshop / journal (specify: _____________)
- [ ] LinkedIn article (long-form)

**Submission deadline:** *(if applicable)*

---

## 2) Content Identity

| Property | Value |
|----------|-------|
| **Working title** | Why CVSS Gets It Wrong: ML-Powered Vulnerability Prioritization with Explainable Features |
| **Content pillar** | AI Security Architecture (40% pillar) |
| **Target audience** | P1: AI security hiring managers, P2: CISOs (Mandiant network) |
| **One-line thesis** | CVSS is a static formula from 2005; an ML model trained on real exploit data reveals what actually makes vulnerabilities dangerous — and the top features aren't what CVSS weights highest. |
| **What was shipped** | https://github.com/rexcoleman/vuln-prioritization-ml- |

### Voice Check

This post MUST pass the builder-voice test:

| Test | Pass? |
|------|-------|
| References something you built (not theoretical) | [ ] |
| Shows work (code, architecture, data) not just opinions | [ ] |
| Avoids "5 Tips" / "Why You Should" / "The Future of" framing | [ ] |
| Includes at least one architecture diagram | [ ] |
| Links to a GitHub repo with working code | [ ] |

---

## 3) Draft Structure

| # | Section | Content | Estimated Length | Status |
|---|---------|---------|-----------------|--------|
| 1 | Hook | Problem statement — why this matters, in 2-3 sentences | 2-3 sentences | |
| 2 | Context | What you built, what tools/frameworks you used, why this approach | 1 paragraph | |
| 3 | Architecture | System-level diagram + explanation of key design decisions | 1-2 paragraphs + diagram | |
| 4 | Key Findings | 3-5 findings with evidence (numbers, charts, code) | 3-5 subsections | |
| 5 | Code Examples | Runnable snippets that prove the claims | 2-3 code blocks | |
| 6 | What I Learned / What Broke | Honest reflection on failures and surprises | 1-2 paragraphs | |
| 7 | Conclusion + Links | Summary + links to repo, govML, related posts | 1 paragraph | |

### Architecture Diagram Requirement

Every technical post MUST include at least one architecture diagram showing system-level design. Tools: Mermaid (renders in Hugo/GitHub/dev.to), Excalidraw (hand-drawn aesthetic), or draw.io.

**Diagram captures:** Data flow from NVD API → ExploitDB label join → Feature engineering (NLP + temporal + metadata) → Model training → CVSS/EPSS comparison → SHAP explainability → Adversarial evaluation

---

## 4) Evidence Inventory

Map key claims to supporting artifacts from the project:

| Claim | Evidence | Source File / Figure |
|-------|---------|---------------------|
| *(e.g., "Feature controllability enables 100% detection of noise attacks")* | *(metric, chart)* | *(FINDINGS.md §X, outputs/figures/Y.png)* |
| | | |
| | | |

---

## 5) Distribution Checklist

### 5.1 Pre-Publication

- [ ] Draft reviewed for builder voice (§2 voice check passes)
- [ ] Architecture diagram finalized and embedded
- [ ] Code examples tested and runnable
- [ ] All claims traceable to evidence inventory (§4)
- [ ] Links to repo and govML included
- [ ] No anti-claims present (grep for "superior", "prove", "novel", "always", "never", "best")

### 5.2 Publish

- [ ] Published on Hugo site (canonical URL: `{{CANONICAL_URL}}`)
- [ ] Emailed via Substack (with email-specific intro paragraph)
- [ ] Canonical URL set in Substack post metadata

### 5.3 Cross-Post (within 24 hours of publication)

- [ ] Cross-posted to dev.to with canonical URL → `{{CANONICAL_URL}}`
- [ ] Cross-posted to Hashnode with canonical URL → `{{CANONICAL_URL}}`
- [ ] LinkedIn native text post written (key insight + 3-5 bullets)
- [ ] Blog link added as first comment on LinkedIn post (not in post body)
- [ ] Mastodon (infosec.exchange) post with link *(if account active)*

### 5.4 Post-Publication (within 48 hours)

- [ ] Submitted to Hacker News *(only for strong technical posts)*
- [ ] Respond to comments on all platforms
- [ ] Update govML LESSONS_LEARNED.md with any publishing friction

---

## 6) Metrics to Track

| Metric | Source | Check After |
|--------|--------|-------------|
| Blog page views | Cloudflare / Plausible Analytics | 7 days |
| Substack open rate | Substack dashboard | 3 days |
| LinkedIn impressions | LinkedIn analytics | 7 days |
| dev.to / Hashnode views | Platform dashboards | 7 days |
| GitHub repo traffic spike | GitHub Insights | 7 days |
| Hacker News points *(if submitted)* | HN | 24 hours |

---

## 7) Change Control Triggers

The following changes require a `CONTRACT_CHANGE` commit:

- Target venue or audience
- Working title or one-sentence thesis
- Draft structure (section additions/removals)
- Distribution channel list
