# LinkedIn Post — Vulnerability Prioritization

> Paste as native LinkedIn text. Add blog link as FIRST COMMENT (not in post body — LinkedIn algorithm deprioritizes external links).

---

From my time at FireEye/Mandiant, I saw security teams burn hours patching CVSS 9.8s that never got exploited — while CVSS 7.5s got weaponized and led to breaches.

CVSS measures severity. Attackers measure opportunity.

I trained ML models on 338,000 real CVEs to find out what actually predicts exploitation. Here's what the data says:

CVSS predicts exploitability with AUC 0.66. Barely better than a coin flip.
ML models: AUC 0.90-0.93. A 24+ point improvement.

But here's the honest part most researchers skip:

The #1 predictor is EPSS percentile — which is ITSELF another ML model trained on richer data. Remove EPSS features and every model drops to ~0.68 AUC. The model was largely delegating to EPSS.

So what does ML actually add?

1. Dual ground truth matters. Combining ExploitDB + CISA KEV labels pushes tuned XGBoost to 0.928 AUC — because different sources capture different facets of exploitation.

2. Without EPSS, public metadata still beats CVSS. Vendor history, CWE patterns, and temporal features provide modest but real signal (~0.68-0.78 AUC depending on ground truth).

3. Feature controllability is an architectural defense. 0% adversarial evasion because the top features are things attackers can't control. Validated across two security domains.

4. For organizations without threat intel subscriptions, a model trained on public NVD data gets meaningful predictions — especially for CISA KEV prediction (0.784 AUC without EPSS).

The real contribution isn't "ML beats CVSS." It's quantifying how much EPSS contributes and showing what's possible with public data alone.

Full code, data pipeline, and SHAP visualizations are open source. 7 algorithms, 5 seeds, 167 tests.

What does your team actually use for vuln prioritization? I'm curious if CVSS is still the default.

#AISecurity #MachineLearning #Cybersecurity #VulnerabilityManagement #BuildInPublic

---

> First comment: "Full write-up with architecture diagram and SHAP plots: [blog URL]"
> Second comment: "Code + governed pipeline: github.com/rexcoleman/vuln-prioritization-ml-"
> Third comment: "Built with govML — open-source ML governance framework: github.com/rexcoleman/govML"
