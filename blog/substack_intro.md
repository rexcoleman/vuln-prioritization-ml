# Substack Email Intro

> Paste this BEFORE the full blog post content in Substack editor. This is the email-specific hook that appears in inbox previews.

---

**Subject line:** CVSS predicts exploitation at AUC 0.66. I trained models that hit 0.93.

If you've ever triaged vulnerabilities using CVSS scores, you know the feeling: a "Critical 9.8" sits unpatched for months because your team knows it's not actually exploitable in your environment, while a "High 7.5" gets weaponized next week.

CVSS scores severity, not exploitability. It's a static formula that predicts real-world exploitation with AUC 0.66 — barely better than a coin flip.

I trained ML models on 338,000 real CVEs with dual ground truth (ExploitDB + CISA KEV) to find out what actually predicts exploitation. The results surprised me — and then I had to be honest about what was really driving the predictions.

Spoiler: EPSS does the heavy lifting. But there's a practical story for organizations that don't have EPSS access.

Read on for the full analysis: 7 algorithms, SHAP explainability, EPSS circularity, and why feature controllability makes this model hard to game.
