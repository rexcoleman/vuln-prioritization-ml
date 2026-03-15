# ADVERSARIAL EVALUATION

<!-- version: 1.0 -->
<!-- created: 2026-03-11 -->
<!-- last_validated_against: none -->

> **Activation:** This template is OPTIONAL. Include it when your project involves adversarial
> robustness evaluation, security-sensitive ML, or when the project specification requires
> robustness analysis. Delete if not applicable.

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
- See [EXPERIMENT_CONTRACT](EXPERIMENT_CONTRACT.tmpl.md) §2 for compute budgets (adversarial budget draws from the same pool)
- See [METRICS_CONTRACT](METRICS_CONTRACT.tmpl.md) §2 for baseline metric definitions
- See [DATA_CONTRACT](DATA_CONTRACT.tmpl.md) §4 for leakage prevention (adversarial examples must not leak test data)

**Downstream (depends on this contract):**
- See [FIGURES_TABLES_CONTRACT](FIGURES_TABLES_CONTRACT.tmpl.md) §3 for robustness figures
- See [REPORT_ASSEMBLY_PLAN](../report/REPORT_ASSEMBLY_PLAN.tmpl.md) for adversarial analysis section placement
- See [RISK_REGISTER](../management/RISK_REGISTER.tmpl.md) for adversarial-specific risk entries

## Customization Guide

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `Adversarial ML on Network Intrusion Detection` | Project name | Image Classification Robustness Study |
| `{{THREAT_MODEL}}` | Adversary capability description | White-box, Lp-bounded, untargeted |
| `{{PERTURBATION_NORM}}` | Norm constraint | L∞, L2, L1 |
| `{{EPSILON_VALUES}}` | Perturbation budget values | [0.01, 0.03, 0.1, 0.3] |
| `{{ATTACK_METHODS}}` | Attack algorithms to evaluate | FGSM, PGD-20, AutoAttack |
| `{{DEFENSE_METHODS}}` | Defense methods (if applicable) | Adversarial training, input preprocessing |
| `{{ROBUSTNESS_METRICS}}` | Metrics for adversarial evaluation | Robust accuracy, certified radius |
| `docs/PROJECT_BRIEF.md` | Tier 1 authority document | Project requirements spec |
| `null # No external FAQ` | Tier 2 authority document | FAQ or clarifications document |
| `docs/ADVERSARIAL_EVALUATION.md` | Tier 3 authority document | Advisory clarifications |

---

## 1) Purpose & Scope

This contract defines the adversarial evaluation protocol for the **Adversarial ML on Network Intrusion Detection** project. It specifies the threat model, attack and defense methods, robustness metrics, and disclosure requirements.

---

## 2) Threat Model Definition

The threat model MUST be defined before any adversarial evaluation begins. It constrains the adversary's knowledge, capability, and goals.

| Property | Value |
|----------|-------|
| **Adversary knowledge** | {{THREAT_MODEL}} *(white-box / black-box / grey-box)* |
| **Adversary goal** | *(untargeted misclassification / targeted misclassification / confidence reduction)* |
| **Perturbation type** | *(input perturbation / data poisoning / reward perturbation / environment modification)* |
| **Perturbation norm** | {{PERTURBATION_NORM}} |
| **Perturbation budget (ε)** | {{EPSILON_VALUES}} |
| **Attack surface** | *(test-time inputs / training data / reward signal / environment dynamics)* |

### 2b) Formal Threat Model (YAML)

```yaml
threat_model:
  adversary_knowledge: gray_box  # Attacker knows CVSS/EPSS exist, can influence metadata
  adversary_capability:
    perturbation_type: semantic  # Modify CVE text, vendor disclosure patterns
    perturbation_budget:
      norm: semantic
      epsilon: "Can modify CVE description text, reference URLs, vendor information"
    access:
      - cve_description  # Via responsible disclosure or fake CVEs
      - vendor_metadata  # Via vendor reputation manipulation (slow)
    constraints:
      - "Cannot modify EPSS scores directly (computed by First.org)"
      - "Cannot modify historical exploitation labels"
      - "Cannot modify CVSS base scores retroactively"
  adversary_goal: integrity  # Cause mis-prioritization (high-risk vulns deprioritized)
  attack_surface:
    controllable_features: [description_keywords, reference_count, vendor_name]
    observable_features: [epss_score, cvss_base_score, cwe_category]
    system_features: [publication_date, exploitation_label, vendor_cve_count]
  attacks_tested:
    - type: text_keyword_perturbation
      sophistication: low
      tool: custom (modify description keywords)
    - type: reference_count_manipulation
      sophistication: low
      tool: custom (add/remove reference URLs)
  attacks_NOT_tested:
    - type: adversarial_NLP_attack
      reason: "Text features are bag-of-words; gradient attacks not applicable to tree models"
    - type: EPSS_score_manipulation
      reason: "EPSS is external system; cannot be perturbed by attacker"
    - type: temporal_poisoning
      reason: "Historical labels cannot be retroactively modified"
  limitation_acknowledgment: |
    Single-seed results (42). Adversarial evaluation limited to controllability
    analysis — classifying features by attacker influence. No adaptive adversary
    or ML-specific attack (evasion, poisoning) tested.
```

**Rule:** The threat model MUST be documented in the report Methods section before adversarial experiments run.

**Verification:** Report Methods section contains a threat model paragraph specifying all properties above. `config_resolved.yaml` records `threat_model`, `perturbation_norm`, and `epsilon` for every adversarial run.

---

## 3) Perturbation Types

### 3.1 Input Perturbations (Evasion Attacks)

Additive perturbations to test-time inputs within an Lp-norm ball.

| Attack | Type | Parameters | When to Use |
|--------|------|-----------|-------------|
| **FGSM** | Single-step, white-box | ε | Fast baseline; lower bound on vulnerability |
| **PGD** | Multi-step, white-box | ε, step_size, n_steps | Standard strong attack; primary evaluation |
| **AutoAttack** | Ensemble, white-box | ε | Gold-standard; use for final robustness claims |
| **Square Attack** | Black-box, query-based | ε, n_queries | When white-box access is not assumed |
| *(add project-specific)* | | | |

**Budget rule:** Attack iterations (PGD steps, query count) MUST be logged in `summary.json`. Total adversarial compute budget MUST be reported alongside standard evaluation budget.

#### Attack Selection by Model Type

Not all attacks work with all model types. Select attacks based on model differentiability:

| Model Type | Gradient Access? | Recommended Attacks | Avoid |
|-----------|-----------------|--------------------| ------|
| **Neural networks** (PyTorch, TF) | Yes | FGSM, PGD, AutoAttack, C&W | — |
| **sklearn tree ensembles** (RF, XGBoost, GBM) | No | ZOO, HopSkipJump, noise baseline | FGSM, PGD (will fail — `EstimatorError`) |
| **sklearn linear models** (SVM, LogReg) | Partial (via decision function) | ZOO, HopSkipJump | PGD (may work via ART wrapper but unreliable) |
| **Black-box API** | No | Square Attack, HopSkipJump, ZOO | All white-box attacks |

**Rule:** If your model is not differentiable, do NOT attempt gradient-based attacks. Use zeroth-order optimization (ZOO) as primary and decision-based (HopSkipJump) as validation. Random noise is acceptable as a sanity check baseline but is NOT a substitute for optimization-based attacks.

### 3.1b Feature Controllability Matrix (Security ML)

> **Activation:** Include when your adversarial evaluation targets a domain where not all features
> are equally perturbable (e.g., network IDS, malware detection, fraud detection).

For security ML projects, features divide into categories by who controls them:

| Category | Features | Rationale |
|----------|----------|-----------|
| **Attacker-controllable** | *(list features the attacker can modify)* | *(why: e.g., attacker controls packet timing, payload content)* |
| **Defender-observable only** | *(list features set by OS/network/environment)* | *(why: e.g., TCP flags set by receiver OS stack)* |
| **Environment-determined** | *(list features neither party controls)* | *(why: e.g., time-of-day, network load)* |

**Constraint enforcement:** Constrained attacks MUST multiply perturbation vectors by a binary feature mask (1=controllable, 0=not controllable). Both constrained and unconstrained results MUST be reported.

**Domain expertise source:** *(Cite the domain knowledge used to classify features — protocol specs, system documentation, threat models.)*

### 3.2 Data Poisoning

Corruption of training data to degrade model performance or implant backdoors.

| Property | Value |
|----------|-------|
| **Poison fraction** | *(e.g., 1%, 5%, 10% of training set)* |
| **Poison strategy** | *(label flip / gradient-based / backdoor pattern)* |
| **Detection method** | *(spectral signatures / activation clustering / none)* |

### 3.3 Reward Perturbation (RL)

> **Activation:** Include when evaluating RL agent robustness to reward corruption.

| Property | Value |
|----------|-------|
| **Perturbation type** | *(additive noise / adversarial reward / delayed reward)* |
| **Perturbation magnitude** | *(e.g., ±0.1 reward units)* |
| **Frequency** | *(every step / episodic / random fraction)* |

### 3.4 Environment Modification (RL)

> **Activation:** Include when evaluating RL agent robustness to environment changes.

| Property | Value |
|----------|-------|
| **Modification type** | *(transition noise / observation noise / action perturbation)* |
| **Magnitude** | *(e.g., Gaussian noise σ=0.01)* |
| **Evaluation protocol** | *(train in clean → test in perturbed / train in perturbed → test in clean)* |

---

## 4) Robustness Metrics

### 4.1 Required Metrics

| Metric | Definition | When Required |
|--------|-----------|---------------|
| **Clean accuracy** | Standard accuracy on unperturbed test set | Always (baseline) |
| **Robust accuracy** | Accuracy under strongest attack at each ε | Always |
| **Attack success rate** | Fraction of correctly-classified inputs that are misclassified after attack | Always |
| **Accuracy drop** | Clean accuracy − Robust accuracy | Always |
| **Certified radius** | Provably guaranteed perturbation radius (if using certified defense) | When certified defenses are evaluated |

**Verification:** `final_eval_results.json` contains `clean_accuracy`, `robust_accuracy_eps_{ε}`, and `attack_success_rate_eps_{ε}` for each ε value.

### 4.2 Reporting Requirements

- Robustness MUST be reported at multiple ε values, not just a single point
- Results MUST include both clean and robust accuracy to show the accuracy-robustness tradeoff
- Seed dispersion MUST be reported (median + IQR across seeds)
- Attack hyperparameters (steps, step size, restarts) MUST be disclosed

---

## 5) Adversarial Budget Accounting

### 5.1 Compute Budget

Adversarial evaluation has its own compute cost that MUST be tracked separately.

| Budget Component | Unit | How Counted |
|-----------------|------|-------------|
| Attack generation | forward + backward passes | Per-sample: n_steps × (1 forward + 1 backward) for PGD |
| Robustness evaluation | forward passes | Per-sample: 1 forward per attack variant per ε |
| Adversarial training *(if applicable)* | grad_evals | Same as standard training budget, logged separately |

### 5.2 Logging

Every adversarial evaluation run MUST log in `summary.json`:

```json
{
  "adversarial": {
    "attack": "{{ATTACK_METHOD}}",
    "epsilon": 0.03,
    "attack_steps": 20,
    "attack_step_size": 0.003,
    "n_restarts": 1,
    "total_attack_forward_passes": 0,
    "total_attack_backward_passes": 0,
    "clean_accuracy": 0.0,
    "robust_accuracy": 0.0,
    "attack_success_rate": 0.0
  }
}
```

---

## 6) Evaluation Protocol

### 6.1 Standard Evaluation Sequence

```
1. Evaluate clean accuracy on unperturbed test set
2. For each ε in {{EPSILON_VALUES}}:
   a. Generate adversarial examples using each attack method
   b. Evaluate robust accuracy on adversarial examples
   c. Log attack success rate
3. Report accuracy-robustness curve (clean → strongest attack at each ε)
```

### 6.2 Defense Evaluation (if applicable)

| Property | Requirement |
|----------|------------|
| **Adaptive attacks** | Defenses MUST be evaluated against adaptive attacks that are aware of the defense mechanism |
| **Obfuscated gradients check** | If using gradient-masking defenses, MUST verify with black-box attacks (Square Attack) |
| **No security through obscurity** | Defense mechanism MUST be fully disclosed; robustness claims assume white-box access |

**Verification:** For each defense, at least one adaptive attack is included. Black-box attack results are reported alongside white-box results.

---

## 7) Adversarial Baselines

Every adversarial evaluation MUST include baselines for context:

| Baseline | Purpose |
|----------|---------|
| **Undefended model** | Clean accuracy upper bound; robustness lower bound |
| **Random perturbation** | Distinguishes adversarial vulnerability from noise sensitivity |
| **Strongest known attack** | Upper bound on vulnerability (AutoAttack recommended) |

---

## 8) Disclosure Rules

### 8.1 Report Disclosures

The report MUST disclose:

- Complete threat model definition (§2)
- All attack methods, hyperparameters, and implementation sources
- All defense methods and training procedures (if applicable)
- Clean vs robust accuracy at every evaluated ε
- Compute cost of adversarial evaluation
- Any limitations of the evaluation (e.g., attacks not tested, threat models not covered)

### 8.2 Figure and Table Requirements

- Accuracy-robustness curves MUST show clean accuracy as the y-intercept (ε=0)
- Tables MUST include both clean and robust columns
- Captions MUST state the attack method, ε value, and number of seeds

---

## 9) Acceptance Criteria

- [ ] Threat model documented before adversarial experiments
- [ ] Clean accuracy baseline established
- [ ] Robust accuracy evaluated at all specified ε values
- [ ] Attack hyperparameters logged in `config_resolved.yaml`
- [ ] Adversarial compute budget tracked in `summary.json`
- [ ] Baselines included (undefended, random perturbation)
- [ ] Seed dispersion reported for all adversarial metrics
- [ ] Report discloses all required information (§8.1)

---

## 10) Change Control Triggers

The following changes require a `CONTRACT_CHANGE` commit:

- Threat model definition (adversary knowledge, goal, perturbation type/norm)
- ε values or attack method list
- Robustness metric definitions
- Defense methods or adversarial training protocol
- Evaluation protocol or baseline list
- Disclosure requirements

---

## Appendix B: Systems Security Evaluation (Optional)

> **Activation:** Include this appendix when your project involves systems-level security analysis
> (buffer overflows, memory corruption, race conditions in security-critical code). Delete if
> not applicable.

### B.1 Sanitizer-Based Vulnerability Detection

| Tool | What It Finds | Build Flag | When Required |
|------|---------------|-----------|---------------|
| **AddressSanitizer (ASan)** | Buffer overflows, use-after-free, stack overflow | `-fsanitize=address` | Always |
| **MemorySanitizer (MSan)** | Uninitialized memory reads | `-fsanitize=memory` | When processing untrusted input |
| **UndefinedBehaviorSanitizer (UBSan)** | Integer overflow, null deref, type confusion | `-fsanitize=undefined` | Always |
| **ThreadSanitizer (TSan)** | Data races, lock-order inversions | `-fsanitize=thread` | When project uses concurrency |

**Rule:** All test suites MUST pass under ASan + UBSan with zero findings before any security claims. See [BUILD_SYSTEM_CONTRACT](BUILD_SYSTEM_CONTRACT.tmpl.md) §5 for sanitizer build governance.

### B.2 Fuzzing Protocol

| Property | Value |
|----------|-------|
| **Fuzzer** | *(e.g., AFL++, libFuzzer, honggfuzz)* |
| **Corpus** | *(e.g., seed inputs from test suite + manually crafted edge cases)* |
| **Duration** | *(e.g., minimum 1 hour per target, or until coverage plateau)* |
| **Targets** | *(list functions/interfaces to fuzz)* |

**Rule:** Fuzzing MUST target all functions that process external input. Crashes found by fuzzing MUST be triaged, fixed, and added to the regression test suite.

**Verification:** Fuzzing coverage report shows all target functions reached. Zero crashes in final fuzzing pass.

### B.3 Static Analysis

| Tool | Purpose | When Required |
|------|---------|---------------|
| **Clang Static Analyzer** | Buffer overflows, null derefs, logic errors | Always |
| **cppcheck** | Undefined behavior, resource leaks | C/C++ projects |
| **Coverity** | Deep path analysis | If available |

**Rule:** Static analysis MUST produce zero high-severity findings before release. Medium-severity findings MUST be triaged (fix or document suppression with justification).

### B.4 Security Evaluation Reporting

The report MUST include:
- Sanitizer findings summary (total found, total fixed, any suppressions)
- Fuzzing results (duration, corpus size, unique crashes found and resolved)
- Static analysis summary (findings by severity, resolution status)
- Residual risk assessment for any unresolved findings
