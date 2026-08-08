# Route A Paper Revision Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert the current manuscript into a mathematically scoped, claim-evidence-aligned paper on aggregation discriminability and the limits of simple structural diagnostics.

**Architecture:** Work in two gates. Gate 1 is a submission-safety rewrite that removes false or unsupported claims and adds the exact self-feature mixing result. Gate 2 builds direct synthetic validation and a fair, fixed-target diagnostic benchmark before any positive empirical claim is restored.

**Tech Stack:** IEEEtran LaTeX, BibTeX, Python 3, NumPy, JSON per-seed artifacts, unittest/pytest-style verification, MiKTeX.

---

### Task 1: Add a Claim-Safety Regression Audit

**Files:**
- Create: `scripts/audit_route_a_claims.py`
- Create: `tests/test_route_a_claims.py`
- Inspect: `main_ieee_journal.tex`
- Inspect: `sections/01_intro_information_budget.tex`
- Inspect: `sections/04_information_budget_medium.tex`
- Inspect: `sections/08_discussion_medium.tex`

**Step 1: Write failing tests**

Test that active manuscript sources reject the following patterns: `Structure Information Bound`, `Efficiency \\in [-1, 1]`, `32/36`, `7/9`, `12/12`, `regardless of architecture`, and language calling an edge shuffle degree-preserving without an accompanying degree checksum statement.

**Step 2: Run the focused test**

Run: `python -m unittest tests.test_route_a_claims -v`

Expected: FAIL on the current manuscript and list every source location.

**Step 3: Implement the audit utility**

Implement a deterministic scanner over only the files included by `main_ieee_journal.tex`. Ignore LaTeX comments and return a nonzero exit code with `path:line:pattern` diagnostics.

**Step 4: Re-run the test**

Run: `python -m unittest tests.test_route_a_claims -v`

Expected: the scanner tests PASS while its fixture representing the current unsafe manuscript reports all banned claims.

**Step 5: Commit**

Commit message: `test(paper): add route A claim-safety audit`

### Task 2: Rewrite the Abstract and Contribution Contract

**Files:**
- Modify: `main_ieee_journal.tex:55-68`
- Modify: `sections/01_intro_information_budget.tex:9-68`
- Reference: `docs/plans/2026-08-08-paper-route-a-design.md`

**Step 1: Create a claim-evidence map**

Add `docs/claim_evidence_map.md` with columns for claim, evidence artifact/proposition, status, and permitted wording. Mark the exact fixed-degree result as supported, the self-mixing result as pending until Task 4, and all selector success claims as unsupported.

**Step 2: Rewrite the Abstract**

Use three messages only: problem, scoped method, supported result/limitation. Remove all three historical selector scores and avoid `framework`, `predictor`, `novel threshold`, and `automatic model selection`.

**Step 3: Rewrite the Introduction contributions**

Replace the Information Budget principle with an arithmetic-headroom observation. State the negative diagnostic question explicitly and reduce the contribution list to the three items in the design document.

**Step 4: Run claim audit and compile**

Run: `python scripts/audit_route_a_claims.py`

Run: `pdflatex -interaction=nonstopmode -halt-on-error main_ieee_journal.tex`

Expected: no banned Abstract/Introduction claims; LaTeX exits 0.

**Step 5: Commit**

Commit message: `docs(paper): align abstract and contributions with audited evidence`

### Task 3: Remove the Invalid Information-Budget Theory

**Files:**
- Modify: `sections/04_information_budget_medium.tex:6-145`
- Modify: `sections/07_experiments_medium.tex:24-55`
- Modify: `sections/02_related_work_information_budget.tex:9-43`

**Step 1: Add negative assertions to the claim audit**

Require that active sources contain no theorem or proposition bounding classification-error improvement by `I(Y;G|X)/log C`, no bounded efficiency range, and no statement that small positive headroom bounds negative aggregation damage.

**Step 2: Replace the section's role**

Retitle it `Accuracy Headroom Is Not a Structural Predictor`. Keep only the arithmetic inequality for positive improvement, explain that negative damage is unbounded by that headroom, and retain the shared-term warning.

**Step 3: Delete invalid material**

Remove the Structure Information Bound, the efficiency decomposition/range, and tables or predictions whose only support is the arithmetic ceiling.

**Step 4: Verify**

Run the claim audit and compile. Expected: both exit 0; no dangling references to removed tables or propositions.

**Step 5: Commit**

Commit message: `fix(theory): remove invalid information-budget claims`

### Task 4: Add the Exact Self-Feature Mixing Proposition

**Files:**
- Modify: `sections/04_snr_theory_medium.tex:35-165`
- Create: `tests/test_self_feature_mixing.py`
- Create: `scripts/verify_self_feature_mixing.py`

**Step 1: Write numerical moment tests**

For binary labels, Gaussian features, fixed degree `d`, correlation `rho`, strength `s`, and operator `Z_alpha = alpha X_v + (1-alpha) A_v`, simulate class-conditional samples with fixed seeds. Test empirical conditional mean and covariance against the derived moments.

**Step 2: Test boundary reductions**

Require: `alpha=0` reduces to the existing neighbor-only ratio; `alpha=1` gives ratio 1; `rho=1` removes label-mixture variance; and the denominator remains positive for the declared parameter domain.

**Step 3: State and prove the proposition**

Add

`kappa_alpha = [alpha + (1-alpha)rho]^2 / [alpha^2 + (1-alpha)^2/d + ((1-alpha)^2/d)(1-rho^2)s]`.

Explicitly state fixed degree, conditional independence, independent self/neighbor feature noise, binary Gaussian features, and moment-based Mahalanobis discriminability. State that the aggregate is generally a Gaussian mixture and that this is not a Bayes-error formula.

**Step 4: Separate theory from GCN claims**

Describe the operator as an analyzable linear surrogate. Do not equate `alpha` with a learned GCN coefficient without an additional approximation argument.

**Step 5: Verify and commit**

Run: `python -m unittest tests.test_self_feature_mixing -v`

Run the claim audit and compile. Commit message: `feat(theory): derive self-feature aggregation discriminability`

### Task 5: Remove KS and Architecture Overclaims

**Files:**
- Modify: `sections/04_snr_theory_medium.tex:135-165`
- Modify: `sections/08_discussion_medium.tex:20-36`
- Modify: `sections/02_related_work_information_budget.tex:13-43`

**Step 1: Tighten the contextual comparison**

Retain only the observation that a signal-squared-times-branching-factor form also appears in broadcasting. Distinguish regular-tree forward degree, Poisson offspring mean, finite-depth GNN neighborhoods, and self-looped normalized aggregation.

**Step 2: Delete universal statements**

Remove claims that layers beyond a KS threshold reduce discriminability regardless of architecture or that the manuscript proves an SBM detectability threshold.

**Step 3: Add missing direct related work**

Verify and cite the recent primary works on heterophily separability, CSBM GCN embeddings, feature-versus-structure noise, and CSBM thresholds. Use DOI/proceedings metadata rather than copying unverified BibTeX.

**Step 4: Verify and commit**

Run citation-key checks, claim audit, BibTeX, and full LaTeX compilation. Commit message: `docs(theory): scope broadcasting context and update related work`

### Task 6: Reframe ADR and Two-Hop Recovery as Descriptive Diagnostics

**Files:**
- Modify: `sections/05_aggregation_damage_medium.tex`
- Modify: `sections/06_dual_heterophily_medium.tex`
- Modify: `sections/03_motivating_observations_medium.tex`
- Modify: `sections/07_experiments_medium.tex:109-170`

**Step 1: Standardize ADR**

Define ADR once as an absolute accuracy difference in percentage points. Remove all ratio language and reconcile code/artifact labels before using historical values.

**Step 2: Stabilize two-hop reporting**

Report `h2`, `h2-h1`, class-prior baseline, and uncertainty. Retain `h2/h1` only with a warning about instability near zero.

**Step 3: Remove the selector algorithm**

Replace the prescriptive algorithm with a descriptive diagnostic checklist. Do not map a structural statistic directly to a winning model family.

**Step 4: Verify and commit**

Run claim audit and compile. Commit message: `docs(paper): reframe ADR and two-hop recovery as diagnostics`

### Task 7: Build Direct Synthetic Validation of the Theory

**Files:**
- Create: `experiments/validate_discriminability_formula.py`
- Create: `configs/discriminability_grid.json`
- Create: `tests/test_discriminability_experiment.py`
- Create: `results/discriminability/<run_id>/seed_*.json`
- Create: `scripts/summarize_discriminability.py`

**Step 1: Freeze the estimand and grid**

Primary estimand: relative error between empirical moment-based discriminability and exact `kappa_alpha`. Secondary estimand: error of the `rho^2 d` approximation. Grid over `d`, `rho`, `s`, and `alpha`; fix seeds before running.

**Step 2: Add schema and failure tests**

Require one immutable JSON per configuration/seed, with config, seed, Git commit, package versions, sample count, empirical moments, exact prediction, approximation prediction, and exception status.

**Step 3: Implement and dry-run**

Use a tiny grid in tests. Expected: deterministic repeated runs and no aggregate file mutation by the experiment process.

**Step 4: Run the frozen grid**

Do not change the grid after inspecting outcomes. Summarize median absolute relative error and 95% bootstrap intervals. Plot exact-versus-empirical and approximation bias by feature strength.

**Step 5: Commit**

Commit code/config before full execution; commit audited summaries separately after provenance checks.

### Task 8: Rebuild the Diagnostic Benchmark with a Fixed Target

**Files:**
- Create: `docs/preregistration_diagnostic_benchmark.md`
- Create: `experiments/evaluate_diagnostics.py`
- Create: `tests/test_diagnostic_scoring.py`
- Modify: `sections/07_experiments_medium.tex`

**Step 1: Freeze one target**

Use one operational target across every dataset: whether the best graph model selected from a predeclared model set improves over a tuned MLP by more than a predeclared practical margin on held-out test data. Define ties before seeing results.

**Step 2: Freeze baselines and scoring**

Include always-MLP, always-graph, 50/50 expectation, homophily-only, degree-only, homophily-plus-degree, validation-set model selection, two-hop-only, and the historical combined rule. Apply identical tie handling. For abstaining methods report coverage, selective risk, full-set regret, and risk--coverage curves.

**Step 3: Prevent leakage**

Compute diagnostics only from information allowed at decision time. If full test labels are required for homophily, label the analysis post hoc rather than model selection.

**Step 4: Preserve paired outcomes**

Store dataset/model/seed records separately. Report paired bootstrap confidence intervals and effect sizes. Use Holm correction for predeclared families of comparisons; do not interpret non-significance as equivalence.

**Step 5: Restore only supported manuscript claims**

If the combined diagnostic does not improve regret or risk at matched coverage, make that negative result the empirical conclusion. Do not restore historical headline accuracies as evidence of increment.

### Task 9: Correct the Edge Intervention or Remove Its Causal Interpretation

**Files:**
- Inspect/modify: the edge-shuffle experiment source brought into scope for rerun
- Create: `tests/test_degree_preserving_shuffle.py`
- Modify: `sections/04_information_budget_medium.tex`
- Modify: `sections/07_experiments_medium.tex:57-84`

**Step 1: Test graph invariants**

Require identical node count, edge count, and sorted degree sequence before and after randomization, with no silent loss from self-loop deletion or duplicate-edge collapse.

**Step 2: Implement double-edge swaps or downgrade the claim**

If a verified degree-preserving implementation is unavailable, call the existing result an exploratory edge randomization and remove causal wording.

**Step 3: Separate intervention effects**

Report changes in homophily, component structure, assortativity, and clustering so the experiment is not interpreted as isolating one mechanism.

**Step 4: Verify and commit**

Run invariant tests and compile. Commit message: `fix(experiment): make edge intervention claims auditable`

### Task 10: Final Claim-Evidence and Submission-Safety Review

**Files:**
- Modify: `sections/08_discussion_medium.tex`
- Modify: `sections/09_conclusion_medium.tex`
- Modify: `docs/claim_evidence_map.md`
- Create: `docs/route_a_adversarial_review.md`

**Step 1: Reverse-outline every active section**

Record each paragraph's topic sentence and supporting evidence. Remove paragraphs that do not map to the three contribution claims.

**Step 2: Run an adversarial review**

Score novelty, soundness, significance, reproducibility, statistical rigor, and claim-evidence alignment. Treat any unresolved soundness issue as submission-blocking.

**Step 3: Run the full verification sequence**

Run unit tests, claim audit, JSON provenance audit, BibTeX, two LaTeX passes, and scan the log for undefined references/citations and multiply defined labels.

Expected: all tests pass; zero undefined citations/references; every Abstract claim is `supported` in the claim-evidence map.

**Step 4: Render and inspect the PDF**

Check equations, tables, figure readability, float placement, and page overflow. Record remaining layout warnings.

**Step 5: Commit**

Commit message: `docs(paper): complete route A submission-safety review`

