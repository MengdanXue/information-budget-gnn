# Route A Adversarial Review

- Review date: 2026-08-08
- Assumed submission type: full journal paper
Assumed venue standard: IEEE TNNLS-level soundness, novelty, empirical depth,
and reproducibility

## Overall assessment

**Current decision: No-Go for journal submission.** The revision has converted
an overclaimed theory-and-selector story into a substantially more defensible
scoped mechanism study. The fixed-degree derivation now retains label-mixture
covariance, the self-feature extension is explicit, and the direct simulation
is auditable. However, the empirical decision-value contribution is still
prospective rather than executed, and the legacy edge intervention has not been
regenerated with the verified degree-matched implementation. For a full TNNLS
paper, those are material completeness gaps rather than cosmetic limitations.

The paper is suitable as a controlled technical report or thesis chapter in its
current form. It becomes a plausible journal submission only if the new
diagnostic benchmark produces a stable nontrivial result (positive or negative)
from clean paired records, the cross-protocol legacy tables are consolidated,
and the paper is positioned as a mechanism-and-evaluation study rather than a
new threshold theory.

## Structured summary

- **Problem:** Explain why graph aggregation can help or harm node
  classification and test whether simple structural diagnostics predict that
  difference.
- **Theory:** Exact first-two-moment Mahalanobis discriminability ratios for a
  binary Gaussian fixed-degree conditional-neighbor model, including a
  prescribed center--neighbor mixture.
- **Direct evidence:** A frozen 400-configuration by 5-seed grid (2,000 records)
  with 5,000 samples per class. Median absolute relative error is 0.94% over the
  1,920 records with nonzero exact ratio.
- **Empirical evidence:** Protocol-specific MLP--GNN accuracy differences,
  archived two-hop label statistics, paired H2GCN--MLP results, a negative
  six-dataset retrospective selector comparison, and explicitly downgraded
  legacy edge randomization.
- **What is not established:** A Bayes-error formula, a stochastic-block-model
  recovery threshold, a trained-GNN guarantee, a universal homophily interval,
  a validated prospective selector, or a causal effect isolated by edge
  randomization.

## Reverse outline of every active section

### Abstract

1. Opens with the aggregation-help-or-harm problem and limits the question to
   discriminability plus diagnostic reliability.
2. States the two exact moment formulas and their non-claims.
3. Reports the frozen direct-validation sample count and main error estimates.
4. Reports the negative diagnostic evidence and demotes accuracy headroom to an
   arithmetic identity.

**Evidence map:** Items 2--3 map to Proposition 1/2 and Task 7 artifacts; item 4
maps to the audited paired two-hop results and six-dataset selector audit.

### I. Introduction

1. Defines the practical aggregation question.
2. Explains why homophily omits degree, feature, covariance, and ego-retention
   effects.
3. Introduces the fixed-degree ratio and its exact scope.
4. Reports direct empirical-moment validation.
5. Defines accuracy difference and two-hop recovery as descriptions, not
   theories.
6. Limits headroom to a one-sided arithmetic ceiling.
7. Lists three contributions: scoped derivation, direct validation, and a
   negative diagnostic study.
8. Provides the paper roadmap.

**Reviewer check:** Every contribution now has either current evidence or an
explicitly negative result. The prospective benchmark is not listed as a
completed contribution.

### II. Related Work

1. Aggregation subsection distinguishes the surrogate from GCN, GraphSAGE,
   attention, and scalable training.
2. Heterophily subsection positions global homophily against class-conditional
   and distributional alternatives.
3. Feature--structure subsection separates arithmetic headroom from feature
   informativeness and explicit noise models.
4. Theory subsection separates moment discriminability from WL expressiveness,
   information flow, oversmoothing, CSBM recovery, and finite-depth embeddings.
5. Positioning subsection states the restricted calculation, direct-validation
   standard, and negative diagnostic evaluation.

**Reviewer check:** The novelty is an application-specific exact calculation
and audit standard, not a new branching threshold. That is honest but modest for
TNNLS.

### III. Motivating Observations

1. Declares all examples descriptive.
2. Presents the archived U-shaped homophily sweep.
3. Reports architecture-specific amplitudes and avoids a causal attribution to
   ego retention.
4. Presents same-homophily rows with arithmetic headroom.
5. Notes that the nine-dataset correlation does not support predictive use.
6. Replaces mixed-protocol heterophily point estimates with one paired protocol
   and uncertainty intervals.
7. Connects the mismatch between two-hop recovery and H2GCN outcomes to the
   diagnostic question.
8. Restates the three puzzles as motivation rather than solved phenomena.

**Evidence gap:** The U-shape and same-h tables remain legacy aggregates without
the complete per-seed provenance of Task 7. They should not carry inferential
language.

### IV. Scoped Aggregation Discriminability Analysis

1. Defines notation and the binary Gaussian conditional-neighbor model.
2. Defines first-two-moment Mahalanobis discriminability and disclaims a full
   distributional distance.
3. Derives conditional mean, covariance, and exact neighbor-only ratio.
4. Derives the prescribed center--neighbor ratio.
5. Checks alpha=0, alpha=1, perfect correlation, and cancellation boundaries.
6. Derives the within-model improvement condition and feature-dependent danger
   interval.
7. Quantifies the approximation error of rho-squared times degree.
8. Separates the weak-feature resemblance from Kesten--Stigum broadcasting.
9. Rejects scalar multi-class, practical-GCN, Bayes-error, and random-degree
   extensions.

**Soundness status:** The algebra is internally coherent and unit-tested. The
remaining limitation is external validity, not an identified algebraic error.

### V. Accuracy Headroom Is Not a Structural Predictor

1. Proves the positive-gain ceiling from the range of accuracy.
2. Shows why it does not bound negative damage or create a bounded efficiency
   ratio.
3. Distinguishes observed tuned-MLP accuracy from optimal feature-only accuracy.
4. Explains shared-term correlation bias.
5. Reports the audited nine-dataset correlation and permutation null.
6. Restricts headroom to scale reporting and ceiling-effect description.

**Reviewer check:** This section is a correction of a prior claim, not a novel
theorem. It is useful methodological hygiene but should not be sold as a primary
contribution.

### VI. Aggregation Damage Analysis

1. Defines ADR as a signed percentage-point difference, not a ratio.
2. Reports the mid-homophily synthetic point estimates.
3. Uses mutual-information chain rule only for the uncompressed concatenated
   representation.
4. Explicitly denies a finite-training guarantee for GraphSAGE.
5. Reports architecture amplitudes as consistency evidence, not causal proof.
6. Gives a matched-comparison checklist.
7. Reports four real-dataset point estimates and marks missing uncertainty.

**Evidence gap:** The exact real-dataset ADR table lacks paired uncertainty and
cannot support an architecture ranking.

### VII. Two-Hop Recovery as a Heterophily Diagnostic

1. Defines h1, h2, delta-h, optional ratio R, and class-prior baseline h0.
2. Explains the instability of R and the need for h0.
3. Marks h0 and structural uncertainty unavailable in the archive.
4. Reports five structural point estimates without significance language.
5. Reports paired H2GCN--MLP intervals for the same five datasets.
6. Shows counterexamples to both directions of a two-hop selector.
7. Notes H2GCN component confounding.
8. Gives a descriptive reporting checklist and rejects fixed recovery cutoffs.

**Strength:** The failures are retained rather than hidden, which materially
improves credibility.

### VIII. Experimental Validation

1. Defines datasets, models, and the distinction between paired and aggregate
   evidence.
2. Reports arithmetic headroom as a transcription/scale check only.
3. Reports the frozen 2,000-record direct moment validation and approximation
   bias.
4. Audits legacy edge randomization, including the self-loop/duplicate failure,
   and states that the verified replacement has not regenerated the table.
5. Reports the negative retrospective selector audit.
6. Describes the prospective target, margin, baselines, leakage checks,
   abstention fallback, and Holm correction without inventing results.
7. Omits unreproducible legacy significance tests.
8. Reports the two-hop point estimates and paired model intervals.
9. Ends with a reporting checklist instead of an automatic selector.

**Primary missing experiment:** The prospective diagnostic benchmark has no
conforming new dataset/model/seed records, so the paper cannot claim predictive
or decision value beyond the historical negative audit.

### IX. Discussion

1. Gives three possible mechanisms for the U-shape, same-h differences, and
   architecture dependence.
2. Delimits broadcasting, finite-depth CSBM, and recovery theory.
3. Lists binary, fixed-degree, homogeneous, Gaussian, finite-graph, and static
   scope conditions.
4. Rejects finite-n transfer rates and universal sample-size cutoffs.
5. Restricts practical use to exploratory diagnostics and prospective testing.
6. Marks the prospective selector and verified edge rerun as unexecuted.
7. Gives modest broader-impact and baseline-coverage statements.
8. Frames adaptive use, random degree, semi-supervision, and heterogeneous
   graphs as future work.

### X. Conclusion

1. Restates the paper as a scoped mechanism study with negative diagnostic
   evidence.
2. Reports the exact expression and frozen direct-validation result.
3. Restates the diagnostic, headroom, prospective-selector, and edge-rerun
   limitations.
4. Defines theoretical significance as moment analysis rather than a trained-GNN
   theorem.
5. Lists exact binary fixed-degree assumptions and concrete next steps.
6. States precisely which artifacts are reproducible and which outcomes remain
   pending.

## Numerical and internal-consistency audit

- U-shape amplitudes are arithmetically consistent: 0.60 - (-18.75) = 19.35;
  0.55 - (-5.30) = 5.85; and 0.65 - (-0.10) = 0.75 percentage points.
- Headroom values equal one minus MLP accuracy in the reported rows, up to the
  displayed rounding.
- Legacy edge-randomization drops are consistent: 12.6 - (-34.8) = 47.4;
  2.8 - (-32.3) = 35.1; and 0.1 - (-17.4) = 17.5 points.
- Two-hop deltas and ratios agree with the displayed h1/h2 values up to rounding.
- Task 7 summary counts agree: 2,000 successes, 1,920 defined relative errors,
  and 400 neighbor-only approximation records.
- The Task 7 text agrees with `summary.json`: primary median 0.009381 and 95%
  interval [0.008628, 0.010536]; secondary median absolute error 0.293397 and
  interval [0.241613, 0.365217].
- A cross-protocol inconsistency was removed from the motivating section:
  Chameleon point accuracies from one archived panel were previously presented
  beside paired H2GCN--MLP intervals from another protocol.
- Bibliography and internal references compile through BibTeX. External citation
  metadata was not re-audited in this final pass.

## Strengths

1. The theoretical claim is now narrow enough to be true and testable.
2. Label-mixture covariance, the major omitted term in the rejected derivation,
   is retained.
3. Direct validation is frozen, deterministic, immutable, and byte-reproducible.
4. Negative results and missing provenance are visible.
5. The prospective scorer uses one target, common tie handling, explicit
   abstention, dataset-level resampling, and leakage rejection.
6. The new edge randomizer verifies graph invariants and reports concurrent
   structural changes.

## Major weaknesses and author questions

1. **Novelty:** Is an exact fixed-degree moment calculation plus negative
   diagnostic audit sufficient for TNNLS after the Kesten--Stigum resemblance is
   acknowledged? The paper needs a sharper significance argument or a more
   empirical venue positioning.
2. **Prospective evidence:** Why is the central diagnostic benchmark specified
   but not executed? For a full paper, infrastructure alone is not evidence.
3. **Legacy dependence:** Why retain multiple legacy aggregate tables if their
   exact per-seed provenance is absent? Consolidating or removing them may be
   stronger than adding caveats.
4. **Edge intervention:** Will the degree-matched rerun reproduce the large
   archived drops after edge count and degree sequence are truly fixed?
5. **Generalization:** The theory is binary, fixed-degree, one-hop, Gaussian,
   and conditionally independent. Which empirical claim genuinely transfers
   beyond that model?
6. **Model-selection value:** If the prospective combined rule does not beat
   homophily/degree/validation baselines at matched coverage, is there enough
   paper left without a diagnostic contribution? The manuscript should be ready
   to make that negative result central.

## Minor issues

- `latexmk` is unavailable locally because MiKTeX cannot find Perl; the document
  is instead verified with pdflatex, BibTeX, and repeated pdflatex passes.
- The IEEE template emits existing underfull-box and anonymous-author warnings.
- The archived U-shape and real-dataset ADR tables still need a common provenance
  manifest if they remain in a submission package.
- “ADR” is retained as an acronym for a difference; define it once and avoid the
  historical word “ratio” everywhere.

## Top actions: start here

1. Generate clean immutable dataset/model/seed records for the prospective
   diagnostic benchmark and run the frozen evaluator without changing its
   target or thresholds after seeing outcomes.
2. Run the verified degree-matched edge intervention on the three archived
   datasets or remove Table XII and Figure 3 from the journal submission.
3. Consolidate every active real/synthetic accuracy table onto named protocols
   with split IDs, seeds, model-selection rules, and source artifacts.
4. Decide the paper's final identity after the prospective result: mechanism
   study with a negative selector audit, or stop the journal submission if only
   descriptive legacy tables remain.
5. Add a compact experiment manifest mapping every active number to a committed
   source record.
6. Reassess venue fit. A scoped empirical/theory note or technical report may be
   more realistic than TNNLS unless the new benchmark supplies broader evidence.
7. Re-run an external citation metadata audit before submission.
8. Remove any remaining unused legacy paper sections from the submission
   package even if they are not included by the active main file.
9. Add authors and final anonymization metadata only at the appropriate
   submission stage.
10. Perform one final clean-environment build and repository quick-start test.

## Five-dimension self-review

- **Contribution:** Clear and honest, but modest relative to a full TNNLS paper.
- **Writing clarity:** Active sections now use stable terms and separate
  mechanism, diagnostic, and arithmetic claims.
- **Experimental strength:** Direct simulation is strong; real-data decision
  evidence remains incomplete.
- **Evaluation completeness:** Missing prospective selector and degree-matched
  edge reruns are decisive gaps.
- **Method soundness:** Scoped derivation and new audit code are sound under
  stated assumptions; external validity is deliberately unresolved.

## Confidence

High. The review uses the active LaTeX sources, generated PDF, unit tests,
frozen Task 7 records, Task 8 scoring contract, Task 9 source audit, and direct
numerical consistency checks. Confidence about venue acceptance remains lower
because no external reviewers or newly generated real-data benchmark outcomes
are available.
