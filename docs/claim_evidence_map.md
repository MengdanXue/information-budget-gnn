# Route A Claim--Evidence Map

This map is the contract for claims in the active manuscript. A claim may appear in the Abstract as a result only when its status is **supported** and its evidence is available as an auditable proposition or artifact.

| Claim | Evidence | Status | Permitted wording |
|---|---|---|---|
| Fixed-degree neighbor averaging has ratio $\rho^2d/[1+(1-\rho^2)s]$ under the stated binary Gaussian conditional-neighbor model. | Proposition and derivation in `sections/04_snr_theory_medium.tex`. | Supported within assumptions | “Exact moment-based discriminability ratio under the stated fixed-degree model.” |
| The denominator is label-mixture covariance and the $\rho^2d$ expression is a noise-dominated approximation. | Conditional covariance derivation in `sections/04_snr_theory_medium.tex`. | Supported within assumptions | “The approximation can be inaccurate for strong features.” |
| The aggregate distribution generally remains a Gaussian mixture. | Conditional construction in the fixed-degree proposition. | Supported within assumptions | “The ratio is not a Bayes-error formula.” |
| A prescribed self-feature-plus-neighbor operator has the stated $\kappa_\alpha$ expression. | Proposition~`prop:self_neighbor_mixing` and `tests/test_self_feature_mixing.py`, including boundary reductions and a fixed-seed moment simulation. | Supported within assumptions | “Exact for the prescribed linear surrogate,” not “the exact behavior of GCN.” |
| The exact expression matches empirical moments across a frozen parameter grid. | 2,000 immutable records under `results/discriminability/route_a_grid_v1/` and the audited summary/figure; median absolute relative error 0.94% over nonzero exact ratios. | Supported for the prescribed surrogate grid | “Direct moment validation of the scoped surrogate,” not “validation of trained-GNN accuracy.” |
| The historical combined diagnostic improves model-selection regret over simple baselines under the prospective target. | `docs/preregistration_diagnostic_benchmark.md` and `experiments/evaluate_diagnostics.py` define the future test, but no conforming new model records exist. | Pending prospective rerun | Do not report a prospective selector headline; keep the audited historical negative result. |
| Two-hop recovery does not uniquely determine the best model. | Audited 10-seed paired results for six heterophilic datasets; failures on Chameleon and Squirrel. | Supported for the audited protocol | “Insufficient as a stand-alone selector,” not “never useful.” |
| The historical combined rule does not improve over trivial baselines under matched scoring. | `paper_release/results/selector_baseline_audit.json` in the audited release branch: historical rule and always-graph both select correctly on 5/6 datasets. | Supported for the audited six-dataset comparison | “No demonstrated incremental value in the audited comparison.” |
| $1-\mathrm{Acc}_{\mathrm{MLP}}$ bounds positive accuracy gain. | Arithmetic range of accuracy. | Supported but tautological | “Arithmetic positive-gain headroom,” never “information-theoretic predictor.” |
| Positive headroom bounds negative aggregation damage. | None; counterexamples exist. | Rejected | Must not appear. |
| Classification-error improvement is bounded by $I(Y;G\mid X)/\log C$. | None; the stated inequality is false. | Rejected | Delete the Structure Information Bound. |
| Efficiency defined as gain divided by headroom lies in $[-1,1]$. | None; negative values can have magnitude greater than one. | Rejected | Delete the range claim and decomposition. |
| The legacy edge shuffle is degree preserving and isolates topology causally. | Source audit confirms discarded self-loops and collapsed duplicate edges; the tested replacement in `experiments/degree_preserving_edge_randomization.py` has not generated the legacy model outcomes. | Rejected for legacy results; replacement pending rerun | Keep the archived result exploratory. A future degree-matched rerun must report concurrent structural changes and must not claim isolation of homophily. |
| Selector scores 32/36, 7/9, and 12/12 demonstrate predictive value. | Historical scoring gives ties/abstentions favorable treatment and omits matched trivial baselines. | Rejected as incremental evidence | May appear only in an explicitly retrospective audit table with matched baselines and caveats. |

## Abstract outline

1. **Problem:** aggregation can improve or degrade node classification, and homophily alone does not resolve the variation.
2. **Scoped method:** exact fixed-degree moment calculation retaining label-mixture covariance.
3. **Supported finding and limitation:** simple diagnostics describe mechanisms but fail as stand-alone selectors in the audited comparison.

## Introduction reverse outline

1. Define the aggregation-help-or-harm question.
2. Explain why homophily alone omits relevant variables.
3. State the restricted model and exact neighbor-only result.
4. Define aggregation damage and two-hop recovery as descriptive quantities.
5. Demote accuracy headroom to an arithmetic observation.
6. Separate the supported theory, pending direct validation, and supported negative diagnostic study.
