# Route A Paper Repositioning Design

## Decision

Reposition the manuscript as a scoped aggregation-discriminability analysis plus a controlled negative study of simple structural diagnostics. The paper will not present Information Budget as a new theory, and it will not claim that the historical diagnostic rules solve automatic model selection.

## Research Question

Under explicit, controlled assumptions, when does neighbor aggregation improve or damage feature discriminability, and are simple graph statistics sufficient to predict that change on real datasets?

## Contribution Structure

1. **Scoped exact analysis.** Retain the exact fixed-degree, binary Gaussian, neighbor-only Mahalanobis discriminability ratio, including label-mixture covariance. Extend it to a self-feature-plus-neighbor operator so the analyzed operator is closer to a GCN layer with a self contribution.
2. **Direct theory validation.** Measure empirical class-conditional moments and discriminability in synthetic data while varying degree, edge correlation, feature strength, and self-feature weight. Compare the exact formula with the noise-dominated approximation rather than treating downstream accuracy as a direct theorem test.
3. **Controlled negative diagnostic study.** Evaluate homophily, two-hop recovery, and the historical combined rule against trivial and operational baselines using a single prediction target, paired seeds, regret, coverage, and held-out evaluation. Report failure to add value when that is the result.

## Claims Removed or Downgraded

- Delete the false Structure Information Bound.
- Delete the claim that efficiency lies in $[-1,1]$.
- Treat $1-\mathrm{Acc}_{\mathrm{MLP}}$ only as arithmetic positive-gain headroom.
- Remove 32/36, 7/9, and 12/12 from the Abstract and contribution claims until they are compared fairly with trivial baselines.
- Use Kesten--Stigum only as related context; do not claim a new threshold or architecture-independent oversmoothing law.
- Treat ADR as an absolute descriptive performance difference, not a ratio or theory.
- Treat two-hop recovery as a descriptive statistic, not an automatic selector.
- Replace the degree-preserving causal claim unless a true degree-preserving intervention with checksums is rerun.

## Evidence Policy

Every major Abstract and Introduction claim must map to one of: an exact proposition with explicit assumptions, a reproducible per-seed artifact, or a clearly labeled limitation/negative result. Retrospective aggregate JSON files may motivate experiments but cannot support confirmatory performance claims.

## Success Criteria

- No known false proposition or invalid range claim remains.
- The analyzed operator and experimental operator are explicitly distinguished.
- The prediction target is fixed before rerunning selector experiments.
- Trivial baselines are reported under identical tie and abstention scoring.
- All confirmatory results retain per-seed records and protocol metadata.
- The manuscript compiles cleanly and passes an automated banned-claim audit.

