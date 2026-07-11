# Audited Paper Release

This directory contains the minimal, traceable artifacts used by the July 2026 paper revision. The paper is now framed around aggregation SNR dynamics, aggregation damage, and heterophily diagnostics; Information Budget is retained only as arithmetic accuracy headroom.

## Contents

- `results/external_validation_results.json`: nine-dataset external validation.
- `results/dual_heterophily_results.json`: protocol-specific two-hop recovery and model results.
- `results/ogb_full_experiments_results.json`: successful three-seed ogbn-arxiv run and the recorded ogbn-products OOM failure.
- `results/budget_shared_term_audit.json`: deterministic shared-term correlation audit.
- `results/dual_heterophily_10seed_results.json`: raw paired results for six datasets, five models, and seeds 0--9.
- `results/dual_heterophily_10seed_summary.json`: means, sample standard deviations, paired differences, and bootstrap 95% intervals.
- `scripts/analyze_budget_shared_term.py`: reproduces the correlation audit.
- `scripts/generate_audited_tables.py`: generates LaTeX rows from the audited heterophily JSON.
- `scripts/summarize_multiseed_results.py`: validates seed pairing and regenerates the ten-seed statistical summary and LaTeX rows.

## Reproduce the Statistical Audit

From the repository root:

```bash
python paper_release/scripts/analyze_budget_shared_term.py
```

Expected headline values:

- observed Pearson `r = -0.255`;
- bootstrap 95% CI `[-0.743, 0.523]`;
- shared-term permutation-null mean `r = 0.669`.

## Generate Audited Table Rows

```bash
python paper_release/scripts/generate_audited_tables.py
```

Generated files are written to `paper_release/tables/generated/`.

## Scope

This release does not claim a successful ogbn-products experiment. It also does not reproduce legacy significance tests whose paired per-seed samples are absent. See the root `REPRODUCIBILITY_STATUS.md` for details.

The two-hop recovery ratio is a descriptive structural statistic, not an automatic architecture selector. The included paired ten-seed rerun shows that strong recovery does not consistently imply that H2GCN beats MLP, while Chameleon and Squirrel benefit from graph models despite absent measured two-hop recovery.

## Reproduce the Ten-Seed Summary

```bash
python paper_release/scripts/summarize_multiseed_results.py \
  paper_release/results/dual_heterophily_10seed_results.json \
  --summary paper_release/results/dual_heterophily_10seed_summary.json \
  --latex paper_release/tables/generated/dual_heterophily_10seed_rows.tex
```
