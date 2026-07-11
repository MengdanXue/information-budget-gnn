# Audited Paper Release

This directory contains the minimal, traceable artifacts used by the July 2026 revision of **The Information Budget of Graph Neural Networks: When Features Are Enough**.

## Contents

- `results/external_validation_results.json`: nine-dataset external validation.
- `results/dual_heterophily_results.json`: protocol-specific two-hop recovery and model results.
- `results/ogb_full_experiments_results.json`: successful three-seed ogbn-arxiv run and the recorded ogbn-products OOM failure.
- `results/budget_shared_term_audit.json`: deterministic shared-term correlation audit.
- `scripts/analyze_budget_shared_term.py`: reproduces the correlation audit.
- `scripts/generate_audited_tables.py`: generates LaTeX rows from the audited heterophily JSON.

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
