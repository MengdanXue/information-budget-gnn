# Reproducibility Status

Updated: 2026-07-11

## Verified

- The repository URL in the supplementary PDF resolves to this public repository.
- The audited `ogbn-arxiv` result contains three seeds and official splits.
- The two-hop recovery table is generated from one result source in the paper artifact.
- The shared-term Budget audit uses 10,000 permutations and 10,000 bootstrap samples with a fixed seed.

## Not Yet Verified

- The current `ogbn-products` run terminated with CUDA out-of-memory and is not evidence of a successful experiment.
- Several legacy aggregate result files do not retain paired per-seed samples.
- Legacy Wilcoxon, Friedman, and Mann--Whitney values should not be cited until their underlying samples are restored or rerun.
- Historical result directories contain multiple protocols and must not be combined without recording split, seed, preprocessing, and model configuration.

## Required Result Metadata

Every new experiment should save:

- dataset version and preprocessing;
- train/validation/test split identifier;
- random seed;
- model and optimizer configuration;
- per-seed validation and test metrics;
- runtime and hardware;
- source commit hash.

## Release Policy

Only results referenced by the paper should appear in the release manifest. Historical or exploratory files should be retained in an archive rather than presented as paper evidence.
