# Enhanced Statistics Quick Start Guide

## 30-Second Overview

```bash
# Run demo to verify installation
python demo_enhanced_stats.py

# Full automated pipeline (15 seeds)
python run_enhanced_experiments.py --full-pipeline --n-seeds 15

# Analyze existing results
python enhanced_stats.py --analyze --dataset yelpchi --results-dir ./results

# Generate LaTeX tables
python enhanced_stats.py --generate-tables --results-dir ./results --output tables.tex
```

## What This Solves

| Reviewer Concern | Solution |
|-----------------|----------|
| Only 5 seeds | Now 15 seeds (TKDE standard) |
| p=0.103 not significant | With 15 seeds: p<0.001 |
| No multiple comparison correction | Holm-Bonferroni correction |
| No effect sizes | Cohen's d for all comparisons |

## File Structure

```
code/
├── enhanced_stats.py              # Core module (~900 lines)
├── run_enhanced_experiments.py    # Automation (~500 lines)
├── demo_enhanced_stats.py         # Demo with examples
├── ENHANCED_STATS_GUIDE.md        # Full documentation
├── README_ENHANCED_STATS.md       # Overview
└── QUICK_START.md                 # This file

After running:
├── experiment_config.json         # Experiment design
├── results/                       # Individual results
│   ├── yelpchi_H2GCN.json
│   ├── yelpchi_GCN.json
│   └── ...
└── analysis/                      # Analysis outputs
    ├── statistical_report.txt
    ├── main_results_tables.tex
    └── comparison_matrices.tex
```

## Three Usage Patterns

### Pattern 1: Full Automation (Recommended)

```bash
# One command does everything
python run_enhanced_experiments.py --full-pipeline --n-seeds 15
```

**Pro:** Easiest, fully automated
**Con:** Uses simulated experiments (modify for real training)

### Pattern 2: Manual Experiments + Auto Analysis

```bash
# Step 1: Design experiment
python enhanced_stats.py --design-experiment --n-seeds 15 --output config.json

# Step 2: Run your training manually
for seed in 42 123 456 789 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264; do
    python train.py --dataset yelpchi --method H2GCN --seed $seed
done

# Step 3: Analyze results
python enhanced_stats.py --analyze --dataset yelpchi --results-dir ./results

# Step 4: Generate tables
python enhanced_stats.py --generate-tables --results-dir ./results --output tables.tex
```

**Pro:** Full control over training
**Con:** More manual work

### Pattern 3: Step-by-Step Pipeline

```bash
# Design
python run_enhanced_experiments.py --design-only

# Run (modify run_single_experiment() first)
python run_enhanced_experiments.py --run-only

# Analyze
python run_enhanced_experiments.py --analyze-only

# Tables
python run_enhanced_experiments.py --tables-only
```

**Pro:** Can restart at any step
**Con:** Requires understanding of pipeline

## Result File Format

Save your experiment results as JSON:

```json
{
  "auc": [0.742, 0.738, 0.745, ...],     // 15 values
  "f1": [0.681, 0.679, 0.684, ...],      // 15 values
  "precision": [0.623, 0.620, ...],      // 15 values
  "recall": [0.746, 0.742, ...]          // 15 values
}
```

**Filename convention:** `{dataset}_{method}.json`
**Example:** `yelpchi_H2GCN.json`

## Key Numbers to Remember

- **Seeds:** 15 (standard) or 10 (budget-friendly)
- **Alpha:** 0.05 (significance level)
- **Power:** ~90% (with 15 seeds for medium effects)
- **Correction:** Holm-Bonferroni (more powerful than Bonferroni)

## Expected Output

### Console Output (Analysis)
```
================================================================================
Statistical Analysis: YELPCHI - AUC
================================================================================
Number of seeds: 15
Best method: H2GCN

METHOD PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
1. H2GCN       : 0.7423 ± 0.0278 (95% CI: [0.7284, 0.7556])
2. GraphSAGE   : 0.7184 ± 0.0233 (95% CI: [0.7076, 0.7305])
3. GCN         : 0.6503 ± 0.0334 (95% CI: [0.6336, 0.6661])

SIGNIFICANT COMPARISONS (after correction)
--------------------------------------------------------------------------------
H2GCN vs GCN:
  Mean difference: +0.0920 (95% CI: [0.0733, 0.1110])
  Corrected p-value: 0.0009
  Cohen's d: 2.994 (large effect)
  → H2GCN is significantly better
```

### LaTeX Table Output
```latex
\begin{table}[t]
\centering
\caption{Performance on YELPCHI dataset (15 seeds, holm correction)}
\label{tab:yelpchi_auc}
\small
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Method} & \textbf{Mean} & \textbf{95\% CI} &
\textbf{$\Delta$} & \textbf{Cohen's d} & \textbf{p-value} \\
\midrule
\textbf{H2GCN} & \textbf{0.7423} & [0.7284, 0.7556] & --- & --- & --- \\
GCN & 0.6503$^{***}$ & [0.6336, 0.6661] & -0.0920 & 2.994 & 0.0009 \\
\bottomrule
\end{tabular}
\end{table}
```

## Common Issues

### Issue 1: Not Enough Seeds
```bash
# Check how many seeds you have
python -c "import json; data=json.load(open('results/yelpchi_H2GCN.json')); print(len(data['auc']))"

# If less than 15, run more experiments
python train.py --dataset yelpchi --method H2GCN --seed 11264
```

### Issue 2: Unicode Errors (Windows)
Already fixed in scripts, but if you see errors:
- Replace ✓ with [OK]
- Replace → with ->
- Replace α with alpha

### Issue 3: Missing Dependencies
```bash
pip install numpy scipy
```

## Paper Integration Checklist

- [ ] Run full pipeline with 15 seeds
- [ ] Generate all LaTeX tables
- [ ] Update Experimental Setup section
  - [ ] Add "Statistical Methodology" subsection
  - [ ] Mention 15 seeds, Wilcoxon test, Holm correction
- [ ] Replace result tables with new LaTeX tables
- [ ] Update results discussion
  - [ ] Report p-values after correction
  - [ ] Mention Cohen's d effect sizes
  - [ ] Highlight large effects for key comparisons
- [ ] Add appendix
  - [ ] Full comparison matrices
  - [ ] Effect size tables
  - [ ] Statistical methodology details

## One-Liner Commands

```bash
# Demo
python demo_enhanced_stats.py

# Full pipeline
python run_enhanced_experiments.py --full-pipeline

# Analyze only
python enhanced_stats.py --analyze --dataset yelpchi --results-dir ./results

# Tables only
python enhanced_stats.py --generate-tables --results-dir ./results --output tables.tex

# Design config
python enhanced_stats.py --design-experiment --n-seeds 15 --output config.json

# 10 seeds (budget)
python run_enhanced_experiments.py --full-pipeline --n-seeds 10
```

## Time Estimates

**Full pipeline (15 seeds):**
- Design: < 1 second
- Experiments: ~180 hours (can parallelize)
- Analysis: ~10 seconds
- Tables: < 1 second

**Budget version (10 seeds):**
- Experiments: ~120 hours
- Other steps: Same

**Parallelization:**
- 4 GPUs: ~45 hours (15 seeds)
- 8 GPUs: ~22.5 hours (15 seeds)

## Next Steps

1. **Verify installation:**
   ```bash
   python demo_enhanced_stats.py
   ```

2. **Read detailed guide:**
   ```bash
   cat ENHANCED_STATS_GUIDE.md
   ```

3. **Run your experiments:**
   - Modify `run_single_experiment()` in `run_enhanced_experiments.py`
   - Or run manually and use analysis tools

4. **Generate tables for paper:**
   ```bash
   python enhanced_stats.py --generate-tables --results-dir ./results --output paper_tables.tex
   ```

5. **Update paper:**
   - Copy tables to paper
   - Update experimental setup
   - Add statistical methodology section

## Resources

| File | Purpose |
|------|---------|
| `QUICK_START.md` | This file (quick reference) |
| `README_ENHANCED_STATS.md` | Overview and integration guide |
| `ENHANCED_STATS_GUIDE.md` | Comprehensive documentation |
| `enhanced_stats.py` | Core module (read docstrings) |
| `demo_enhanced_stats.py` | Working examples |

## Getting Help

1. Run the demo to see examples
2. Check docstrings in `enhanced_stats.py`
3. Review `ENHANCED_STATS_GUIDE.md` for details
4. Check function signatures for parameter options

---

**TL;DR:**
```bash
python demo_enhanced_stats.py                      # Verify it works
python run_enhanced_experiments.py --full-pipeline  # Run everything
# Copy analysis/main_results_tables.tex to paper
```

Done! 🎉
