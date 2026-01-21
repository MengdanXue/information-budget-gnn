# NAA Ablation Study Guide

## Overview

This document describes the systematic ablation study for the **Numerical-Aware Attention (NAA)** mechanism in the FSD-GNN paper. The ablation study validates the contribution of each component through controlled experiments.

## NAA Mechanism Components

The NAA mechanism consists of three key components:

### 1. Log-scale Normalization
```
x̃ = sign(x) · log(1 + |x|)
```
- **Purpose**: Compress large numerical values to prevent dominance
- **Benefit**: Handles fraud detection features with extreme ranges (e.g., transaction amounts)

### 2. Feature Importance Weights
```
w_i = sigmoid(θ_i)  (learnable per-feature weights)
x_weighted = x̃ · w_i
```
- **Purpose**: Learn which features are most predictive
- **Benefit**: Automatically downweight noisy features, upweight informative ones

### 3. Adaptive Gating
```
h = λ · h_feat + (1 - λ) · h_struct
```
- **Purpose**: Balance feature-based (MLP) and structure-based (GCN) representations
- **Benefit**: Adapts to dataset characteristics (feature quality vs graph quality)

## Ablation Experiments

### Experiment A1: NAA vs NAA w/o Log-scale
**Question**: Does log-scale normalization help?

**Hypothesis**: Yes, especially for datasets with extreme numerical ranges.

**Expected Outcome**:
- Datasets with normalized features (e.g., Elliptic): small improvement
- Datasets with raw numerical values (e.g., IEEE-CIS): significant improvement

### Experiment A2: NAA vs NAA w/o Feature Weights
**Question**: Do learnable feature importance weights help?

**Hypothesis**: Yes, especially for high-dimensional datasets with noisy features.

**Expected Outcome**:
- High-dim datasets (e.g., Amazon, IEEE-CIS): significant improvement
- Low-dim datasets (e.g., YelpChi): smaller improvement

### Experiment A3: NAA vs NAA w/o Adaptive Gate
**Question**: Is adaptive gating necessary?

**Hypothesis**: Yes, fixed equal weighting (0.5/0.5) is suboptimal.

**Expected Outcome**:
- Datasets with good graph structure: gate learns to favor structure
- Datasets with poor graph structure: gate learns to favor features

### Experiment A4: Lambda Sensitivity Analysis
**Question**: How sensitive is performance to λ?

**Test Values**: λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
- λ = 0.0: Pure structure (GCN only)
- λ = 0.5: Balanced
- λ = 1.0: Pure features (MLP only)

**Expected Outcome**:
- Optimal λ varies by dataset
- Demonstrates that neither pure GCN nor pure MLP is optimal

## Usage

### Basic Usage

```bash
# Run ablation study on Elliptic dataset
python ablation_study.py \
    --data_path ./processed/elliptic_graph.pkl \
    --dataset_name "Elliptic" \
    --output_dir ./ablation_results/elliptic

# Run on IEEE-CIS dataset
python ablation_study.py \
    --data_path ./processed/ieee_cis_graph.pkl \
    --dataset_name "IEEE-CIS" \
    --output_dir ./ablation_results/ieee_cis
```

### Advanced Usage

```bash
# Custom seeds for more robust results
python ablation_study.py \
    --data_path ./processed/elliptic_graph.pkl \
    --dataset_name "Elliptic" \
    --seeds 42 123 456 789 1024 2048 3072 4096 5120 6144 \
    --output_dir ./ablation_results/elliptic_10seeds

# Use CPU if no GPU available
python ablation_study.py \
    --data_path ./processed/elliptic_graph.pkl \
    --dataset_name "Elliptic" \
    --device cpu \
    --output_dir ./ablation_results/elliptic_cpu
```

## Output Files

The script generates three main outputs:

### 1. `ablation_table.tex`
LaTeX table showing component ablation results:

```latex
\begin{table}[t]
\centering
\caption{Ablation Study: NAA Component Analysis on Elliptic}
\label{tab:ablation_naa}
\begin{tabular}{lccccc}
\toprule
\textbf{Model Variant} & \textbf{AUC-ROC} & \textbf{F1 Score} & ...
\midrule
Baseline GCN & 0.650 ± 0.023 & 0.612 ± 0.018 & ...
NAA (Full) & \textbf{0.802 ± 0.015} & \textbf{0.758 ± 0.012} & ...
NAA w/o Log-scale & 0.785 ± 0.019 & 0.741 ± 0.015 & ...
...
\bottomrule
\end{tabular}
\end{table}
```

### 2. `lambda_sensitivity_table.tex`
LaTeX table showing λ sensitivity analysis:

```latex
\begin{table}[t]
\centering
\caption{Sensitivity Analysis: Impact of $\lambda$ on Elliptic}
\label{tab:lambda_sensitivity}
\begin{tabular}{lcccc}
\toprule
\textbf{$\lambda$ Value} & \textbf{AUC-ROC} & \textbf{F1 Score} & ...
\midrule
0.00 & 0.650 ± 0.023 & 0.612 ± 0.018 & ...
0.25 & 0.765 ± 0.017 & 0.721 ± 0.014 & ...
0.50 & \textbf{0.802 ± 0.015} & \textbf{0.758 ± 0.012} & ...
...
\bottomrule
\end{tabular}
\end{table}
```

### 3. `ablation_results.json`
Complete results in JSON format for further analysis:

```json
{
  "dataset": "Elliptic",
  "seeds": [42, 123, 456, 789, 1024],
  "ablation": {
    "NAA (Full)": {
      "auc_mean": 0.802,
      "auc_std": 0.015,
      "f1_mean": 0.758,
      ...
    },
    ...
  },
  "lambda_sensitivity": {...},
  "significance_tests": {...}
}
```

## Expected Runtime

For a typical fraud detection dataset (50k nodes, 200k edges, 300 features):

- Single configuration, single seed: ~2-3 minutes on GPU
- Full ablation (5 configs × 5 seeds): ~50-75 minutes on GPU
- Lambda sensitivity (5 configs × 5 seeds): ~50-75 minutes on GPU
- **Total**: ~2-3 hours on GPU for complete ablation study

## Interpreting Results

### Component Importance Ranking

1. **Critical** (Δ AUC > 0.03): Component is essential
2. **Important** (0.01 < Δ AUC ≤ 0.03): Component provides significant benefit
3. **Helpful** (0.005 < Δ AUC ≤ 0.01): Component provides marginal benefit
4. **Negligible** (Δ AUC ≤ 0.005): Component has minimal impact

### Statistical Significance

The script reports paired t-tests comparing each variant to NAA (Full):
- **p < 0.001**: Highly significant (marked ***)
- **p < 0.01**: Significant (marked **)
- **p < 0.05**: Marginally significant (marked *)
- **p ≥ 0.05**: Not significant (marked n.s.)

### Lambda Interpretation

- **λ < 0.3**: Graph structure is high quality, rely more on GCN
- **0.3 ≤ λ ≤ 0.7**: Balanced approach, both features and structure are useful
- **λ > 0.7**: Features are more informative than graph structure

## Example Results

### Elliptic Dataset (Expected)

```
Component Ablation:
- Baseline GCN:              0.650 ± 0.023  (baseline)
- NAA (Full):                0.802 ± 0.015  (+15.2% improvement)
- NAA w/o Log-scale:         0.785 ± 0.019  (-0.017, significant)
- NAA w/o Feature Weights:   0.774 ± 0.021  (-0.028, significant)
- NAA w/o Adaptive Gate:     0.769 ± 0.018  (-0.033, significant)

Lambda Sensitivity:
- λ = 0.0 (pure structure):  0.650 ± 0.023
- λ = 0.25:                  0.765 ± 0.017
- λ = 0.50:                  0.802 ± 0.015  (optimal)
- λ = 0.75:                  0.791 ± 0.016
- λ = 1.0 (pure features):   0.754 ± 0.019

Key Finding: All three components are important for Elliptic.
Optimal λ ≈ 0.5 suggests balanced feature-structure approach.
```

### IEEE-CIS Dataset (Expected)

```
Component Ablation:
- Baseline GCN:              0.687 ± 0.028
- NAA (Full):                0.749 ± 0.021  (+6.2% improvement)
- NAA w/o Log-scale:         0.712 ± 0.025  (-0.037, highly significant)
- NAA w/o Feature Weights:   0.728 ± 0.023  (-0.021, significant)
- NAA w/o Adaptive Gate:     0.734 ± 0.022  (-0.015, significant)

Lambda Sensitivity:
- λ = 0.0 (pure structure):  0.687 ± 0.028
- λ = 0.25:                  0.718 ± 0.024
- λ = 0.50:                  0.749 ± 0.021  (optimal)
- λ = 0.75:                  0.741 ± 0.023
- λ = 1.0 (pure features):   0.698 ± 0.026

Key Finding: Log-scale normalization is critical for IEEE-CIS (raw numerical features).
Optimal λ ≈ 0.5, but slightly favoring features would also work.
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution**:
```bash
# Reduce hidden dimension
# Edit ablation_study.py, line ~22:
# Change hidden_dim=128 to hidden_dim=64

# Or use CPU
python ablation_study.py --device cpu ...
```

### Issue: Experiments take too long
**Solution**:
```bash
# Use fewer seeds
python ablation_study.py --seeds 42 123 456 ...
```

### Issue: Results are unstable
**Solution**:
```bash
# Use more seeds for robust results
python ablation_study.py --seeds 42 123 456 789 1024 2048 3072 4096 5120 6144 ...
```

## Paper Integration

### Recommended Paper Structure

**Section 5.3: Ablation Study**

1. **Component Analysis** (Table from `ablation_table.tex`)
   - Show full ablation results
   - Discuss importance of each component
   - Report statistical significance

2. **Lambda Sensitivity** (Table from `lambda_sensitivity_table.tex`)
   - Show performance across λ values
   - Discuss optimal λ for different datasets
   - Connect to FSD theory (dilution affects optimal λ)

3. **Key Insights**
   - All three components contribute to NAA's effectiveness
   - Log-scale normalization is critical for raw numerical features
   - Feature weights help in high-dimensional settings
   - Adaptive gating learns dataset-specific balance

### Example Text for Paper

```latex
\subsection{Ablation Study}

To validate the contribution of each NAA component, we conduct systematic
ablation experiments. Table~\ref{tab:ablation_naa} shows results on the
Elliptic dataset.

\textbf{Component Importance.} Removing any single component degrades
performance, demonstrating that all three are necessary:
(1) Log-scale normalization improves AUC by 1.7\% (p < 0.01),
(2) Feature importance weights contribute 2.8\% (p < 0.001),
(3) Adaptive gating adds 3.3\% (p < 0.001).

\textbf{Lambda Sensitivity.} Table~\ref{tab:lambda_sensitivity} shows
performance across different $\lambda$ values. The optimal value
$\lambda \approx 0.5$ indicates that both feature-based and structure-based
representations are important for Elliptic. Pure structure ($\lambda = 0$)
matches baseline GCN, while pure features ($\lambda = 1$) underperform,
confirming the benefit of graph structure.
```

## Related Files

- `daaa_model.py`: Main DAAA model implementation (uses NAA components)
- `train_ieee_cis.py`: Training script for experiments
- `ablation_delta_agg.py`: Ablation for δ_agg metric
- `generate_latex_tables.py`: General LaTeX table generation

## Citation

If you use this ablation study framework, please cite:

```bibtex
@article{fsd-gnn-2024,
  title={FSD-GNN: Feature-Structure Disentanglement for Fraud Detection on Graphs},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2024}
}
```

## Contact

For questions or issues with the ablation study:
- Check the code comments in `ablation_study.py`
- Review this README
- Contact: [your-email@example.com]
