# Ablation Study Quick Start Guide

## TL;DR

Run the full ablation study on all datasets in one command:

**Windows:**
```bash
run_ablation_study.bat
```

**Linux/Mac:**
```bash
bash run_ablation_study.sh
```

## Prerequisites

1. **Python 3.8+** with required packages:
   ```bash
   pip install torch torch_geometric scikit-learn scipy numpy
   ```

2. **Processed datasets** in `processed/` directory:
   - `elliptic_graph.pkl`
   - `ieee_cis_graph.pkl`
   - `yelpchi_graph.pkl`
   - `amazon_graph.pkl`

3. **GPU recommended** (but not required):
   - With GPU: ~2-3 hours for full ablation
   - CPU only: ~6-8 hours for full ablation

## Single Dataset Example

Run ablation on just Elliptic dataset:

```bash
python ablation_study.py \
    --data_path processed/elliptic_graph.pkl \
    --dataset_name "Elliptic" \
    --output_dir ablation_results/elliptic
```

## Understanding the Output

After running, you'll get three files per dataset:

### 1. `ablation_table.tex`
Ready-to-use LaTeX table for your paper:
```latex
\begin{table}[t]
...
NAA (Full) & \textbf{0.802 ± 0.015} & ...
NAA w/o Log-scale & 0.785 ± 0.019 & ...
...
\end{table}
```

### 2. `lambda_sensitivity_table.tex`
LaTeX table showing λ sensitivity:
```latex
\begin{table}[t]
...
0.50 & \textbf{0.802 ± 0.015} & ...
...
\end{table}
```

### 3. `ablation_results.json`
Complete results for further analysis:
```json
{
  "ablation": {
    "NAA (Full)": {
      "auc_mean": 0.802,
      "auc_std": 0.015,
      ...
    }
  },
  "significance_tests": {...}
}
```

## Customization

### Use More Seeds for Robustness

Default uses 5 seeds. For publication-quality results, use 10:

```bash
python ablation_study.py \
    --data_path processed/elliptic_graph.pkl \
    --dataset_name "Elliptic" \
    --seeds 42 123 456 789 1024 2048 3072 4096 5120 6144 \
    --output_dir ablation_results/elliptic_10seeds
```

### CPU-Only Mode

If no GPU available:

```bash
python ablation_study.py \
    --data_path processed/elliptic_graph.pkl \
    --dataset_name "Elliptic" \
    --device cpu \
    --output_dir ablation_results/elliptic
```

## Quick Results Check

After running, check results summary:

```bash
# View JSON summary
python -m json.tool ablation_results/elliptic/ablation_results.json

# Or just look at the printed output in terminal
```

## Integrating into Your Paper

1. Copy the LaTeX tables to your paper:
   ```bash
   cp ablation_results/elliptic/ablation_table.tex paper/tables/
   ```

2. Include in your LaTeX:
   ```latex
   \input{tables/ablation_table}
   ```

3. Reference in text:
   ```latex
   Table~\ref{tab:ablation_naa} shows that all three NAA components
   contribute to performance, with adaptive gating providing the
   largest improvement (\Delta AUC = 0.033, p < 0.001).
   ```

## Troubleshooting

### Problem: Out of memory

**Solution 1**: Reduce batch size (edit line ~22 in ablation_study.py)
```python
hidden_dim=64  # instead of 128
```

**Solution 2**: Use CPU
```bash
--device cpu
```

### Problem: Takes too long

**Solution**: Use fewer seeds
```bash
--seeds 42 123 456  # just 3 seeds
```

### Problem: Results look wrong

**Solution**: Check data preprocessing
```python
# In Python:
import pickle
with open('processed/elliptic_graph.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"Nodes: {data['features'].shape[0]}")
print(f"Features: {data['features'].shape[1]}")
print(f"Edges: {data['edge_index'].shape[1]}")
```

## Expected Results Preview

### Elliptic
- Baseline GCN: ~0.650 AUC
- NAA (Full): ~0.802 AUC (+15.2% improvement)
- Best λ: ~0.50

### IEEE-CIS
- Baseline GCN: ~0.687 AUC
- NAA (Full): ~0.749 AUC (+6.2% improvement)
- Best λ: ~0.50

### Key Insights
1. All three components are important
2. Log-scale normalization critical for raw numerical features
3. Feature weights help in high-dimensional settings
4. Optimal λ around 0.5 for most datasets

## Time Estimates

| Dataset | Nodes | Features | Time (5 seeds) | Time (10 seeds) |
|---------|-------|----------|----------------|-----------------|
| Elliptic | 46K | 165 | ~60 min | ~120 min |
| IEEE-CIS | 144K | 394 | ~90 min | ~180 min |
| YelpChi | 45K | 32 | ~45 min | ~90 min |
| Amazon | 11K | 767 | ~50 min | ~100 min |

**Total**: ~4-5 hours (5 seeds) or ~8-10 hours (10 seeds)

## What Gets Tested

The ablation study runs:

1. **5 model variants**:
   - Baseline GCN
   - NAA (Full)
   - NAA w/o Log-scale
   - NAA w/o Feature Weights
   - NAA w/o Adaptive Gate

2. **5 λ values**:
   - λ = 0.0 (pure structure)
   - λ = 0.25
   - λ = 0.5
   - λ = 0.75
   - λ = 1.0 (pure features)

3. **Multiple seeds** (5 or 10) for each configuration

**Total experiments per dataset**: 10 configs × N seeds

## Advanced: Python API

You can also import and use directly in Python:

```python
from ablation_study import run_ablation_experiment, load_data

# Load data
data = load_data('processed/elliptic_graph.pkl')

# Define custom configs
configs = {
    'My Custom NAA': {
        'type': 'naa',
        'use_log_scale': True,
        'use_feature_weights': True,
        'use_adaptive_gate': True,
        'fixed_lambda': 0.3  # Try λ=0.3
    }
}

# Run experiments
results = run_ablation_experiment(
    configs=configs,
    data=data,
    seeds=[42, 123, 456],
    device='cuda'
)

print(results)
```

## Citation

If you use this ablation framework:

```bibtex
@article{fsd-gnn-2024,
  title={FSD-GNN: Feature-Structure Disentanglement for Fraud Detection},
  author={[Your Name]},
  year={2024}
}
```

## Support

For issues:
1. Check the full README: `ABLATION_STUDY_README.md`
2. Review code comments in `ablation_study.py`
3. Check example results in this guide
