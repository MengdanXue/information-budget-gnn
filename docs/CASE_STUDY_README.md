# Case Study Visualization for FSD Paper

## Overview

This script (`generate_case_study.py`) creates publication-quality visualizations demonstrating how NAA's attention mechanism correctly identifies fraudulent transactions in the Elliptic Bitcoin dataset.

## Generated Visualizations

### 1. Multi-Case Summary (`case_study_elliptic.pdf`)
- Side-by-side comparison of 3-5 fraud cases
- Shows NAA-GCN vs GAT predictions
- Visualizes 1-hop neighborhood for each case
- Node size indicates prediction confidence
- Node color indicates true label (red=fraud, blue=legitimate)

### 2. Attention Comparison (`case_study_attention_comparison.pdf`)
- **NAA Feature Importance**: Top-K features weighted by NAA
- **GAT Neighbor Attention**: Which neighbors GAT focuses on
- **Feature Distribution**: Original vs NAA-weighted features
- **Statistics Panel**: Quantitative comparison of attention mechanisms

### 3. Detailed Neighborhood (`case_study_node_*_neighborhood.pdf`)
- 2-hop neighborhood subgraph for selected fraud node
- NAA vs GAT predictions side-by-side
- Confidence scores for all nodes
- Highlights why NAA makes correct predictions

## Key Insights Demonstrated

1. **Feature-level vs Neighbor-level Attention**
   - NAA focuses on discriminative FEATURES
   - GAT focuses on NEIGHBOR relationships
   - In fraud detection, features often matter more than graph structure

2. **Homophily vs Heterophily**
   - NAA adapts to low homophily (fraud mixed with legitimate)
   - GAT assumes high homophily (similar nodes connected)
   - Elliptic has low feature homophily (ρ_FS ≈ 0.31)

3. **Interpretability**
   - NAA: "This transaction is fraud because features X, Y, Z are unusual"
   - GAT: "This transaction is fraud because neighbors A, B, C are suspicious"
   - NAA provides more actionable insights for analysts

## Usage

### Basic Usage

```bash
python generate_case_study.py \
    --data_dir ./data \
    --output_dir ../figures \
    --num_cases 3
```

### With Pre-trained Models

If you already have trained models:

```bash
python generate_case_study.py \
    --data_dir ./data \
    --output_dir ../figures \
    --num_cases 5 \
    --skip_training
```

### Full Training

```bash
python generate_case_study.py \
    --data_dir ./data \
    --output_dir ../figures \
    --num_cases 3 \
    --device cuda
```

## Prerequisites

### 1. Elliptic Dataset

Download from: https://www.kaggle.com/ellipticco/elliptic-data-set

Place the following files in `./data/`:
- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

### 2. Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch >= 1.10.0`
- `torch-geometric >= 2.0.0`
- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `matplotlib >= 3.5.0`
- `seaborn >= 0.11.0`
- `scikit-learn >= 1.0.0`
- `networkx >= 2.6.0`

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | `./data` | Directory containing Elliptic CSV files |
| `--output_dir` | str | `../figures` | Directory to save generated figures |
| `--num_cases` | int | 3 | Number of fraud cases to visualize |
| `--device` | str | `cuda` | Device to use (`cuda` or `cpu`) |
| `--skip_training` | flag | False | Load pre-trained models instead of training |

## Output Files

All files are saved in `{output_dir}`:

1. **case_study_elliptic.pdf**
   - Main figure for paper
   - Multi-case comparison
   - Publication-ready quality

2. **case_study_attention_comparison.pdf**
   - Detailed attention mechanism analysis
   - Feature importance vs neighbor attention
   - Statistical comparisons

3. **case_study_node_{id}_neighborhood.pdf**
   - Per-node detailed analysis
   - 2-hop neighborhood visualization
   - NAA vs GAT side-by-side

4. **Model checkpoints** (if training):
   - `naa_gcn_elliptic.pt`
   - `gat_elliptic.pt`

## Expected Runtime

- **With training**: ~10-20 minutes (200 epochs, early stopping)
- **Without training**: ~2-3 minutes (inference + visualization only)
- **GPU recommended**: Training is 5-10x faster on GPU

## Troubleshooting

### "No high-confidence fraud cases found"

Lower the confidence threshold in the code:
```python
selected_nodes, node_info = select_fraud_cases(
    model_naa, data, num_cases=5,
    confidence_threshold=0.85,  # Lower from 0.9
    device=device
)
```

### CUDA out of memory

Reduce batch size or use CPU:
```bash
python generate_case_study.py --device cpu
```

### Missing Elliptic data

The script will automatically process raw CSV files if `elliptic_weber_split.pkl` doesn't exist. Ensure you have the three CSV files in `--data_dir`.

## Customization

### Change number of hops in neighborhood

Edit `visualize_neighborhood_subgraph` call:
```python
fig = visualize_neighborhood_subgraph(
    node_id, data, model_naa, model_gat,
    num_hops=3,  # Change from 2 to 3
    device=device
)
```

### Select different fraud cases

Modify selection criteria in `select_fraud_cases`:
```python
# Select cases with specific characteristics
correct_high_conf = (
    (fraud_preds == 1) &
    (fraud_probs >= 0.85) &
    (degree_mask)  # Add degree constraints
)
```

### Adjust figure aesthetics

Modify matplotlib parameters at the end of the script:
```python
plt.rcParams['font.size'] = 12  # Increase font size
plt.rcParams['figure.dpi'] = 150  # Higher resolution
```

## Paper Integration

### Main Text

Use `case_study_elliptic.pdf` in the "Case Study" section:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/case_study_elliptic.pdf}
    \caption{Case study showing NAA-GCN correctly identifying
             fraudulent Bitcoin transactions. NAA achieves higher
             confidence than GAT by focusing on discriminative features
             rather than neighbor relationships.}
    \label{fig:case_study}
\end{figure}
```

### Supplementary Material

Use `case_study_attention_comparison.pdf` for detailed analysis:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/case_study_attention_comparison.pdf}
    \caption{Detailed attention mechanism comparison. Top: NAA feature
             importance weights highlight discriminative features. Middle:
             GAT attention focuses on neighbor relationships. Bottom:
             NAA's weighted features show stronger discrimination.}
    \label{fig:attention_comparison}
\end{figure}
```

## Citation

If you use this visualization code, please cite:

```bibtex
@article{fsd2024,
  title={Feature Set Dilution: A Unified Framework for
         Understanding GNN Performance in Fraud Detection},
  author={Your Name et al.},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024}
}
```

## Contact

For questions or issues, please contact:
- Email: [your-email]
- GitHub: [your-repo]

## License

This code is part of the FSD framework research project.
