# Information Budget of Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

Research artifact for **"The Information Budget of Graph Neural Networks: When Features Are Enough"**.

> **Audit status (July 2026):** the paper and repository are being aligned to a single set of traceable result files. Claims not supported by the audited artifacts are marked below instead of being presented as completed validation.

## Key Contributions

1. **SNR Framework**: under sparse-CSBM, local-tree, and feature-noise-dominance assumptions, $\kappa \approx \rho^2 k$ is a leading-order diagnostic.
2. **Information Headroom**: $(1 - \text{Acc}_{\text{MLP}})$ is the arithmetic room for accuracy improvement, not an independent predictive theorem.
3. **ADR Analysis**: reports absolute accuracy loss relative to MLP; near-zero-denominator ratios are intentionally avoided.
4. **Two-Hop Recovery**: $R=h_2/h_1$ diagnoses one recovery mechanism but does not uniquely select the best model family.

## Quick Start

```bash
# Clone repository
git clone https://github.com/MengdanXue/information-budget-gnn.git
cd information-budget-gnn

# Install dependencies
pip install -r requirements.txt

# Download datasets (automatic via PyG)
python scripts/download_datasets.py

# Run experiments
python experiments/h_sweep_experiment.py        # CSBM synthetic experiments
python experiments/cross_model_hsweep.py        # Multi-model comparison
python experiments/real_dataset_validation.py   # Real dataset validation
python experiments/ogb_full_experiments.py      # OGB; audited success currently limited to ogbn-arxiv
```

## Datasets

Datasets are downloaded automatically via PyTorch Geometric. Total ~2GB for standard datasets.

| Dataset | Nodes | Edges | h | Source |
|---------|-------|-------|---|--------|
| Cora | 2,708 | 10,556 | 0.81 | PyG |
| CiteSeer | 3,327 | 9,104 | 0.74 | PyG |
| PubMed | 19,717 | 88,648 | 0.80 | PyG |
| Texas | 183 | 325 | 0.11 | PyG |
| Wisconsin | 251 | 515 | 0.20 | PyG |
| Chameleon | 2,277 | 36,101 | 0.24 | PyG |
| Actor | 7,600 | 30,019 | 0.22 | PyG |
| ogbn-arxiv | 169,343 | 1.17M | 0.66 | OGB |
| ogbn-products | 2.4M | 123M | 0.81 | OGB (current audited run: CUDA OOM) |

To download OGB datasets:
```bash
pip install ogb
python scripts/download_datasets.py --include-ogb
```

## Repository Structure

```
information-budget-gnn/
├── README.md
├── requirements.txt
├── LICENSE
├── scripts/
│   └── download_datasets.py      # Download all datasets
├── experiments/                   # 156 experiment scripts
│   ├── h_sweep_experiment.py     # CSBM homophily sweep
│   ├── cross_model_hsweep.py     # Multi-model comparison
│   ├── ogb_full_experiments.py   # OGB large-scale validation
│   ├── h2gcn_validation.py       # H2GCN heterophily validation
│   ├── heterophily_baselines.py  # LINKX, MixHop, FAGCN
│   ├── csbm_deviation_metrics.py # CSBM deviation analysis
│   └── baselines/                # Baseline implementations
├── src/
│   ├── models/                   # MLP, GCN, GraphSAGE, H2GCN
│   └── metrics/                  # SNR, ADR, homophily
├── results/                      # 91 JSON result files
│   ├── ogb_full_experiments_results.json
│   ├── h2gcn_validation_results.json
│   ├── heterophily_baselines_results.json
│   ├── cross_model_hsweep_results.json
│   └── ...
├── figures/                      # All paper figures (PDF + PNG)
│   ├── u_shape_*.pdf
│   ├── information_budget_*.pdf
│   ├── phase_diagram_*.pdf
│   └── defense/, validation/
└── data/                         # Downloaded automatically
```

## Audited Results

### Information Headroom

The inequality

```text
Acc_GNN - Acc_MLP <= 1 - Acc_MLP
```

is algebraically equivalent to `Acc_GNN <= 1`. The current paper uses it as a headroom diagnostic. On nine traceable external datasets, the correlation between headroom and best-candidate-GNN advantage is `r=-0.255` with a bootstrap interval crossing zero. See `results/budget_shared_term_audit.json` in the audited paper artifact.

### ADR by Architecture at h=0.5

| Model | Type | Accuracy loss vs MLP |
|-------|------|----------------------|
| GCN | Replace | 18.9 percentage points |
| GAT | Replace + attention | 5.3 percentage points |
| GraphSAGE | Concatenate | 0.1 percentage points |

### Two-Hop Recovery Diagnostic

| Dataset | h₁ | h₂ | R | Best model in audited protocol |
|---------|----|----|---|--------------------------------|
| Texas | 0.087 | 0.571 | 6.56 | H2GCN |
| Wisconsin | 0.192 | 0.425 | 2.21 | MLP |
| Cornell | 0.127 | 0.392 | 3.07 | H2GCN |
| Chameleon | 0.231 | 0.213 | 0.92 | GCN |
| Squirrel | 0.222 | 0.197 | 0.88 | GCN |

These results show why `R` must not be interpreted as a complete model-selection rule.

## Usage

### Compute SNR Metrics for Your Dataset

```python
from src.metrics import compute_snr_ratio, compute_adr, classify_heterophily

# Load your graph
data = load_graph("your_dataset")

# Compute metrics
h = compute_homophily(data)
k = compute_avg_degree(data)
kappa = compute_snr_ratio(h, k)  # κ = ρ²k

print(f"Homophily h: {h:.3f}")
print(f"SNR ratio κ: {kappa:.3f}")
print(f"Leading-order CSBM diagnostic: {kappa > 1}")

# For heterophily datasets
if h < 0.3:
    R = compute_2hop_recovery(data)
    het_type = classify_heterophily(R)
    print(f"Heterophily type: {het_type}")
```

### Model Selection Algorithm

```python
from src.decision import select_model

recommendation = select_model(
    h=0.35,        # homophily
    k=12,          # average degree
    mlp_acc=0.82,  # MLP baseline accuracy
    R=2.1          # 2-hop recovery ratio (optional)
)
print(recommendation)
# Treat the output as a screening recommendation, then validate models empirically.
```

## Citation

```bibtex
@article{xue2025information,
  title={The Information Budget of Graph Neural Networks: When Features Are Enough},
  author={Xue, Mengdan},
  journal={Manuscript},
  year={2026},
  note={Under review}
}
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.0+
- OGB (for large-scale experiments)
- NumPy, SciPy, Scikit-learn

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- Mengdan Xue: 17326961775@163.com
- Issues: https://github.com/MengdanXue/information-budget-gnn/issues
