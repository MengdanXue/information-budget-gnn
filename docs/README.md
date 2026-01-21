# Trust Regions of Graph Propagation

Code for the paper: **"Trust Regions of Graph Propagation: When to Use GNNs and When Not To"**

## Overview

This repository provides code for:
1. **SPI (Structural Predictability Index)** computation: `SPI = |2h - 1|`
2. **U-Shape Pattern** discovery through controlled H-Sweep experiments
3. **Feature-Pattern Duality** analysis via semi-synthetic experiments
4. **Cross-model validation** (GCN, GAT, GraphSAGE)
5. **Statistical significance tests** with comprehensive diagnostics

## Key Findings

- **U-Shape Pattern**: In synthetic experiments, GNN advantage follows a U-shape across homophily spectrum
  - GCN wins at both h < 0.3 and h > 0.7
  - GCN loses up to 18% at h ≈ 0.5
- **Feature-Pattern Duality**: Real features produce monotonic (not U-shaped) patterns
  - U-shape is a synthetic artifact
  - Real heterophilic neighbors are feature-orthogonal (noise), not feature-opposite (signal)
- **Trust Region**: SPI > 0.4 with h > 0.5 achieves reliable GNN advantage
- **GraphSAGE Robustness**: Sampling-based aggregation shows minimal U-shape amplitude (0.7% vs GCN's 18.9%)

## Installation

```bash
pip install -r requirements.txt
```

Or install dependencies directly:
```bash
pip install torch torch-geometric numpy scipy scikit-learn matplotlib pandas tqdm seaborn networkx
```

## Quick Start

### 1. Compute SPI for a dataset

```python
import torch

def compute_spi(edge_index, labels):
    """Compute Structural Predictability Index"""
    src, dst = edge_index
    h = (labels[src] == labels[dst]).float().mean().item()
    spi = abs(2 * h - 1)
    return spi, h

# Example usage
spi, h = compute_spi(edge_index, labels)
print(f"Homophily: {h:.3f}, SPI: {spi:.3f}")

if h > 0.5 and spi > 0.4:
    print("Trust Region: GNN recommended")
elif h < 0.5:
    print("Low homophily: Use MLP or heterophily-aware GNN")
else:
    print("Uncertain Region: Consider feature quality")
```

### 2. Run Cross-Model H-Sweep Experiment

```bash
python cross_model_hsweep.py
```

This runs the core U-shape validation across GCN, GAT, and GraphSAGE.

### 3. Run Semi-Synthetic Experiment (Feature-Pattern Duality)

```bash
python semi_synthetic_hsweep.py
```

This validates whether U-shape holds with real features (Cora, CiteSeer, PubMed).

### 4. Run Feature Similarity Analysis

```bash
python feature_similarity_gap_analysis.py
```

This computes the feature similarity gap that explains the monotonic pattern.

### 5. Run Statistical Analysis

```bash
python comprehensive_statistical_analysis.py
```

## File Structure

```
code/
├── README.md                              # This file
├── requirements.txt                       # Dependencies
├── .gitignore                             # Git ignore rules
├── LICENSE                                # MIT License
│
├── # Core Experiments
├── cross_model_hsweep.py                  # U-shape H-sweep (synthetic)
├── cross_model_hsweep_enhanced.py         # Enhanced version with more metrics
├── semi_synthetic_hsweep.py               # Semi-synthetic validation
│
├── # SPI Framework
├── spi_correlation_analysis.py            # SPI-advantage correlation
├── spi_guided_gating.py                   # SPI-guided soft gating mechanism
├── analyze_spi_validation.py              # SPI validation analysis
│
├── # Feature Analysis
├── feature_similarity_gap_analysis.py     # Feature similarity gap
├── feature_sufficiency_theory_v3.py       # Feature sufficiency framework
│
├── # Statistical Analysis
├── comprehensive_statistical_analysis.py  # Full statistical validation
├── statistical_significance_tests.py      # Significance tests
├── enhanced_statistical_analysis.py       # Bootstrap & AIC/BIC analysis
│
├── # Baseline Models
├── baselines/
│   ├── baseline_models.py                 # GCN, GAT, GraphSAGE, H2GCN
│   ├── data_loaders.py                    # Dataset loading utilities
│   ├── run_baselines.py                   # Run all baselines
│   └── README.md                          # Baseline documentation
│
├── # Results
├── cross_model_hsweep_results.json        # H-sweep results
├── semi_synthetic_hsweep_results.json     # Semi-synthetic results
└── comprehensive_statistical_results.json # Statistical results
```

## Experiments

### 1. U-Shape Pattern (Synthetic Data)

| h | SPI | MLP | GCN | GAT | SAGE | Best |
|---|-----|-----|-----|-----|------|------|
| 0.1 | 0.80 | 99.2 | 99.8 | 99.7 | **99.9** | SAGE |
| 0.5 | 0.00 | 99.2 | 80.5 | 93.9 | **99.1** | SAGE |
| 0.9 | 0.80 | 99.4 | **99.8** | 99.6 | 99.6 | GCN |

### 2. Semi-Synthetic Validation (Real Features)

| Dataset | h=0.15 GCN-MLP | h=0.55 GCN-MLP | h=0.75 GCN-MLP | Pattern |
|---------|----------------|----------------|----------------|---------|
| Cora | -31.4% | +3.0% | +24.0% | Monotonic |
| CiteSeer | -25.2% | +3.9% | +10.9% | Monotonic |
| PubMed | -17.1% | -8.2% | +5.1% | Monotonic |

**Key Finding**: Real features produce monotonic (not U-shaped) patterns.

### 3. Feature Similarity Analysis

| Dataset | h | Intra-Sim | Inter-Sim | Gap | GNN Adv |
|---------|---|-----------|-----------|-----|---------|
| Cora | 0.18 | 0.176 | 0.062 | 0.115 | -31.4% |
| Cora | 0.78 | 0.173 | 0.127 | 0.046 | +24.0% |

Correlation: r = **-0.755** (larger gap → worse GNN performance)

## Datasets

We validate on 12+ real-world datasets:

| Category | Datasets | Homophily Range |
|----------|----------|-----------------|
| High h (>0.7) | Cora, CiteSeer, PubMed, Photo, Computers | 0.74-0.83 |
| Mid h (0.3-0.7) | ogbn-arxiv, Amazon-ratings, Tolokers | 0.38-0.66 |
| Low h (<0.3) | Texas, Wisconsin, Cornell, Squirrel, Chameleon | 0.05-0.24 |

## Citation

```bibtex
@article{trust_regions_gnn_2025,
  title={Trust Regions of Graph Propagation: When to Use GNNs and When Not To},
  author={Xue, Mengdan},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2025},
  note={Under review}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds on PyTorch Geometric and uses datasets from OGB, Planetoid, and WebKB benchmarks.
