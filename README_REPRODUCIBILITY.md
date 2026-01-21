# Information Budget Theory - Reproducibility Package

## Overview

This package contains all code, configurations, and random seeds needed to reproduce the experiments in our paper "Information Budget Theory: When Graph Structure Helps GNNs".

## Directory Structure

```
code/
├── README_REPRODUCIBILITY.md     # This file
├── PREREGISTRATION.md            # Frozen decision rules (before experiments)
├── CAUSAL_EVIDENCE_CHAIN.md      # Evidence chain summary
│
├── Core Experiments
│   ├── information_budget_experiment.py      # Edge shuffle, feature degradation, same-h pairs
│   ├── csbm_falsifiable_experiment.py        # CSBM synthetic data prediction
│   ├── dual_heterophily_experiment.py        # Type A/B heterophily validation
│   ├── external_validation_experiment.py     # External dataset validation
│   ├── mlp_tuning_experiment.py              # MLP-only tuning
│   └── symmetric_tuning_experiment.py        # Symmetric MLP+GNN tuning
│
├── Results
│   ├── information_budget_results.json
│   ├── csbm_falsifiable_results.json
│   ├── dual_heterophily_results.json
│   ├── external_validation_results.json
│   ├── mlp_tuning_results.json
│   └── symmetric_tuning_results.json
│
├── Visualization
│   ├── information_budget_visualization.py
│   └── figures/
│
└── requirements.txt
```

## Requirements

```
torch>=1.12.0
torch-geometric>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
```

## Random Seeds

All experiments use the following seed scheme for reproducibility:

```python
# Base seed
BASE_SEED = 42

# For multiple runs (n_runs = 5)
for run in range(n_runs):
    seed = BASE_SEED + run * 100
    # Seeds: 42, 142, 242, 342, 442
```

## Experiment Reproduction

### 1. Information Budget Experiments

```bash
# Run core experiments (edge shuffle, feature degradation, same-h pairs, ADR)
python information_budget_experiment.py

# Expected output: information_budget_results.json
# Key results:
#   - Edge shuffle: Cora +12.6% → -34.8%
#   - Feature degradation: 9/9 within budget
#   - Same-h pairs: 7/7 support hypothesis
```

### 2. CSBM Falsifiable Prediction

```bash
# Run CSBM synthetic data experiment
python csbm_falsifiable_experiment.py

# Expected output: csbm_falsifiable_results.json
# Key results:
#   - Overall accuracy: 88.9% (32/36)
#   - High-h: 100%, Mid-h: 100%, Low-h: 90%
```

### 3. External Validation

```bash
# Validate on external datasets
python external_validation_experiment.py

# Expected output: external_validation_results.json
# Key results:
#   - Overall accuracy: 77.8% (7/9)
#   - High-h: 83%, Low-h: 67%
```

### 4. Symmetric Tuning

```bash
# Run symmetric hyperparameter tuning
python symmetric_tuning_experiment.py

# Expected output: symmetric_tuning_results.json
# Key results:
#   - MLP tuning gain: +1.4%
#   - GCN tuning gain: +1.8%
#   - Fairness confirmed
```

## Configuration Files

### Frozen Decision Rules (from PREREGISTRATION.md)

```python
def predict_winner(homophily: float, mlp_acc: float):
    budget = 1 - mlp_acc
    spi = abs(2 * homophily - 1)

    if mlp_acc > 0.95:
        return "MLP"
    if homophily > 0.75 and budget > 0.05:
        return "GNN"
    if homophily < 0.25 and budget > 0.05:
        return "GNN"
    if 0.35 <= homophily <= 0.65:
        if budget > 0.4:
            return "GNN"
        return "MLP"
    if spi * budget > 0.15:
        return "GNN"
    return "MLP"
```

### Model Hyperparameters (Default)

| Parameter | Value |
|-----------|-------|
| Hidden dim | 64 |
| Num layers | 2 |
| Dropout | 0.5 |
| Learning rate | 0.01 |
| Weight decay | 5e-4 |
| Patience | 30 |
| Max epochs | 200 |

### CSBM Graph Generation

| Parameter | Value |
|-----------|-------|
| Nodes | 1000 |
| Classes | 2 |
| Features | 100 |
| Avg degree | 10 |
| Runs per config | 5 |

## Verification Checksums

| File | SHA-256 (first 16 chars) |
|------|--------------------------|
| PREREGISTRATION.md | 6CBDED43CE22E5AA |
| csbm_falsifiable_experiment.py | [compute after final] |
| information_budget_experiment.py | [compute after final] |

## Expected Results Summary

| Experiment | Key Metric | Expected Value |
|------------|-----------|----------------|
| Edge Shuffle | Cora GCN drop | ~47% |
| Budget Validation | Within budget | 9/9 (100%) |
| Same-h Pairs | Support rate | 7/7 (100%) |
| CSBM Prediction | Accuracy | ~89% |
| External Validation | Accuracy | ~78% |
| Symmetric Tuning | Fairness | MLP≈GNN gain |

## Hardware Used

- GPU: NVIDIA CUDA-capable
- CPU: Intel/AMD x64
- Memory: 16GB+ recommended
- OS: Windows 10/11, Linux, macOS

## Estimated Runtime

| Experiment | Approximate Time |
|------------|------------------|
| Information Budget | 10-15 min |
| CSBM Prediction | 15-20 min |
| External Validation | 10-15 min |
| Symmetric Tuning | 20-30 min |
| **Total** | **~1 hour** |

## Contact

For questions about reproducibility, please open an issue on GitHub.

## License

MIT License - See LICENSE file for details.

---

**Note**: All random seeds and configurations are fixed. Results should be reproducible within small variance (±1-2%) due to GPU non-determinism.
