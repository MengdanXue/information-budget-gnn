# Pre-registration Document: Information Budget Theory
## Frozen Decision Rules for GNN vs MLP Prediction

**Document Created**: 2025-01-16
**Status**: FROZEN - Do not modify after experiments

---

## 1. Core Hypothesis

**Information Budget Principle**: The maximum possible GNN advantage over MLP is bounded by:

```
GNN_max_gain ≤ (1 - MLP_accuracy) = Budget
```

**Rationale**: GNN can only improve upon the residual variance that features (MLP) cannot explain.

---

## 2. Frozen Decision Rules

The following rules are **fixed before any CSBM experiments** and will not be modified based on results:

### Rule 1: Very High MLP Accuracy
```python
if mlp_acc > 0.95:
    return "MLP"  # Budget too small for GNN to help
```

### Rule 2: Extreme Homophily Regions (Trust Regions)
```python
if homophily > 0.75 and budget > 0.05:
    return "GNN"  # High-h trust region

if homophily < 0.25 and budget > 0.05:
    return "GNN"  # Low-h trust region (pattern heterophily)
```

### Rule 3: Mid-Homophily Uncertainty Zone
```python
if 0.35 <= homophily <= 0.65:
    if budget > 0.4:
        return "GNN"  # Large budget might allow some gain
    return "MLP"  # Structure is noise in mid-h
```

### Rule 4: Intermediate Regions
```python
spi = abs(2 * homophily - 1)  # Structural Predictability Index
if spi * budget > 0.15:
    return "GNN"
else:
    return "MLP"
```

---

## 3. Key Thresholds (Frozen)

| Parameter | Threshold | Justification |
|-----------|-----------|---------------|
| MLP accuracy ceiling | 0.95 | Beyond this, budget < 0.05 is negligible |
| High-h boundary | 0.75 | Standard definition in GNN literature |
| Low-h boundary | 0.25 | Standard definition in GNN literature |
| Mid-h range | [0.35, 0.65] | Uncertainty zone where h ≈ 0.5 |
| Minimum budget for trust regions | 0.05 | 5% residual variance needed |
| Large budget threshold | 0.4 | 40% residual allows mid-h GNN gains |
| SPI × Budget threshold | 0.15 | Empirically derived from prior work |

---

## 4. Metric Definitions (Frozen)

### Homophily (h)
```python
def compute_homophily(edge_index, labels):
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()
```

### Information Budget (B)
```python
budget = 1 - mlp_accuracy
```

### Structural Predictability Index (SPI)
```python
spi = abs(2 * homophily - 1)
```

---

## 5. CSBM Experiment Parameters (Frozen)

### Graph Generation
- Nodes: 1000
- Classes: 2
- Features: 100
- Average degree: 10
- Runs per configuration: 5

### Parameter Grid
- Homophily values: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
- Feature informativeness: [0.1, 0.3, 0.5, 0.7]
- Total configurations: 36

### Feature Generation
```python
effective_info = feature_informativeness ** 2
class_centers = np.random.randn(n_classes, n_features) * 0.5
features[i] = (1 - effective_info) * noise + effective_info * class_signal
```

---

## 6. Model Configurations (Frozen)

### MLP
- Hidden: 64
- Layers: 2
- Dropout: 0.5
- Learning rate: 0.01
- Weight decay: 5e-4

### GCN
- Hidden: 64
- Layers: 2
- Dropout: 0.5
- Learning rate: 0.01
- Weight decay: 5e-4

### GraphSAGE
- Hidden: 64
- Layers: 2
- Dropout: 0.5
- Learning rate: 0.01
- Weight decay: 5e-4

---

## 7. Evaluation Protocol (Frozen)

### Winner Determination
```python
if best_gnn > mlp_mean + 0.01:
    actual_winner = "GNN"
elif mlp_mean > best_gnn + 0.01:
    actual_winner = "MLP"
else:
    actual_winner = "Tie"  # Within 1% is considered a tie
```

### Prediction Correctness
- If actual_winner == "Tie": prediction is always correct
- Otherwise: prediction must match actual_winner

---

## 8. Commitment Statement

**I hereby commit that:**

1. All decision rules above were defined BEFORE running the CSBM experiments
2. The thresholds (0.95, 0.75, 0.25, 0.35, 0.65, 0.05, 0.4, 0.15) are fixed and will not be tuned
3. Any modification to these rules after seeing results will be clearly documented as "post-hoc refinement"
4. The initial experiment results (even if prediction accuracy is low) will be reported honestly

**Signed**: FSD Framework Research Team
**Date**: 2025-01-16

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-16 | Initial frozen version |

---

## 10. SHA-256 Hash of This Document

To verify this document has not been modified, compute:
```
sha256sum PREREGISTRATION.md
```

The hash at creation time will be recorded in the git commit.
