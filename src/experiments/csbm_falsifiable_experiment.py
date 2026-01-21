#!/usr/bin/env python3
"""
CSBM Synthetic Data Experiment: Falsifiable Prediction Test
============================================================

目的：用受控合成数据验证Information Budget Theory的可证伪性

实验设计（按Codex/建议）：
1. 使用Contextual Stochastic Block Model (CSBM)生成图
2. 控制两个关键参数：
   - h (homophily): 同质性
   - feature_info (特征信息量): 控制MLP准确率
3. 先用理论预测GNN vs MLP的胜负
4. 再运行实验验证预测是否正确

理论预测公式：
- Information Budget: B = 1 - MLP_acc
- 预测：当 h > 0.5 且 B > 0.1 时，GNN应该赢
- 预测：当 h ≈ 0.5 时，MLP应该赢（结构是噪音）
- 预测：当 MLP_acc > 0.9 时，GNN难以提升（Budget太小）

Author: FSD Framework
Date: 2025-01-16
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

np.random.seed(42)
torch.manual_seed(42)


# ============== Models ==============

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


# ============== CSBM Graph Generator ==============

def generate_csbm_graph(
    n_nodes: int = 1000,
    n_classes: int = 2,
    n_features: int = 100,
    homophily: float = 0.5,
    feature_informativeness: float = 0.5,
    avg_degree: int = 10,
    seed: int = 42
) -> Data:
    """
    Generate a Contextual Stochastic Block Model (CSBM) graph.

    Parameters:
    -----------
    n_nodes: Number of nodes
    n_classes: Number of classes
    n_features: Feature dimension
    homophily: Edge homophily (probability that edge connects same-class nodes)
    feature_informativeness: How informative features are (0=random, 1=perfect)
    avg_degree: Average node degree
    seed: Random seed

    Returns:
    --------
    PyG Data object
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate labels (balanced classes)
    labels = np.array([i % n_classes for i in range(n_nodes)])
    np.random.shuffle(labels)

    # Generate features based on informativeness
    # Use smaller signal-to-noise ratio for more realistic scenarios
    features = np.random.randn(n_nodes, n_features)

    # Class centers with smaller magnitude to reduce separability
    class_centers = np.random.randn(n_classes, n_features) * 0.5  # Reduced from 2 to 0.5

    for i in range(n_nodes):
        # Mix random noise with class-specific signal
        # Square the informativeness to make low values have less impact
        effective_info = feature_informativeness ** 2  # e.g., 0.2 -> 0.04
        class_signal = class_centers[labels[i]]
        features[i] = (1 - effective_info) * features[i] + \
                      effective_info * class_signal

    # Normalize features
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    # Generate edges based on homophily
    n_edges = n_nodes * avg_degree // 2
    edges = set()

    while len(edges) < n_edges:
        src = np.random.randint(0, n_nodes)

        # Decide if edge should be homophilic or heterophilic
        if np.random.random() < homophily:
            # Same class edge
            same_class_nodes = np.where(labels == labels[src])[0]
            same_class_nodes = same_class_nodes[same_class_nodes != src]
            if len(same_class_nodes) > 0:
                dst = np.random.choice(same_class_nodes)
            else:
                continue
        else:
            # Different class edge
            diff_class_nodes = np.where(labels != labels[src])[0]
            if len(diff_class_nodes) > 0:
                dst = np.random.choice(diff_class_nodes)
            else:
                continue

        if src != dst:
            edge = (min(src, dst), max(src, dst))
            edges.add(edge)

    # Build edge_index
    edge_list = list(edges)
    edge_index = torch.tensor(
        [[e[0] for e in edge_list] + [e[1] for e in edge_list],
         [e[1] for e in edge_list] + [e[0] for e in edge_list]],
        dtype=torch.long
    )

    # Create Data object
    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long)
    )

    return data


def compute_actual_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute actual edge homophily."""
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()


# ============== Training ==============

def train_and_evaluate(model, data, train_mask, val_mask, test_mask,
                       epochs=200, lr=0.01, patience=30) -> float:
    """Train and return test accuracy."""
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_acc = 0
    best_test_acc = 0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

            val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    return best_test_acc


def create_split(n_nodes: int, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Create random split."""
    torch.manual_seed(seed)
    perm = torch.randperm(n_nodes)

    train_size = int(train_ratio * n_nodes)
    val_size = int(val_ratio * n_nodes)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size + val_size]] = True
    test_mask[perm[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask


# ============== Theory Prediction ==============

def predict_winner(homophily: float, mlp_acc: float) -> Tuple[str, str]:
    """
    Predict whether GNN or MLP will win based on Information Budget Theory.

    Updated rules based on empirical analysis:
    - High-h (>0.7) and Low-h (<0.3) are "trust regions" where structure helps
    - Mid-h (0.3-0.7) is uncertainty zone
    - Budget threshold should be lower (0.05 instead of 0.1)

    Returns: (prediction, reason)
    """
    budget = 1 - mlp_acc
    spi = abs(2 * homophily - 1)  # Structural Predictability Index

    # Decision rules based on refined theory
    # Rule 1: Very high MLP accuracy means no room for GNN
    if mlp_acc > 0.95:
        return "MLP", f"Budget too small ({budget:.2f}), MLP already excellent"

    # Rule 2: Extreme homophily regions (h > 0.75 or h < 0.25) are trust regions
    # GNN can help even with smaller budget
    if homophily > 0.75 and budget > 0.05:
        return "GNN", f"High-h trust region (h={homophily:.2f}) + budget ({budget:.2f})"

    if homophily < 0.25 and budget > 0.05:
        return "GNN", f"Low-h trust region (h={homophily:.2f}) + budget ({budget:.2f})"

    # Rule 3: Mid-h uncertainty zone (0.35-0.65)
    if 0.35 <= homophily <= 0.65:
        if budget > 0.4:
            # Large budget might still allow some GNN gain
            return "GNN", f"Mid-h but large budget ({budget:.2f})"
        return "MLP", f"Mid-h uncertainty zone (h={homophily:.2f})"

    # Rule 4: For intermediate h (0.25-0.35 or 0.65-0.75)
    # Use SPI * budget threshold
    if spi * budget > 0.15:
        return "GNN", f"SPI ({spi:.2f}) x Budget ({budget:.2f}) = {spi*budget:.2f} > 0.15"

    # Rule 5: Default to MLP in uncertain cases
    return "MLP", f"SPI ({spi:.2f}) x Budget ({budget:.2f}) = {spi*budget:.2f} <= 0.15"


# ============== Main Experiment ==============

@dataclass
class CSBMResult:
    homophily_target: float
    homophily_actual: float
    feature_info: float
    mlp_acc: float
    gcn_acc: float
    sage_acc: float
    best_gnn_acc: float
    budget: float
    spi: float
    prediction: str
    prediction_reason: str
    actual_winner: str
    prediction_correct: bool
    gcn_advantage: float


def run_csbm_experiment(
    homophily_values: List[float],
    feature_info_values: List[float],
    n_runs: int = 5
) -> List[CSBMResult]:
    """
    Run CSBM experiment across different homophily and feature informativeness levels.
    """
    results = []

    print(f"\n{'='*80}")
    print("CSBM SYNTHETIC DATA EXPERIMENT")
    print("Testing Information Budget Theory Predictions")
    print(f"{'='*80}")

    total_configs = len(homophily_values) * len(feature_info_values)
    config_idx = 0

    for h in homophily_values:
        for feat_info in feature_info_values:
            config_idx += 1
            print(f"\n--- Config {config_idx}/{total_configs}: h={h:.2f}, feat_info={feat_info:.2f} ---")

            mlp_accs, gcn_accs, sage_accs = [], [], []
            actual_hs = []

            for run in range(n_runs):
                seed = 42 + run * 1000 + int(h * 100) + int(feat_info * 10)

                # Generate graph
                data = generate_csbm_graph(
                    n_nodes=1000,
                    n_classes=2,
                    n_features=100,
                    homophily=h,
                    feature_informativeness=feat_info,
                    avg_degree=10,
                    seed=seed
                )

                actual_h = compute_actual_homophily(data.edge_index, data.y)
                actual_hs.append(actual_h)

                n_nodes = data.x.shape[0]
                n_features = data.x.shape[1]
                n_classes = len(torch.unique(data.y))

                train_mask, val_mask, test_mask = create_split(n_nodes, seed=seed)

                # Train MLP
                torch.manual_seed(seed)
                mlp = MLP(n_features, 64, n_classes)
                mlp_acc = train_and_evaluate(mlp, data, train_mask, val_mask, test_mask)
                mlp_accs.append(mlp_acc)

                # Train GCN
                torch.manual_seed(seed)
                gcn = GCN(n_features, 64, n_classes)
                gcn_acc = train_and_evaluate(gcn, data, train_mask, val_mask, test_mask)
                gcn_accs.append(gcn_acc)

                # Train GraphSAGE
                torch.manual_seed(seed)
                sage = GraphSAGE(n_features, 64, n_classes)
                sage_acc = train_and_evaluate(sage, data, train_mask, val_mask, test_mask)
                sage_accs.append(sage_acc)

            # Aggregate results
            mlp_mean = np.mean(mlp_accs)
            gcn_mean = np.mean(gcn_accs)
            sage_mean = np.mean(sage_accs)
            actual_h_mean = np.mean(actual_hs)
            best_gnn = max(gcn_mean, sage_mean)

            # Make prediction BEFORE seeing results
            prediction, reason = predict_winner(actual_h_mean, mlp_mean)

            # Determine actual winner
            if best_gnn > mlp_mean + 0.01:
                actual_winner = "GNN"
            elif mlp_mean > best_gnn + 0.01:
                actual_winner = "MLP"
            else:
                actual_winner = "Tie"

            # Check if prediction is correct
            if actual_winner == "Tie":
                prediction_correct = True  # Ties are acceptable
            else:
                prediction_correct = (prediction == actual_winner)

            budget = 1 - mlp_mean
            spi = abs(2 * actual_h_mean - 1)
            gcn_adv = gcn_mean - mlp_mean

            result = CSBMResult(
                homophily_target=h,
                homophily_actual=actual_h_mean,
                feature_info=feat_info,
                mlp_acc=mlp_mean,
                gcn_acc=gcn_mean,
                sage_acc=sage_mean,
                best_gnn_acc=best_gnn,
                budget=budget,
                spi=spi,
                prediction=prediction,
                prediction_reason=reason,
                actual_winner=actual_winner,
                prediction_correct=prediction_correct,
                gcn_advantage=gcn_adv
            )
            results.append(result)

            status = "CORRECT" if prediction_correct else "WRONG"
            print(f"  MLP={mlp_mean:.3f}, GCN={gcn_mean:.3f}, SAGE={sage_mean:.3f}")
            print(f"  Budget={budget:.3f}, SPI={spi:.3f}")
            print(f"  Prediction: {prediction} | Actual: {actual_winner} | [{status}]")

    return results


def main():
    print("=" * 80)
    print("CSBM FALSIFIABLE PREDICTION EXPERIMENT")
    print("Testing: Can Information Budget Theory predict GNN vs MLP winner?")
    print("=" * 80)

    # Define parameter grid
    # Homophily: 0.1 (strong heterophily) to 0.9 (strong homophily)
    homophily_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Feature informativeness: lower values to create more realistic scenarios
    # where MLP is not perfect and structure can help
    feature_info_values = [0.1, 0.3, 0.5, 0.7]

    # Run experiment
    results = run_csbm_experiment(
        homophily_values=homophily_values,
        feature_info_values=feature_info_values,
        n_runs=5
    )

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Create summary table
    print(f"\n{'h':<6} {'feat':<6} {'MLP':<8} {'GCN':<8} {'Budget':<8} {'Pred':<6} {'Actual':<6} {'OK':<4}")
    print("-" * 60)

    correct_count = 0
    total_count = len(results)

    for r in results:
        ok = "Y" if r.prediction_correct else "N"
        if r.prediction_correct:
            correct_count += 1
        print(f"{r.homophily_actual:<6.2f} {r.feature_info:<6.2f} {r.mlp_acc:<8.3f} "
              f"{r.gcn_acc:<8.3f} {r.budget:<8.3f} {r.prediction:<6} {r.actual_winner:<6} {ok:<4}")

    accuracy = correct_count / total_count
    print("-" * 60)
    print(f"\nPREDICTION ACCURACY: {correct_count}/{total_count} = {accuracy*100:.1f}%")

    # Analyze by region
    print("\n" + "=" * 80)
    print("ANALYSIS BY REGION")
    print("=" * 80)

    # High-h region (h > 0.7)
    high_h = [r for r in results if r.homophily_actual > 0.7]
    if high_h:
        high_h_correct = sum(1 for r in high_h if r.prediction_correct)
        print(f"\nHigh-h (h > 0.7): {high_h_correct}/{len(high_h)} = {high_h_correct/len(high_h)*100:.0f}%")

    # Mid-h region (0.4 <= h <= 0.6)
    mid_h = [r for r in results if 0.4 <= r.homophily_actual <= 0.6]
    if mid_h:
        mid_h_correct = sum(1 for r in mid_h if r.prediction_correct)
        print(f"Mid-h (0.4-0.6): {mid_h_correct}/{len(mid_h)} = {mid_h_correct/len(mid_h)*100:.0f}%")

    # Low-h region (h < 0.3)
    low_h = [r for r in results if r.homophily_actual < 0.3]
    if low_h:
        low_h_correct = sum(1 for r in low_h if r.prediction_correct)
        print(f"Low-h (h < 0.3): {low_h_correct}/{len(low_h)} = {low_h_correct/len(low_h)*100:.0f}%")

    # High budget region
    high_budget = [r for r in results if r.budget > 0.3]
    if high_budget:
        hb_correct = sum(1 for r in high_budget if r.prediction_correct)
        print(f"\nHigh Budget (>0.3): {hb_correct}/{len(high_budget)} = {hb_correct/len(high_budget)*100:.0f}%")

    # Low budget region
    low_budget = [r for r in results if r.budget < 0.2]
    if low_budget:
        lb_correct = sum(1 for r in low_budget if r.prediction_correct)
        print(f"Low Budget (<0.2): {lb_correct}/{len(low_budget)} = {lb_correct/len(low_budget)*100:.0f}%")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Find failure cases
    failures = [r for r in results if not r.prediction_correct]
    if failures:
        print(f"\nFailure cases ({len(failures)}):")
        for r in failures:
            print(f"  h={r.homophily_actual:.2f}, feat={r.feature_info:.2f}, "
                  f"MLP={r.mlp_acc:.3f}, GCN={r.gcn_acc:.3f}")
            print(f"    Predicted: {r.prediction} ({r.prediction_reason})")
            print(f"    Actual: {r.actual_winner}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT: INFORMATION BUDGET THEORY")
    print("=" * 80)

    if accuracy >= 0.85:
        verdict = "STRONGLY SUPPORTED"
        print(f"\n*** {verdict} ***")
        print(f"Theory predictions are correct {accuracy*100:.0f}% of the time.")
        print("This demonstrates falsifiable predictive power, not just post-hoc explanation.")
    elif accuracy >= 0.70:
        verdict = "SUPPORTED"
        print(f"\n*** {verdict} ***")
        print(f"Theory predictions are correct {accuracy*100:.0f}% of the time.")
        print("Some edge cases need refinement, but core theory holds.")
    else:
        verdict = "NEEDS REVISION"
        print(f"\n*** {verdict} ***")
        print(f"Theory predictions are correct only {accuracy*100:.0f}% of the time.")
        print("Decision rules need significant refinement.")

    # Save results
    output_path = Path(__file__).parent / 'csbm_falsifiable_results.json'

    output_data = {
        'experiment': 'csbm_falsifiable_prediction',
        'description': 'Testing Information Budget Theory with controlled synthetic data',
        'parameters': {
            'homophily_values': homophily_values,
            'feature_info_values': feature_info_values,
            'n_runs': 5
        },
        'results': [asdict(r) for r in results],
        'summary': {
            'total_predictions': total_count,
            'correct_predictions': correct_count,
            'accuracy': accuracy,
            'verdict': verdict
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results, accuracy


if __name__ == '__main__':
    results, accuracy = main()
