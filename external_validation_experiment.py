#!/usr/bin/env python3
"""
External Dataset Validation Experiment
========================================

Purpose: Validate Information Budget Theory on additional external datasets
beyond the original CSBM synthetic data.

Datasets to test:
1. OGB datasets (ogbn-arxiv, ogbn-products subset)
2. Heterogeneous benchmark datasets (Actor, Roman-empire)
3. Additional citation networks (CoraFull, DBLP)

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
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, Actor, WikipediaNetwork
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
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


# ============== Utilities ==============

def compute_homophily(edge_index, labels):
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()


def predict_winner(homophily: float, mlp_acc: float) -> Tuple[str, str]:
    """
    Predict winner using FROZEN rules from PREREGISTRATION.md
    """
    budget = 1 - mlp_acc
    spi = abs(2 * homophily - 1)

    if mlp_acc > 0.95:
        return "MLP", f"Budget too small ({budget:.2f})"

    if homophily > 0.75 and budget > 0.05:
        return "GNN", f"High-h trust region (h={homophily:.2f})"

    if homophily < 0.25 and budget > 0.05:
        return "GNN", f"Low-h trust region (h={homophily:.2f})"

    if 0.35 <= homophily <= 0.65:
        if budget > 0.4:
            return "GNN", f"Mid-h but large budget ({budget:.2f})"
        return "MLP", f"Mid-h uncertainty zone (h={homophily:.2f})"

    if spi * budget > 0.15:
        return "GNN", f"SPI ({spi:.2f}) x Budget ({budget:.2f}) > 0.15"

    return "MLP", f"SPI ({spi:.2f}) x Budget ({budget:.2f}) <= 0.15"


def train_and_evaluate(model, data, train_mask, val_mask, test_mask,
                       epochs=200, lr=0.01, patience=30) -> float:
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


# ============== Main Experiment ==============

@dataclass
class ExternalValidationResult:
    dataset: str
    category: str  # high-h, mid-h, low-h
    n_nodes: int
    n_edges: int
    n_classes: int
    homophily: float
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


def validate_on_dataset(data, dataset_name: str, n_runs: int = 5) -> ExternalValidationResult:
    """Run validation on a single dataset"""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    n_nodes = data.x.shape[0]
    n_features = data.x.shape[1]
    n_classes = len(torch.unique(data.y))
    n_edges = data.edge_index.shape[1] // 2
    h = compute_homophily(data.edge_index, data.y)

    print(f"Nodes: {n_nodes}, Features: {n_features}, Classes: {n_classes}, Edges: {n_edges}")
    print(f"Homophily: {h:.3f}")

    # Categorize
    if h > 0.7:
        category = "high-h"
    elif h < 0.3:
        category = "low-h"
    else:
        category = "mid-h"

    mlp_accs, gcn_accs, sage_accs = [], [], []

    for run in range(n_runs):
        seed = 42 + run * 100
        train_mask, val_mask, test_mask = create_split(n_nodes, seed=seed)

        # MLP
        torch.manual_seed(seed)
        mlp = MLP(n_features, 64, n_classes)
        mlp_acc = train_and_evaluate(mlp, data, train_mask, val_mask, test_mask)
        mlp_accs.append(mlp_acc)

        # GCN
        torch.manual_seed(seed)
        gcn = GCN(n_features, 64, n_classes)
        gcn_acc = train_and_evaluate(gcn, data, train_mask, val_mask, test_mask)
        gcn_accs.append(gcn_acc)

        # GraphSAGE
        torch.manual_seed(seed)
        sage = GraphSAGE(n_features, 64, n_classes)
        sage_acc = train_and_evaluate(sage, data, train_mask, val_mask, test_mask)
        sage_accs.append(sage_acc)

    mlp_mean = np.mean(mlp_accs)
    gcn_mean = np.mean(gcn_accs)
    sage_mean = np.mean(sage_accs)
    best_gnn = max(gcn_mean, sage_mean)

    # Make prediction using frozen rules
    prediction, reason = predict_winner(h, mlp_mean)

    # Determine actual winner
    if best_gnn > mlp_mean + 0.01:
        actual_winner = "GNN"
    elif mlp_mean > best_gnn + 0.01:
        actual_winner = "MLP"
    else:
        actual_winner = "Tie"

    # Check correctness
    if actual_winner == "Tie":
        prediction_correct = True
    else:
        prediction_correct = (prediction == actual_winner)

    budget = 1 - mlp_mean
    spi = abs(2 * h - 1)
    gcn_adv = gcn_mean - mlp_mean

    status = "CORRECT" if prediction_correct else "WRONG"
    print(f"\nResults:")
    print(f"  MLP: {mlp_mean:.3f}, GCN: {gcn_mean:.3f}, SAGE: {sage_mean:.3f}")
    print(f"  Budget: {budget:.3f}, SPI: {spi:.3f}")
    print(f"  Prediction: {prediction} | Actual: {actual_winner} | [{status}]")
    print(f"  Reason: {reason}")

    return ExternalValidationResult(
        dataset=dataset_name,
        category=category,
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_classes=n_classes,
        homophily=h,
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


def main():
    print("="*80)
    print("EXTERNAL DATASET VALIDATION")
    print("Testing Information Budget Theory on diverse real-world datasets")
    print("="*80)

    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)

    results = []

    # ============== High-h Datasets ==============
    print("\n" + "="*80)
    print("HIGH-HOMOPHILY DATASETS (Expected: GNN wins when budget allows)")
    print("="*80)

    # Planetoid
    for name in ['Cora', 'CiteSeer', 'PubMed']:
        try:
            dataset = Planetoid(root=str(data_dir), name=name)
            result = validate_on_dataset(dataset[0], name)
            results.append(asdict(result))
        except Exception as e:
            print(f"Error on {name}: {e}")

    # Coauthor
    try:
        dataset = Coauthor(root=str(data_dir), name='CS')
        result = validate_on_dataset(dataset[0], 'Coauthor-CS')
        results.append(asdict(result))
    except Exception as e:
        print(f"Error on Coauthor-CS: {e}")

    # Amazon
    for name in ['Computers', 'Photo']:
        try:
            dataset = Amazon(root=str(data_dir), name=name)
            result = validate_on_dataset(dataset[0], f'Amazon-{name}')
            results.append(asdict(result))
        except Exception as e:
            print(f"Error on Amazon-{name}: {e}")

    # ============== Mid-h Datasets ==============
    print("\n" + "="*80)
    print("MID-HOMOPHILY DATASETS (Expected: MLP wins in uncertainty zone)")
    print("="*80)

    # Actor
    try:
        dataset = Actor(root=str(data_dir))
        result = validate_on_dataset(dataset[0], 'Actor')
        results.append(asdict(result))
    except Exception as e:
        print(f"Error on Actor: {e}")

    # ============== Low-h Datasets ==============
    print("\n" + "="*80)
    print("LOW-HOMOPHILY DATASETS (Expected: depends on heterophily type)")
    print("="*80)

    # Wikipedia networks
    for name in ['chameleon', 'squirrel']:
        try:
            dataset = WikipediaNetwork(root=str(data_dir), name=name)
            result = validate_on_dataset(dataset[0], f'Wikipedia-{name}')
            results.append(asdict(result))
        except Exception as e:
            print(f"Error on Wikipedia-{name}: {e}")

    # ============== Summary ==============
    print("\n" + "="*80)
    print("EXTERNAL VALIDATION SUMMARY")
    print("="*80)

    print(f"\n{'Dataset':<20} {'h':<6} {'Cat':<8} {'MLP':<7} {'GCN':<7} {'Pred':<5} {'Actual':<6} {'OK'}")
    print("-" * 75)

    correct_count = 0
    for r in results:
        ok = "Y" if r['prediction_correct'] else "N"
        if r['prediction_correct']:
            correct_count += 1
        print(f"{r['dataset']:<20} {r['homophily']:<6.3f} {r['category']:<8} "
              f"{r['mlp_acc']:<7.3f} {r['gcn_acc']:<7.3f} {r['prediction']:<5} "
              f"{r['actual_winner']:<6} {ok}")

    accuracy = correct_count / len(results) if results else 0
    print("-" * 75)
    print(f"\nOVERALL PREDICTION ACCURACY: {correct_count}/{len(results)} = {accuracy*100:.1f}%")

    # By category
    print("\nAccuracy by Category:")
    for cat in ['high-h', 'mid-h', 'low-h']:
        cat_results = [r for r in results if r['category'] == cat]
        if cat_results:
            cat_correct = sum(1 for r in cat_results if r['prediction_correct'])
            print(f"  {cat}: {cat_correct}/{len(cat_results)} = {cat_correct/len(cat_results)*100:.0f}%")

    # Failure analysis
    failures = [r for r in results if not r['prediction_correct']]
    if failures:
        print(f"\nFailure Cases ({len(failures)}):")
        for r in failures:
            print(f"  {r['dataset']}: h={r['homophily']:.3f}, MLP={r['mlp_acc']:.3f}, GCN={r['gcn_acc']:.3f}")
            print(f"    Predicted: {r['prediction']} ({r['prediction_reason']})")
            print(f"    Actual: {r['actual_winner']}")

    # Save results
    output_path = Path(__file__).parent / 'external_validation_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'external_dataset_validation',
            'description': 'Testing frozen prediction rules on diverse real-world datasets',
            'results': results,
            'summary': {
                'total_datasets': len(results),
                'correct_predictions': correct_count,
                'accuracy': accuracy
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results, accuracy


if __name__ == '__main__':
    results, accuracy = main()
