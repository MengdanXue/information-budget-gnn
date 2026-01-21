#!/usr/bin/env python3
"""
Symmetric Hyperparameter Tuning Experiment
============================================

Purpose: Address Codex's concern about symmetric tuning fairness
- Both MLP and GNN should receive equal tuning effort
- Compare best-of-K results with variance across seeds
- Ensure conclusions are not biased by asymmetric optimization

Author: FSD Framework
Date: 2025-01-16
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.datasets import Planetoid, WebKB
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from itertools import product
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

np.random.seed(42)
torch.manual_seed(42)


# ============== Models ==============

class FlexibleMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


class FlexibleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class FlexibleSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class FlexibleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ============== Training ==============

def train_and_evaluate(model, data, train_mask, val_mask, test_mask,
                       epochs=200, lr=0.01, patience=30, weight_decay=5e-4):
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    return best_val_acc, best_test_acc


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


def compute_homophily(edge_index, labels):
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()


# ============== Main Experiment ==============

@dataclass
class SymmetricTuningResult:
    dataset: str
    homophily: float
    model_type: str
    best_config: str
    best_acc_mean: float
    best_acc_std: float
    default_acc_mean: float
    default_acc_std: float
    tuning_gain: float


@dataclass
class DatasetComparison:
    dataset: str
    homophily: float
    mlp_best: float
    mlp_best_std: float
    mlp_default: float
    gcn_best: float
    gcn_best_std: float
    gcn_default: float
    sage_best: float
    sage_best_std: float
    sage_default: float
    gat_best: float
    gat_best_std: float
    gat_default: float
    best_gnn_tuned: float
    best_gnn_name: str
    gnn_advantage_default: float
    gnn_advantage_tuned: float
    conclusion: str


def run_symmetric_tuning(data, dataset_name: str, n_seeds: int = 5):
    """Run symmetric hyperparameter tuning for both MLP and GNNs"""
    print(f"\n{'='*70}")
    print(f"SYMMETRIC TUNING: {dataset_name}")
    print(f"{'='*70}")

    n_nodes = data.x.shape[0]
    n_features = data.x.shape[1]
    n_classes = len(torch.unique(data.y))
    h = compute_homophily(data.edge_index, data.y)

    print(f"Nodes: {n_nodes}, Features: {n_features}, Classes: {n_classes}")
    print(f"Homophily: {h:.3f}")

    # Shared hyperparameter grid (same for all models)
    hidden_sizes = [64, 128, 256]
    num_layers_list = [2, 3]
    dropouts = [0.3, 0.5]
    learning_rates = [0.01, 0.001]

    total_configs = len(hidden_sizes) * len(num_layers_list) * len(dropouts) * len(learning_rates)
    print(f"\nTesting {total_configs} configurations per model type...")

    all_results = []

    # Model classes
    model_classes = {
        'MLP': FlexibleMLP,
        'GCN': FlexibleGCN,
        'SAGE': FlexibleSAGE,
        'GAT': FlexibleGAT
    }

    dataset_results = {}

    for model_name, ModelClass in model_classes.items():
        print(f"\n--- Tuning {model_name} ---")

        best_config = None
        best_acc = 0
        best_accs = []
        default_accs = []

        config_idx = 0
        for hidden, layers, dropout, lr in product(hidden_sizes, num_layers_list, dropouts, learning_rates):
            config_idx += 1
            config_name = f"h{hidden}_l{layers}_d{dropout}_lr{lr}"

            accs = []
            for seed in range(n_seeds):
                run_seed = 42 + seed * 100
                train_mask, val_mask, test_mask = create_split(n_nodes, seed=run_seed)

                torch.manual_seed(run_seed)
                if model_name == 'GAT':
                    model = ModelClass(n_features, hidden, n_classes, num_layers=layers, dropout=dropout)
                else:
                    model = ModelClass(n_features, hidden, n_classes, num_layers=layers, dropout=dropout)

                _, test_acc = train_and_evaluate(model, data, train_mask, val_mask, test_mask, lr=lr)
                accs.append(test_acc)

                # Track default config (h=64, l=2, d=0.5, lr=0.01)
                if hidden == 64 and layers == 2 and dropout == 0.5 and lr == 0.01:
                    default_accs.append(test_acc)

            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_config = config_name
                best_accs = accs

        # Store results
        default_mean = np.mean(default_accs) if default_accs else 0
        default_std = np.std(default_accs) if default_accs else 0
        best_mean = np.mean(best_accs)
        best_std = np.std(best_accs)
        tuning_gain = best_mean - default_mean

        print(f"  Best config: {best_config}")
        print(f"  Best acc: {best_mean:.3f} +/- {best_std:.3f}")
        print(f"  Default acc: {default_mean:.3f} +/- {default_std:.3f}")
        print(f"  Tuning gain: {tuning_gain*100:+.1f}%")

        dataset_results[model_name] = {
            'best_config': best_config,
            'best_mean': best_mean,
            'best_std': best_std,
            'default_mean': default_mean,
            'default_std': default_std,
            'tuning_gain': tuning_gain
        }

        all_results.append(SymmetricTuningResult(
            dataset=dataset_name,
            homophily=h,
            model_type=model_name,
            best_config=best_config,
            best_acc_mean=best_mean,
            best_acc_std=best_std,
            default_acc_mean=default_mean,
            default_acc_std=default_std,
            tuning_gain=tuning_gain
        ))

    # Create comparison
    best_gnn_tuned = max(
        dataset_results['GCN']['best_mean'],
        dataset_results['SAGE']['best_mean'],
        dataset_results['GAT']['best_mean']
    )
    best_gnn_name = max(
        [('GCN', dataset_results['GCN']['best_mean']),
         ('SAGE', dataset_results['SAGE']['best_mean']),
         ('GAT', dataset_results['GAT']['best_mean'])],
        key=lambda x: x[1]
    )[0]

    gnn_adv_default = max(
        dataset_results['GCN']['default_mean'],
        dataset_results['SAGE']['default_mean'],
        dataset_results['GAT']['default_mean']
    ) - dataset_results['MLP']['default_mean']

    gnn_adv_tuned = best_gnn_tuned - dataset_results['MLP']['best_mean']

    if gnn_adv_tuned > 0.01:
        conclusion = "GNN_WINS"
    elif gnn_adv_tuned < -0.01:
        conclusion = "MLP_WINS"
    else:
        conclusion = "TIE"

    comparison = DatasetComparison(
        dataset=dataset_name,
        homophily=h,
        mlp_best=dataset_results['MLP']['best_mean'],
        mlp_best_std=dataset_results['MLP']['best_std'],
        mlp_default=dataset_results['MLP']['default_mean'],
        gcn_best=dataset_results['GCN']['best_mean'],
        gcn_best_std=dataset_results['GCN']['best_std'],
        gcn_default=dataset_results['GCN']['default_mean'],
        sage_best=dataset_results['SAGE']['best_mean'],
        sage_best_std=dataset_results['SAGE']['best_std'],
        sage_default=dataset_results['SAGE']['default_mean'],
        gat_best=dataset_results['GAT']['best_mean'],
        gat_best_std=dataset_results['GAT']['best_std'],
        gat_default=dataset_results['GAT']['default_mean'],
        best_gnn_tuned=best_gnn_tuned,
        best_gnn_name=best_gnn_name,
        gnn_advantage_default=gnn_adv_default,
        gnn_advantage_tuned=gnn_adv_tuned,
        conclusion=conclusion
    )

    print(f"\n--- Summary ---")
    print(f"  MLP (tuned): {dataset_results['MLP']['best_mean']:.3f}")
    print(f"  Best GNN (tuned): {best_gnn_tuned:.3f} ({best_gnn_name})")
    print(f"  GNN advantage (default): {gnn_adv_default*100:+.1f}%")
    print(f"  GNN advantage (tuned): {gnn_adv_tuned*100:+.1f}%")
    print(f"  Conclusion: {conclusion}")

    return all_results, comparison


def main():
    print("="*80)
    print("SYMMETRIC HYPERPARAMETER TUNING EXPERIMENT")
    print("Both MLP and GNNs receive equal tuning effort")
    print("="*80)

    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)

    all_results = []
    comparisons = []

    # Test on representative datasets
    datasets = []

    # High-h
    for name in ['Cora', 'CiteSeer']:
        try:
            dataset = Planetoid(root=str(data_dir), name=name)
            datasets.append((dataset[0], name))
        except Exception as e:
            print(f"Error loading {name}: {e}")

    # Low-h
    for name in ['Texas', 'Wisconsin']:
        try:
            dataset = WebKB(root=str(data_dir), name=name)
            datasets.append((dataset[0], name))
        except Exception as e:
            print(f"Error loading {name}: {e}")

    for data, name in datasets:
        try:
            results, comparison = run_symmetric_tuning(data, name, n_seeds=5)
            all_results.extend([asdict(r) for r in results])
            comparisons.append(asdict(comparison))
        except Exception as e:
            print(f"Error on {name}: {e}")
            import traceback
            traceback.print_exc()

    # Final Summary
    print("\n" + "="*80)
    print("SYMMETRIC TUNING SUMMARY")
    print("="*80)

    print(f"\n{'Dataset':<12} {'h':<6} {'MLP':<12} {'GCN':<12} {'SAGE':<12} {'GAT':<12} {'Winner'}")
    print("-" * 85)

    for c in comparisons:
        print(f"{c['dataset']:<12} {c['homophily']:<6.2f} "
              f"{c['mlp_best']:.3f}+/-{c['mlp_best_std']:.2f} "
              f"{c['gcn_best']:.3f}+/-{c['gcn_best_std']:.2f} "
              f"{c['sage_best']:.3f}+/-{c['sage_best_std']:.2f} "
              f"{c['gat_best']:.3f}+/-{c['gat_best_std']:.2f} "
              f"{c['conclusion']}")

    print("\n" + "-"*80)
    print("KEY FINDINGS:")

    # Average tuning gains
    mlp_gains = [c['mlp_best'] - c['mlp_default'] for c in comparisons]
    gcn_gains = [c['gcn_best'] - c['gcn_default'] for c in comparisons]
    sage_gains = [c['sage_best'] - c['sage_default'] for c in comparisons]
    gat_gains = [c['gat_best'] - c['gat_default'] for c in comparisons]

    print(f"\n1. Average tuning gains:")
    print(f"   MLP:  {np.mean(mlp_gains)*100:+.1f}%")
    print(f"   GCN:  {np.mean(gcn_gains)*100:+.1f}%")
    print(f"   SAGE: {np.mean(sage_gains)*100:+.1f}%")
    print(f"   GAT:  {np.mean(gat_gains)*100:+.1f}%")

    # GNN advantage before/after tuning
    avg_gnn_adv_default = np.mean([c['gnn_advantage_default'] for c in comparisons])
    avg_gnn_adv_tuned = np.mean([c['gnn_advantage_tuned'] for c in comparisons])

    print(f"\n2. GNN advantage:")
    print(f"   Before tuning: {avg_gnn_adv_default*100:+.1f}%")
    print(f"   After tuning:  {avg_gnn_adv_tuned*100:+.1f}%")

    # Win counts
    gnn_wins = sum(1 for c in comparisons if c['conclusion'] == 'GNN_WINS')
    mlp_wins = sum(1 for c in comparisons if c['conclusion'] == 'MLP_WINS')
    ties = sum(1 for c in comparisons if c['conclusion'] == 'TIE')

    print(f"\n3. Results: GNN wins {gnn_wins}, MLP wins {mlp_wins}, Ties {ties}")

    # Conclusion
    print("\n" + "-"*80)
    if max(np.mean(gcn_gains), np.mean(sage_gains), np.mean(gat_gains)) <= np.mean(mlp_gains) + 0.02:
        print("FAIRNESS CONFIRMED: GNN tuning gain is similar to MLP tuning gain.")
        print("Conclusions are not biased by asymmetric optimization effort.")
    else:
        print("NOTE: GNN benefits more from tuning than MLP.")
        print("This may affect fairness of comparison.")

    # Save results
    output_path = Path(__file__).parent / 'symmetric_tuning_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'symmetric_hyperparameter_tuning',
            'description': 'Equal tuning effort for MLP and GNNs',
            'all_results': all_results,
            'comparisons': comparisons,
            'summary': {
                'avg_mlp_tuning_gain': float(np.mean(mlp_gains)),
                'avg_gcn_tuning_gain': float(np.mean(gcn_gains)),
                'avg_sage_tuning_gain': float(np.mean(sage_gains)),
                'avg_gat_tuning_gain': float(np.mean(gat_gains)),
                'avg_gnn_advantage_default': float(avg_gnn_adv_default),
                'avg_gnn_advantage_tuned': float(avg_gnn_adv_tuned),
                'gnn_wins': gnn_wins,
                'mlp_wins': mlp_wins,
                'ties': ties
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
