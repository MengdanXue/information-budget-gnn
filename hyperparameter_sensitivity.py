#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis
====================================

Test whether SPI predictions are robust across different hyperparameter choices.
Critical for TKDE: reviewers need to know results aren't cherry-picked.

Tests:
1. Learning rate: 0.001, 0.01, 0.1
2. Hidden dimensions: 32, 64, 128, 256
3. Number of layers: 2, 3, 4
4. Dropout: 0.3, 0.5, 0.7

Author: FSD Framework
Date: 2025-01-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from itertools import product

from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.transforms import NormalizeFeatures

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================
# Models
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ============================================================
# Training
# ============================================================

def compute_homophily(edge_index, y, num_nodes):
    row, col = edge_index
    same_label = (y[row] == y[col]).float()
    return same_label.mean().item()


def get_split(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(num_nodes)
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True

    return train_mask, val_mask, test_mask


def train_and_eval(model_class, data, train_mask, val_mask, test_mask,
                   hidden=64, num_layers=2, dropout=0.5, lr=0.01, epochs=200, patience=30):
    in_channels = data.x.size(1)
    out_channels = data.y.max().item() + 1

    model = model_class(in_channels, hidden, out_channels, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val = 0
    best_test = 0
    patience_counter = 0

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
            if val_acc > best_val:
                best_val = val_acc
                best_test = (pred[test_mask] == data.y[test_mask]).float().mean().item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

    return best_test


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 70)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Load datasets (high-h, low-h, mid-h representatives)
    datasets = {}

    # High-h: Cora (h=0.81)
    dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
    data = dataset[0].to(device)
    h = compute_homophily(data.edge_index, data.y, data.num_nodes)
    datasets['Cora'] = {'data': data, 'h': h, 'category': 'high-h'}
    print(f"Loaded Cora: h={h:.3f}")

    # Low-h: Texas (h=0.11)
    dataset = WebKB(root='./data', name='Texas', transform=NormalizeFeatures())
    data = dataset[0].to(device)
    h = compute_homophily(data.edge_index, data.y, data.num_nodes)
    datasets['Texas'] = {'data': data, 'h': h, 'category': 'low-h'}
    print(f"Loaded Texas: h={h:.3f}")

    # Mid-h: CiteSeer (h=0.74)
    dataset = Planetoid(root='./data', name='CiteSeer', transform=NormalizeFeatures())
    data = dataset[0].to(device)
    h = compute_homophily(data.edge_index, data.y, data.num_nodes)
    datasets['CiteSeer'] = {'data': data, 'h': h, 'category': 'mid-h'}
    print(f"Loaded CiteSeer: h={h:.3f}")

    # Hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1]
    hidden_dims = [32, 64, 128, 256]
    num_layers_list = [2, 3, 4]
    dropouts = [0.3, 0.5, 0.7]

    results = {
        'experiment': 'Hyperparameter Sensitivity Analysis',
        'datasets': {},
        'summary': {}
    }

    n_runs = 3  # Quick runs for sensitivity
    seeds = [42, 123, 456]

    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (h={dataset_info['h']:.3f}, {dataset_info['category']})")
        print(f"{'='*60}")

        data = dataset_info['data']
        dataset_results = {
            'homophily': dataset_info['h'],
            'category': dataset_info['category'],
            'experiments': []
        }

        # Test learning rate sensitivity
        print("\n--- Learning Rate Sensitivity ---")
        for lr in learning_rates:
            mlp_scores = []
            gcn_scores = []

            for seed in seeds:
                train_mask, val_mask, test_mask = get_split(data.num_nodes, seed=seed)
                train_mask = train_mask.to(device)
                val_mask = val_mask.to(device)
                test_mask = test_mask.to(device)

                mlp_acc = train_and_eval(MLP, data, train_mask, val_mask, test_mask, lr=lr)
                gcn_acc = train_and_eval(GCN, data, train_mask, val_mask, test_mask, lr=lr)

                mlp_scores.append(mlp_acc)
                gcn_scores.append(gcn_acc)

            mlp_mean = np.mean(mlp_scores)
            gcn_mean = np.mean(gcn_scores)
            gcn_adv = gcn_mean - mlp_mean
            winner = 'GCN' if gcn_adv > 0.02 else ('MLP' if gcn_adv < -0.02 else 'Tie')

            dataset_results['experiments'].append({
                'param': 'lr', 'value': lr,
                'mlp': mlp_mean, 'gcn': gcn_mean, 'gcn_adv': gcn_adv, 'winner': winner
            })
            print(f"  lr={lr}: MLP={mlp_mean*100:.1f}%, GCN={gcn_mean*100:.1f}%, Adv={gcn_adv*100:+.1f}%, Winner={winner}")

        # Test hidden dimension sensitivity
        print("\n--- Hidden Dimension Sensitivity ---")
        for hidden in hidden_dims:
            mlp_scores = []
            gcn_scores = []

            for seed in seeds:
                train_mask, val_mask, test_mask = get_split(data.num_nodes, seed=seed)
                train_mask = train_mask.to(device)
                val_mask = val_mask.to(device)
                test_mask = test_mask.to(device)

                mlp_acc = train_and_eval(MLP, data, train_mask, val_mask, test_mask, hidden=hidden)
                gcn_acc = train_and_eval(GCN, data, train_mask, val_mask, test_mask, hidden=hidden)

                mlp_scores.append(mlp_acc)
                gcn_scores.append(gcn_acc)

            mlp_mean = np.mean(mlp_scores)
            gcn_mean = np.mean(gcn_scores)
            gcn_adv = gcn_mean - mlp_mean
            winner = 'GCN' if gcn_adv > 0.02 else ('MLP' if gcn_adv < -0.02 else 'Tie')

            dataset_results['experiments'].append({
                'param': 'hidden', 'value': hidden,
                'mlp': mlp_mean, 'gcn': gcn_mean, 'gcn_adv': gcn_adv, 'winner': winner
            })
            print(f"  hidden={hidden}: MLP={mlp_mean*100:.1f}%, GCN={gcn_mean*100:.1f}%, Adv={gcn_adv*100:+.1f}%, Winner={winner}")

        # Test num_layers sensitivity
        print("\n--- Number of Layers Sensitivity ---")
        for num_layers in num_layers_list:
            mlp_scores = []
            gcn_scores = []

            for seed in seeds:
                train_mask, val_mask, test_mask = get_split(data.num_nodes, seed=seed)
                train_mask = train_mask.to(device)
                val_mask = val_mask.to(device)
                test_mask = test_mask.to(device)

                mlp_acc = train_and_eval(MLP, data, train_mask, val_mask, test_mask, num_layers=num_layers)
                gcn_acc = train_and_eval(GCN, data, train_mask, val_mask, test_mask, num_layers=num_layers)

                mlp_scores.append(mlp_acc)
                gcn_scores.append(gcn_acc)

            mlp_mean = np.mean(mlp_scores)
            gcn_mean = np.mean(gcn_scores)
            gcn_adv = gcn_mean - mlp_mean
            winner = 'GCN' if gcn_adv > 0.02 else ('MLP' if gcn_adv < -0.02 else 'Tie')

            dataset_results['experiments'].append({
                'param': 'num_layers', 'value': num_layers,
                'mlp': mlp_mean, 'gcn': gcn_mean, 'gcn_adv': gcn_adv, 'winner': winner
            })
            print(f"  layers={num_layers}: MLP={mlp_mean*100:.1f}%, GCN={gcn_mean*100:.1f}%, Adv={gcn_adv*100:+.1f}%, Winner={winner}")

        # Test dropout sensitivity
        print("\n--- Dropout Sensitivity ---")
        for dropout in dropouts:
            mlp_scores = []
            gcn_scores = []

            for seed in seeds:
                train_mask, val_mask, test_mask = get_split(data.num_nodes, seed=seed)
                train_mask = train_mask.to(device)
                val_mask = val_mask.to(device)
                test_mask = test_mask.to(device)

                mlp_acc = train_and_eval(MLP, data, train_mask, val_mask, test_mask, dropout=dropout)
                gcn_acc = train_and_eval(GCN, data, train_mask, val_mask, test_mask, dropout=dropout)

                mlp_scores.append(mlp_acc)
                gcn_scores.append(gcn_acc)

            mlp_mean = np.mean(mlp_scores)
            gcn_mean = np.mean(gcn_scores)
            gcn_adv = gcn_mean - mlp_mean
            winner = 'GCN' if gcn_adv > 0.02 else ('MLP' if gcn_adv < -0.02 else 'Tie')

            dataset_results['experiments'].append({
                'param': 'dropout', 'value': dropout,
                'mlp': mlp_mean, 'gcn': gcn_mean, 'gcn_adv': gcn_adv, 'winner': winner
            })
            print(f"  dropout={dropout}: MLP={mlp_mean*100:.1f}%, GCN={gcn_mean*100:.1f}%, Adv={gcn_adv*100:+.1f}%, Winner={winner}")

        results['datasets'][dataset_name] = dataset_results

    # Summary analysis
    print("\n" + "=" * 70)
    print("SUMMARY: SPI Prediction Robustness to Hyperparameters")
    print("=" * 70)

    for dataset_name, dataset_results in results['datasets'].items():
        h = dataset_results['homophily']
        spi = abs(2 * h - 1)
        spi_prediction = 'GCN' if spi > 0.4 else 'MLP'

        experiments = dataset_results['experiments']
        winners = [e['winner'] for e in experiments]
        gcn_wins = winners.count('GCN')
        mlp_wins = winners.count('MLP')
        ties = winners.count('Tie')

        # Check if SPI prediction is consistent
        if spi_prediction == 'GCN':
            consistent = gcn_wins + ties
        else:
            consistent = mlp_wins + ties

        consistency_rate = consistent / len(experiments)

        results['summary'][dataset_name] = {
            'h': h,
            'spi': spi,
            'spi_prediction': spi_prediction,
            'gcn_wins': gcn_wins,
            'mlp_wins': mlp_wins,
            'ties': ties,
            'total_experiments': len(experiments),
            'consistency_rate': consistency_rate
        }

        print(f"\n{dataset_name} (h={h:.2f}, SPI={spi:.2f}):")
        print(f"  SPI Prediction: {spi_prediction}")
        print(f"  Results: GCN wins {gcn_wins}, MLP wins {mlp_wins}, Ties {ties}")
        print(f"  Consistency with SPI: {consistency_rate*100:.1f}%")

    # Overall robustness
    total_consistent = sum(s['consistency_rate'] * s['total_experiments'] for s in results['summary'].values())
    total_experiments = sum(s['total_experiments'] for s in results['summary'].values())
    overall_consistency = total_consistent / total_experiments

    results['overall_consistency'] = overall_consistency
    print(f"\n{'='*70}")
    print(f"OVERALL SPI CONSISTENCY ACROSS HYPERPARAMETERS: {overall_consistency*100:.1f}%")
    print(f"{'='*70}")

    # Save results
    output_path = Path(__file__).parent / 'hyperparameter_sensitivity_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    results = main()
