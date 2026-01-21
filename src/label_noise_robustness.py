#!/usr/bin/env python3
"""
Label Noise Robustness Experiment
==================================

Test whether SPI predictions are robust when labels are noisy.
Critical: Real-world labels often have annotation errors.

Tests:
- Inject label noise: 0%, 5%, 10%, 20%, 30%
- Compare: True SPI vs Noisy SPI
- Measure: Does SPI still predict GNN reliability?

Author: FSD Framework
Date: 2025-01-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
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
# Utility Functions
# ============================================================

def compute_homophily(edge_index, y, num_nodes):
    """Compute edge homophily ratio"""
    row, col = edge_index
    same_label = (y[row] == y[col]).float()
    return same_label.mean().item()


def inject_label_noise(y, noise_rate, num_classes, seed=42):
    """Inject symmetric label noise"""
    np.random.seed(seed)
    y_noisy = y.clone()
    n = len(y)
    n_flip = int(n * noise_rate)

    flip_indices = np.random.choice(n, n_flip, replace=False)
    for idx in flip_indices:
        current_label = y[idx].item()
        # Flip to a random different label
        new_label = np.random.choice([l for l in range(num_classes) if l != current_label])
        y_noisy[idx] = new_label

    return y_noisy


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


def train_and_eval(model_class, data, y_train, y_true, train_mask, val_mask, test_mask,
                   hidden=64, lr=0.01, epochs=200, patience=30):
    """Train with noisy labels, evaluate on true labels"""
    in_channels = data.x.size(1)
    out_channels = y_true.max().item() + 1

    model = model_class(in_channels, hidden, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val = 0
    best_test = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # Train on noisy labels
        loss = F.cross_entropy(out[train_mask], y_train[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

            # Validate and test on TRUE labels
            val_acc = (pred[val_mask] == y_true[val_mask]).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                best_test = (pred[test_mask] == y_true[test_mask]).float().mean().item()
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
    print("LABEL NOISE ROBUSTNESS EXPERIMENT")
    print("=" * 70)

    # Noise levels to test
    noise_rates = [0.0, 0.05, 0.10, 0.20, 0.30]

    # Load datasets
    datasets = {}
    for name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='./data', name=name, transform=NormalizeFeatures())
        data = dataset[0].to(device)
        h = compute_homophily(data.edge_index, data.y, data.num_nodes)
        datasets[name] = {'data': data, 'h': h, 'num_classes': dataset.num_classes}
        print(f"Loaded {name}: h={h:.3f}, classes={dataset.num_classes}")

    results = {
        'experiment': 'Label Noise Robustness',
        'noise_rates': noise_rates,
        'datasets': {}
    }

    n_runs = 5
    seeds = [42 + i * 100 for i in range(n_runs)]

    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (h={dataset_info['h']:.3f})")
        print(f"{'='*60}")

        data = dataset_info['data']
        num_classes = dataset_info['num_classes']
        y_true = data.y.clone()

        dataset_results = {
            'homophily_true': dataset_info['h'],
            'num_classes': num_classes,
            'noise_experiments': []
        }

        for noise_rate in noise_rates:
            print(f"\n--- Noise Rate: {noise_rate*100:.0f}% ---")

            mlp_scores = []
            gcn_scores = []
            noisy_h_values = []

            for seed in seeds:
                # Inject label noise
                y_noisy = inject_label_noise(y_true, noise_rate, num_classes, seed=seed)
                y_noisy = y_noisy.to(device)

                # Compute homophily with noisy labels
                noisy_h = compute_homophily(data.edge_index, y_noisy, data.num_nodes)
                noisy_h_values.append(noisy_h)

                # Get split
                train_mask, val_mask, test_mask = get_split(data.num_nodes, seed=seed)
                train_mask = train_mask.to(device)
                val_mask = val_mask.to(device)
                test_mask = test_mask.to(device)

                # Train with noisy labels, evaluate on true labels
                mlp_acc = train_and_eval(MLP, data, y_noisy, y_true, train_mask, val_mask, test_mask)
                gcn_acc = train_and_eval(GCN, data, y_noisy, y_true, train_mask, val_mask, test_mask)

                mlp_scores.append(mlp_acc)
                gcn_scores.append(gcn_acc)

            mlp_mean = np.mean(mlp_scores)
            mlp_std = np.std(mlp_scores)
            gcn_mean = np.mean(gcn_scores)
            gcn_std = np.std(gcn_scores)
            gcn_adv = gcn_mean - mlp_mean
            noisy_h_mean = np.mean(noisy_h_values)

            # SPI predictions
            true_spi = abs(2 * dataset_info['h'] - 1)
            noisy_spi = abs(2 * noisy_h_mean - 1)
            spi_prediction = 'GCN' if true_spi > 0.4 else 'MLP'
            actual_winner = 'GCN' if gcn_adv > 0.02 else ('MLP' if gcn_adv < -0.02 else 'Tie')

            dataset_results['noise_experiments'].append({
                'noise_rate': noise_rate,
                'h_noisy': noisy_h_mean,
                'spi_true': true_spi,
                'spi_noisy': noisy_spi,
                'mlp_acc': mlp_mean,
                'mlp_std': mlp_std,
                'gcn_acc': gcn_mean,
                'gcn_std': gcn_std,
                'gcn_advantage': gcn_adv,
                'spi_prediction': spi_prediction,
                'actual_winner': actual_winner,
                'prediction_correct': spi_prediction == actual_winner or actual_winner == 'Tie'
            })

            print(f"  Noisy h: {noisy_h_mean:.3f} (true: {dataset_info['h']:.3f})")
            print(f"  MLP: {mlp_mean*100:.1f}% +/- {mlp_std*100:.1f}%")
            print(f"  GCN: {gcn_mean*100:.1f}% +/- {gcn_std*100:.1f}%")
            print(f"  GCN Advantage: {gcn_adv*100:+.1f}%")
            print(f"  SPI Prediction: {spi_prediction}, Actual: {actual_winner}")

        results['datasets'][dataset_name] = dataset_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SPI Robustness to Label Noise")
    print("=" * 70)

    total_correct = 0
    total_experiments = 0

    for dataset_name, dataset_results in results['datasets'].items():
        print(f"\n{dataset_name}:")
        for exp in dataset_results['noise_experiments']:
            status = "Y" if exp['prediction_correct'] else "N"
            print(f"  [{status}] Noise {exp['noise_rate']*100:.0f}%: "
                  f"h={exp['h_noisy']:.3f}, "
                  f"GCN Adv={exp['gcn_advantage']*100:+.1f}%, "
                  f"Pred={exp['spi_prediction']}, Actual={exp['actual_winner']}")

            total_experiments += 1
            if exp['prediction_correct']:
                total_correct += 1

    overall_accuracy = total_correct / total_experiments
    results['summary'] = {
        'total_experiments': total_experiments,
        'correct_predictions': total_correct,
        'accuracy': overall_accuracy
    }

    print(f"\n{'='*70}")
    print(f"OVERALL SPI PREDICTION ACCURACY UNDER LABEL NOISE: {overall_accuracy*100:.1f}% ({total_correct}/{total_experiments})")
    print(f"{'='*70}")

    # Save results
    output_path = Path(__file__).parent / 'label_noise_robustness_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    results = main()
