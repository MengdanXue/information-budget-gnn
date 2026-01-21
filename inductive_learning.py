#!/usr/bin/env python3
"""
Inductive Learning Validation Experiment
=========================================

Test whether SPI framework applies to inductive learning scenarios.
Critical: Many real-world applications require inductive learning (new nodes at test time).

Tests:
- Train on subset of graph, test on unseen nodes
- Compare transductive vs inductive performance
- Verify SPI predictions hold in inductive setting

Author: FSD Framework
Date: 2025-01-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import subgraph

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


class GraphSAGE(nn.Module):
    """GraphSAGE is designed for inductive learning"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
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
    row, col = edge_index
    mask = (row < len(y)) & (col < len(y))
    row, col = row[mask], col[mask]
    same_label = (y[row] == y[col]).float()
    return same_label.mean().item() if len(same_label) > 0 else 0.0


def create_inductive_split(data, train_ratio=0.5, val_ratio=0.1, seed=42):
    """
    Create inductive split: train subgraph has NO overlap with test nodes.
    This simulates real-world scenario where new nodes appear at test time.
    """
    np.random.seed(seed)
    num_nodes = data.num_nodes
    indices = np.random.permutation(num_nodes)

    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_nodes = torch.tensor(indices[:train_size], dtype=torch.long)
    val_nodes = torch.tensor(indices[train_size:train_size+val_size], dtype=torch.long)
    test_nodes = torch.tensor(indices[train_size+val_size:], dtype=torch.long)

    # Create train subgraph (edges only between train nodes)
    # Need to move edge_index to CPU for subgraph operation
    edge_index_cpu = data.edge_index.cpu()
    train_edge_index, _ = subgraph(train_nodes, edge_index_cpu, relabel_nodes=False)

    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True

    return train_mask, val_mask, test_mask, train_edge_index


def train_transductive(model_class, data, train_mask, val_mask, test_mask,
                       hidden=64, lr=0.01, epochs=200, patience=30):
    """Standard transductive training: full graph during training"""
    in_channels = data.x.size(1)
    out_channels = data.y.max().item() + 1

    model = model_class(in_channels, hidden, out_channels).to(device)
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


def train_inductive(model_class, data, train_mask, val_mask, test_mask, train_edge_index,
                    hidden=64, lr=0.01, epochs=200, patience=30):
    """
    Inductive training: train only on train subgraph.
    At test time, use full graph (simulates new nodes appearing).
    """
    in_channels = data.x.size(1)
    out_channels = data.y.max().item() + 1

    model = model_class(in_channels, hidden, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val = 0
    best_test = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Train on subgraph only
        out = model(data.x, train_edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            # Validate/Test on full graph (inductive setting)
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
    print("INDUCTIVE LEARNING VALIDATION EXPERIMENT")
    print("=" * 70)

    # Load datasets
    datasets = {}

    # High-h datasets
    for name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='./data', name=name, transform=NormalizeFeatures())
        data = dataset[0].to(device)
        h = compute_homophily(data.edge_index, data.y, data.num_nodes)
        datasets[name] = {'data': data, 'h': h, 'category': 'high-h'}
        print(f"Loaded {name}: h={h:.3f}")

    # Low-h datasets
    for name in ['Texas', 'Wisconsin', 'Cornell']:
        try:
            dataset = WebKB(root='./data', name=name, transform=NormalizeFeatures())
            data = dataset[0].to(device)
            h = compute_homophily(data.edge_index, data.y, data.num_nodes)
            datasets[name] = {'data': data, 'h': h, 'category': 'low-h'}
            print(f"Loaded {name}: h={h:.3f}")
        except Exception as e:
            print(f"Could not load {name}: {e}")

    results = {
        'experiment': 'Inductive Learning Validation',
        'datasets': {},
        'summary': {}
    }

    n_runs = 5
    seeds = [42 + i * 100 for i in range(n_runs)]

    models = {
        'MLP': MLP,
        'GCN': GCN,
        'GraphSAGE': GraphSAGE
    }

    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (h={dataset_info['h']:.3f}, {dataset_info['category']})")
        print(f"{'='*60}")

        data = dataset_info['data']
        dataset_results = {
            'homophily': dataset_info['h'],
            'category': dataset_info['category'],
            'transductive': {},
            'inductive': {}
        }

        # Run experiments
        for model_name, model_class in models.items():
            trans_scores = []
            ind_scores = []

            for seed in seeds:
                train_mask, val_mask, test_mask, train_edge_index = create_inductive_split(
                    data, train_ratio=0.5, val_ratio=0.1, seed=seed
                )
                train_mask = train_mask.to(device)
                val_mask = val_mask.to(device)
                test_mask = test_mask.to(device)
                train_edge_index = train_edge_index.to(device)

                # Transductive
                trans_acc = train_transductive(model_class, data, train_mask, val_mask, test_mask)
                trans_scores.append(trans_acc)

                # Inductive
                ind_acc = train_inductive(model_class, data, train_mask, val_mask, test_mask, train_edge_index)
                ind_scores.append(ind_acc)

            trans_mean = np.mean(trans_scores)
            trans_std = np.std(trans_scores)
            ind_mean = np.mean(ind_scores)
            ind_std = np.std(ind_scores)
            degradation = (trans_mean - ind_mean) / trans_mean * 100 if trans_mean > 0 else 0

            dataset_results['transductive'][model_name] = {
                'mean': trans_mean, 'std': trans_std
            }
            dataset_results['inductive'][model_name] = {
                'mean': ind_mean, 'std': ind_std, 'degradation': degradation
            }

            print(f"\n{model_name}:")
            print(f"  Transductive: {trans_mean*100:.1f}% +/- {trans_std*100:.1f}%")
            print(f"  Inductive:    {ind_mean*100:.1f}% +/- {ind_std*100:.1f}%")
            print(f"  Degradation:  {degradation:.1f}%")

        # SPI Analysis
        h = dataset_info['h']
        spi = abs(2 * h - 1)
        spi_prediction = 'GNN' if spi > 0.4 else 'MLP'

        # Check winners for both settings
        trans_gcn_adv = dataset_results['transductive']['GCN']['mean'] - dataset_results['transductive']['MLP']['mean']
        ind_gcn_adv = dataset_results['inductive']['GCN']['mean'] - dataset_results['inductive']['MLP']['mean']
        ind_sage_adv = dataset_results['inductive']['GraphSAGE']['mean'] - dataset_results['inductive']['MLP']['mean']

        trans_winner = 'GNN' if trans_gcn_adv > 0.02 else ('MLP' if trans_gcn_adv < -0.02 else 'Tie')
        ind_winner = 'GNN' if max(ind_gcn_adv, ind_sage_adv) > 0.02 else ('MLP' if max(ind_gcn_adv, ind_sage_adv) < -0.02 else 'Tie')

        dataset_results['spi'] = spi
        dataset_results['spi_prediction'] = spi_prediction
        dataset_results['transductive_winner'] = trans_winner
        dataset_results['inductive_winner'] = ind_winner
        dataset_results['spi_correct_transductive'] = (spi_prediction == trans_winner) or (trans_winner == 'Tie')
        dataset_results['spi_correct_inductive'] = (spi_prediction == ind_winner) or (ind_winner == 'Tie')

        print(f"\nSPI Analysis:")
        print(f"  SPI = {spi:.3f}, Prediction = {spi_prediction}")
        print(f"  Transductive Winner: {trans_winner} (GCN Adv = {trans_gcn_adv*100:+.1f}%)")
        print(f"  Inductive Winner: {ind_winner} (GCN Adv = {ind_gcn_adv*100:+.1f}%, SAGE Adv = {ind_sage_adv*100:+.1f}%)")

        results['datasets'][dataset_name] = dataset_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SPI Predictions in Transductive vs Inductive Settings")
    print("=" * 70)

    trans_correct = 0
    ind_correct = 0
    total = len(results['datasets'])

    print(f"\n{'Dataset':<12} {'h':>6} {'SPI':>6} {'Pred':>6} {'Trans':>8} {'Ind':>8} {'Trans OK':>10} {'Ind OK':>10}")
    print("-" * 80)

    for dataset_name, dataset_results in results['datasets'].items():
        h = dataset_results['homophily']
        spi = dataset_results['spi']
        pred = dataset_results['spi_prediction']
        trans_win = dataset_results['transductive_winner']
        ind_win = dataset_results['inductive_winner']
        trans_ok = 'Y' if dataset_results['spi_correct_transductive'] else 'N'
        ind_ok = 'Y' if dataset_results['spi_correct_inductive'] else 'N'

        print(f"{dataset_name:<12} {h:>6.2f} {spi:>6.2f} {pred:>6} {trans_win:>8} {ind_win:>8} {trans_ok:>10} {ind_ok:>10}")

        if dataset_results['spi_correct_transductive']:
            trans_correct += 1
        if dataset_results['spi_correct_inductive']:
            ind_correct += 1

    results['summary'] = {
        'transductive_accuracy': trans_correct / total,
        'inductive_accuracy': ind_correct / total,
        'total_datasets': total
    }

    print(f"\n{'='*70}")
    print(f"SPI Prediction Accuracy:")
    print(f"  Transductive: {trans_correct}/{total} ({trans_correct/total*100:.1f}%)")
    print(f"  Inductive:    {ind_correct}/{total} ({ind_correct/total*100:.1f}%)")
    print(f"{'='*70}")

    # Key Finding: SAGE performance
    print("\n" + "=" * 70)
    print("KEY FINDING: GraphSAGE Inductive Degradation")
    print("=" * 70)

    for dataset_name, dataset_results in results['datasets'].items():
        sage_deg = dataset_results['inductive']['GraphSAGE']['degradation']
        gcn_deg = dataset_results['inductive']['GCN']['degradation']
        mlp_deg = dataset_results['inductive']['MLP']['degradation']

        print(f"{dataset_name}: SAGE deg={sage_deg:.1f}%, GCN deg={gcn_deg:.1f}%, MLP deg={mlp_deg:.1f}%")

    # Save results
    output_path = Path(__file__).parent / 'inductive_learning_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    results = main()
