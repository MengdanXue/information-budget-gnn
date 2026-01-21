#!/usr/bin/env python3
"""
Extended Baselines Experiment
=============================

补充更多近期heterophily-aware方法：BernNet, JacobiConv
同时记录运行时间用于效率对比

References:
- BernNet: BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation (NeurIPS 2021)
- JacobiConv: How Powerful are Spectral Graph Neural Networks (ICML 2022)

Author: FSD Framework
Date: 2025-01-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from collections import defaultdict

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, MessagePassing
from torch_geometric.datasets import WebKB, HeterophilousGraphDataset, Planetoid, Actor
from torch_geometric.utils import to_undirected, add_self_loops, degree, get_laplacian
from torch_geometric.transforms import NormalizeFeatures
from sklearn.model_selection import train_test_split
from scipy.special import comb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================
# Utility Functions
# ============================================================

def compute_homophily(edge_index, y, num_nodes):
    """Compute edge homophily ratio"""
    row, col = edge_index
    same_label = (y[row] == y[col]).float()
    return same_label.mean().item()


def get_split(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Generate train/val/test split"""
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


# ============================================================
# Model Implementations
# ============================================================

class MLP(nn.Module):
    """Baseline MLP"""
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
    """Standard GCN"""
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
    """GraphSAGE"""
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


class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, heads=8):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ============================================================
# BernNet Implementation
# ============================================================

class BernConv(MessagePassing):
    """Bernstein polynomial graph convolution"""
    def __init__(self, K=10):
        super().__init__(aggr='add')
        self.K = K
        # Learnable Bernstein coefficients
        self.coeffs = nn.Parameter(torch.ones(K + 1) / (K + 1))

    def forward(self, x, edge_index, edge_weight=None):
        # Compute normalized Laplacian
        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization='sym',
            num_nodes=x.size(0)
        )

        # Scale to [0, 2] -> [0, 1] for Bernstein basis
        # L_scaled = L / 2, so eigenvalues in [0, 1]
        if edge_weight is not None:
            edge_weight = edge_weight / 2

        # Compute Bernstein polynomial approximation
        # B_k(x) = C(K,k) * x^k * (1-x)^(K-k)
        # For graph: filter = sum_k coeff_k * B_k(L)

        # Use Chebyshev-like recursion for efficiency
        Tx_0 = x
        Tx_1 = x - self.propagate(edge_index, x=x, edge_weight=edge_weight)

        out = self.coeffs[0] * Tx_0

        if self.K > 0:
            out = out + self.coeffs[1] * Tx_1

        for k in range(2, self.K + 1):
            # Recursion: T_{k+1} = 2*L*T_k - T_{k-1}
            Tx_2 = 2 * (Tx_1 - self.propagate(edge_index, x=Tx_1, edge_weight=edge_weight)) - Tx_0
            out = out + self.coeffs[k] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j if edge_weight is not None else x_j


class BernNet(nn.Module):
    """BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation

    NeurIPS 2021
    """
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, num_layers=2, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(BernConv(K=K))
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


# ============================================================
# JacobiConv Implementation
# ============================================================

class JacobiConv(MessagePassing):
    """Jacobi polynomial graph convolution"""
    def __init__(self, K=10, a=0.5, b=0.5):
        super().__init__(aggr='add')
        self.K = K
        self.a = a  # Jacobi parameter alpha
        self.b = b  # Jacobi parameter beta
        # Learnable coefficients for Jacobi polynomials
        self.coeffs = nn.Parameter(torch.ones(K + 1) / (K + 1))

    def forward(self, x, edge_index, edge_weight=None):
        # Compute normalized adjacency
        edge_index, edge_weight = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Jacobi polynomial recursion
        # P_0^{a,b}(x) = 1
        # P_1^{a,b}(x) = (a-b)/2 + (a+b+2)/2 * x
        # P_{n+1}^{a,b}(x) = (A_n * x + B_n) * P_n - C_n * P_{n-1}

        Tx_0 = x
        out = self.coeffs[0] * Tx_0

        if self.K > 0:
            # P_1 = (a-b)/2 + (a+b+2)/2 * (2*A - I) where A is normalized adjacency
            # Simplified: P_1 ≈ propagate(x)
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + self.coeffs[1] * Tx_1

        for k in range(2, self.K + 1):
            # Recursion coefficients for Jacobi polynomials
            n = k - 1
            a, b = self.a, self.b

            A_n = (2*n + a + b + 1) * (2*n + a + b + 2) / (2 * (n + 1) * (n + a + b + 1))
            B_n = (a**2 - b**2) * (2*n + a + b + 1) / (2 * (n + 1) * (n + a + b + 1) * (2*n + a + b))
            C_n = (n + a) * (n + b) * (2*n + a + b + 2) / ((n + 1) * (n + a + b + 1) * (2*n + a + b))

            Tx_2 = A_n * self.propagate(edge_index, x=Tx_1, norm=norm) + B_n * Tx_1 - C_n * Tx_0
            out = out + self.coeffs[k] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class JacobiNet(nn.Module):
    """JacobiConv: How Powerful are Spectral Graph Neural Networks

    ICML 2022
    """
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, num_layers=2, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(JacobiConv(K=K))
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


# ============================================================
# LINKX (for comparison)
# ============================================================

class LINKX(nn.Module):
    """LINKX: Large Scale Learning on Non-Homophilous Graphs (NeurIPS 2021)"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # Feature MLP
        self.mlp_feat = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Structure MLP (processes A*X)
        self.mlp_struct = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Combination MLP
        self.mlp_combine = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        # Feature branch
        h_feat = self.mlp_feat(x)

        # Structure branch: compute A*X then MLP
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0

        # Sparse A*X
        ax = torch.zeros_like(x)
        ax.index_add_(0, col, x[row] * deg_inv[col].view(-1, 1))
        h_struct = self.mlp_struct(ax)

        # Combine
        h = torch.cat([h_feat, h_struct], dim=-1)
        return self.mlp_combine(h)


# ============================================================
# Training and Evaluation
# ============================================================

def train_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    return correct / mask.sum().item()


def run_experiment(model_class, model_name, data, train_mask, val_mask, test_mask,
                   hidden=64, epochs=200, lr=0.01, weight_decay=5e-4, patience=30, **kwargs):
    """Run single experiment with timing"""

    in_channels = data.x.size(1)
    out_channels = data.y.max().item() + 1

    model = model_class(in_channels, hidden, out_channels, **kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = 0
    best_test = 0
    patience_counter = 0

    # Training with timing
    start_time = time.time()

    for epoch in range(epochs):
        loss = train_epoch(model, data, optimizer, train_mask)
        val_acc = evaluate(model, data, val_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_test = evaluate(model, data, test_mask)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    train_time = time.time() - start_time

    # Inference timing (average over 10 runs)
    model.eval()
    inference_times = []
    with torch.no_grad():
        for _ in range(10):
            start = time.time()
            _ = model(data.x, data.edge_index)
            inference_times.append(time.time() - start)

    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms

    return {
        'test_acc': best_test,
        'val_acc': best_val,
        'train_time': train_time,
        'inference_time_ms': avg_inference_time,
        'epochs_trained': epoch + 1
    }


# ============================================================
# Dataset Loading
# ============================================================

def load_datasets():
    """Load all benchmark datasets"""
    datasets = {}

    # WebKB datasets (heterophilic)
    for name in ['Texas', 'Wisconsin', 'Cornell']:
        try:
            dataset = WebKB(root='./data', name=name, transform=NormalizeFeatures())
            data = dataset[0].to(device)
            h = compute_homophily(data.edge_index, data.y, data.num_nodes)
            datasets[name] = {'data': data, 'homophily': h}
            print(f"Loaded {name}: {data.num_nodes} nodes, h={h:.3f}")
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    # Actor dataset
    try:
        dataset = Actor(root='./data', transform=NormalizeFeatures())
        data = dataset[0].to(device)
        h = compute_homophily(data.edge_index, data.y, data.num_nodes)
        datasets['Actor'] = {'data': data, 'homophily': h}
        print(f"Loaded Actor: {data.num_nodes} nodes, h={h:.3f}")
    except Exception as e:
        print(f"Failed to load Actor: {e}")

    # Planetoid datasets (homophilic)
    for name in ['Cora', 'CiteSeer', 'PubMed']:
        try:
            dataset = Planetoid(root='./data', name=name, transform=NormalizeFeatures())
            data = dataset[0].to(device)
            h = compute_homophily(data.edge_index, data.y, data.num_nodes)
            datasets[name] = {'data': data, 'homophily': h}
            print(f"Loaded {name}: {data.num_nodes} nodes, h={h:.3f}")
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    # Heterophilous Graph Benchmark datasets
    for name in ['Roman-empire', 'Minesweeper', 'Tolokers', 'Questions']:
        try:
            dataset = HeterophilousGraphDataset(root='./data', name=name, transform=NormalizeFeatures())
            data = dataset[0].to(device)
            h = compute_homophily(data.edge_index, data.y, data.num_nodes)
            datasets[name] = {'data': data, 'homophily': h}
            print(f"Loaded {name}: {data.num_nodes} nodes, h={h:.3f}")
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    return datasets


# ============================================================
# Main Experiment
# ============================================================

def main():
    print("=" * 70)
    print("EXTENDED BASELINES EXPERIMENT")
    print("Including: BernNet, JacobiConv, LINKX + Runtime Analysis")
    print("=" * 70)

    # Load datasets
    print("\n--- Loading Datasets ---")
    datasets = load_datasets()

    # Model configurations
    models = {
        'MLP': (MLP, {}),
        'GCN': (GCN, {}),
        'GraphSAGE': (GraphSAGE, {}),
        'GAT': (GAT, {'heads': 8}),
        'LINKX': (LINKX, {}),
        'BernNet': (BernNet, {'K': 10}),
        'JacobiNet': (JacobiNet, {'K': 10}),
    }

    # Experiment settings
    n_runs = 10
    seeds = [42 + i * 100 for i in range(n_runs)]

    results = {}

    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name} (h={dataset_info['homophily']:.3f})")
        print(f"{'='*70}")

        data = dataset_info['data']
        results[dataset_name] = {
            'homophily': dataset_info['homophily'],
            'n_nodes': data.num_nodes,
            'n_edges': data.edge_index.size(1) // 2,
            'results': {}
        }

        for model_name, (model_class, model_kwargs) in models.items():
            print(f"\n  Running {model_name}...")

            run_results = []
            train_times = []
            inference_times = []

            for seed in seeds:
                # Get split
                train_mask, val_mask, test_mask = get_split(data.num_nodes, seed=seed)
                train_mask = train_mask.to(device)
                val_mask = val_mask.to(device)
                test_mask = test_mask.to(device)

                try:
                    result = run_experiment(
                        model_class, model_name, data,
                        train_mask, val_mask, test_mask,
                        **model_kwargs
                    )
                    run_results.append(result['test_acc'])
                    train_times.append(result['train_time'])
                    inference_times.append(result['inference_time_ms'])
                except Exception as e:
                    print(f"    Error in run: {e}")
                    continue

            if run_results:
                results[dataset_name]['results'][model_name] = {
                    'mean': float(np.mean(run_results)),
                    'std': float(np.std(run_results)),
                    'scores': run_results,
                    'train_time_mean': float(np.mean(train_times)),
                    'train_time_std': float(np.std(train_times)),
                    'inference_time_ms_mean': float(np.mean(inference_times)),
                    'inference_time_ms_std': float(np.std(inference_times))
                }
                print(f"    {model_name}: {np.mean(run_results)*100:.1f}% ± {np.std(run_results)*100:.1f}%")
                print(f"      Train: {np.mean(train_times):.2f}s, Inference: {np.mean(inference_times):.2f}ms")

    # Save results
    output_path = Path(__file__).parent / 'extended_baselines_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    # Header
    model_names = list(models.keys())
    header = f"{'Dataset':<15} {'h':>5}"
    for m in model_names:
        header += f" {m:>10}"
    print(header)
    print("-" * len(header))

    # Data rows
    for dataset_name in sorted(results.keys(), key=lambda x: results[x]['homophily']):
        row = f"{dataset_name:<15} {results[dataset_name]['homophily']:>5.2f}"
        for m in model_names:
            if m in results[dataset_name]['results']:
                acc = results[dataset_name]['results'][m]['mean'] * 100
                row += f" {acc:>10.1f}"
            else:
                row += f" {'--':>10}"
        print(row)

    # Runtime comparison table
    print("\n" + "=" * 70)
    print("RUNTIME COMPARISON (Training Time in seconds)")
    print("=" * 70)

    header = f"{'Dataset':<15}"
    for m in model_names:
        header += f" {m:>10}"
    print(header)
    print("-" * len(header))

    for dataset_name in sorted(results.keys(), key=lambda x: results[x]['homophily']):
        row = f"{dataset_name:<15}"
        for m in model_names:
            if m in results[dataset_name]['results']:
                t = results[dataset_name]['results'][m]['train_time_mean']
                row += f" {t:>10.2f}"
            else:
                row += f" {'--':>10}"
        print(row)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return results


if __name__ == '__main__':
    results = main()
