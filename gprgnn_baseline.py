"""
GPR-GNN Baseline Comparison
===========================

GPR-GNN (ICLR 2021) 是heterophily-aware GNN的重要baseline。
本实验验证SPI预测与GPR-GNN性能的一致性。

核心问题：GPR-GNN是否能在Q2象限（高FS，低h）改善性能？
假设：即使GPR-GNN有learnable propagation weights，在Q2象限仍会失败。

Reference:
- Adaptive Universal Generalized PageRank Graph Neural Network (ICLR 2021)
- https://github.com/jianhao2016/GPRGNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
import json

from torch_geometric.nn import MessagePassing
from torch_geometric.datasets import WebKB, HeterophilousGraphDataset, Planetoid
from torch_geometric.utils import to_undirected, add_self_loops, degree
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================
# GPR-GNN Implementation
# ============================================================

class GPR_prop(MessagePassing):
    """
    GPR-GNN propagation layer with learnable weights.
    """
    def __init__(self, K, alpha, Init='PPR', Gamma=None, bias=True):
        super(GPR_prop, self).__init__(aggr='add')
        self.K = K
        self.Init = Init
        self.alpha = alpha

        # Initialize GPR weights
        TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
        TEMP[-1] = (1 - alpha) ** K
        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # K-hop propagation with learnable weights
        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GPRGNN(nn.Module):
    """
    GPR-GNN: Generalized PageRank GNN

    Key innovation: Learnable propagation weights that can adapt to
    both homophilic and heterophilic graphs.
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 K=10, alpha=0.1, dropout=0.5):
        super(GPRGNN, self).__init__()

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = GPR_prop(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return x


# ============================================================
# Other Models for Comparison
# ============================================================

class MLP(nn.Module):
    """Baseline MLP (no graph structure)"""
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
    """Standard GCN"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


# ============================================================
# Training and Evaluation
# ============================================================

def train_and_evaluate(model, x, edge_index, labels, train_mask, val_mask, test_mask,
                       lr=0.01, weight_decay=5e-4, epochs=200, patience=50):
    """Train model and return best test accuracy"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(dim=1)
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    return best_test_acc


def compute_homophily(edge_index, labels):
    """Compute edge homophily"""
    src, dst = edge_index.cpu().numpy()
    lab = labels.cpu().numpy()
    return (lab[src] == lab[dst]).mean()


# ============================================================
# Main Experiment
# ============================================================

def run_gprgnn_comparison(n_runs=10):
    """
    Compare GPR-GNN with MLP and GCN across different homophily regimes.
    """

    print("=" * 80)
    print("GPR-GNN BASELINE COMPARISON")
    print("=" * 80)
    print("\nQuestion: Can GPR-GNN's learnable propagation help in Q2 quadrant?")
    print("Hypothesis: Even GPR-GNN will fail when features are sufficient + low h\n")

    # Datasets covering different regimes
    datasets_config = [
        # Q2 Quadrant: High FS, Low h
        ('Texas', WebKB, {'name': 'Texas'}),
        ('Wisconsin', WebKB, {'name': 'Wisconsin'}),
        ('Cornell', WebKB, {'name': 'Cornell'}),
        ('Roman-empire', HeterophilousGraphDataset, {'name': 'Roman-empire'}),
        # High h (Trust Region)
        ('Cora', Planetoid, {'name': 'Cora'}),
        ('CiteSeer', Planetoid, {'name': 'CiteSeer'}),
        ('PubMed', Planetoid, {'name': 'PubMed'}),
    ]

    model_classes = {
        'MLP': MLP,
        'GCN': GCN,
        'GPR-GNN': GPRGNN,
    }

    all_results = {}

    for ds_name, DatasetClass, kwargs in datasets_config:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            # Load dataset
            dataset = DatasetClass(root='./data', **kwargs)
            data = dataset[0]

            x = data.x.to(device)
            edge_index = to_undirected(data.edge_index).to(device)
            labels = data.y.to(device)
            n_nodes = data.num_nodes
            n_features = data.num_features
            n_classes = len(labels.unique())

            h = compute_homophily(edge_index, labels)
            spi = abs(2 * h - 1)

            print(f"  Nodes: {n_nodes}, Features: {n_features}, Classes: {n_classes}")
            print(f"  Homophily: {h:.4f}, SPI: {spi:.4f}")

            # Determine regime
            if h < 0.3:
                regime = "Q2 (Low h)"
            elif h > 0.7:
                regime = "Trust Region (High h)"
            else:
                regime = "Uncertain"
            print(f"  Regime: {regime}")

            results = {model_name: [] for model_name in model_classes}

            for seed in range(n_runs):
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Create splits
                indices = np.arange(n_nodes)
                train_idx, temp_idx = train_test_split(indices, train_size=0.6, random_state=seed)
                val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=seed)

                train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
                val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
                test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
                train_mask[train_idx] = True
                val_mask[val_idx] = True
                test_mask[test_idx] = True

                # Train each model
                for model_name, ModelClass in model_classes.items():
                    if model_name == 'GPR-GNN':
                        model = ModelClass(n_features, 64, n_classes, K=10, alpha=0.1).to(device)
                    else:
                        model = ModelClass(n_features, 64, n_classes).to(device)

                    acc = train_and_evaluate(model, x, edge_index, labels,
                                            train_mask, val_mask, test_mask)
                    results[model_name].append(acc)

            # Print results
            print(f"\n  Results ({n_runs} runs):")
            print(f"  {'Model':>12} {'Mean':>10} {'Std':>10} {'vs MLP':>10}")
            print("  " + "-" * 45)

            mlp_scores = results['MLP']
            mlp_mean = np.mean(mlp_scores)

            for model_name in model_classes:
                scores = results[model_name]
                mean = np.mean(scores)
                std = np.std(scores)
                diff = mean - mlp_mean
                print(f"  {model_name:>12} {mean:>10.3f} {std:>10.3f} {diff:>+10.3f}")

            all_results[ds_name] = {
                'homophily': h,
                'spi': spi,
                'regime': regime,
                'n_nodes': n_nodes,
                'results': {k: {'mean': np.mean(v), 'std': np.std(v), 'scores': v}
                           for k, v in results.items()}
            }

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # ============================================================
    # Summary Analysis
    # ============================================================

    print("\n" + "=" * 80)
    print("SUMMARY: GPR-GNN vs SPI Predictions")
    print("=" * 80)

    print("\n| Dataset | h | SPI | Regime | MLP | GCN | GPR-GNN | Best |")
    print("|---------|-----|-----|--------|-----|-----|---------|------|")

    for ds_name, data in all_results.items():
        h = data['homophily']
        spi = data['spi']
        regime = data['regime'][:10]
        mlp = data['results']['MLP']['mean']
        gcn = data['results']['GCN']['mean']
        gpr = data['results']['GPR-GNN']['mean']

        best = 'MLP' if mlp >= max(gcn, gpr) else ('GCN' if gcn >= gpr else 'GPR-GNN')

        print(f"| {ds_name:13} | {h:.2f} | {spi:.2f} | {regime:10} | {mlp:.3f} | {gcn:.3f} | {gpr:.3f} | {best} |")

    # Q2 Analysis
    print("\n" + "-" * 60)
    print("Q2 Quadrant Analysis (High FS, Low h):")
    print("-" * 60)

    q2_datasets = ['Texas', 'Wisconsin', 'Cornell', 'Roman-empire']
    gprgnn_wins = 0
    mlp_wins = 0

    for ds_name in q2_datasets:
        if ds_name in all_results:
            data = all_results[ds_name]
            mlp = data['results']['MLP']['mean']
            gpr = data['results']['GPR-GNN']['mean']
            gcn = data['results']['GCN']['mean']

            winner = 'MLP' if mlp >= max(gcn, gpr) else ('GPR-GNN' if gpr >= gcn else 'GCN')

            if winner == 'GPR-GNN':
                gprgnn_wins += 1
            elif winner == 'MLP':
                mlp_wins += 1

            print(f"  {ds_name}: GPR-GNN {gpr:.3f} vs MLP {mlp:.3f} -> {winner} wins")

    print(f"\nConclusion: In Q2 quadrant, GPR-GNN wins {gprgnn_wins}/4, MLP wins {mlp_wins}/4")

    if mlp_wins >= 3:
        print("\n[CONFIRMED] Even GPR-GNN cannot overcome the Q2 quadrant challenge.")
        print("This supports our claim: when features are sufficient and h is low,")
        print("NO GNN architecture can help - the structure is fundamentally noise.")
    else:
        print("\n[INCONCLUSIVE] GPR-GNN shows some improvement in Q2 quadrant.")

    # Save results
    results_file = 'gprgnn_baseline_results.json'
    json_results = {}
    for ds_name, data in all_results.items():
        json_results[ds_name] = {
            'homophily': float(data['homophily']),
            'spi': float(data['spi']),
            'regime': data['regime'],
            'n_nodes': int(data['n_nodes']),
            'results': {
                k: {
                    'mean': float(v['mean']),
                    'std': float(v['std']),
                    'scores': [float(s) for s in v['scores']]
                }
                for k, v in data['results'].items()
            }
        }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == '__main__':
    results = run_gprgnn_comparison(n_runs=10)
