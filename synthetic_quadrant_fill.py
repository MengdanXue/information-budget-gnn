"""
Synthetic Dataset Generation for Quadrant Filling
==================================================

生成合成数据集来填补Q2、Q3、Q4象限的空白。
使用cSBM (contextual Stochastic Block Model) 方法。

目标：
- 生成可控的合成数据集
- 验证Two-Factor Framework在更广泛设置下的有效性
- 确保每个象限有足够样本
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import json

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def generate_csbm_graph(n_nodes=1000, n_features=20, homophily=0.5,
                        feature_sep=0.5, avg_degree=15, seed=42):
    """
    Generate contextual SBM graph with controlled homophily and feature separability.

    Parameters:
    - n_nodes: Number of nodes
    - n_features: Feature dimension
    - homophily: Edge homophily (probability same-class edges)
    - feature_sep: Feature separability (higher = more separable classes)
    - avg_degree: Average node degree
    - seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Binary classification
    n_per_class = n_nodes // 2
    labels = torch.tensor([0] * n_per_class + [1] * (n_nodes - n_per_class))

    # Generate features with controlled separability
    # Class 0: centered at [sep, 0, 0, ...]
    # Class 1: centered at [-sep, 0, 0, ...]
    features = torch.randn(n_nodes, n_features)
    features[:n_per_class, 0] += feature_sep
    features[n_per_class:, 0] -= feature_sep

    # Generate edges with controlled homophily
    n_edges = int(avg_degree * n_nodes / 2)
    edges = []

    for _ in range(n_edges):
        u = np.random.randint(n_nodes)
        if np.random.random() < homophily:
            # Same class edge
            if labels[u] == 0:
                v = np.random.randint(n_per_class)
            else:
                v = np.random.randint(n_per_class, n_nodes)
        else:
            # Different class edge
            if labels[u] == 0:
                v = np.random.randint(n_per_class, n_nodes)
            else:
                v = np.random.randint(n_per_class)

        if u != v:
            edges.append([u, v])
            edges.append([v, u])  # Undirected

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return features, edge_index, labels


def train_and_evaluate(model, x, edge_index, labels, train_mask, val_mask, test_mask,
                       epochs=200, patience=50):
    """Train and evaluate model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
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


def evaluate_synthetic(n_nodes, n_features, homophily, feature_sep, n_runs=5):
    """Evaluate on synthetic dataset"""
    mlp_scores = []
    gcn_scores = []

    for seed in range(n_runs):
        # Generate data
        x, edge_index, labels = generate_csbm_graph(
            n_nodes=n_nodes, n_features=n_features,
            homophily=homophily, feature_sep=feature_sep, seed=seed
        )

        x = x.to(device)
        edge_index = edge_index.to(device)
        labels = labels.to(device)

        # Create splits
        n = n_nodes
        perm = torch.randperm(n)
        train_mask = torch.zeros(n, dtype=torch.bool, device=device)
        val_mask = torch.zeros(n, dtype=torch.bool, device=device)
        test_mask = torch.zeros(n, dtype=torch.bool, device=device)
        train_mask[perm[:int(0.6*n)]] = True
        val_mask[perm[int(0.6*n):int(0.8*n)]] = True
        test_mask[perm[int(0.8*n):]] = True

        # MLP
        mlp = MLP(n_features, 64, 2).to(device)
        mlp_acc = train_and_evaluate(mlp, x, edge_index, labels, train_mask, val_mask, test_mask)
        mlp_scores.append(mlp_acc)

        # GCN
        gcn = GCN(n_features, 64, 2).to(device)
        gcn_acc = train_and_evaluate(gcn, x, edge_index, labels, train_mask, val_mask, test_mask)
        gcn_scores.append(gcn_acc)

    return {
        'mlp_mean': np.mean(mlp_scores),
        'mlp_std': np.std(mlp_scores),
        'gcn_mean': np.mean(gcn_scores),
        'gcn_std': np.std(gcn_scores),
        'delta': np.mean(gcn_scores) - np.mean(mlp_scores)
    }


def classify_quadrant(mlp_acc, h, fs_thresh=0.65, h_thresh=0.5):
    """Classify into quadrant"""
    if mlp_acc >= fs_thresh:
        if h >= h_thresh:
            return 'Q1', 'GCN_maybe'
        else:
            return 'Q2', 'MLP'
    else:
        if h >= h_thresh:
            return 'Q3', 'GCN'
        else:
            return 'Q4', 'Uncertain'


def main():
    print("=" * 80)
    print("SYNTHETIC DATASET GENERATION FOR QUADRANT FILLING")
    print("=" * 80)

    # Define configurations for each quadrant
    # Q2: High FS (high feature_sep), Low h
    # Q3: Low FS (low feature_sep), High h
    # Q4: Low FS, Low h

    configs = [
        # Q2 additions: High FS (sep=2.0-3.0), Low h (0.1-0.4)
        {'name': 'cSBM-Q2-1', 'h': 0.15, 'sep': 2.5, 'target_q': 'Q2'},
        {'name': 'cSBM-Q2-2', 'h': 0.25, 'sep': 2.0, 'target_q': 'Q2'},
        {'name': 'cSBM-Q2-3', 'h': 0.35, 'sep': 2.2, 'target_q': 'Q2'},
        {'name': 'cSBM-Q2-4', 'h': 0.10, 'sep': 3.0, 'target_q': 'Q2'},

        # Q3 additions: Low FS (sep=0.3-0.8), High h (0.6-0.9)
        {'name': 'cSBM-Q3-1', 'h': 0.70, 'sep': 0.5, 'target_q': 'Q3'},
        {'name': 'cSBM-Q3-2', 'h': 0.80, 'sep': 0.4, 'target_q': 'Q3'},
        {'name': 'cSBM-Q3-3', 'h': 0.75, 'sep': 0.6, 'target_q': 'Q3'},
        {'name': 'cSBM-Q3-4', 'h': 0.85, 'sep': 0.3, 'target_q': 'Q3'},

        # Q4 additions: Low FS (sep=0.3-0.8), Low h (0.2-0.4)
        {'name': 'cSBM-Q4-1', 'h': 0.25, 'sep': 0.5, 'target_q': 'Q4'},
        {'name': 'cSBM-Q4-2', 'h': 0.35, 'sep': 0.4, 'target_q': 'Q4'},
        {'name': 'cSBM-Q4-3', 'h': 0.30, 'sep': 0.6, 'target_q': 'Q4'},
        {'name': 'cSBM-Q4-4', 'h': 0.40, 'sep': 0.5, 'target_q': 'Q4'},
    ]

    results = []

    print("\nGenerating and evaluating synthetic datasets...\n")

    for cfg in configs:
        print(f"Evaluating {cfg['name']} (target: {cfg['target_q']})...", end=" ", flush=True)

        res = evaluate_synthetic(
            n_nodes=1000, n_features=20,
            homophily=cfg['h'], feature_sep=cfg['sep'],
            n_runs=5
        )

        quadrant, prediction = classify_quadrant(res['mlp_mean'], cfg['h'])
        winner = 'GCN' if res['delta'] > 0.01 else ('MLP' if res['delta'] < -0.01 else 'Tie')

        # Check if correct
        if quadrant == 'Q4':
            correct = True  # Uncertain is always "correct"
        elif prediction == 'MLP':
            correct = winner == 'MLP'
        elif prediction == 'GCN':
            correct = winner == 'GCN'
        else:  # GCN_maybe
            correct = winner in ['GCN', 'Tie']

        result = {
            'name': cfg['name'],
            'target_quadrant': cfg['target_q'],
            'homophily': cfg['h'],
            'feature_sep': cfg['sep'],
            'mlp_mean': res['mlp_mean'],
            'gcn_mean': res['gcn_mean'],
            'delta': res['delta'],
            'actual_quadrant': quadrant,
            'prediction': prediction,
            'winner': winner,
            'correct': correct
        }
        results.append(result)

        status = "OK" if correct else "WRONG"
        match = "MATCH" if quadrant == cfg['target_q'] else "MISMATCH"
        print(f"h={cfg['h']:.2f} MLP={res['mlp_mean']:.3f} Delta={res['delta']:+.3f} "
              f"Q={quadrant} [{match}] [{status}]")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Count by quadrant
    from collections import defaultdict
    quadrant_results = defaultdict(list)
    for r in results:
        quadrant_results[r['actual_quadrant']].append(r)

    print("\nResults by Quadrant:")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        qr = quadrant_results[q]
        if qr:
            correct = sum(1 for r in qr if r['correct'])
            print(f"  {q}: {len(qr)} datasets, {correct}/{len(qr)} correct")

    # Overall accuracy
    decisive = [r for r in results if r['actual_quadrant'] != 'Q4']
    correct = sum(1 for r in decisive if r['correct'])
    print(f"\nOverall decisive accuracy: {correct}/{len(decisive)} = {correct/len(decisive)*100:.1f}%")

    # Save results
    with open('synthetic_quadrant_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: synthetic_quadrant_results.json")

    return results


if __name__ == '__main__':
    results = main()
