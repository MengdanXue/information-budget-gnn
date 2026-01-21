"""
Expand Heterophilic Dataset Validation
======================================

尝试添加更多异配图数据集来验证Direction_Residual
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.datasets import (
    HeterophilousGraphDataset, WikipediaNetwork, WebKB, Actor
)
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


def compute_direction(data):
    """计算Direction指标"""
    edge_index = data.edge_index
    labels = data.y
    n_nodes = data.num_nodes
    n_classes = len(labels.unique())

    # 计算每个节点的邻居标签分布
    src, dst = edge_index
    neighbor_labels = labels[dst]

    # 统计每个节点的邻居标签计数
    node_neighbor_counts = torch.zeros(n_nodes, n_classes)
    for i in range(edge_index.size(1)):
        node_idx = src[i].item()
        label = neighbor_labels[i].item()
        node_neighbor_counts[node_idx, label] += 1

    # 计算dominant neighbor class
    dominant_classes = node_neighbor_counts.argmax(dim=1)

    # 计算match rate
    has_neighbors = node_neighbor_counts.sum(dim=1) > 0
    matches = (dominant_classes == labels) & has_neighbors
    match_rate = matches.float().sum() / has_neighbors.float().sum()

    # 计算class-corrected baseline
    class_counts = torch.bincount(labels, minlength=n_classes).float()
    class_priors = class_counts / n_nodes
    corrected_baseline = class_priors.max().item()

    # Direction = match_rate - baseline
    direction = match_rate.item() - corrected_baseline

    # Edge homophily
    same_label = (labels[src] == labels[dst]).float().mean().item()

    return {
        'direction': direction,
        'homophily': same_label,
        'match_rate': match_rate.item(),
        'baseline': corrected_baseline
    }


def train_and_evaluate(model, x, edge_index, labels, train_mask, val_mask, test_mask, epochs=200):
    """训练并评估模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_acc = 0
    best_test_acc = 0
    patience = 0
    max_patience = 50

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
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break

    return best_val_acc, best_test_acc


def evaluate_dataset(name, data, n_runs=5):
    """评估单个数据集"""
    x = data.x.to(device)
    edge_index = to_undirected(data.edge_index).to(device)
    labels = data.y.to(device)
    n_nodes = data.num_nodes
    n_features = data.num_features
    n_classes = len(labels.unique())

    # 计算Direction和Homophily
    metrics = compute_direction(data)

    results = {'gcn': [], 'mlp': []}

    for run in range(n_runs):
        # 创建随机split
        indices = np.arange(n_nodes)
        train_idx, temp_idx = train_test_split(indices, train_size=0.6, random_state=42+run)
        val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=42+run)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # GCN
        gcn = GCN(n_features, 64, n_classes).to(device)
        _, gcn_acc = train_and_evaluate(gcn, x, edge_index, labels, train_mask, val_mask, test_mask)
        results['gcn'].append(gcn_acc)

        # MLP
        mlp = MLP(n_features, 64, n_classes).to(device)
        _, mlp_acc = train_and_evaluate(mlp, x, edge_index, labels, train_mask, val_mask, test_mask)
        results['mlp'].append(mlp_acc)

        print(f"    Run {run+1}/{n_runs}: GCN={gcn_acc:.3f}, MLP={mlp_acc:.3f}")

    return {
        'dataset': name,
        'n_nodes': n_nodes,
        'n_features': n_features,
        'n_classes': n_classes,
        'homophily': metrics['homophily'],
        'direction': metrics['direction'],
        'gcn_mean': np.mean(results['gcn']),
        'gcn_std': np.std(results['gcn']),
        'mlp_mean': np.mean(results['mlp']),
        'mlp_std': np.std(results['mlp']),
        'gcn_mlp': np.mean(results['gcn']) - np.mean(results['mlp']),
    }


def main():
    print("=" * 80)
    print("EXPANDING HETEROPHILIC DATASET VALIDATION")
    print("=" * 80)

    all_results = []

    # 尝试加载HeterophilousGraphDataset中的数据集
    hetero_datasets = ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions']

    for name in hetero_datasets:
        print(f"\n{'='*60}")
        print(f"Trying: {name}")
        print(f"{'='*60}")

        try:
            dataset = HeterophilousGraphDataset(root='./data', name=name)
            data = dataset[0]
            print(f"  Loaded: {data.num_nodes} nodes, {data.num_edges} edges")

            result = evaluate_dataset(name, data, n_runs=3)
            all_results.append(result)

            print(f"\n  Summary: h={result['homophily']:.3f}, Dir={result['direction']:+.3f}")
            print(f"  GCN: {result['gcn_mean']:.3f}, MLP: {result['mlp_mean']:.3f}")
            print(f"  GCN-MLP: {result['gcn_mlp']:+.3f}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # 加载线性回归系数
    with open('pattern_direction_enhanced_results.json', 'r') as f:
        direction_data = json.load(f)['results']

    all_h = np.array([r['edge_homophily'] for r in direction_data])
    all_dir = np.array([r['direction_corrected'] for r in direction_data])
    slope, intercept, _, _, _ = stats.linregress(all_h, all_dir)

    print("\n" + "=" * 80)
    print("EXPANDED RESULTS SUMMARY")
    print("=" * 80)

    if all_results:
        print(f"\nUsing coefficients: Direction = {slope:.3f} * h + {intercept:.3f}")
        print(f"\n{'Dataset':>15} {'h':>8} {'Dir':>8} {'Residual':>10} {'GCN-MLP':>10} {'Prediction':>12}")
        print("-" * 70)

        for r in sorted(all_results, key=lambda x: x['homophily']):
            residual = r['direction'] - (slope * r['homophily'] + intercept)

            # 预测
            if residual > 0.1:
                pred = "GCN wins"
                correct = r['gcn_mlp'] > 0
            elif residual < -0.1:
                pred = "MLP wins"
                correct = r['gcn_mlp'] < 0
            else:
                pred = "Neutral"
                correct = None

            status = "Y" if correct else ("N" if correct is False else "?")

            print(f"{r['dataset']:>15} {r['homophily']:>8.3f} {r['direction']:>+8.3f} "
                  f"{residual:>+10.3f} {r['gcn_mlp']:>+10.3f} {pred:>10} {status}")

        # 保存结果
        output = {
            'experiment': 'expanded_heterophilic_validation',
            'n_datasets': len(all_results),
            'results': all_results
        }

        with open('expanded_heterophilic_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: expanded_heterophilic_results.json")

    else:
        print("\nNo new datasets could be loaded successfully.")
        print("Falling back to synthetic data validation...")


if __name__ == '__main__':
    main()
