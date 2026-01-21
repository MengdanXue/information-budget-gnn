"""
Validate Pattern Predictability Metric
======================================

验证Pattern Score能否预测GCN-MLP的性能差异

关键假说：Pattern Score越高，GCN相对MLP的优势越大
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from pattern_predictability_metric import metric_pattern_predictability
from patterned_heterophily_experiment import create_patterned_heterophily_graph
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def train_model(model, x, edge_index, labels, train_mask, val_mask, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_acc = 0

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
            if val_acc > best_val_acc:
                best_val_acc = val_acc

    return best_val_acc


def run_validation():
    print("=" * 70)
    print("VALIDATING: Pattern Score predicts GCN-MLP Advantage")
    print("=" * 70)

    # 加载Cora
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    labels = data.y
    n_nodes = data.num_nodes
    n_features = data.num_features
    n_classes = dataset.num_classes
    n_edges = data.edge_index.shape[1] // 2

    # 创建固定的splits
    indices = np.arange(n_nodes)
    train_idx, temp_idx = train_test_split(indices, train_size=0.6, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=42)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    x_device = data.x.to(device)
    labels_device = labels.to(device)
    train_mask_device = train_mask.to(device)
    val_mask_device = val_mask.to(device)

    results = []

    # 测试不同的h和alpha组合
    h_values = [0.1, 0.2, 0.3, 0.5, 0.7]
    alpha_values = [0.0, 0.5, 1.0]

    print("\nRunning experiments...")

    for target_h in h_values:
        for alpha in alpha_values:
            # 创建图
            edge_index = create_patterned_heterophily_graph(
                n_nodes, labels, n_edges, alpha, target_h
            )

            # 计算Pattern Score
            metrics = metric_pattern_predictability(edge_index, labels)
            pattern_score = metrics['pattern_score']

            # 训练模型
            edge_index_device = edge_index.to(device)

            gcn = GCN(n_features, 64, n_classes).to(device)
            gcn_acc = train_model(gcn, x_device, edge_index_device, labels_device,
                                 train_mask_device, val_mask_device)

            mlp = MLP(n_features, 64, n_classes).to(device)
            mlp_acc = train_model(mlp, x_device, edge_index_device, labels_device,
                                 train_mask_device, val_mask_device)

            gcn_advantage = gcn_acc - mlp_acc

            results.append({
                'target_h': target_h,
                'alpha': alpha,
                'actual_h': metrics['homophily'],
                'pattern_score': pattern_score,
                'gcn_acc': gcn_acc,
                'mlp_acc': mlp_acc,
                'gcn_advantage': gcn_advantage
            })

            print(f"  h={target_h}, alpha={alpha}: Score={pattern_score:.3f}, GCN-MLP={gcn_advantage:+.3f}")

    # 计算相关性
    pattern_scores = [r['pattern_score'] for r in results]
    gcn_advantages = [r['gcn_advantage'] for r in results]

    correlation = np.corrcoef(pattern_scores, gcn_advantages)[0, 1]

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(f"\n{'h':>6} {'alpha':>6} {'Score':>8} {'GCN':>8} {'MLP':>8} {'GCN-MLP':>10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x['pattern_score']):
        print(f"{r['target_h']:>6.1f} {r['alpha']:>6.1f} {r['pattern_score']:>8.3f} "
              f"{r['gcn_acc']:>8.3f} {r['mlp_acc']:>8.3f} {r['gcn_advantage']:>+10.3f}")

    print("\n" + "=" * 70)
    print(f"CORRELATION: Pattern Score vs GCN-MLP Advantage = {correlation:.3f}")
    print("=" * 70)

    if correlation > 0.7:
        print("\n[STRONG] Pattern Score is a strong predictor of GCN advantage!")
    elif correlation > 0.5:
        print("\n[MODERATE] Pattern Score is a moderate predictor of GCN advantage.")
    else:
        print("\n[WEAK] Pattern Score is a weak predictor.")

    # 保存结果
    output = {
        'experiment': 'pattern_score_validation',
        'correlation': correlation,
        'results': results
    }

    with open('pattern_score_validation_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    return correlation, results


if __name__ == '__main__':
    run_validation()
