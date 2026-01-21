"""
Comprehensive GCN Evaluation on All Datasets
=============================================

目标：收集所有15个数据集的GCN vs MLP性能数据
验证Direction_Residual的预测力
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
    WebKB, Planetoid, WikipediaNetwork, Actor,
    Amazon, Coauthor, CitationFull
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


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

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


def load_dataset(name):
    """加载数据集"""
    name_lower = name.lower()

    try:
        if name_lower in ['texas', 'wisconsin', 'cornell']:
            dataset = WebKB(root='./data', name=name.capitalize())
            return dataset[0], 'WebKB'
        elif name_lower in ['squirrel', 'chameleon']:
            dataset = WikipediaNetwork(root='./data', name=name_lower)
            return dataset[0], 'Wikipedia'
        elif name_lower in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root='./data', name=name.capitalize())
            return dataset[0], 'Planetoid'
        elif name_lower == 'actor':
            dataset = Actor(root='./data/Actor')
            return dataset[0], 'Actor'
        elif name_lower in ['computers', 'photo']:
            dataset = Amazon(root='./data', name=name.capitalize())
            return dataset[0], 'Amazon'
        elif name_lower in ['cs', 'physics']:
            dataset = Coauthor(root='./data', name=name.upper())
            return dataset[0], 'Coauthor'
        else:
            return None, None
    except Exception as e:
        print(f"  Error loading {name}: {e}")
        return None, None


def evaluate_dataset(name, data, n_runs=5):
    """评估单个数据集"""

    x = data.x.to(device)
    edge_index = to_undirected(data.edge_index).to(device)
    labels = data.y.to(device)
    n_nodes = data.num_nodes
    n_features = data.num_features
    n_classes = len(labels.unique())

    # 计算homophily
    src, dst = data.edge_index
    homophily = (data.y[src] == data.y[dst]).float().mean().item()

    results = {
        'gcn': [], 'gat': [], 'sage': [], 'mlp': []
    }

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

        # GAT
        gat = GAT(n_features, 64, n_classes).to(device)
        _, gat_acc = train_and_evaluate(gat, x, edge_index, labels, train_mask, val_mask, test_mask)
        results['gat'].append(gat_acc)

        # SAGE
        sage = GraphSAGE(n_features, 64, n_classes).to(device)
        _, sage_acc = train_and_evaluate(sage, x, edge_index, labels, train_mask, val_mask, test_mask)
        results['sage'].append(sage_acc)

        # MLP
        mlp = MLP(n_features, 64, n_classes).to(device)
        _, mlp_acc = train_and_evaluate(mlp, x, edge_index, labels, train_mask, val_mask, test_mask)
        results['mlp'].append(mlp_acc)

        print(f"    Run {run+1}/{n_runs}: GCN={gcn_acc:.3f}, GAT={gat_acc:.3f}, "
              f"SAGE={sage_acc:.3f}, MLP={mlp_acc:.3f}")

    return {
        'dataset': name,
        'n_nodes': n_nodes,
        'n_features': n_features,
        'n_classes': n_classes,
        'homophily': homophily,
        'gcn_mean': np.mean(results['gcn']),
        'gcn_std': np.std(results['gcn']),
        'gat_mean': np.mean(results['gat']),
        'gat_std': np.std(results['gat']),
        'sage_mean': np.mean(results['sage']),
        'sage_std': np.std(results['sage']),
        'mlp_mean': np.mean(results['mlp']),
        'mlp_std': np.std(results['mlp']),
        'gcn_mlp': np.mean(results['gcn']) - np.mean(results['mlp']),
        'gat_mlp': np.mean(results['gat']) - np.mean(results['mlp']),
        'sage_mlp': np.mean(results['sage']) - np.mean(results['mlp']),
    }


def run_comprehensive_evaluation():
    """运行全面评估"""

    print("=" * 90)
    print("COMPREHENSIVE GCN EVALUATION")
    print("=" * 90)

    # 优先测试异配数据集（我们最关心的）
    datasets = [
        # 异配 (低h)
        'Texas', 'Wisconsin', 'Cornell',
        'Squirrel', 'Chameleon',
        'Actor',
        # 同质 (高h) - 作为对照
        'Cora', 'CiteSeer', 'PubMed',
        'Computers', 'Photo',
    ]

    all_results = []

    for name in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        data, source = load_dataset(name)
        if data is None:
            print(f"  Skipped")
            continue

        result = evaluate_dataset(name, data, n_runs=5)
        all_results.append(result)

        print(f"\n  Summary: h={result['homophily']:.3f}")
        print(f"  GCN: {result['gcn_mean']:.3f} +/- {result['gcn_std']:.3f}")
        print(f"  GAT: {result['gat_mean']:.3f} +/- {result['gat_std']:.3f}")
        print(f"  SAGE: {result['sage_mean']:.3f} +/- {result['sage_std']:.3f}")
        print(f"  MLP: {result['mlp_mean']:.3f} +/- {result['mlp_std']:.3f}")
        print(f"  GCN-MLP: {result['gcn_mlp']:+.3f}")

    # 汇总
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)

    print(f"\n{'Dataset':>12} {'h':>6} {'GCN':>8} {'GAT':>8} {'SAGE':>8} {'MLP':>8} {'GCN-MLP':>10}")
    print("-" * 80)

    for r in sorted(all_results, key=lambda x: x['homophily']):
        print(f"{r['dataset']:>12} {r['homophily']:>6.3f} "
              f"{r['gcn_mean']:>8.3f} {r['gat_mean']:>8.3f} "
              f"{r['sage_mean']:>8.3f} {r['mlp_mean']:>8.3f} "
              f"{r['gcn_mlp']:>+10.3f}")

    # 保存结果
    output = {
        'experiment': 'comprehensive_gcn_evaluation',
        'n_datasets': len(all_results),
        'results': all_results
    }

    with open('comprehensive_gcn_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: comprehensive_gcn_results.json")

    return all_results


if __name__ == '__main__':
    results = run_comprehensive_evaluation()
