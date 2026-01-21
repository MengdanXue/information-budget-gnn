"""
Heterophilic Benchmark Validation
=================================

在经典的异配数据集上验证Pattern Score

数据集：
- Texas, Wisconsin, Cornell (WebKB)
- Squirrel, Chameleon (Wikipedia networks)
- Actor (Film-Actor network)

假说：Pattern Score能预测这些数据集上GCN vs MLP的表现
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.model_selection import train_test_split
from pattern_predictability_metric import metric_pattern_predictability
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


def train_model(model, x, edge_index, labels, train_mask, val_mask, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_acc = 0
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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break

    return best_val_acc


def evaluate_dataset(dataset_name, data, n_runs=5):
    """在一个数据集上评估所有模型"""

    x = data.x
    edge_index = data.edge_index
    labels = data.y
    n_nodes = data.num_nodes
    n_features = data.num_features
    n_classes = len(labels.unique())

    # 计算Pattern Score
    metrics = metric_pattern_predictability(edge_index, labels)

    print(f"\nDataset: {dataset_name}")
    print(f"  Nodes: {n_nodes}, Features: {n_features}, Classes: {n_classes}")
    print(f"  Edges: {edge_index.shape[1] // 2}")
    print(f"  Homophily: {metrics['homophily']:.3f}")
    print(f"  Pattern Score: {metrics['pattern_score']:.3f}")

    # 运行多次实验
    results = {
        'gcn': [],
        'gat': [],
        'sage': [],
        'mlp': []
    }

    x_device = x.to(device)
    edge_index_device = edge_index.to(device)
    labels_device = labels.to(device)

    for run in range(n_runs):
        # 创建随机split
        indices = np.arange(n_nodes)
        train_idx, temp_idx = train_test_split(indices, train_size=0.6, random_state=42+run)
        val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=42+run)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        train_mask_device = train_mask.to(device)
        val_mask_device = val_mask.to(device)

        # 训练GCN
        gcn = GCN(n_features, 64, n_classes).to(device)
        gcn_acc = train_model(gcn, x_device, edge_index_device, labels_device,
                             train_mask_device, val_mask_device)
        results['gcn'].append(gcn_acc)

        # 训练GAT
        gat = GAT(n_features, 64, n_classes).to(device)
        gat_acc = train_model(gat, x_device, edge_index_device, labels_device,
                             train_mask_device, val_mask_device)
        results['gat'].append(gat_acc)

        # 训练SAGE
        sage = GraphSAGE(n_features, 64, n_classes).to(device)
        sage_acc = train_model(sage, x_device, edge_index_device, labels_device,
                              train_mask_device, val_mask_device)
        results['sage'].append(sage_acc)

        # 训练MLP
        mlp = MLP(n_features, 64, n_classes).to(device)
        mlp_acc = train_model(mlp, x_device, edge_index_device, labels_device,
                             train_mask_device, val_mask_device)
        results['mlp'].append(mlp_acc)

        print(f"  Run {run+1}/{n_runs}: GCN={gcn_acc:.3f}, GAT={gat_acc:.3f}, "
              f"SAGE={sage_acc:.3f}, MLP={mlp_acc:.3f}")

    # 计算平均值
    gcn_mean = np.mean(results['gcn'])
    gat_mean = np.mean(results['gat'])
    sage_mean = np.mean(results['sage'])
    mlp_mean = np.mean(results['mlp'])

    return {
        'dataset': dataset_name,
        'n_nodes': n_nodes,
        'n_features': n_features,
        'n_classes': n_classes,
        'homophily': metrics['homophily'],
        'pattern_score': metrics['pattern_score'],
        'metrics': metrics,
        'gcn_acc': gcn_mean,
        'gat_acc': gat_mean,
        'sage_acc': sage_mean,
        'mlp_acc': mlp_mean,
        'gcn_advantage': gcn_mean - mlp_mean,
        'gat_advantage': gat_mean - mlp_mean,
        'sage_advantage': sage_mean - mlp_mean,
        'gcn_std': np.std(results['gcn']),
        'gat_std': np.std(results['gat']),
        'sage_std': np.std(results['sage']),
        'mlp_std': np.std(results['mlp'])
    }


def run_benchmark_validation():
    print("=" * 80)
    print("HETEROPHILIC BENCHMARK VALIDATION")
    print("Testing Pattern Score on Classic Heterophilic Datasets")
    print("=" * 80)

    all_results = []

    # WebKB datasets
    print("\n" + "=" * 80)
    print("WebKB Datasets (Citation Networks)")
    print("=" * 80)

    for name in ['Texas', 'Wisconsin', 'Cornell']:
        try:
            dataset = WebKB(root='./data', name=name)
            data = dataset[0]
            result = evaluate_dataset(name, data, n_runs=5)
            all_results.append(result)
        except Exception as e:
            print(f"\nError loading {name}: {e}")

    # Wikipedia datasets
    print("\n" + "=" * 80)
    print("Wikipedia Networks")
    print("=" * 80)

    for name in ['Squirrel', 'Chameleon']:
        try:
            dataset = WikipediaNetwork(root='./data', name=name.lower())
            data = dataset[0]
            result = evaluate_dataset(name, data, n_runs=5)
            all_results.append(result)
        except Exception as e:
            print(f"\nError loading {name}: {e}")

    # Actor dataset
    print("\n" + "=" * 80)
    print("Actor Network")
    print("=" * 80)

    try:
        dataset = Actor(root='./data/Actor')
        data = dataset[0]
        result = evaluate_dataset('Actor', data, n_runs=5)
        all_results.append(result)
    except Exception as e:
        print(f"\nError loading Actor: {e}")

    # 总结
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"\n{'Dataset':>12} {'h':>6} {'Score':>8} {'GCN':>8} {'GAT':>8} {'SAGE':>8} {'MLP':>8} {'GCN-MLP':>10}")
    print("-" * 85)

    for r in all_results:
        print(f"{r['dataset']:>12} {r['homophily']:>6.3f} {r['pattern_score']:>8.3f} "
              f"{r['gcn_acc']:>8.3f} {r['gat_acc']:>8.3f} {r['sage_acc']:>8.3f} "
              f"{r['mlp_acc']:>8.3f} {r['gcn_advantage']:>+10.3f}")

    # 计算相关性
    if len(all_results) > 2:
        pattern_scores = [r['pattern_score'] for r in all_results]
        gcn_advantages = [r['gcn_advantage'] for r in all_results]
        gat_advantages = [r['gat_advantage'] for r in all_results]
        sage_advantages = [r['sage_advantage'] for r in all_results]

        corr_gcn = np.corrcoef(pattern_scores, gcn_advantages)[0, 1]
        corr_gat = np.corrcoef(pattern_scores, gat_advantages)[0, 1]
        corr_sage = np.corrcoef(pattern_scores, sage_advantages)[0, 1]

        print("\n" + "=" * 80)
        print("CORRELATION: Pattern Score vs GNN Advantage")
        print("=" * 80)
        print(f"  GCN:        r = {corr_gcn:.3f}")
        print(f"  GAT:        r = {corr_gat:.3f}")
        print(f"  GraphSAGE:  r = {corr_sage:.3f}")

        if corr_gcn > 0.6:
            print(f"\n[SUCCESS] Pattern Score is a strong predictor across heterophilic benchmarks!")
        elif corr_gcn > 0.4:
            print(f"\n[MODERATE] Pattern Score shows moderate predictive power.")
        else:
            print(f"\n[WEAK] Pattern Score shows weak correlation.")

    # 保存结果
    output = {
        'experiment': 'heterophilic_benchmark_validation',
        'results': all_results
    }

    with open('heterophilic_benchmark_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: heterophilic_benchmark_results.json")

    return all_results


if __name__ == '__main__':
    run_benchmark_validation()
