"""
Phase Diagram Experiment: Feature Quality vs Homophily
验证假说：U-Shape只在特征弱时出现

实验设计：
1. 固定数据集(Cora)
2. 通过Feature Corruption控制特征质量
3. 通过Edge Rewiring控制同质性h
4. 画出2D相图：x轴=h, y轴=特征质量, z轴=GCN-MLP差异

关键假说：
- 特征强(MLP>90%): 单调递增曲线
- 特征中等(MLP~80%): 可能出现U-Shape
- 特征弱(MLP~50-70%): 明显U-Shape
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def corrupt_features(x: torch.Tensor, corruption_level: float) -> torch.Tensor:
    """
    通过添加噪声来降低特征质量

    corruption_level: 0 = 原始特征, 1 = 完全随机噪声
    """
    if corruption_level == 0:
        return x.clone()

    noise = torch.randn_like(x) * x.std()
    corrupted = (1 - corruption_level) * x + corruption_level * noise
    return corrupted


def mask_features(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    通过随机mask来降低特征质量

    mask_ratio: 0 = 全部保留, 1 = 全部mask
    """
    if mask_ratio == 0:
        return x.clone()

    mask = torch.rand(x.shape[1]) > mask_ratio
    masked = x.clone()
    masked[:, ~mask] = 0
    return masked


def reduce_feature_dimensions(x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """
    通过PCA降维来降低特征质量
    保留前keep_ratio比例的主成分
    """
    if keep_ratio >= 1:
        return x.clone()

    # 简单实现：随机选择部分特征维度
    n_keep = max(1, int(x.shape[1] * keep_ratio))
    indices = torch.randperm(x.shape[1])[:n_keep]
    reduced = torch.zeros_like(x)
    reduced[:, indices] = x[:, indices]
    return reduced


def compute_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """计算边同质性"""
    src, dst = edge_index
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()


def rewire_edges_to_target_h(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    target_h: float,
    n_nodes: int,
    max_iterations: int = 10000
) -> torch.Tensor:
    """通过Edge Rewiring调整同质性到目标值"""
    edges = set()
    src, dst = edge_index.cpu().numpy()
    for i in range(len(src)):
        if src[i] < dst[i]:
            edges.add((src[i], dst[i]))
        else:
            edges.add((dst[i], src[i]))

    labels_np = labels.cpu().numpy()

    def get_h():
        same = sum(1 for u, v in edges if labels_np[u] == labels_np[v])
        return same / len(edges) if edges else 0

    current_h = get_h()

    for _ in range(max_iterations):
        if abs(current_h - target_h) < 0.01:
            break

        edge_list = list(edges)
        idx = np.random.randint(len(edge_list))
        u, v = edge_list[idx]

        if current_h > target_h:
            # 需要增加异质边：找同质边换成异质边
            if labels_np[u] == labels_np[v]:
                # 找一个不同标签的节点
                candidates = [n for n in range(n_nodes)
                             if labels_np[n] != labels_np[u] and n != u and n != v]
                if candidates:
                    new_v = np.random.choice(candidates)
                    new_edge = (min(u, new_v), max(u, new_v))
                    if new_edge not in edges:
                        edges.remove((u, v))
                        edges.add(new_edge)
                        current_h = get_h()
        else:
            # 需要增加同质边
            if labels_np[u] != labels_np[v]:
                candidates = [n for n in range(n_nodes)
                             if labels_np[n] == labels_np[u] and n != u and n != v]
                if candidates:
                    new_v = np.random.choice(candidates)
                    new_edge = (min(u, new_v), max(u, new_v))
                    if new_edge not in edges:
                        edges.remove((u, v))
                        edges.add(new_edge)
                        current_h = get_h()

    # 转换回edge_index
    edge_list = list(edges)
    src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
    dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]

    return torch.tensor([src, dst], dtype=torch.long)


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
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = 0
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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 30:
                    break

    return best_val_acc


def run_phase_diagram_experiment():
    """运行相图实验"""

    print("=" * 80)
    print("PHASE DIAGRAM EXPERIMENT")
    print("Feature Quality vs Homophily → GCN-MLP Performance")
    print("=" * 80)

    # 加载数据
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]

    original_x = data.x.clone()
    original_edge_index = data.edge_index.clone()
    labels = data.y.clone()
    n_nodes = data.num_nodes
    n_features = data.num_features
    n_classes = dataset.num_classes

    original_h = compute_homophily(original_edge_index, labels)
    print(f"\nDataset: Cora")
    print(f"Nodes: {n_nodes}, Features: {n_features}, Classes: {n_classes}")
    print(f"Original homophily: {original_h:.3f}")

    # 实验参数
    corruption_levels = [0, 0.3, 0.5, 0.7, 0.9]  # 特征corruption程度
    h_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # 目标同质性
    n_runs = 3  # 每个配置运行次数

    results = []

    # 创建固定的train/val/test splits
    indices = np.arange(n_nodes)
    train_idx, temp_idx = train_test_split(indices, train_size=0.6, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=42)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # 相图实验
    for corruption in corruption_levels:
        print(f"\n{'='*60}")
        print(f"Feature Corruption Level: {corruption}")
        print(f"{'='*60}")

        # 创建corrupted特征
        corrupted_x = corrupt_features(original_x, corruption)

        # 先测试MLP baseline（不使用结构）
        x_device = corrupted_x.to(device)
        labels_device = labels.to(device)
        train_mask_device = train_mask.to(device)
        val_mask_device = val_mask.to(device)

        mlp_accs = []
        for run in range(n_runs):
            mlp = MLP(n_features, 64, n_classes).to(device)
            # MLP不需要edge_index，但为了接口一致，传入dummy
            dummy_edge = torch.zeros(2, 0, dtype=torch.long).to(device)
            acc = train_model(mlp, x_device, dummy_edge, labels_device,
                            train_mask_device, val_mask_device)
            mlp_accs.append(acc)

        mlp_baseline = np.mean(mlp_accs)
        print(f"MLP Baseline (no structure): {mlp_baseline:.3f}")

        for target_h in h_values:
            print(f"\n--- Target h = {target_h} ---")

            gcn_accs = []
            mlp_accs_h = []
            actual_hs = []

            for run in range(n_runs):
                # Edge rewiring
                rewired_edge_index = rewire_edges_to_target_h(
                    original_edge_index, labels, target_h, n_nodes
                )
                actual_h = compute_homophily(rewired_edge_index, labels)
                actual_hs.append(actual_h)

                # 移到device
                edge_index_device = rewired_edge_index.to(device)

                # 训练GCN
                gcn = GCN(n_features, 64, n_classes).to(device)
                gcn_acc = train_model(gcn, x_device, edge_index_device, labels_device,
                                     train_mask_device, val_mask_device)
                gcn_accs.append(gcn_acc)

                # MLP在这个配置下（用于对比）
                mlp = MLP(n_features, 64, n_classes).to(device)
                mlp_acc = train_model(mlp, x_device, edge_index_device, labels_device,
                                     train_mask_device, val_mask_device)
                mlp_accs_h.append(mlp_acc)

            gcn_mean = np.mean(gcn_accs)
            gcn_std = np.std(gcn_accs)
            mlp_mean = np.mean(mlp_accs_h)
            actual_h_mean = np.mean(actual_hs)
            gcn_advantage = gcn_mean - mlp_mean

            result = {
                'corruption_level': corruption,
                'target_h': target_h,
                'actual_h': actual_h_mean,
                'mlp_baseline': mlp_baseline,
                'gcn_acc': gcn_mean,
                'gcn_std': gcn_std,
                'mlp_acc': mlp_mean,
                'gcn_advantage': gcn_advantage
            }
            results.append(result)

            print(f"  h={actual_h_mean:.2f}, GCN={gcn_mean:.3f}±{gcn_std:.3f}, "
                  f"MLP={mlp_mean:.3f}, GCN-MLP={gcn_advantage:+.3f}")

    # 打印相图总结
    print("\n" + "=" * 80)
    print("PHASE DIAGRAM SUMMARY")
    print("=" * 80)
    print("\nGCN-MLP Advantage by Feature Quality and Homophily:")
    print("\n" + " " * 15 + "".join(f"h={h:.1f}  " for h in h_values))
    print("-" * 60)

    for corruption in corruption_levels:
        row_results = [r for r in results if r['corruption_level'] == corruption]
        mlp_baseline = row_results[0]['mlp_baseline']
        advantages = [r['gcn_advantage'] for r in sorted(row_results, key=lambda x: x['target_h'])]

        row_str = f"c={corruption:.1f} (MLP={mlp_baseline:.2f}): "
        row_str += "  ".join(f"{a:+.2f}" for a in advantages)
        print(row_str)

    # 分析U-Shape存在性
    print("\n" + "=" * 80)
    print("U-SHAPE ANALYSIS BY CORRUPTION LEVEL")
    print("=" * 80)

    for corruption in corruption_levels:
        row_results = sorted([r for r in results if r['corruption_level'] == corruption],
                            key=lambda x: x['target_h'])

        advantages = [r['gcn_advantage'] for r in row_results]
        h_vals = [r['actual_h'] for r in row_results]

        # 检查是否存在U-Shape
        # U-Shape特征：两端比中间高
        low_h_adv = np.mean(advantages[:2])  # h=0.1, 0.3
        mid_h_adv = advantages[2]  # h=0.5
        high_h_adv = np.mean(advantages[3:])  # h=0.7, 0.9

        is_u_shape = low_h_adv > mid_h_adv and high_h_adv > mid_h_adv
        pattern = "U-SHAPE" if is_u_shape else "MONOTONIC"

        mlp_baseline = row_results[0]['mlp_baseline']
        print(f"\nCorruption={corruption:.1f} (MLP baseline={mlp_baseline:.2f}):")
        print(f"  Low-h advantage (h<0.4): {low_h_adv:+.3f}")
        print(f"  Mid-h advantage (h≈0.5): {mid_h_adv:+.3f}")
        print(f"  High-h advantage (h>0.6): {high_h_adv:+.3f}")
        print(f"  Pattern: {pattern}")

    # 保存结果
    output = {
        'experiment': 'phase_diagram',
        'description': 'Feature quality vs homophily phase diagram',
        'corruption_levels': corruption_levels,
        'h_values': h_values,
        'n_runs': n_runs,
        'results': results
    }

    output_path = 'D:/Users/11919/Documents/毕业论文/paper/code/phase_diagram_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    run_phase_diagram_experiment()
