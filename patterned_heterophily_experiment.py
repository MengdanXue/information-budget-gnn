"""
Patterned Heterophily Experiment
================================

关键实验：验证U-Shape是由"patterned heterophily"而非"low h"导致

实验设计：
1. 固定h=0.2（低同质性）
2. 变化pattern strength α从0到1
   - α=0: 纯随机异配（Random Heterophily）
   - α=1: 完全结构化异配（Patterned Heterophily，如Class A只连Class B）
3. 观察GCN-MLP差异随α的变化

假说：
- α=0（随机）: GCN应该输（如Phase Diagram所示，-30%左右）
- α=1（有pattern）: GCN应该能学到反向关系，可能赢

如果假说成立，证明：U-Shape的关键是pattern，不是h本身
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def compute_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """计算边同质性"""
    src, dst = edge_index
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()


def create_patterned_heterophily_graph(
    n_nodes: int,
    labels: torch.Tensor,
    n_edges: int,
    pattern_strength: float,
    target_h: float = 0.2
) -> torch.Tensor:
    """
    创建具有可控pattern strength的异配图

    pattern_strength (α):
    - 0: 完全随机的异配边（Random Heterophily）
    - 1: 完全结构化的异配边（Patterned Heterophily）

    Patterned Heterophily规则:
    - Class i 优先连接 Class (i+1) % n_classes
    - 这创造了一个可预测的"轮转"模式
    """
    labels_np = labels.cpu().numpy()
    n_classes = len(np.unique(labels_np))

    # 计算每个类的节点
    class_nodes = {c: np.where(labels_np == c)[0] for c in range(n_classes)}

    edges = set()
    n_homo_edges = int(n_edges * target_h)  # 同质边数量
    n_hetero_edges = n_edges - n_homo_edges  # 异质边数量

    # 1. 添加同质边（保持target_h）
    for _ in range(n_homo_edges * 2):  # 多尝试一些，因为可能有重复
        if len(edges) >= n_homo_edges:
            break
        c = np.random.randint(n_classes)
        if len(class_nodes[c]) >= 2:
            u, v = np.random.choice(class_nodes[c], 2, replace=False)
            edge = (min(u, v), max(u, v))
            if edge not in edges:
                edges.add(edge)

    # 2. 添加异质边（根据pattern_strength混合）
    n_patterned = int(n_hetero_edges * pattern_strength)
    n_random = n_hetero_edges - n_patterned

    # 2a. Patterned异质边: Class i -> Class (i+1) % n_classes
    for _ in range(n_patterned * 2):
        if len(edges) >= n_homo_edges + n_patterned:
            break
        c1 = np.random.randint(n_classes)
        c2 = (c1 + 1) % n_classes  # 轮转规则
        if len(class_nodes[c1]) > 0 and len(class_nodes[c2]) > 0:
            u = np.random.choice(class_nodes[c1])
            v = np.random.choice(class_nodes[c2])
            edge = (min(u, v), max(u, v))
            if edge not in edges:
                edges.add(edge)

    # 2b. Random异质边: 随机选择不同类
    for _ in range(n_random * 2):
        if len(edges) >= n_homo_edges + n_patterned + n_random:
            break
        c1 = np.random.randint(n_classes)
        c2 = np.random.randint(n_classes)
        while c2 == c1:
            c2 = np.random.randint(n_classes)
        if len(class_nodes[c1]) > 0 and len(class_nodes[c2]) > 0:
            u = np.random.choice(class_nodes[c1])
            v = np.random.choice(class_nodes[c2])
            edge = (min(u, v), max(u, v))
            if edge not in edges:
                edges.add(edge)

    # 转换为edge_index
    edge_list = list(edges)
    src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
    dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]

    return torch.tensor([src, dst], dtype=torch.long)


def compute_pattern_predictability(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算邻居标签分布对节点标签的可预测性
    使用简单规则：如果邻居主要是Class (label+1)%n_classes，则预测正确
    """
    labels_np = labels.cpu().numpy()
    n_classes = len(np.unique(labels_np))
    n_nodes = len(labels_np)

    src, dst = edge_index.cpu().numpy()

    correct = 0
    total = 0

    for node in range(n_nodes):
        # 找到所有邻居
        neighbors = dst[src == node]
        if len(neighbors) == 0:
            continue

        # 统计邻居标签分布
        neighbor_labels = labels_np[neighbors]
        label_counts = np.bincount(neighbor_labels, minlength=n_classes)

        # 检查是否符合轮转模式
        expected_neighbor_class = (labels_np[node] + 1) % n_classes
        if label_counts[expected_neighbor_class] == label_counts.max():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0


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


def run_patterned_heterophily_experiment():
    print("=" * 80)
    print("PATTERNED HETEROPHILY EXPERIMENT")
    print("Hypothesis: U-Shape is caused by patterned heterophily, not low h")
    print("=" * 80)

    # 加载数据
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]

    original_x = data.x.clone()
    labels = data.y.clone()
    n_nodes = data.num_nodes
    n_features = data.num_features
    n_classes = dataset.num_classes
    n_edges = data.edge_index.shape[1] // 2  # 无向边数量

    print(f"\nDataset: Cora")
    print(f"Nodes: {n_nodes}, Features: {n_features}, Classes: {n_classes}")
    print(f"Edges: {n_edges}")

    # 实验参数
    pattern_strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    target_h_values = [0.1, 0.2, 0.3]  # 低h区域
    n_runs = 5

    # 创建固定的train/val/test splits
    indices = np.arange(n_nodes)
    train_idx, temp_idx = train_test_split(indices, train_size=0.6, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=42)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    results = []

    for target_h in target_h_values:
        print(f"\n{'='*60}")
        print(f"Target Homophily: h = {target_h}")
        print(f"{'='*60}")

        for alpha in pattern_strengths:
            print(f"\n--- Pattern Strength α = {alpha} ---")

            gcn_accs = []
            mlp_accs = []
            actual_hs = []
            predictabilities = []

            for run in range(n_runs):
                # 创建patterned heterophily图
                edge_index = create_patterned_heterophily_graph(
                    n_nodes, labels, n_edges, alpha, target_h
                )

                actual_h = compute_homophily(edge_index, labels)
                actual_hs.append(actual_h)

                pred_score = compute_pattern_predictability(edge_index, labels)
                predictabilities.append(pred_score)

                # 移到device
                x_device = original_x.to(device)
                edge_index_device = edge_index.to(device)
                labels_device = labels.to(device)
                train_mask_device = train_mask.to(device)
                val_mask_device = val_mask.to(device)

                # 训练GCN
                gcn = GCN(n_features, 64, n_classes).to(device)
                gcn_acc = train_model(gcn, x_device, edge_index_device, labels_device,
                                     train_mask_device, val_mask_device)
                gcn_accs.append(gcn_acc)

                # 训练MLP
                mlp = MLP(n_features, 64, n_classes).to(device)
                mlp_acc = train_model(mlp, x_device, edge_index_device, labels_device,
                                     train_mask_device, val_mask_device)
                mlp_accs.append(mlp_acc)

            gcn_mean = np.mean(gcn_accs)
            gcn_std = np.std(gcn_accs)
            mlp_mean = np.mean(mlp_accs)
            actual_h_mean = np.mean(actual_hs)
            pred_mean = np.mean(predictabilities)
            gcn_advantage = gcn_mean - mlp_mean

            result = {
                'target_h': target_h,
                'pattern_strength': alpha,
                'actual_h': actual_h_mean,
                'pattern_predictability': pred_mean,
                'gcn_acc': gcn_mean,
                'gcn_std': gcn_std,
                'mlp_acc': mlp_mean,
                'gcn_advantage': gcn_advantage
            }
            results.append(result)

            print(f"  h={actual_h_mean:.3f}, Pattern Pred={pred_mean:.3f}")
            print(f"  GCN={gcn_mean:.3f}±{gcn_std:.3f}, MLP={mlp_mean:.3f}, GCN-MLP={gcn_advantage:+.3f}")

    # 总结
    print("\n" + "=" * 80)
    print("PATTERNED HETEROPHILY RESULTS SUMMARY")
    print("=" * 80)

    for target_h in target_h_values:
        print(f"\n--- h = {target_h} ---")
        h_results = [r for r in results if r['target_h'] == target_h]
        h_results = sorted(h_results, key=lambda x: x['pattern_strength'])

        print(f"{'α':>6} {'Pred':>8} {'GCN':>8} {'MLP':>8} {'GCN-MLP':>10}")
        print("-" * 50)
        for r in h_results:
            print(f"{r['pattern_strength']:>6.1f} {r['pattern_predictability']:>8.3f} "
                  f"{r['gcn_acc']:>8.3f} {r['mlp_acc']:>8.3f} {r['gcn_advantage']:>+10.3f}")

    # 验证假说
    print("\n" + "=" * 80)
    print("HYPOTHESIS VERIFICATION")
    print("=" * 80)

    for target_h in target_h_values:
        h_results = [r for r in results if r['target_h'] == target_h]
        h_results = sorted(h_results, key=lambda x: x['pattern_strength'])

        alpha_0 = h_results[0]  # α=0, 随机
        alpha_1 = h_results[-1]  # α=1, 有pattern

        improvement = alpha_1['gcn_advantage'] - alpha_0['gcn_advantage']

        print(f"\nh = {target_h}:")
        print(f"  α=0 (Random):    GCN-MLP = {alpha_0['gcn_advantage']:+.3f}")
        print(f"  α=1 (Patterned): GCN-MLP = {alpha_1['gcn_advantage']:+.3f}")
        print(f"  Improvement: {improvement:+.3f}")

        if improvement > 0.05:
            print(f"  [YES] HYPOTHESIS SUPPORTED: Pattern helps GCN!")
        elif improvement > 0:
            print(f"  [~] Weak support: Small improvement")
        else:
            print(f"  [NO] HYPOTHESIS NOT SUPPORTED")

    # 保存结果
    output = {
        'experiment': 'patterned_heterophily',
        'description': 'Test if U-Shape is caused by patterned heterophily',
        'hypothesis': 'U-Shape appears when heterophily is patterned, not random',
        'pattern_strengths': pattern_strengths,
        'target_h_values': target_h_values,
        'n_runs': n_runs,
        'results': results
    }

    output_path = 'D:/Users/11919/Documents/毕业论文/paper/code/patterned_heterophily_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    run_patterned_heterophily_experiment()
