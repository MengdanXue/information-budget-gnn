"""
Pattern Predictability Metric
=============================

定义一个可在真实图上计算的"结构可预测性"指标

关键思路：
- 如果邻居的标签分布是可预测的（有pattern），则M高
- 如果邻居的标签分布是随机的（无pattern），则M低

候选指标：
1. Neighborhood Label Entropy (NLE): 邻居标签分布的熵
2. Label Transition Matrix Rank (LTMR): 标签转移矩阵的有效秩
3. Conditional Predictability (CP): 用邻居标签预测节点标签的准确率

最终选择：Conditional Predictability (CP)
- 直观：如果知道邻居标签能帮助预测节点标签，说明有pattern
- 可计算：不需要生成器参数
- 可验证：可以在真实数据上测量
"""

import torch
import numpy as np
from collections import Counter
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


def compute_neighborhood_label_distribution(edge_index: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """
    计算每个节点的邻居标签分布
    返回: (n_nodes, n_classes) 的矩阵
    """
    labels_np = labels.cpu().numpy()
    n_nodes = len(labels_np)
    n_classes = len(np.unique(labels_np))

    src, dst = edge_index.cpu().numpy()

    # 初始化邻居标签计数
    neighbor_label_counts = np.zeros((n_nodes, n_classes))

    for i in range(len(src)):
        u, v = src[i], dst[i]
        neighbor_label_counts[u, labels_np[v]] += 1

    # 归一化为分布
    row_sums = neighbor_label_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除0
    neighbor_label_dist = neighbor_label_counts / row_sums

    return neighbor_label_dist


def compute_label_transition_matrix(edge_index: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """
    计算标签转移矩阵 P(neighbor_label | node_label)
    返回: (n_classes, n_classes) 的矩阵
    """
    labels_np = labels.cpu().numpy()
    n_classes = len(np.unique(labels_np))

    src, dst = edge_index.cpu().numpy()

    # 计算转移计数
    transition_counts = np.zeros((n_classes, n_classes))

    for i in range(len(src)):
        u, v = src[i], dst[i]
        transition_counts[labels_np[u], labels_np[v]] += 1

    # 归一化
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_counts / row_sums

    return transition_matrix


def metric_neighborhood_entropy(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """
    指标1: Neighborhood Label Entropy (NLE)
    低熵 = 邻居标签集中 = 高可预测性
    高熵 = 邻居标签分散 = 低可预测性

    返回: 1 - normalized_entropy (越高越可预测)
    """
    neighbor_dist = compute_neighborhood_label_distribution(edge_index, labels)
    n_classes = neighbor_dist.shape[1]

    # 计算每个节点的邻居标签熵
    entropies = []
    for i in range(len(neighbor_dist)):
        if neighbor_dist[i].sum() > 0:
            ent = entropy(neighbor_dist[i] + 1e-10)  # 避免log(0)
            entropies.append(ent)

    if not entropies:
        return 0.0

    # 归一化（最大熵是log(n_classes)）
    max_entropy = np.log(n_classes)
    avg_entropy = np.mean(entropies)
    normalized = avg_entropy / max_entropy if max_entropy > 0 else 0

    # 返回1-entropy，使得高值=高可预测性
    return 1 - normalized


def metric_transition_matrix_structure(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """
    指标2: Label Transition Matrix Structure (LTMS)
    如果转移矩阵有明显的结构（非均匀），说明有pattern

    返回: 1 - normalized_entropy_of_transition_matrix
    """
    transition_matrix = compute_label_transition_matrix(edge_index, labels)
    n_classes = transition_matrix.shape[0]

    # 计算转移矩阵的"非均匀性"
    # 均匀分布的熵最高，有结构的熵较低
    entropies = []
    for i in range(n_classes):
        if transition_matrix[i].sum() > 0:
            ent = entropy(transition_matrix[i] + 1e-10)
            entropies.append(ent)

    if not entropies:
        return 0.0

    max_entropy = np.log(n_classes)
    avg_entropy = np.mean(entropies)
    normalized = avg_entropy / max_entropy if max_entropy > 0 else 0

    return 1 - normalized


def metric_conditional_predictability(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """
    指标3: Conditional Predictability (CP)
    用邻居标签分布预测节点标签的准确率

    这是最直接的指标：如果邻居信息有用，准确率会高于随机
    """
    neighbor_dist = compute_neighborhood_label_distribution(edge_index, labels)
    labels_np = labels.cpu().numpy()

    # 过滤掉没有邻居的节点
    has_neighbors = neighbor_dist.sum(axis=1) > 0
    X = neighbor_dist[has_neighbors]
    y = labels_np[has_neighbors]

    if len(np.unique(y)) < 2 or len(y) < 10:
        return 0.0

    # 用逻辑回归做5折交叉验证
    try:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X, y, cv=min(5, len(y)), scoring='accuracy')
        return scores.mean()
    except:
        return 0.0


def metric_pattern_predictability(edge_index: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    综合Pattern Predictability指标
    返回多个子指标
    """
    # 计算同质性作为baseline
    src, dst = edge_index
    labels_tensor = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
    homophily = (labels_tensor[src] == labels_tensor[dst]).float().mean().item()

    # 计算各个指标
    nle = metric_neighborhood_entropy(edge_index, labels)
    ltms = metric_transition_matrix_structure(edge_index, labels)
    cp = metric_conditional_predictability(edge_index, labels)

    # 综合指标: 取CP作为主指标（最直观）
    # 但如果h很高，CP自然会高，所以要除以一个baseline
    random_baseline = 1.0 / len(np.unique(labels.cpu().numpy()))
    cp_lift = (cp - random_baseline) / (1 - random_baseline) if cp > random_baseline else 0

    return {
        'homophily': homophily,
        'neighborhood_entropy': nle,  # 1 - entropy, 越高越可预测
        'transition_structure': ltms,  # 1 - entropy, 越高越有结构
        'conditional_predictability': cp,  # 用邻居预测标签的准确率
        'predictability_lift': cp_lift,  # 相对随机baseline的提升
        'pattern_score': (nle + ltms + cp_lift) / 3  # 综合分数
    }


def test_on_patterned_data():
    """在我们的patterned heterophily数据上测试指标"""
    from patterned_heterophily_experiment import create_patterned_heterophily_graph
    from torch_geometric.datasets import Planetoid

    print("=" * 70)
    print("TESTING PATTERN PREDICTABILITY METRIC")
    print("=" * 70)

    # 加载Cora
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    labels = data.y
    n_nodes = data.num_nodes
    n_edges = data.edge_index.shape[1] // 2

    print(f"\nDataset: Cora, Nodes: {n_nodes}, Edges: {n_edges}")

    # 测试不同的pattern strength
    print("\n" + "-" * 70)
    print("Pattern Strength vs Predictability Metrics")
    print("-" * 70)

    target_h = 0.1
    pattern_strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    print(f"\n{'α':>6} {'h':>8} {'NLE':>8} {'LTMS':>8} {'CP':>8} {'Lift':>8} {'Score':>8}")
    print("-" * 70)

    for alpha in pattern_strengths:
        edge_index = create_patterned_heterophily_graph(
            n_nodes, labels, n_edges, alpha, target_h
        )

        metrics = metric_pattern_predictability(edge_index, labels)

        print(f"{alpha:>6.1f} {metrics['homophily']:>8.3f} "
              f"{metrics['neighborhood_entropy']:>8.3f} "
              f"{metrics['transition_structure']:>8.3f} "
              f"{metrics['conditional_predictability']:>8.3f} "
              f"{metrics['predictability_lift']:>8.3f} "
              f"{metrics['pattern_score']:>8.3f}")

    # 测试真实数据
    print("\n" + "-" * 70)
    print("Real Dataset Metrics")
    print("-" * 70)

    datasets = ['Cora', 'CiteSeer', 'PubMed']

    print(f"\n{'Dataset':>12} {'h':>8} {'NLE':>8} {'LTMS':>8} {'CP':>8} {'Lift':>8} {'Score':>8}")
    print("-" * 70)

    for name in datasets:
        try:
            dataset = Planetoid(root='./data', name=name)
            data = dataset[0]

            metrics = metric_pattern_predictability(data.edge_index, data.y)

            print(f"{name:>12} {metrics['homophily']:>8.3f} "
                  f"{metrics['neighborhood_entropy']:>8.3f} "
                  f"{metrics['transition_structure']:>8.3f} "
                  f"{metrics['conditional_predictability']:>8.3f} "
                  f"{metrics['predictability_lift']:>8.3f} "
                  f"{metrics['pattern_score']:>8.3f}")
        except Exception as e:
            print(f"{name:>12} Error: {e}")


if __name__ == '__main__':
    test_on_patterned_data()
