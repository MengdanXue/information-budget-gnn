"""
Unified FSD Metrics Computation - TKDE Version
统一的FSD指标计算代码，解决之前两种实现不一致的问题

关键修复:
1. δ_agg统一使用"节点与聚合邻居的相似度"定义 (compute_dilution.py的方法)
   δ_agg = E[d_i × (1 - cos(x_i, mean(x_neighbors)))]
   这更准确反映了GNN mean aggregation的实际行为

2. 明确区分两种相似度:
   - S_pairwise: 与各邻居的平均相似度 = mean_j(cos(x_i, x_j))
   - S_agg: 与聚合后邻居的相似度 = cos(x_i, normalize(mean(x_neighbors)))

   δ_agg使用S_agg，因为它直接衡量mean aggregation的效果

Metrics:
- ρ_FS: Feature-Structure Alignment Score
- δ_agg: Aggregation Dilution (使用S_agg定义)
- h: Homophily ratio
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree
import json
from datetime import datetime


def compute_feature_similarity_matrix(x, batch_size=1000):
    """计算特征余弦相似度矩阵（分批处理以节省内存）"""
    x_norm = F.normalize(x, p=2, dim=1)
    n = x.size(0)

    if n <= batch_size:
        return torch.mm(x_norm, x_norm.t())

    sim_matrix = torch.zeros(n, n, device=x.device)
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)
            sim_matrix[i:end_i, j:end_j] = torch.mm(
                x_norm[i:end_i], x_norm[j:end_j].t()
            )
    return sim_matrix


def compute_rho_fs(data, device='cpu', n_samples=100000):
    """
    计算Feature-Structure Alignment Score (ρ_FS)
    ρ_FS = E[S_feat(i,j) | (i,j)∈E] - E[S_feat(i,j) | (i,j)∉E]
    """
    edge_index = data.edge_index.to(device)
    x = data.x.to(device)
    n = x.size(0)

    x_norm = F.normalize(x, p=2, dim=1)

    # 计算边的特征相似度
    edge_sims = (x_norm[edge_index[0]] * x_norm[edge_index[1]]).sum(dim=1)
    mean_edge_sim = edge_sims.mean().item()

    # 采样非边的特征相似度
    adj_set = set(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()))

    non_edge_sims = []
    sampled = 0
    max_attempts = n_samples * 10
    attempts = 0

    while sampled < n_samples and attempts < max_attempts:
        i, j = np.random.randint(0, n, 2)
        if i != j and (i, j) not in adj_set:
            sim = (x_norm[i] * x_norm[j]).sum().item()
            non_edge_sims.append(sim)
            sampled += 1
        attempts += 1

    mean_non_edge_sim = np.mean(non_edge_sims)

    rho_fs = mean_edge_sim - mean_non_edge_sim

    return {
        'rho_fs': rho_fs,
        'mean_edge_sim': mean_edge_sim,
        'mean_non_edge_sim': mean_non_edge_sim,
        'n_edges': edge_index.size(1),
        'n_samples': sampled
    }


def compute_delta_agg_unified(data, device='cpu'):
    """
    统一的Aggregation Dilution计算

    使用定义: δ_agg = E[d_i × (1 - S_agg(i))]
    其中 S_agg(i) = cos(x_i, normalize(mean(x_neighbors)))

    这是正确的定义，因为:
    1. GNN的mean aggregation产生 mean(x_neighbors)
    2. S_agg衡量的是节点与聚合后信息的相似度
    3. δ_agg反映了aggregation导致的信息损失

    Returns:
        dict: 包含delta_agg, mean_degree, mean_sim_to_agg, 以及各degree组的分析
    """
    edge_index = data.edge_index.to(device)
    x = data.x.to(device)
    n = x.size(0)

    # 归一化特征
    x_norm = F.normalize(x, p=2, dim=1)

    # 计算每个节点的度数
    deg = degree(edge_index[0], n).float()

    # 构建邻接表
    adj_list = [[] for _ in range(n)]
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].append(dst)

    # 计算每个节点的dilution
    dilutions = []
    degrees = []
    sims_to_agg = []
    sims_pairwise = []  # 用于对比

    for i in range(n):
        neighbors = adj_list[i]
        if len(neighbors) == 0:
            continue

        d = len(neighbors)
        degrees.append(d)

        # 方法1: S_agg - 与聚合邻居的相似度 (正确的δ_agg定义)
        neighbor_feats = x_norm[neighbors]
        mean_neighbor = neighbor_feats.mean(dim=0)
        mean_neighbor_norm = F.normalize(mean_neighbor.unsqueeze(0), p=2, dim=1).squeeze()
        sim_agg = (x_norm[i] * mean_neighbor_norm).sum().item()
        sims_to_agg.append(sim_agg)

        # δ_agg = d × (1 - sim_agg)
        dilution = d * (1 - sim_agg)
        dilutions.append(dilution)

        # 方法2: S_pairwise - 与各邻居的平均相似度 (仅用于对比)
        pairwise_sims = (x_norm[i:i+1] @ neighbor_feats.t()).squeeze()
        sim_pairwise = pairwise_sims.mean().item() if pairwise_sims.dim() > 0 else pairwise_sims.item()
        sims_pairwise.append(sim_pairwise)

    dilutions = np.array(dilutions)
    degrees = np.array(degrees)
    sims_to_agg = np.array(sims_to_agg)
    sims_pairwise = np.array(sims_pairwise)

    # 按度数分组分析
    degree_analysis = {}
    thresholds = [(1, 10), (11, 50), (51, 100), (101, 200), (201, float('inf'))]

    for low, high in thresholds:
        if high == float('inf'):
            mask = degrees > low
            label = f'degree_{low}+'
        else:
            mask = (degrees >= low) & (degrees <= high)
            label = f'degree_{low}_{high}'

        if mask.sum() > 0:
            degree_analysis[label] = {
                'count': int(mask.sum()),
                'mean_dilution': float(dilutions[mask].mean()),
                'mean_sim_to_agg': float(sims_to_agg[mask].mean()),
                'mean_sim_pairwise': float(sims_pairwise[mask].mean())
            }

    return {
        'delta_agg': float(dilutions.mean()),
        'delta_agg_std': float(dilutions.std()),
        'mean_degree': float(degrees.mean()),
        'mean_sim_to_agg': float(sims_to_agg.mean()),
        'mean_sim_pairwise': float(sims_pairwise.mean()),
        'n_nodes_with_neighbors': len(degrees),
        'degree_analysis': degree_analysis,
        # 验证公式: δ_agg ≠ mean_degree × (1 - mean_sim_to_agg)
        # 因为 E[d×(1-s)] ≠ E[d]×(1-E[s])
        'naive_estimate': float(degrees.mean() * (1 - sims_to_agg.mean())),
        'actual_vs_naive_ratio': float(dilutions.mean() / (degrees.mean() * (1 - sims_to_agg.mean()) + 1e-8))
    }


def compute_homophily(data):
    """
    计算Homophily ratio
    h = 边连接同类节点的比例
    """
    edge_index = data.edge_index
    y = data.y

    same_label = (y[edge_index[0]] == y[edge_index[1]]).float()
    homophily = same_label.mean().item()

    return homophily


def compute_all_fsd_metrics(data, name, device='cpu'):
    """计算完整的FSD指标集"""
    print(f"\nComputing unified FSD metrics for {name}...")

    # 基本统计
    n_nodes = data.x.size(0)
    n_edges = data.edge_index.size(1)
    n_features = data.x.size(1)

    # FSD指标
    rho_result = compute_rho_fs(data, device)
    delta_result = compute_delta_agg_unified(data, device)

    # Homophily (如果有标签)
    homophily = None
    if hasattr(data, 'y') and data.y is not None:
        try:
            homophily = compute_homophily(data)
        except:
            homophily = None

    return {
        'name': name,
        'timestamp': datetime.now().isoformat(),
        'basic_stats': {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'n_features': n_features,
            'mean_degree': delta_result['mean_degree']
        },
        'fsd_metrics': {
            'rho_fs': rho_result['rho_fs'],
            'delta_agg': delta_result['delta_agg'],
            'homophily': homophily
        },
        'rho_fs_details': rho_result,
        'delta_agg_details': delta_result,
        'method_prediction': predict_best_method(
            rho_result['rho_fs'],
            delta_result['delta_agg'],
            homophily,
            n_features
        )
    }


def predict_best_method(rho_fs, delta_agg, homophily, n_features):
    """
    基于FSD框架的方法选择预测

    注意: 这是基于实验观察的分析框架，不是先验预测。
    阈值来源于对4个数据集的后验分析。
    """
    predictions = []
    reasoning = []

    # 规则1: 高δ_agg (>10) → 采样/拼接方法
    # 理论依据: Theorem 3.3 (Sampling Bounds Dilution)
    if delta_agg is not None and delta_agg > 10:
        predictions.append("H2GCN/GraphSAGE")
        reasoning.append(f"High δ_agg ({delta_agg:.1f} > 10) indicates severe aggregation dilution. "
                        f"Sampling (GraphSAGE) or concatenation (H2GCN) methods bound this effect.")

    # 规则2: 低δ_agg + 高维特征 → NAA
    elif delta_agg is not None and delta_agg < 5 and n_features > 100:
        predictions.append("NAA")
        reasoning.append(f"Low δ_agg ({delta_agg:.1f} < 5) with high-dim features ({n_features} > 100). "
                        f"Feature-aware attention can leverage rich numerical information.")

    # 规则3: 负ρ_FS → 异质感知方法
    elif rho_fs is not None and rho_fs < -0.05:
        predictions.append("H2GCN")
        reasoning.append(f"Negative ρ_FS ({rho_fs:.3f} < -0.05) indicates heterophily. "
                        f"2-hop aggregation in H2GCN can reach same-class nodes.")

    # 规则4: 高ρ_FS + 高同质性 → 标准GCN/GAT
    elif rho_fs is not None and rho_fs > 0.3 and homophily is not None and homophily > 0.6:
        predictions.append("GCN/GAT")
        reasoning.append(f"High ρ_FS ({rho_fs:.3f} > 0.3) and homophily ({homophily:.2f} > 0.6). "
                        f"Standard mean aggregation is effective.")

    # 中等稀释度
    elif delta_agg is not None and 5 <= delta_agg <= 10:
        predictions.append("Mixed (DAAA/NAA may help)")
        reasoning.append(f"Medium δ_agg ({delta_agg:.1f} in [5,10]). "
                        f"Adaptive methods like DAAA may provide incremental benefit.")

    else:
        predictions.append("Standard methods")
        reasoning.append("No strong indicator. Standard GCN/GAT should work reasonably.")

    return {
        'predicted_method': predictions[0],
        'reasoning': reasoning[0],
        'confidence': 'high' if delta_agg is not None and (delta_agg > 10 or delta_agg < 5) else 'medium',
        'note': 'This is a post-hoc analysis framework based on 4 datasets, not a validated a priori prediction tool.'
    }


def validate_delta_agg_formula():
    """
    验证δ_agg公式，解释为什么 E[d×(1-s)] ≠ E[d]×(1-E[s])
    """
    print("\n" + "="*70)
    print("δ_agg Formula Validation")
    print("="*70)

    print("""
δ_agg的正确定义:
    δ_agg = E[d_i × (1 - S_agg(i))]

其中:
    S_agg(i) = cos(x_i, normalize(mean(x_neighbors)))

重要说明:
    E[d × (1-s)] ≠ E[d] × (1-E[s])

原因: Jensen不等式和度数-相似度的相关性
    - 高度数节点往往有更多样的邻居 → 更低的S_agg
    - 这导致 d × (1-s) 的期望高于简单乘积

例如 (IEEE-CIS):
    - mean_degree = 47.65
    - mean_sim_to_agg = 0.84
    - 朴素估计: 47.65 × (1-0.84) = 7.62
    - 实际δ_agg: 11.25
    - 比值: 11.25 / 7.62 = 1.48 (高48%)

这个差异正是δ_agg的价值所在 - 它捕获了度数对信息损失的放大效应。
""")


if __name__ == '__main__':
    validate_delta_agg_formula()

    print("\n" + "="*70)
    print("Unified FSD Metrics Computation")
    print("="*70)
    print("""
使用方法:
    from compute_fsd_metrics_unified import compute_all_fsd_metrics

    metrics = compute_all_fsd_metrics(data, 'dataset_name')
    print(f"δ_agg = {metrics['fsd_metrics']['delta_agg']:.2f}")
    print(f"Predicted method: {metrics['method_prediction']['predicted_method']}")
""")
