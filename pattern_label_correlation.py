"""
Pattern-Label Correlation Analysis
===================================

关键问题：为什么WebKB有清晰的pattern却让GCN失败？

假说：Pattern的"方向"与标签的关系决定了GCN能否利用它
- 正相关Pattern: 邻居的主导类能预测节点标签 → GCN有用
- 负相关Pattern: 邻居的主导类与节点标签负相关 → GCN有害
"""

import torch
import numpy as np
from collections import defaultdict
from torch_geometric.datasets import WebKB, Planetoid, WikipediaNetwork
import json


def compute_pattern_label_correlation(edge_index, labels):
    """
    计算Pattern方向与标签的相关性

    对每个节点：
    1. 找到邻居的主导类（最多的邻居标签）
    2. 计算P(node_label | dominant_neighbor_class)
    3. 如果P > random baseline，pattern正相关；否则负相关
    """
    src, dst = edge_index.numpy()
    labels_np = labels.numpy()
    n_nodes = len(labels_np)  # 修正：使用实际节点数
    n_classes = len(np.unique(labels_np))

    # 统计：给定邻居主导类，节点是什么类
    # dominant_class -> node_class -> count
    dominant_to_node = defaultdict(lambda: defaultdict(int))

    # 每个节点的分析
    node_analyses = []

    for node in range(n_nodes):
        neighbors = dst[src == node]
        if len(neighbors) == 0:
            continue

        node_label = labels_np[node]
        neighbor_labels = labels_np[neighbors]

        # 找到邻居的主导类
        label_counts = np.bincount(neighbor_labels, minlength=n_classes)
        dominant_neighbor_class = label_counts.argmax()
        dominant_ratio = label_counts[dominant_neighbor_class] / len(neighbors)

        # 统计
        dominant_to_node[dominant_neighbor_class][node_label] += 1

        node_analyses.append({
            'node': node,
            'node_label': int(node_label),
            'dominant_neighbor_class': int(dominant_neighbor_class),
            'dominant_ratio': float(dominant_ratio),
            'match': int(dominant_neighbor_class == node_label)
        })

    # 计算全局的Pattern-Label相关性
    # 对于每个dominant_class，看它预测节点标签的准确率
    pattern_accuracies = {}

    for dom_class, node_dist in dominant_to_node.items():
        total = sum(node_dist.values())
        if total == 0:
            continue

        # 如果pattern是"邻居主导类=X → 节点类=X"，准确率是多少
        correct = node_dist[dom_class]
        accuracy = correct / total

        pattern_accuracies[dom_class] = {
            'accuracy': accuracy,
            'total_nodes': total,
            'correct': correct
        }

    # 全局Pattern-Label相关性
    # 修改：分析所有有邻居的节点，不限制dominant_ratio
    total_nodes_analyzed = len(node_analyses)
    total_matches = sum(a['match'] for a in node_analyses)

    if total_nodes_analyzed > 0:
        global_match_rate = total_matches / total_nodes_analyzed
    else:
        global_match_rate = 0

    # 额外统计：只看dominant_ratio > 0.5的节点
    strong_pattern_nodes = sum(a['dominant_ratio'] > 0.5 for a in node_analyses)
    strong_pattern_matches = sum(a['match'] for a in node_analyses if a['dominant_ratio'] > 0.5)

    if strong_pattern_nodes > 0:
        strong_pattern_match_rate = strong_pattern_matches / strong_pattern_nodes
    else:
        strong_pattern_match_rate = 0

    random_baseline = 1.0 / n_classes

    # Pattern方向性得分
    # > 0: Pattern与标签正相关（有用）
    # < 0: Pattern与标签负相关（有害）
    pattern_direction_score = global_match_rate - random_baseline

    return {
        'global_match_rate': global_match_rate,
        'random_baseline': random_baseline,
        'pattern_direction_score': pattern_direction_score,
        'pattern_accuracies': pattern_accuracies,
        'node_analyses': node_analyses,
        'total_nodes': total_nodes_analyzed,
        'strong_pattern_nodes': strong_pattern_nodes,
        'strong_pattern_match_rate': strong_pattern_match_rate
    }


def analyze_dataset(dataset_name, data):
    """分析一个数据集的Pattern-Label相关性"""

    edge_index = data.edge_index
    labels = data.y

    result = compute_pattern_label_correlation(edge_index, labels)

    print(f"\n{dataset_name}:")
    print(f"  Total nodes analyzed: {result['total_nodes']}")
    print(f"  Strong pattern nodes (>50% dominant): {result['strong_pattern_nodes']}")
    print(f"  Global Match Rate: {result['global_match_rate']:.3f}")
    print(f"  Strong Pattern Match Rate: {result['strong_pattern_match_rate']:.3f}")
    print(f"  Random Baseline: {result['random_baseline']:.3f}")
    print(f"  Pattern Direction Score: {result['pattern_direction_score']:+.3f}")

    if result['pattern_direction_score'] > 0.1:
        print(f"  ==> POSITIVE Pattern: Neighbors predict node label (GCN should work)")
    elif result['pattern_direction_score'] < -0.1:
        print(f"  ==> NEGATIVE Pattern: Neighbors anti-predict node label (GCN harmful)")
    else:
        print(f"  ==> NEUTRAL Pattern: No useful signal in structure")

    # 详细分析每个类的pattern
    print(f"\n  Per-class pattern accuracy:")
    for dom_class, stats in result['pattern_accuracies'].items():
        print(f"    Dominant class {dom_class}: {stats['accuracy']:.3f} "
              f"({stats['correct']}/{stats['total_nodes']})")

    return result


def run_analysis():
    print("=" * 80)
    print("PATTERN-LABEL CORRELATION ANALYSIS")
    print("Hypothesis: WebKB has NEGATIVE pattern (neighbors anti-predict labels)")
    print("=" * 80)

    all_results = {}

    # WebKB datasets
    print("\n" + "=" * 80)
    print("WebKB Datasets (Expected: NEGATIVE pattern)")
    print("=" * 80)

    for name in ['Texas', 'Wisconsin', 'Cornell']:
        try:
            dataset = WebKB(root='./data', name=name)
            data = dataset[0]
            result = analyze_dataset(name, data)
            all_results[name] = result
        except Exception as e:
            print(f"\nError loading {name}: {e}")

    # Wikipedia datasets
    print("\n" + "=" * 80)
    print("Wikipedia Networks (Expected: NEGATIVE/NEUTRAL pattern)")
    print("=" * 80)

    for name in ['squirrel', 'chameleon']:
        try:
            dataset = WikipediaNetwork(root='./data', name=name)
            data = dataset[0]
            result = analyze_dataset(name.capitalize(), data)
            all_results[name.capitalize()] = result
        except Exception as e:
            print(f"\nError loading {name}: {e}")

    # Planetoid datasets
    print("\n" + "=" * 80)
    print("Planetoid Datasets (Expected: POSITIVE pattern)")
    print("=" * 80)

    for name in ['Cora', 'CiteSeer', 'PubMed']:
        try:
            dataset = Planetoid(root='./data', name=name)
            data = dataset[0]
            result = analyze_dataset(name, data)
            all_results[name] = result
        except Exception as e:
            print(f"\nError loading {name}: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Pattern Direction Scores")
    print("=" * 80)

    print(f"\n{'Dataset':>12} {'Match Rate':>12} {'Baseline':>10} {'Direction':>10} {'Type':>15}")
    print("-" * 70)

    for dataset_name, result in all_results.items():
        match_rate = result['global_match_rate']
        baseline = result['random_baseline']
        direction = result['pattern_direction_score']

        if direction > 0.1:
            ptype = "POSITIVE"
        elif direction < -0.1:
            ptype = "NEGATIVE"
        else:
            ptype = "NEUTRAL"

        print(f"{dataset_name:>12} {match_rate:>12.3f} {baseline:>10.3f} {direction:>+10.3f} {ptype:>15}")

    # 保存结果
    # 转换node_analyses为可JSON序列化的格式
    output = {}
    for dataset_name, result in all_results.items():
        output[dataset_name] = {
            'global_match_rate': result['global_match_rate'],
            'strong_pattern_match_rate': result['strong_pattern_match_rate'],
            'random_baseline': result['random_baseline'],
            'pattern_direction_score': result['pattern_direction_score'],
            'total_nodes': result['total_nodes'],
            'strong_pattern_nodes': result['strong_pattern_nodes'],
            'pattern_accuracies': {
                str(k): v for k, v in result['pattern_accuracies'].items()
            }
        }

    with open('pattern_label_correlation_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: pattern_label_correlation_results.json")

    return all_results


if __name__ == '__main__':
    run_analysis()
