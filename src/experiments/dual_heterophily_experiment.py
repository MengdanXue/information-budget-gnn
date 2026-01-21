#!/usr/bin/env python3
"""
Dual Heterophily Types Validation Experiment
=============================================

验证双类型异质性假说：
- Type A (WebKB): 特征相似的异质性，2-hop可恢复，H2GCN有效
- Type B (Wikipedia): 特征正交的异质性，2-hop不可恢复，MLP最优

关键指标：
1. 邻居特征相似度分布
2. 2-hop label recovery ratio
3. H2GCN vs MLP vs GCN 性能对比

Author: FSD Framework
Date: 2025-01-16
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
from torch_geometric.datasets import WebKB, WikipediaNetwork, Planetoid
from torch_geometric.utils import to_undirected, add_self_loops
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

np.random.seed(42)
torch.manual_seed(42)


# ============== Models ==============

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class H2GCN(nn.Module):
    """
    Simplified H2GCN: uses 1-hop and 2-hop aggregation separately, then concatenates
    Based on "Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs"
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # Feature transform
        self.fc0 = nn.Linear(in_channels, hidden_channels)

        # 1-hop aggregation
        self.conv1 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)

        # 2-hop aggregation (will be computed manually)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)

        # Final classifier on concatenated features
        # ego + 1-hop + 2-hop = 3 * hidden_channels
        self.fc_out = nn.Linear(hidden_channels * 3, out_channels)

    def forward(self, x, edge_index):
        # Initial feature transform
        h0 = F.relu(self.fc0(x))
        h0 = F.dropout(h0, p=self.dropout, training=self.training)

        # 1-hop aggregation (neighbors only, no self-loop)
        h1 = self.conv1(h0, edge_index)
        h1 = F.relu(h1)

        # 2-hop aggregation: aggregate from 1-hop representations
        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)

        # Concatenate ego, 1-hop, 2-hop
        h_concat = torch.cat([h0, h1, h2], dim=1)
        h_concat = F.dropout(h_concat, p=self.dropout, training=self.training)

        return self.fc_out(h_concat)


class LINKX(nn.Module):
    """
    Simplified LINKX: separate MLPs for features and structure, then combine
    Based on "Large Scale Learning on Non-Homophilous Graphs"
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # Feature MLP
        self.mlp_feat = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Structure MLP (on aggregated features)
        self.conv = GCNConv(in_channels, hidden_channels, add_self_loops=True)
        self.mlp_struct = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Combination
        self.fc_out = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, x, edge_index):
        # Feature branch
        h_feat = self.mlp_feat(x)

        # Structure branch
        h_agg = self.conv(x, edge_index)
        h_struct = self.mlp_struct(h_agg)

        # Combine
        h_combined = torch.cat([h_feat, h_struct], dim=1)
        h_combined = F.dropout(h_combined, p=self.dropout, training=self.training)

        return self.fc_out(h_combined)


# ============== Metrics ==============

def compute_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute edge homophily (1-hop)"""
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()


def compute_2hop_homophily(edge_index: torch.Tensor, labels: torch.Tensor, n_nodes: int) -> float:
    """Compute 2-hop homophily"""
    # Build adjacency list
    adj = defaultdict(set)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj[src].add(dst)
        adj[dst].add(src)

    # Find 2-hop neighbors (excluding 1-hop)
    two_hop_same = 0
    two_hop_total = 0

    labels_np = labels.cpu().numpy()

    for node in range(min(n_nodes, 1000)):  # Sample for efficiency
        one_hop = adj[node]
        two_hop = set()
        for neighbor in one_hop:
            two_hop.update(adj[neighbor])
        two_hop -= one_hop  # Remove 1-hop neighbors
        two_hop.discard(node)  # Remove self

        for neighbor2 in two_hop:
            two_hop_total += 1
            if labels_np[node] == labels_np[neighbor2]:
                two_hop_same += 1

    if two_hop_total == 0:
        return 0.0
    return two_hop_same / two_hop_total


def compute_neighbor_feature_similarity(features: torch.Tensor, edge_index: torch.Tensor,
                                        labels: torch.Tensor, sample_size: int = 10000) -> Dict:
    """
    Compute feature similarity statistics for different-class neighbors
    """
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()

    # Get different-class edges
    diff_class_mask = labels_np[src] != labels_np[dst]
    src_diff = src[diff_class_mask]
    dst_diff = dst[diff_class_mask]

    if len(src_diff) == 0:
        return {'mean_sim': 0, 'std_sim': 0, 'opposite_frac': 0, 'orthogonal_frac': 0, 'similar_frac': 0}

    # Sample if too many
    if len(src_diff) > sample_size:
        idx = np.random.choice(len(src_diff), sample_size, replace=False)
        src_diff, dst_diff = src_diff[idx], dst_diff[idx]

    # Normalize features
    norms = np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8
    features_norm = features_np / norms

    # Compute cosine similarities
    sims = np.sum(features_norm[src_diff] * features_norm[dst_diff], axis=1)

    return {
        'mean_sim': float(np.mean(sims)),
        'std_sim': float(np.std(sims)),
        'min_sim': float(np.min(sims)),
        'max_sim': float(np.max(sims)),
        'opposite_frac': float((sims < -0.1).mean()),  # < -0.1
        'orthogonal_frac': float((np.abs(sims) <= 0.1).mean()),  # [-0.1, 0.1]
        'similar_frac': float((sims > 0.1).mean()),  # > 0.1
        'n_negative': int((sims < 0).sum()),
        'n_total': len(sims)
    }


def compute_2hop_recovery_ratio(edge_index: torch.Tensor, labels: torch.Tensor, n_nodes: int) -> float:
    """
    Compute 2-hop label recovery ratio = h_2hop / h_1hop

    If > 1: 2-hop neighbors are more likely to share labels than 1-hop (Type A)
    If < 1: 2-hop is even worse than 1-hop (Type B)
    """
    h_1hop = compute_homophily(edge_index, labels)
    h_2hop = compute_2hop_homophily(edge_index, labels, n_nodes)

    if h_1hop == 0:
        return float('inf') if h_2hop > 0 else 1.0

    return h_2hop / h_1hop


# ============== Training ==============

def train_and_evaluate(model, data, train_mask, val_mask, test_mask,
                       epochs=200, lr=0.01, patience=30) -> float:
    """Train and return test accuracy"""
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_acc = 0
    best_test_acc = 0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

            val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    return best_test_acc


def create_split(n_nodes: int, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Create random split"""
    torch.manual_seed(seed)
    perm = torch.randperm(n_nodes)

    train_size = int(train_ratio * n_nodes)
    val_size = int(val_ratio * n_nodes)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size + val_size]] = True
    test_mask[perm[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask


# ============== Main Experiment ==============

@dataclass
class DualHeterophilyResult:
    dataset: str
    heterophily_type: str  # "Type A" or "Type B"
    n_nodes: int
    n_edges: int
    n_classes: int
    homophily_1hop: float
    homophily_2hop: float
    recovery_ratio: float
    neighbor_sim_mean: float
    neighbor_sim_std: float
    opposite_frac: float
    orthogonal_frac: float
    similar_frac: float
    mlp_acc: float
    gcn_acc: float
    sage_acc: float
    h2gcn_acc: float
    linkx_acc: float
    best_model: str
    h2gcn_vs_mlp: float
    linkx_vs_mlp: float
    hypothesis_supported: bool


def run_dual_heterophily_experiment(data: Data, dataset_name: str, n_runs: int = 10) -> DualHeterophilyResult:
    """
    Run dual heterophily validation on a single dataset
    """
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")

    n_nodes = data.x.shape[0]
    n_features = data.x.shape[1]
    n_classes = len(torch.unique(data.y))
    n_edges = data.edge_index.shape[1] // 2

    # Ensure undirected
    edge_index = to_undirected(data.edge_index)
    data = Data(x=data.x, edge_index=edge_index, y=data.y)

    print(f"Nodes: {n_nodes}, Features: {n_features}, Classes: {n_classes}, Edges: {n_edges}")

    # Compute metrics
    h_1hop = compute_homophily(edge_index, data.y)
    h_2hop = compute_2hop_homophily(edge_index, data.y, n_nodes)
    recovery_ratio = compute_2hop_recovery_ratio(edge_index, data.y, n_nodes)

    print(f"\nHomophily Analysis:")
    print(f"  1-hop homophily: {h_1hop:.3f}")
    print(f"  2-hop homophily: {h_2hop:.3f}")
    print(f"  2-hop recovery ratio: {recovery_ratio:.2f}x")

    # Neighbor feature similarity
    sim_stats = compute_neighbor_feature_similarity(data.x, edge_index, data.y)

    print(f"\nNeighbor Feature Similarity (different-class edges):")
    print(f"  Mean similarity: {sim_stats['mean_sim']:.3f}")
    print(f"  Opposite (<-0.1): {sim_stats['opposite_frac']*100:.1f}%")
    print(f"  Orthogonal (±0.1): {sim_stats['orthogonal_frac']*100:.1f}%")
    print(f"  Similar (>0.1): {sim_stats['similar_frac']*100:.1f}%")

    # Determine heterophily type
    if recovery_ratio > 1.5 or sim_stats['mean_sim'] > 0.2:
        het_type = "Type A"
        print(f"\n  → Classified as TYPE A (Feature-Similar Heterophily)")
    else:
        het_type = "Type B"
        print(f"\n  → Classified as TYPE B (Feature-Orthogonal Heterophily)")

    # Train models
    print(f"\nTraining models ({n_runs} runs)...")

    models_results = {
        'MLP': [], 'GCN': [], 'GraphSAGE': [], 'H2GCN': [], 'LINKX': []
    }

    for run in range(n_runs):
        seed = 42 + run * 100
        train_mask, val_mask, test_mask = create_split(n_nodes, seed=seed)

        # MLP
        torch.manual_seed(seed)
        mlp = MLP(n_features, 64, n_classes)
        mlp_acc = train_and_evaluate(mlp, data, train_mask, val_mask, test_mask)
        models_results['MLP'].append(mlp_acc)

        # GCN
        torch.manual_seed(seed)
        gcn = GCN(n_features, 64, n_classes)
        gcn_acc = train_and_evaluate(gcn, data, train_mask, val_mask, test_mask)
        models_results['GCN'].append(gcn_acc)

        # GraphSAGE
        torch.manual_seed(seed)
        sage = GraphSAGE(n_features, 64, n_classes)
        sage_acc = train_and_evaluate(sage, data, train_mask, val_mask, test_mask)
        models_results['GraphSAGE'].append(sage_acc)

        # H2GCN
        torch.manual_seed(seed)
        h2gcn = H2GCN(n_features, 64, n_classes)
        h2gcn_acc = train_and_evaluate(h2gcn, data, train_mask, val_mask, test_mask)
        models_results['H2GCN'].append(h2gcn_acc)

        # LINKX
        torch.manual_seed(seed)
        linkx = LINKX(n_features, 64, n_classes)
        linkx_acc = train_and_evaluate(linkx, data, train_mask, val_mask, test_mask)
        models_results['LINKX'].append(linkx_acc)

        print(f"  Run {run+1}: MLP={mlp_acc:.3f} GCN={gcn_acc:.3f} SAGE={sage_acc:.3f} H2GCN={h2gcn_acc:.3f} LINKX={linkx_acc:.3f}")

    # Aggregate results
    means = {k: np.mean(v) for k, v in models_results.items()}
    stds = {k: np.std(v) for k, v in models_results.items()}

    print(f"\n--- Results Summary ---")
    for model_name in ['MLP', 'GCN', 'GraphSAGE', 'H2GCN', 'LINKX']:
        print(f"  {model_name}: {means[model_name]*100:.1f}% ± {stds[model_name]*100:.1f}%")

    best_model = max(means, key=means.get)
    h2gcn_vs_mlp = means['H2GCN'] - means['MLP']
    linkx_vs_mlp = means['LINKX'] - means['MLP']

    print(f"\n  Best model: {best_model}")
    print(f"  H2GCN vs MLP: {h2gcn_vs_mlp*100:+.1f}%")
    print(f"  LINKX vs MLP: {linkx_vs_mlp*100:+.1f}%")

    # Check hypothesis
    # Type A: H2GCN should help (H2GCN > MLP)
    # Type B: MLP should be best or close
    if het_type == "Type A":
        hypothesis_supported = h2gcn_vs_mlp > 0 or linkx_vs_mlp > 0
    else:
        hypothesis_supported = best_model in ['MLP', 'GraphSAGE', 'LINKX'] or h2gcn_vs_mlp < 0.02

    print(f"\n  Hypothesis supported: {'YES' if hypothesis_supported else 'NO'}")

    return DualHeterophilyResult(
        dataset=dataset_name,
        heterophily_type=het_type,
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_classes=n_classes,
        homophily_1hop=h_1hop,
        homophily_2hop=h_2hop,
        recovery_ratio=recovery_ratio,
        neighbor_sim_mean=sim_stats['mean_sim'],
        neighbor_sim_std=sim_stats['std_sim'],
        opposite_frac=sim_stats['opposite_frac'],
        orthogonal_frac=sim_stats['orthogonal_frac'],
        similar_frac=sim_stats['similar_frac'],
        mlp_acc=means['MLP'],
        gcn_acc=means['GCN'],
        sage_acc=means['GraphSAGE'],
        h2gcn_acc=means['H2GCN'],
        linkx_acc=means['LINKX'],
        best_model=best_model,
        h2gcn_vs_mlp=h2gcn_vs_mlp,
        linkx_vs_mlp=linkx_vs_mlp,
        hypothesis_supported=hypothesis_supported
    )


def main():
    print("=" * 80)
    print("DUAL HETEROPHILY TYPES VALIDATION EXPERIMENT")
    print("Type A: Feature-Similar (WebKB) vs Type B: Feature-Orthogonal (Wikipedia)")
    print("=" * 80)

    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)

    results = []

    # WebKB datasets (expected Type A)
    print("\n" + "=" * 80)
    print("WEBKB DATASETS (Expected: Type A - Feature-Similar Heterophily)")
    print("=" * 80)

    for name in ['Texas', 'Wisconsin', 'Cornell']:
        try:
            dataset = WebKB(root=str(data_dir), name=name)
            result = run_dual_heterophily_experiment(dataset[0], name, n_runs=10)
            results.append(asdict(result))
        except Exception as e:
            print(f"Error on {name}: {e}")
            import traceback
            traceback.print_exc()

    # Wikipedia datasets (expected Type B)
    print("\n" + "=" * 80)
    print("WIKIPEDIA DATASETS (Expected: Type B - Feature-Orthogonal Heterophily)")
    print("=" * 80)

    for name in ['chameleon', 'squirrel']:
        try:
            dataset = WikipediaNetwork(root=str(data_dir), name=name)
            result = run_dual_heterophily_experiment(dataset[0], name, n_runs=10)
            results.append(asdict(result))
        except Exception as e:
            print(f"Error on {name}: {e}")
            import traceback
            traceback.print_exc()

    # Add a high-h dataset for comparison
    print("\n" + "=" * 80)
    print("HIGH HOMOPHILY DATASET (Control Group)")
    print("=" * 80)

    try:
        dataset = Planetoid(root=str(data_dir), name='Cora')
        result = run_dual_heterophily_experiment(dataset[0], 'Cora', n_runs=10)
        results.append(asdict(result))
    except Exception as e:
        print(f"Error on Cora: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: DUAL HETEROPHILY TYPES")
    print("=" * 80)

    print(f"\n{'Dataset':<15} {'Type':<10} {'h_1hop':<8} {'h_2hop':<8} {'Recovery':<10} {'MLP':<8} {'H2GCN':<8} {'LINKX':<8} {'Best':<10} {'Support'}")
    print("-" * 110)

    type_a_results = []
    type_b_results = []

    for r in results:
        support = "YES" if r['hypothesis_supported'] else "NO"
        print(f"{r['dataset']:<15} {r['heterophily_type']:<10} {r['homophily_1hop']:<8.3f} {r['homophily_2hop']:<8.3f} "
              f"{r['recovery_ratio']:<10.2f} {r['mlp_acc']*100:<8.1f} {r['h2gcn_acc']*100:<8.1f} "
              f"{r['linkx_acc']*100:<8.1f} {r['best_model']:<10} {support}")

        if r['heterophily_type'] == 'Type A':
            type_a_results.append(r)
        else:
            type_b_results.append(r)

    print("\n" + "-" * 80)
    print("KEY FINDINGS:")
    print("-" * 80)

    if type_a_results:
        print(f"\nType A (Feature-Similar) - {len(type_a_results)} datasets:")
        avg_recovery = np.mean([r['recovery_ratio'] for r in type_a_results])
        avg_sim = np.mean([r['neighbor_sim_mean'] for r in type_a_results])
        avg_h2gcn_gain = np.mean([r['h2gcn_vs_mlp'] for r in type_a_results])
        print(f"  Average 2-hop recovery ratio: {avg_recovery:.2f}x")
        print(f"  Average neighbor similarity: {avg_sim:.3f}")
        print(f"  Average H2GCN gain over MLP: {avg_h2gcn_gain*100:+.1f}%")
        print(f"  → H2GCN/LINKX {'EFFECTIVE' if avg_h2gcn_gain > 0 else 'NOT EFFECTIVE'}")

    if type_b_results:
        print(f"\nType B (Feature-Orthogonal) - {len(type_b_results)} datasets:")
        avg_recovery = np.mean([r['recovery_ratio'] for r in type_b_results])
        avg_sim = np.mean([r['neighbor_sim_mean'] for r in type_b_results])
        avg_h2gcn_gain = np.mean([r['h2gcn_vs_mlp'] for r in type_b_results])
        print(f"  Average 2-hop recovery ratio: {avg_recovery:.2f}x")
        print(f"  Average neighbor similarity: {avg_sim:.3f}")
        print(f"  Average H2GCN gain over MLP: {avg_h2gcn_gain*100:+.1f}%")
        best_models = [r['best_model'] for r in type_b_results]
        print(f"  Best models: {', '.join(best_models)}")
        print(f"  → Structure-aware methods {'INEFFECTIVE' if avg_h2gcn_gain <= 0 else 'SOMEWHAT EFFECTIVE'}")

    # Hypothesis validation
    total_supported = sum(1 for r in results if r['hypothesis_supported'])
    print(f"\nOVERALL HYPOTHESIS VALIDATION: {total_supported}/{len(results)} ({total_supported/len(results)*100:.0f}%)")

    # Save results
    output_path = Path(__file__).parent / 'dual_heterophily_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'dual_heterophily_types',
            'description': 'Validation of Type A (Feature-Similar) vs Type B (Feature-Orthogonal) Heterophily',
            'results': results,
            'summary': {
                'total_datasets': len(results),
                'hypothesis_supported': total_supported,
                'support_rate': total_supported / len(results) if results else 0
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
