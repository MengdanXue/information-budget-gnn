"""
GraphSAGE Ablation Study: Concat vs Replace
============================================

验证三AI审稿人的关键问题：
GraphSAGE在Q2象限的异常表现是否因为concatenation架构？

实验设计：
1. SAGE-Concat (原版): self-features与neighbor aggregation拼接
2. SAGE-Replace (消融): 用aggregation替换self-features (类似GCN)
3. GCN-Concat (对照): GCN + self-features拼接
4. GCN (baseline): 原版GCN

预期结果：
- 如果SAGE-Replace在Q2象限也失败 → 证明concat是关键
- 如果GCN-Concat在Q2象限改善 → 进一步证明concat假设
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
import json
from collections import defaultdict

from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.datasets import WebKB, HeterophilousGraphDataset
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================
# Model Definitions
# ============================================================

class MLP(nn.Module):
    """Baseline MLP (no graph structure)"""
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
    """Standard GCN: Replace self-features with aggregation"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GCN_Concat(nn.Module):
    """GCN with Concatenation: Preserve self-features"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # After concat: hidden_channels (agg) + in_channels (self) = hidden_channels + in_channels
        self.fc_combine = nn.Linear(hidden_channels + in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc_out = nn.Linear(hidden_channels + hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1: GCN + concat with original features
        x_agg = F.relu(self.conv1(x, edge_index))
        x_combined = torch.cat([x_agg, x], dim=1)  # Concat!
        x_combined = F.relu(self.fc_combine(x_combined))
        x_combined = F.dropout(x_combined, p=self.dropout, training=self.training)

        # Layer 2: GCN + concat
        x_agg2 = self.conv2(x_combined, edge_index)
        x_out = torch.cat([x_agg2, x_combined], dim=1)  # Concat again!
        return self.fc_out(x_out)


class GraphSAGE_Concat(nn.Module):
    """Original GraphSAGE: Concat self-features with neighbor aggregation"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        # SAGEConv already does concat internally
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE_Replace(nn.Module):
    """Modified GraphSAGE: Replace self-features (like GCN)

    This is a custom implementation that removes the concat behavior
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        # Use root_weight=False to disable self-loop (no concat)
        self.conv1 = SAGEConv(in_channels, hidden_channels, root_weight=False)
        self.conv2 = SAGEConv(hidden_channels, out_channels, root_weight=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


# ============================================================
# Training and Evaluation
# ============================================================

def train_and_evaluate(model, x, edge_index, labels, train_mask, val_mask, test_mask,
                       lr=0.01, weight_decay=5e-4, epochs=200, patience=50):
    """Train model and return best test accuracy"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    best_test_acc = 0
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
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    return best_test_acc


def compute_homophily(edge_index, labels):
    """Compute edge homophily"""
    src, dst = edge_index.cpu().numpy()
    lab = labels.cpu().numpy()
    return (lab[src] == lab[dst]).mean()


# ============================================================
# Main Ablation Experiment
# ============================================================

def run_ablation(n_runs=10):
    """Run ablation study on Q2 quadrant datasets"""

    print("=" * 80)
    print("GRAPHSAGE ABLATION STUDY: CONCAT VS REPLACE")
    print("=" * 80)
    print("\nHypothesis: GraphSAGE's robustness in Q2 quadrant comes from concatenation")
    print("Test: If SAGE-Replace fails like GCN, hypothesis is confirmed\n")

    # Q2 Quadrant datasets (High FS + Low h)
    datasets_config = [
        ('Texas', WebKB, {'name': 'Texas'}),
        ('Wisconsin', WebKB, {'name': 'Wisconsin'}),
        ('Cornell', WebKB, {'name': 'Cornell'}),
        ('Roman-empire', HeterophilousGraphDataset, {'name': 'Roman-empire'}),
    ]

    # Models to test
    model_classes = {
        'MLP': MLP,
        'GCN': GCN,
        'GCN-Concat': GCN_Concat,
        'SAGE-Concat': GraphSAGE_Concat,
        'SAGE-Replace': GraphSAGE_Replace,
    }

    all_results = {}

    for ds_name, DatasetClass, kwargs in datasets_config:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        # Load dataset
        dataset = DatasetClass(root='./data', **kwargs)
        data = dataset[0]

        x = data.x.to(device)
        edge_index = to_undirected(data.edge_index).to(device)
        labels = data.y.to(device)
        n_nodes = data.num_nodes
        n_features = data.num_features
        n_classes = len(labels.unique())

        h = compute_homophily(edge_index, labels)
        print(f"  Nodes: {n_nodes}, Features: {n_features}, Classes: {n_classes}")
        print(f"  Homophily: {h:.4f}")

        results = {model_name: [] for model_name in model_classes}

        for seed in range(n_runs):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create splits
            indices = np.arange(n_nodes)
            train_idx, temp_idx = train_test_split(indices, train_size=0.6, random_state=seed)
            val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=seed)

            train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

            # Train each model
            for model_name, ModelClass in model_classes.items():
                model = ModelClass(n_features, 64, n_classes).to(device)
                acc = train_and_evaluate(model, x, edge_index, labels,
                                        train_mask, val_mask, test_mask)
                results[model_name].append(acc)

        # Statistical analysis
        print(f"\n  Results ({n_runs} runs):")
        print(f"  {'Model':>15} {'Mean':>10} {'Std':>10} {'vs MLP':>10}")
        print("  " + "-" * 50)

        mlp_scores = results['MLP']
        mlp_mean = np.mean(mlp_scores)

        stats_results = {}
        for model_name in model_classes:
            scores = results[model_name]
            mean = np.mean(scores)
            std = np.std(scores)
            diff = mean - mlp_mean
            print(f"  {model_name:>15} {mean:>10.3f} {std:>10.3f} {diff:>+10.3f}")

            # Statistical test vs MLP
            if model_name != 'MLP':
                t_stat, p_value = stats.ttest_rel(scores, mlp_scores)
                stats_results[model_name] = {
                    'mean': mean,
                    'std': std,
                    'diff': diff,
                    't_stat': t_stat,
                    'p_value': p_value
                }

        # Print statistical significance
        print(f"\n  Statistical Tests (vs MLP):")
        for model_name, stat in stats_results.items():
            sig = "***" if stat['p_value'] < 0.001 else ("**" if stat['p_value'] < 0.01 else ("*" if stat['p_value'] < 0.05 else ""))
            print(f"  {model_name:>15}: t={stat['t_stat']:>7.3f}, p={stat['p_value']:.4f} {sig}")

        all_results[ds_name] = {
            'homophily': h,
            'n_nodes': n_nodes,
            'results': {k: {'mean': np.mean(v), 'std': np.std(v), 'scores': v}
                       for k, v in results.items()},
            'stats': stats_results
        }

    # ============================================================
    # Summary Analysis
    # ============================================================

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)

    print("\n1. Hypothesis Test: Does concatenation explain GraphSAGE's robustness?")
    print("-" * 60)

    concat_helps = 0
    replace_fails = 0

    for ds_name, data in all_results.items():
        mlp_mean = data['results']['MLP']['mean']
        sage_concat = data['results']['SAGE-Concat']['mean']
        sage_replace = data['results']['SAGE-Replace']['mean']
        gcn = data['results']['GCN']['mean']
        gcn_concat = data['results']['GCN-Concat']['mean']

        sage_concat_wins = sage_concat > mlp_mean
        sage_replace_wins = sage_replace > mlp_mean
        gcn_concat_wins = gcn_concat > mlp_mean
        gcn_wins = gcn > mlp_mean

        print(f"\n  {ds_name} (h={data['homophily']:.3f}):")
        print(f"    SAGE-Concat vs MLP: {sage_concat - mlp_mean:+.3f} {'WIN' if sage_concat_wins else 'LOSE'}")
        print(f"    SAGE-Replace vs MLP: {sage_replace - mlp_mean:+.3f} {'WIN' if sage_replace_wins else 'LOSE'}")
        print(f"    GCN-Concat vs MLP: {gcn_concat - mlp_mean:+.3f} {'WIN' if gcn_concat_wins else 'LOSE'}")
        print(f"    GCN vs MLP: {gcn - mlp_mean:+.3f} {'WIN' if gcn_wins else 'LOSE'}")

        if sage_concat_wins and not sage_replace_wins:
            concat_helps += 1
        if not sage_replace_wins and not gcn_wins:
            replace_fails += 1

    print("\n2. Conclusion:")
    print("-" * 60)
    print(f"  Datasets where SAGE-Concat wins but SAGE-Replace loses: {concat_helps}/4")
    print(f"  Datasets where both SAGE-Replace and GCN lose: {replace_fails}/4")

    if concat_helps >= 2:
        print("\n  [SUPPORTED] HYPOTHESIS CONFIRMED: Concatenation is key to GraphSAGE's robustness")
    else:
        print("\n  [INCONCLUSIVE] Need more investigation")

    # Create summary table for paper
    print("\n3. Table for Paper:")
    print("-" * 60)
    print("| Dataset | h | MLP | GCN | GCN-Cat | SAGE-Cat | SAGE-Rep |")
    print("|---------|-----|-----|-----|---------|----------|----------|")
    for ds_name, data in all_results.items():
        h = data['homophily']
        mlp = data['results']['MLP']['mean']
        gcn = data['results']['GCN']['mean'] - mlp
        gcn_cat = data['results']['GCN-Concat']['mean'] - mlp
        sage_cat = data['results']['SAGE-Concat']['mean'] - mlp
        sage_rep = data['results']['SAGE-Replace']['mean'] - mlp
        print(f"| {ds_name:13} | {h:.3f} | {mlp:.3f} | {gcn:+.3f} | {gcn_cat:+.3f} | {sage_cat:+.3f} | {sage_rep:+.3f} |")

    # Save results
    results_file = 'graphsage_ablation_results.json'

    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for ds_name, data in all_results.items():
        json_results[ds_name] = {
            'homophily': float(data['homophily']),
            'n_nodes': int(data['n_nodes']),
            'results': {
                k: {
                    'mean': float(v['mean']),
                    'std': float(v['std']),
                    'scores': [float(s) for s in v['scores']]
                }
                for k, v in data['results'].items()
            }
        }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == '__main__':
    results = run_ablation(n_runs=10)
