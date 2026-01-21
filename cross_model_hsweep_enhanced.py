"""
Enhanced Cross-Model H-Sweep Experiment
- Increased runs: 5 -> 10 for better statistical power
- Added Bootstrap confidence intervals
- Added effect size (Cohen's d) analysis

P1 improvement for TKDE submission.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from pathlib import Path
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ============== Model Definitions ==============

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

class GCN(nn.Module):
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

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=0.5)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(nn.Module):
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

# ============== Data Generation ==============

def generate_controlled_graph(n_nodes=1000, n_features=20, n_classes=2,
                               target_h=0.5, feature_separability=0.5, seed=42):
    set_seed(seed)

    labels = np.array([i % n_classes for i in range(n_nodes)])
    np.random.shuffle(labels)

    features = np.random.randn(n_nodes, n_features)
    class_centers = {}
    for c in range(n_classes):
        class_centers[c] = np.random.randn(n_features) * feature_separability * 2

    for i in range(n_nodes):
        features[i] += class_centers[labels[i]]

    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    n_edges = n_nodes * 15 // 2
    edge_list = []
    same_class_edges = 0
    diff_class_edges = 0

    target_same = int(n_edges * target_h)
    target_diff = n_edges - target_same

    for _ in range(target_same * 3):
        if same_class_edges >= target_same:
            break
        c = np.random.randint(n_classes)
        class_nodes = np.where(labels == c)[0]
        if len(class_nodes) < 2:
            continue
        i, j = np.random.choice(class_nodes, 2, replace=False)
        if i != j and (i, j) not in edge_list and (j, i) not in edge_list:
            edge_list.append((i, j))
            same_class_edges += 1

    for _ in range(target_diff * 3):
        if diff_class_edges >= target_diff:
            break
        c1 = np.random.randint(n_classes)
        c2 = (c1 + 1) % n_classes
        class1_nodes = np.where(labels == c1)[0]
        class2_nodes = np.where(labels == c2)[0]
        i = np.random.choice(class1_nodes)
        j = np.random.choice(class2_nodes)
        if (i, j) not in edge_list and (j, i) not in edge_list:
            edge_list.append((i, j))
            diff_class_edges += 1

    edge_index = np.array(edge_list).T
    edge_index = np.hstack([edge_index, edge_index[[1, 0]]])

    actual_h = same_class_edges / (same_class_edges + diff_class_edges) if (same_class_edges + diff_class_edges) > 0 else 0

    return {
        'features': torch.tensor(features, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.long),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'h_target': target_h,
        'h_actual': actual_h,
        'n_nodes': n_nodes,
        'n_edges': edge_index.shape[1]
    }

# ============== Training ==============

def train_and_evaluate(model, data, epochs=200, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    x = data['features'].to(device)
    edge_index = data['edge_index'].to(device)
    y = data['labels'].to(device)
    n = len(y)

    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[perm[:int(0.6*n)]] = True
    val_mask[perm[int(0.6*n):int(0.8*n)]] = True
    test_mask[perm[int(0.8*n):]] = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            pred = out.argmax(dim=1)
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

    return best_test_acc

# ============== Statistical Analysis ==============

def bootstrap_ci(data, confidence=0.95, n_bootstrap=10000):
    """Compute bootstrap confidence interval."""
    bootstrap_means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    return lower, upper

def cohens_d(x, y):
    """Compute Cohen's d effect size."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0

# ============== Main Experiment ==============

def run_enhanced_hsweep():
    """Run enhanced H-sweep with 10 runs and bootstrap CI."""

    h_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_runs = 10  # Increased from 5 to 10
    feature_separability = 0.5
    hidden_channels = 64

    model_classes = {
        'MLP': MLP,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE
    }

    results = []
    all_raw_data = defaultdict(lambda: defaultdict(list))

    print("="*70)
    print("Enhanced Cross-Model H-Sweep Experiment (N=10 runs)")
    print("="*70)

    for h in h_values:
        print(f"\n[h = {h}]")
        model_accs = {name: [] for name in model_classes}

        for run in range(n_runs):
            seed = 42 + run * 100 + int(h * 1000)
            data = generate_controlled_graph(
                target_h=h,
                feature_separability=feature_separability,
                seed=seed
            )

            for model_name, model_class in model_classes.items():
                if model_name == 'GAT':
                    model = model_class(20, hidden_channels, 2, heads=4)
                else:
                    model = model_class(20, hidden_channels, 2)

                acc = train_and_evaluate(model, data)
                model_accs[model_name].append(acc)
                all_raw_data[h][model_name].append(acc)

        # Compute statistics with bootstrap CI
        h_result = {
            'h': h,
            'h_actual': data['h_actual'],
            'n_runs': n_runs
        }

        mlp_accs = model_accs['MLP']

        for model_name in model_classes:
            accs = model_accs[model_name]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs, ddof=1)
            ci_lower, ci_upper = bootstrap_ci(accs)

            h_result[f'{model_name}_acc'] = mean_acc
            h_result[f'{model_name}_std'] = std_acc
            h_result[f'{model_name}_ci_lower'] = ci_lower
            h_result[f'{model_name}_ci_upper'] = ci_upper

            print(f"  {model_name}: {mean_acc:.4f} +/- {std_acc:.4f} [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")

        # Compute advantages with effect sizes
        for model_name in ['GCN', 'GAT', 'GraphSAGE']:
            gnn_accs = model_accs[model_name]
            advantage = np.mean(gnn_accs) - np.mean(mlp_accs)

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(gnn_accs, mlp_accs)

            # Cohen's d
            d = cohens_d(gnn_accs, mlp_accs)

            # Bootstrap CI for advantage
            advantages = [g - m for g, m in zip(gnn_accs, mlp_accs)]
            adv_ci_lower, adv_ci_upper = bootstrap_ci(advantages)

            h_result[f'{model_name}_advantage'] = advantage
            h_result[f'{model_name}_advantage_ci_lower'] = adv_ci_lower
            h_result[f'{model_name}_advantage_ci_upper'] = adv_ci_upper
            h_result[f'{model_name}_vs_MLP_ttest_p'] = p_value
            h_result[f'{model_name}_vs_MLP_cohens_d'] = d

            sig = '*' if p_value < 0.05 else ''
            sig = '**' if p_value < 0.01 else sig
            print(f"  {model_name} vs MLP: {advantage:+.4f} [CI: {adv_ci_lower:+.4f}, {adv_ci_upper:+.4f}] p={p_value:.4f}{sig} d={d:.3f}")

        results.append(h_result)

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY: Zone-wise Analysis with Enhanced Statistics")
    print("="*70)

    zone_stats = {
        'trust_low': {'h_range': 'h < 0.3', 'results': [r for r in results if r['h'] < 0.3]},
        'uncertain': {'h_range': '0.3 <= h <= 0.7', 'results': [r for r in results if 0.3 <= r['h'] <= 0.7]},
        'trust_high': {'h_range': 'h > 0.7', 'results': [r for r in results if r['h'] > 0.7]}
    }

    for zone_name, zone_data in zone_stats.items():
        print(f"\n{zone_name.upper()} ({zone_data['h_range']}):")
        for model in ['GCN', 'GAT', 'GraphSAGE']:
            advantages = [r[f'{model}_advantage'] for r in zone_data['results']]
            mean_adv = np.mean(advantages)
            wins = sum(1 for a in advantages if a > 0)
            total = len(advantages)
            print(f"  {model}: mean advantage = {mean_adv:+.4f}, wins {wins}/{total}")

    # Save results
    output = {
        'experiment': 'cross_model_hsweep_enhanced',
        'models': list(model_classes.keys()),
        'h_values': h_values,
        'n_runs': n_runs,
        'feature_separability': feature_separability,
        'results': results,
        'zone_summary': {
            'trust_region_low': {
                'h_range': 'h < 0.3',
                'GCN_mean_advantage': np.mean([r['GCN_advantage'] for r in results if r['h'] < 0.3]),
                'GAT_mean_advantage': np.mean([r['GAT_advantage'] for r in results if r['h'] < 0.3]),
                'GraphSAGE_mean_advantage': np.mean([r['GraphSAGE_advantage'] for r in results if r['h'] < 0.3])
            },
            'uncertainty_zone': {
                'h_range': '0.3 <= h <= 0.7',
                'GCN_mean_advantage': np.mean([r['GCN_advantage'] for r in results if 0.3 <= r['h'] <= 0.7]),
                'GAT_mean_advantage': np.mean([r['GAT_advantage'] for r in results if 0.3 <= r['h'] <= 0.7]),
                'GraphSAGE_mean_advantage': np.mean([r['GraphSAGE_advantage'] for r in results if 0.3 <= r['h'] <= 0.7])
            },
            'trust_region_high': {
                'h_range': 'h > 0.7',
                'GCN_mean_advantage': np.mean([r['GCN_advantage'] for r in results if r['h'] > 0.7]),
                'GAT_mean_advantage': np.mean([r['GAT_advantage'] for r in results if r['h'] > 0.7]),
                'GraphSAGE_mean_advantage': np.mean([r['GraphSAGE_advantage'] for r in results if r['h'] > 0.7])
            }
        }
    }

    output_path = Path(__file__).parent / "cross_model_hsweep_enhanced_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return output

if __name__ == "__main__":
    results = run_enhanced_hsweep()
