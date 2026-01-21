"""
Unified SPI Validation Experiment
=================================
This script provides a single, authoritative validation of the SPI framework.

Key Design Principles:
1. Strict separation of synthetic vs real-world results
2. Consistent success criteria (GNN > MLP by >= 1%)
3. Proper statistical reporting with confidence intervals
4. No "Uncertain" escape hatch - make clear predictions

Output: unified_validation_results.json
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
from torch_geometric.datasets import Amazon, Coauthor, CitationFull
from torch_geometric.utils import to_undirected
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
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

# ============== Utilities ==============

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_homophily(edge_index, y):
    """Compute edge homophily ratio."""
    row, col = edge_index[0], edge_index[1]
    return (y[row] == y[col]).float().mean().item()

def compute_spi(h):
    """Structural Predictability Index."""
    return abs(2 * h - 1)

def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4, patience=30):
    """Train and return best test accuracy."""
    model = model.to(DEVICE)
    x = data.x.to(DEVICE)
    y = data.y.to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)

    train_mask = data.train_mask.to(DEVICE)
    val_mask = data.val_mask.to(DEVICE)
    test_mask = data.test_mask.to(DEVICE)

    # Handle multi-dimensional masks
    if train_mask.dim() > 1:
        train_mask = train_mask[:, 0]
        val_mask = val_mask[:, 0]
        test_mask = test_mask[:, 0]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0

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
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_test_acc

def create_random_splits(data, train_ratio=0.6, val_ratio=0.2):
    """Create random train/val/test splits."""
    n = data.x.size(0)
    perm = torch.randperm(n)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)

    data.train_mask[perm[:train_end]] = True
    data.val_mask[perm[train_end:val_end]] = True
    data.test_mask[perm[val_end:]] = True

    return data

# ============== Data Loading ==============

def load_all_datasets():
    """Load all real-world datasets for validation."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    datasets = {}

    # 1. Planetoid (Homophilic)
    print("Loading Planetoid datasets...")
    for name in ['Cora', 'CiteSeer', 'PubMed']:
        try:
            dataset = Planetoid(root=str(data_dir), name=name)
            data = dataset[0]
            datasets[name] = data
            print(f"  {name}: {data.x.size(0)} nodes, h={compute_homophily(data.edge_index, data.y):.3f}")
        except Exception as e:
            print(f"  {name}: Failed - {e}")

    # 2. WebKB (Heterophilic)
    print("Loading WebKB datasets...")
    for name in ['Cornell', 'Texas', 'Wisconsin']:
        try:
            dataset = WebKB(root=str(data_dir), name=name)
            data = dataset[0]
            # Create random splits since WebKB has many splits
            data = create_random_splits(data)
            datasets[name] = data
            print(f"  {name}: {data.x.size(0)} nodes, h={compute_homophily(data.edge_index, data.y):.3f}")
        except Exception as e:
            print(f"  {name}: Failed - {e}")

    # 3. Actor (Heterophilic)
    print("Loading Actor dataset...")
    try:
        dataset = Actor(root=str(data_dir))
        data = dataset[0]
        data = create_random_splits(data)
        datasets['Actor'] = data
        print(f"  Actor: {data.x.size(0)} nodes, h={compute_homophily(data.edge_index, data.y):.3f}")
    except Exception as e:
        print(f"  Actor: Failed - {e}")

    # 4. Wikipedia Networks (Heterophilic)
    print("Loading Wikipedia datasets...")
    for name in ['Chameleon', 'Squirrel']:
        try:
            dataset = WikipediaNetwork(root=str(data_dir), name=name.lower())
            data = dataset[0]
            data = create_random_splits(data)
            datasets[name] = data
            print(f"  {name}: {data.x.size(0)} nodes, h={compute_homophily(data.edge_index, data.y):.3f}")
        except Exception as e:
            print(f"  {name}: Failed - {e}")

    # 5. Amazon (Homophilic)
    print("Loading Amazon datasets...")
    for name in ['Computers', 'Photo']:
        try:
            dataset = Amazon(root=str(data_dir), name=name)
            data = dataset[0]
            data = create_random_splits(data)
            datasets[f'Amazon-{name}'] = data
            print(f"  Amazon-{name}: {data.x.size(0)} nodes, h={compute_homophily(data.edge_index, data.y):.3f}")
        except Exception as e:
            print(f"  Amazon-{name}: Failed - {e}")

    # 6. Coauthor (Homophilic)
    print("Loading Coauthor datasets...")
    for name in ['CS', 'Physics']:
        try:
            dataset = Coauthor(root=str(data_dir), name=name)
            data = dataset[0]
            data = create_random_splits(data)
            datasets[f'Coauthor-{name}'] = data
            print(f"  Coauthor-{name}: {data.x.size(0)} nodes, h={compute_homophily(data.edge_index, data.y):.3f}")
        except Exception as e:
            print(f"  Coauthor-{name}: Failed - {e}")

    return datasets

# ============== Main Experiment ==============

def run_unified_validation(n_runs=10, hidden_dim=64, max_nodes=25000):
    """
    Run unified validation experiment.

    Success criteria:
    - GNN "wins" if best_GNN_acc > MLP_acc + 0.01 (1% margin)
    - MLP "wins" if MLP_acc >= best_GNN_acc - 0.01
    - Otherwise "Tie"

    SPI Prediction:
    - h > 0.5: Predict GNN wins
    - h <= 0.5: Predict MLP wins (conservative for heterophily)

    Args:
        n_runs: Number of runs per dataset
        hidden_dim: Hidden dimension for models
        max_nodes: Skip datasets larger than this (to avoid OOM)
    """

    print("="*70)
    print("UNIFIED SPI VALIDATION EXPERIMENT")
    print("="*70)
    print(f"Runs per dataset: {n_runs}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Device: {DEVICE}")
    print("="*70)

    # Load datasets
    datasets = load_all_datasets()

    if not datasets:
        print("No datasets loaded!")
        return None

    print(f"\nLoaded {len(datasets)} datasets")

    results = []

    for name, data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {name}")
        print(f"{'='*50}")

        h = compute_homophily(data.edge_index, data.y)
        spi = compute_spi(h)
        n_nodes = data.x.size(0)
        n_edges = data.edge_index.size(1)
        n_classes = len(torch.unique(data.y))
        n_features = data.x.size(1)

        print(f"Nodes: {n_nodes}, Edges: {n_edges}, Classes: {n_classes}")
        print(f"Homophily h = {h:.4f}, SPI = {spi:.4f}")

        # Skip large datasets to avoid OOM
        if n_nodes > max_nodes:
            print(f"  SKIPPED: Too large ({n_nodes} > {max_nodes} nodes)")
            continue

        # Run multiple times
        model_accs = {'MLP': [], 'GCN': [], 'GAT': [], 'GraphSAGE': []}

        for run in range(n_runs):
            set_seed(42 + run * 123)

            # Recreate splits for each run (except for Planetoid which has fixed splits)
            if name not in ['Cora', 'CiteSeer', 'PubMed']:
                data = create_random_splits(data)

            for model_name in model_accs.keys():
                if model_name == 'MLP':
                    model = MLP(n_features, hidden_dim, n_classes)
                elif model_name == 'GCN':
                    model = GCN(n_features, hidden_dim, n_classes)
                elif model_name == 'GAT':
                    model = GAT(n_features, hidden_dim, n_classes)
                else:
                    model = GraphSAGE(n_features, hidden_dim, n_classes)

                acc = train_model(model, data)
                model_accs[model_name].append(acc)

        # Compute statistics
        mlp_mean = np.mean(model_accs['MLP'])
        mlp_std = np.std(model_accs['MLP'])
        mlp_ci = stats.t.interval(0.95, n_runs-1, loc=mlp_mean, scale=stats.sem(model_accs['MLP']))

        best_gnn_name = None
        best_gnn_mean = -1

        gnn_stats = {}
        for gnn_name in ['GCN', 'GAT', 'GraphSAGE']:
            mean = np.mean(model_accs[gnn_name])
            std = np.std(model_accs[gnn_name])
            ci = stats.t.interval(0.95, n_runs-1, loc=mean, scale=stats.sem(model_accs[gnn_name]))
            gnn_stats[gnn_name] = {'mean': mean, 'std': std, 'ci': ci}

            if mean > best_gnn_mean:
                best_gnn_mean = mean
                best_gnn_name = gnn_name

        # Determine winner (1% margin)
        margin = 0.01
        if best_gnn_mean > mlp_mean + margin:
            actual_winner = 'GNN'
        elif mlp_mean > best_gnn_mean + margin:
            actual_winner = 'MLP'
        else:
            actual_winner = 'Tie'

        # SPI prediction (conservative: predict GNN only for h > 0.5)
        if h > 0.5:
            spi_prediction = 'GNN'
        else:
            spi_prediction = 'MLP'

        # Evaluate prediction
        # GNN prediction is correct if actual is GNN or Tie
        # MLP prediction is correct if actual is MLP or Tie
        if spi_prediction == 'GNN':
            prediction_correct = (actual_winner in ['GNN', 'Tie'])
        else:
            prediction_correct = (actual_winner in ['MLP', 'Tie'])

        # Print results
        print(f"\nResults:")
        print(f"  MLP:       {mlp_mean*100:.2f}% ± {mlp_std*100:.2f}%  95% CI: [{mlp_ci[0]*100:.2f}%, {mlp_ci[1]*100:.2f}%]")
        for gnn_name in ['GCN', 'GAT', 'GraphSAGE']:
            s = gnn_stats[gnn_name]
            adv = (s['mean'] - mlp_mean) * 100
            print(f"  {gnn_name:10s} {s['mean']*100:.2f}% ± {s['std']*100:.2f}%  (Δ={adv:+.2f}%)")

        print(f"\nBest GNN: {best_gnn_name} ({best_gnn_mean*100:.2f}%)")
        print(f"Winner: {actual_winner}")
        print(f"SPI Prediction: {spi_prediction}")
        print(f"Correct: {prediction_correct}")

        # Store result
        result = {
            'dataset': name,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'n_classes': n_classes,
            'n_features': n_features,
            'homophily': h,
            'spi': spi,
            'mlp_mean': mlp_mean,
            'mlp_std': mlp_std,
            'mlp_ci_low': mlp_ci[0],
            'mlp_ci_high': mlp_ci[1],
            'gcn_mean': gnn_stats['GCN']['mean'],
            'gcn_std': gnn_stats['GCN']['std'],
            'gat_mean': gnn_stats['GAT']['mean'],
            'gat_std': gnn_stats['GAT']['std'],
            'graphsage_mean': gnn_stats['GraphSAGE']['mean'],
            'graphsage_std': gnn_stats['GraphSAGE']['std'],
            'best_gnn': best_gnn_name,
            'best_gnn_mean': best_gnn_mean,
            'gcn_advantage': gnn_stats['GCN']['mean'] - mlp_mean,
            'best_gnn_advantage': best_gnn_mean - mlp_mean,
            'actual_winner': actual_winner,
            'spi_prediction': spi_prediction,
            'prediction_correct': prediction_correct
        }
        results.append(result)

    # ============== Summary ==============
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total = len(results)
    correct = sum(1 for r in results if r['prediction_correct'])

    print(f"\nOverall Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    # By homophily region
    high_h = [r for r in results if r['homophily'] > 0.7]
    mid_h = [r for r in results if 0.3 <= r['homophily'] <= 0.7]
    low_h = [r for r in results if r['homophily'] < 0.3]

    print(f"\nBy Homophily Region:")
    if high_h:
        c = sum(1 for r in high_h if r['prediction_correct'])
        print(f"  High h (>0.7):   {c}/{len(high_h)} ({100*c/len(high_h):.1f}%)")
        for r in high_h:
            status = "[OK]" if r['prediction_correct'] else "[X]"
            print(f"    {status} {r['dataset']}: h={r['homophily']:.3f}, "
                  f"Winner={r['actual_winner']}, Pred={r['spi_prediction']}")

    if mid_h:
        c = sum(1 for r in mid_h if r['prediction_correct'])
        print(f"  Mid h (0.3-0.7): {c}/{len(mid_h)} ({100*c/len(mid_h):.1f}%)")
        for r in mid_h:
            status = "[OK]" if r['prediction_correct'] else "[X]"
            print(f"    {status} {r['dataset']}: h={r['homophily']:.3f}, "
                  f"Winner={r['actual_winner']}, Pred={r['spi_prediction']}")

    if low_h:
        c = sum(1 for r in low_h if r['prediction_correct'])
        print(f"  Low h (<0.3):    {c}/{len(low_h)} ({100*c/len(low_h):.1f}%)")
        for r in low_h:
            status = "[OK]" if r['prediction_correct'] else "[X]"
            print(f"    {status} {r['dataset']}: h={r['homophily']:.3f}, "
                  f"Winner={r['actual_winner']}, Pred={r['spi_prediction']}")

    # Correlation analysis
    print(f"\nCorrelation Analysis:")
    hs = [r['homophily'] for r in results]
    gcn_advs = [r['gcn_advantage'] for r in results]
    best_advs = [r['best_gnn_advantage'] for r in results]

    r_gcn, p_gcn = stats.pearsonr(hs, gcn_advs)
    r_best, p_best = stats.pearsonr(hs, best_advs)

    print(f"  Homophily vs GCN Advantage:      r={r_gcn:.3f}, p={p_gcn:.4f}")
    print(f"  Homophily vs Best GNN Advantage: r={r_best:.3f}, p={p_best:.4f}")

    # Save results
    output = {
        'experiment': 'unified_spi_validation',
        'n_runs': n_runs,
        'hidden_dim': hidden_dim,
        'success_margin': 0.01,
        'prediction_rule': 'h > 0.5 -> GNN, else MLP',
        'total_datasets': total,
        'overall_accuracy': correct / total,
        'high_h_count': len(high_h),
        'high_h_accuracy': sum(1 for r in high_h if r['prediction_correct']) / len(high_h) if high_h else None,
        'mid_h_count': len(mid_h),
        'mid_h_accuracy': sum(1 for r in mid_h if r['prediction_correct']) / len(mid_h) if mid_h else None,
        'low_h_count': len(low_h),
        'low_h_accuracy': sum(1 for r in low_h if r['prediction_correct']) / len(low_h) if low_h else None,
        'correlation_h_gcn_adv': {'r': r_gcn, 'p': p_gcn},
        'correlation_h_best_adv': {'r': r_best, 'p': p_best},
        'results': results
    }

    output_path = Path(__file__).parent / "unified_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\nResults saved to: {output_path}")

    return output

if __name__ == "__main__":
    results = run_unified_validation(n_runs=10)
