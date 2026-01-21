"""
H2GCN Validation on Heterophilic Datasets

H2GCN: Beyond Homophily in Graph Neural Networks (NeurIPS 2020)
Key components:
1. Ego and neighbor separation
2. Higher-order neighborhoods
3. Combination of intermediate representations

Reference: Zhu et al., "Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs"
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ============== H2GCN Model ==============

class H2GCN(nn.Module):
    """
    H2GCN: Heterophily-aware Graph Convolutional Network

    Key designs for heterophily:
    1. Ego-neighbor separation: Don't mix ego features with neighbor aggregation
    2. Higher-order neighborhoods: Use k-hop neighbors (k=1,2)
    3. Intermediate representation combination: Concat all layer outputs
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Feature transformation (ego)
        self.lin_ego = nn.Linear(in_channels, hidden_channels)

        # Neighbor aggregation layers (k-hop)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_dim, hidden_channels, add_self_loops=False))

        # Final classifier: concat all representations
        # ego + num_layers * neighbor_agg
        final_dim = hidden_channels * (1 + num_layers)
        self.classifier = nn.Linear(final_dim, out_channels)

    def forward(self, x, edge_index):
        # 1. Ego transformation
        h_ego = self.lin_ego(x)
        h_ego = F.relu(h_ego)
        h_ego = F.dropout(h_ego, p=self.dropout, training=self.training)

        representations = [h_ego]

        # 2. Multi-hop neighbor aggregation (without mixing with ego)
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            representations.append(h)

        # 3. Concatenate all representations
        h_combined = torch.cat(representations, dim=1)

        # 4. Final classification
        out = self.classifier(h_combined)

        return out


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


# ============== Data Loading ==============

def compute_homophily(edge_index, labels):
    """Compute edge homophily."""
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()

def load_dataset(name, data_dir):
    """Load dataset from pickle file."""
    possible_paths = [
        data_dir / f"{name}_graph.pkl",
        data_dir / f"{name}.pkl",
        data_dir / f"{name}_weber_split.pkl",
    ]

    pkl_path = None
    for p in possible_paths:
        if p.exists():
            pkl_path = p
            break

    if pkl_path is None:
        return None

    with open(pkl_path, 'rb') as f:
        loaded = pickle.load(f)

    if isinstance(loaded, dict):
        if 'data' in loaded and hasattr(loaded['data'], 'x'):
            data = loaded['data']
            x = data.x
            edge_index = data.edge_index
            y = data.y
            train_mask = getattr(data, 'train_mask', None)
            val_mask = getattr(data, 'val_mask', None)
            test_mask = getattr(data, 'test_mask', None)
        elif 'features' in loaded:
            x = torch.tensor(loaded['features'], dtype=torch.float32)
            edge_index = torch.tensor(loaded['edge_index'], dtype=torch.long)
            y = torch.tensor(loaded['labels'], dtype=torch.long)
            train_mask = loaded.get('train_mask')
            val_mask = loaded.get('val_mask')
            test_mask = loaded.get('test_mask')
            if train_mask is not None:
                train_mask = torch.tensor(train_mask, dtype=torch.bool)
                val_mask = torch.tensor(val_mask, dtype=torch.bool)
                test_mask = torch.tensor(test_mask, dtype=torch.bool)
        else:
            return None
    elif hasattr(loaded, 'x'):
        x = loaded.x
        edge_index = loaded.edge_index
        y = loaded.y
        train_mask = getattr(loaded, 'train_mask', None)
        val_mask = getattr(loaded, 'val_mask', None)
        test_mask = getattr(loaded, 'test_mask', None)
    else:
        return None

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)

    if train_mask is not None and train_mask.dim() > 1:
        train_mask = train_mask[:, 0]
        val_mask = val_mask[:, 0]
        test_mask = test_mask[:, 0]

    n = x.shape[0]
    if train_mask is None:
        perm = torch.randperm(n)
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[perm[:int(0.6*n)]] = True
        val_mask[perm[int(0.6*n):int(0.8*n)]] = True
        test_mask[perm[int(0.8*n):]] = True

    h = compute_homophily(edge_index, y)
    spi = abs(2 * h - 1)

    return {
        'x': x,
        'edge_index': edge_index,
        'y': y,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'homophily': h,
        'spi': spi,
        'n_nodes': n,
        'n_edges': edge_index.shape[1],
        'n_classes': len(torch.unique(y[y >= 0])),
        'n_features': x.shape[1]
    }


# ============== Training ==============

def train_and_evaluate(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    """Train model and return test accuracy."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    x = data['x'].to(device)
    edge_index = data['edge_index'].to(device)
    y = data['y'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0
    best_test_acc = 0
    patience = 50
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


# ============== Main Experiment ==============

def run_h2gcn_validation():
    """Run H2GCN validation on heterophilic datasets."""

    data_dir = Path(__file__).parent / "data"

    # Focus on heterophilic datasets where GCN fails
    heterophilic_datasets = ['texas', 'wisconsin', 'cornell', 'actor', 'chameleon', 'squirrel']

    # Also test on homophilic for comparison
    homophilic_datasets = ['cora', 'citeseer', 'pubmed']

    all_datasets = heterophilic_datasets + homophilic_datasets

    n_runs = 5
    hidden_channels = 64

    print("="*80)
    print("H2GCN VALIDATION ON HETEROPHILIC DATASETS")
    print("="*80)
    print("\nHypothesis: H2GCN should outperform vanilla GCN on low-homophily datasets")
    print("            where GCN currently fails (Wisconsin, Cornell, Actor)")

    results = []

    for dataset_name in all_datasets:
        print(f"\n{'='*60}")
        print(f"[{dataset_name.upper()}]")

        data = load_dataset(dataset_name, data_dir)
        if data is None:
            print(f"  [!] {dataset_name} not found")
            continue

        print(f"  Nodes: {data['n_nodes']}, Edges: {data['n_edges']}")
        print(f"  h = {data['homophily']:.3f}, SPI = {data['spi']:.3f}")

        model_accs = {'MLP': [], 'GCN': [], 'H2GCN': []}

        for run in range(n_runs):
            set_seed(42 + run * 100)

            # MLP
            mlp = MLP(data['n_features'], hidden_channels, data['n_classes'])
            mlp_acc = train_and_evaluate(mlp, data)
            model_accs['MLP'].append(mlp_acc)

            # GCN
            gcn = GCN(data['n_features'], hidden_channels, data['n_classes'])
            gcn_acc = train_and_evaluate(gcn, data)
            model_accs['GCN'].append(gcn_acc)

            # H2GCN
            h2gcn = H2GCN(data['n_features'], hidden_channels, data['n_classes'])
            h2gcn_acc = train_and_evaluate(h2gcn, data)
            model_accs['H2GCN'].append(h2gcn_acc)

        # Compute statistics
        result = {
            'dataset': dataset_name,
            'homophily': data['homophily'],
            'spi': data['spi'],
            'is_heterophilic': dataset_name in heterophilic_datasets
        }

        print(f"\n  Results (mean ± std):")
        for model_name in ['MLP', 'GCN', 'H2GCN']:
            mean_acc = np.mean(model_accs[model_name])
            std_acc = np.std(model_accs[model_name])
            result[f'{model_name}_acc'] = mean_acc
            result[f'{model_name}_std'] = std_acc
            print(f"    {model_name}: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")

        # Compute advantages
        result['GCN_vs_MLP'] = result['GCN_acc'] - result['MLP_acc']
        result['H2GCN_vs_MLP'] = result['H2GCN_acc'] - result['MLP_acc']
        result['H2GCN_vs_GCN'] = result['H2GCN_acc'] - result['GCN_acc']

        # Determine winner
        best_model = max(['MLP', 'GCN', 'H2GCN'], key=lambda m: result[f'{m}_acc'])
        result['best_model'] = best_model

        print(f"\n  Best Model: {best_model}")
        print(f"  GCN vs MLP: {result['GCN_vs_MLP']*100:+.1f}%")
        print(f"  H2GCN vs MLP: {result['H2GCN_vs_MLP']*100:+.1f}%")
        print(f"  H2GCN vs GCN: {result['H2GCN_vs_GCN']*100:+.1f}%")

        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: H2GCN vs GCN on Heterophilic Datasets")
    print("="*80)

    print("\n" + "-"*80)
    print(f"{'Dataset':<15} {'h':>6} {'MLP':>8} {'GCN':>8} {'H2GCN':>8} {'H2GCN>GCN?':>12}")
    print("-"*80)

    hetero_h2gcn_wins = 0
    for r in results:
        h2gcn_better = 'YES' if r['H2GCN_vs_GCN'] > 0 else 'NO'
        if r['is_heterophilic'] and r['H2GCN_vs_GCN'] > 0:
            hetero_h2gcn_wins += 1
        print(f"{r['dataset']:<15} {r['homophily']:>6.2f} "
              f"{r['MLP_acc']*100:>7.1f}% {r['GCN_acc']*100:>7.1f}% {r['H2GCN_acc']*100:>7.1f}% "
              f"{h2gcn_better:>12}")

    print("-"*80)

    hetero_results = [r for r in results if r['is_heterophilic']]
    print(f"\nHeterophilic datasets where H2GCN > GCN: {hetero_h2gcn_wins}/{len(hetero_results)}")

    # Key question: Does H2GCN fix the SPI failures?
    print("\n" + "="*80)
    print("KEY QUESTION: Does H2GCN fix the SPI prediction failures?")
    print("="*80)

    spi_failures = ['wisconsin', 'cornell', 'actor']  # Original SPI failures

    print("\nOriginal failures (SPI predicted GNN but MLP won):")
    for name in spi_failures:
        r = next((x for x in results if x['dataset'] == name), None)
        if r:
            fixed = 'FIXED' if r['H2GCN_acc'] > r['MLP_acc'] else 'STILL FAILS'
            print(f"  {name}: MLP={r['MLP_acc']*100:.1f}%, GCN={r['GCN_acc']*100:.1f}%, "
                  f"H2GCN={r['H2GCN_acc']*100:.1f}% → {fixed}")

    # Save results
    output = {
        'experiment': 'h2gcn_validation',
        'n_runs': n_runs,
        'results': results
    }

    output_path = Path(__file__).parent / "h2gcn_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = run_h2gcn_validation()
