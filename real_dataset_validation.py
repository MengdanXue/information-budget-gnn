"""
Real Dataset Validation for Trust Region Framework
P0 Experiment: Validate U-shape and SPI on real-world datasets

Datasets:
- Homophilic: Cora, CiteSeer, PubMed
- Heterophilic: Texas, Wisconsin, Cornell, Actor, Chameleon, Squirrel
- Injected: Inj-Cora, Inj-Amazon
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set seeds
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ============== Models ==============

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

# ============== Data Loading ==============

def compute_homophily(edge_index, labels):
    """Compute edge homophily."""
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()

def compute_spi(h):
    """Compute Structural Predictability Index."""
    return abs(2 * h - 1)

def load_dataset(name, data_dir):
    """Load dataset from pickle file."""
    # Try different file patterns
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
        print(f"  [!] {name} not found in {data_dir}")
        return None

    with open(pkl_path, 'rb') as f:
        loaded = pickle.load(f)

    # Handle different formats
    if isinstance(loaded, dict):
        # Check if it contains a 'data' key with PyG Data object
        if 'data' in loaded and hasattr(loaded['data'], 'x'):
            data = loaded['data']
            x = data.x
            edge_index = data.edge_index
            y = data.y
            train_mask = getattr(data, 'train_mask', None)
            val_mask = getattr(data, 'val_mask', None)
            test_mask = getattr(data, 'test_mask', None)
        # Old format with 'features', 'edge_index', 'labels' keys
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
            print(f"  [!] Unknown dict format for {name}, keys: {list(loaded.keys())}")
            return None
    elif hasattr(loaded, 'x'):
        x = loaded.x
        edge_index = loaded.edge_index
        y = loaded.y
        train_mask = getattr(loaded, 'train_mask', None)
        val_mask = getattr(loaded, 'val_mask', None)
        test_mask = getattr(loaded, 'test_mask', None)
    else:
        print(f"  [!] Unknown format for {name}")
        return None

    # Convert tensors if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)

    # Handle multi-dimensional masks (multiple splits)
    if train_mask is not None and train_mask.dim() > 1:
        # Take the first split
        train_mask = train_mask[:, 0]
        val_mask = val_mask[:, 0]
        test_mask = test_mask[:, 0]

    # Create random splits if not available
    n = x.shape[0]
    if train_mask is None:
        perm = torch.randperm(n)
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[perm[:int(0.6*n)]] = True
        val_mask[perm[int(0.6*n):int(0.8*n)]] = True
        test_mask[perm[int(0.8*n):]] = True

    # Compute homophily
    h = compute_homophily(edge_index, y)
    spi = compute_spi(h)

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

def run_real_dataset_validation():
    """Run validation on real datasets."""

    data_dir = Path(__file__).parent / "data"

    # Dataset list
    datasets = [
        # Homophilic
        'cora', 'citeseer', 'pubmed',
        # Heterophilic
        'texas', 'wisconsin', 'cornell', 'actor', 'chameleon', 'squirrel',
        # Fraud detection
        'elliptic_weber_split', 'inj_amazon', 'inj_cora',
        # Synthetic controls
        'csbm_high_homo', 'csbm_mid_homo', 'csbm_low_homo',
        'csbm_noisy_feat', 'csbm_clean_feat'
    ]

    model_classes = {
        'MLP': MLP,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE
    }

    n_runs = 5
    hidden_channels = 64

    print("="*70)
    print("Real Dataset Validation for Trust Region Framework")
    print("="*70)

    results = []

    for dataset_name in datasets:
        print(f"\n[{dataset_name.upper()}]")

        # Load data
        data = load_dataset(dataset_name, data_dir)
        if data is None:
            continue

        print(f"  Nodes: {data['n_nodes']}, Edges: {data['n_edges']}, "
              f"Classes: {data['n_classes']}, Features: {data['n_features']}")
        print(f"  Homophily h = {data['homophily']:.3f}, SPI = {data['spi']:.3f}")

        # Run experiments
        model_accs = {name: [] for name in model_classes}

        for run in range(n_runs):
            set_seed(42 + run * 100)

            for model_name, model_class in model_classes.items():
                if model_name == 'GAT':
                    model = model_class(data['n_features'], hidden_channels,
                                       data['n_classes'], heads=4)
                else:
                    model = model_class(data['n_features'], hidden_channels,
                                       data['n_classes'])

                acc = train_and_evaluate(model, data)
                model_accs[model_name].append(acc)

        # Compute statistics
        result = {
            'dataset': dataset_name,
            'n_nodes': data['n_nodes'],
            'n_edges': data['n_edges'],
            'n_classes': data['n_classes'],
            'homophily': data['homophily'],
            'spi': data['spi']
        }

        print(f"  Results (mean +/- std):")
        for model_name in model_classes:
            mean_acc = np.mean(model_accs[model_name])
            std_acc = np.std(model_accs[model_name])
            result[f'{model_name}_acc'] = mean_acc
            result[f'{model_name}_std'] = std_acc
            print(f"    {model_name}: {mean_acc*100:.1f}% +/- {std_acc*100:.1f}%")

        # Compute GNN advantages
        mlp_mean = result['MLP_acc']
        for model_name in ['GCN', 'GAT', 'GraphSAGE']:
            advantage = result[f'{model_name}_acc'] - mlp_mean
            result[f'{model_name}_advantage'] = advantage

        # Determine winner
        best_model = max(['MLP', 'GCN', 'GAT', 'GraphSAGE'],
                        key=lambda m: result[f'{m}_acc'])
        result['best_model'] = best_model

        # SPI prediction
        if data['spi'] > 0.4:
            spi_pred = 'GNN'
        else:
            spi_pred = 'MLP'

        actual_winner = 'GNN' if best_model != 'MLP' else 'MLP'
        result['spi_prediction'] = spi_pred
        result['prediction_correct'] = (spi_pred == actual_winner)

        print(f"  Best: {best_model}, SPI predicts: {spi_pred}, "
              f"Correct: {result['prediction_correct']}")

        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: SPI Prediction Accuracy on Real Datasets")
    print("="*70)

    correct = sum(1 for r in results if r['prediction_correct'])
    total = len(results)
    print(f"\nOverall: {correct}/{total} ({100*correct/total:.1f}%)")

    # By homophily zone
    high_h = [r for r in results if r['homophily'] > 0.7]
    low_h = [r for r in results if r['homophily'] < 0.3]
    mid_h = [r for r in results if 0.3 <= r['homophily'] <= 0.7]

    if high_h:
        high_correct = sum(1 for r in high_h if r['prediction_correct'])
        print(f"High h (>0.7): {high_correct}/{len(high_h)}")

    if mid_h:
        mid_correct = sum(1 for r in mid_h if r['prediction_correct'])
        print(f"Mid h (0.3-0.7): {mid_correct}/{len(mid_h)}")

    if low_h:
        low_correct = sum(1 for r in low_h if r['prediction_correct'])
        print(f"Low h (<0.3): {low_correct}/{len(low_h)}")

    # Save results
    output = {
        'experiment': 'real_dataset_validation',
        'n_runs': n_runs,
        'results': results
    }

    output_path = Path(__file__).parent / "real_dataset_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results

if __name__ == "__main__":
    results = run_real_dataset_validation()
