"""
OGB Full Experiments for Information Budget Paper
Runs MLP, GCN, GraphSAGE, GAT on ogbn-arxiv and ogbn-products
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from datetime import datetime

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Fix PyTorch 2.6+ weights_only issue by monkey-patching torch.load
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
print("Patched torch.load for PyTorch 2.6+ compatibility")

# Try to import OGB
try:
    from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
    OGB_AVAILABLE = True
except ImportError:
    print("OGB not installed. Run: pip install ogb")
    OGB_AVAILABLE = False

# Try to import PyG
try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import NeighborSampler
    from torch_geometric.loader import NeighborLoader
    PYG_AVAILABLE = True
except ImportError:
    print("PyTorch Geometric not installed.")
    PYG_AVAILABLE = False


# ============================================================
# Model Definitions
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x):
        for i, (lin, bn) in enumerate(zip(self.lins[:-1], self.bns)):
            x = lin(x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=8, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ============================================================
# Training Functions
# ============================================================

def train_mlp(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x[train_idx])
    loss = F.cross_entropy(out, data.y[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()


def train_gnn(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)[train_idx]
    loss = F.cross_entropy(out, data.y[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_mlp(model, data, split_idx, evaluator):
    model.eval()
    out = model(data.x)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = {}
    for split in ['train', 'valid', 'test']:
        idx = split_idx[split]
        results[split] = evaluator.eval({
            'y_true': data.y[idx],
            'y_pred': y_pred[idx],
        })['acc']
    return results


@torch.no_grad()
def evaluate_gnn(model, data, split_idx, evaluator):
    model.eval()
    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = {}
    for split in ['train', 'valid', 'test']:
        idx = split_idx[split]
        results[split] = evaluator.eval({
            'y_true': data.y[idx],
            'y_pred': y_pred[idx],
        })['acc']
    return results


def compute_homophily(data):
    """Compute edge homophily - optimized for large graphs"""
    edge_index = data.edge_index
    y = data.y.squeeze()

    # Use GPU if available for faster computation
    if edge_index.is_cuda:
        src_labels = y[edge_index[0]]
        dst_labels = y[edge_index[1]]
        same_label = (src_labels == dst_labels).sum().item()
        total = edge_index.size(1)
    else:
        # CPU version with numpy for speed
        edge_index_np = edge_index.cpu().numpy()
        y_np = y.cpu().numpy()
        same_label = np.sum(y_np[edge_index_np[0]] == y_np[edge_index_np[1]])
        total = edge_index_np.shape[1]

    return same_label / total


# ============================================================
# Main Experiment Function
# ============================================================

def run_ogb_experiment(dataset_name, n_runs=3, epochs=500, patience=50):
    """Run full experiment on an OGB dataset"""

    print(f"\n{'='*60}")
    print(f"Running experiments on {dataset_name}")
    print(f"{'='*60}")

    # Load dataset
    dataset = PygNodePropPredDataset(name=dataset_name, root='data/')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name=dataset_name)

    # Move to device
    data = data.to(device)
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)

    # Dataset statistics
    n_nodes = data.num_nodes
    n_edges = data.edge_index.size(1)
    n_features = data.x.size(1)
    n_classes = dataset.num_classes

    print(f"Nodes: {n_nodes:,}, Edges: {n_edges:,}, Features: {n_features}, Classes: {n_classes}")

    # Compute homophily
    print("Computing homophily...")
    homophily = compute_homophily(data)
    print(f"Homophily: {homophily:.4f}")

    results = {
        'dataset': dataset_name,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'n_features': n_features,
        'n_classes': n_classes,
        'homophily': homophily,
        'models': {}
    }

    # Model configurations
    hidden_channels = 256
    num_layers = 3
    dropout = 0.5
    lr = 0.01

    models_to_run = {
        'MLP': lambda: MLP(n_features, hidden_channels, n_classes, num_layers, dropout),
        'GCN': lambda: GCN(n_features, hidden_channels, n_classes, num_layers, dropout),
        'GraphSAGE': lambda: GraphSAGE(n_features, hidden_channels, n_classes, num_layers, dropout),
    }

    # Only add GAT for smaller datasets (memory constraint)
    if n_nodes < 500000:
        models_to_run['GAT'] = lambda: GAT(n_features, hidden_channels, n_classes, num_layers, heads=8, dropout=dropout)

    for model_name, model_fn in models_to_run.items():
        print(f"\n--- Training {model_name} ---")

        all_test_accs = []
        all_val_accs = []
        all_train_times = []

        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")

            # Initialize model
            model = model_fn().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Training
            best_val_acc = 0
            best_test_acc = 0
            patience_counter = 0
            start_time = time.time()

            is_mlp = model_name == 'MLP'
            train_fn = train_mlp if is_mlp else train_gnn
            eval_fn = evaluate_mlp if is_mlp else evaluate_gnn

            for epoch in range(epochs):
                loss = train_fn(model, data, train_idx, optimizer)

                if (epoch + 1) % 10 == 0:
                    eval_results = eval_fn(model, data, split_idx, evaluator)
                    val_acc = eval_results['valid']
                    test_acc = eval_results['test']

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_test_acc = test_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience // 10:
                        print(f"  Early stopping at epoch {epoch + 1}")
                        break

            train_time = time.time() - start_time
            all_test_accs.append(best_test_acc)
            all_val_accs.append(best_val_acc)
            all_train_times.append(train_time)

            print(f"  Val: {best_val_acc:.4f}, Test: {best_test_acc:.4f}, Time: {train_time:.1f}s")

        results['models'][model_name] = {
            'test_acc_mean': float(np.mean(all_test_accs)),
            'test_acc_std': float(np.std(all_test_accs)),
            'val_acc_mean': float(np.mean(all_val_accs)),
            'val_acc_std': float(np.std(all_val_accs)),
            'train_time_mean': float(np.mean(all_train_times)),
            'all_test_accs': all_test_accs,
            'all_val_accs': all_val_accs,
        }

        print(f"{model_name} Final: Test {np.mean(all_test_accs):.4f} ± {np.std(all_test_accs):.4f}")

    # Compute Information Budget analysis
    if 'MLP' in results['models']:
        mlp_acc = results['models']['MLP']['test_acc_mean']
        budget = 1 - mlp_acc

        best_gnn_acc = 0
        best_gnn_name = None
        for name, res in results['models'].items():
            if name != 'MLP' and res['test_acc_mean'] > best_gnn_acc:
                best_gnn_acc = res['test_acc_mean']
                best_gnn_name = name

        gnn_advantage = best_gnn_acc - mlp_acc if best_gnn_name else 0

        results['information_budget_analysis'] = {
            'mlp_accuracy': mlp_acc,
            'information_budget': budget,
            'best_gnn': best_gnn_name,
            'best_gnn_accuracy': best_gnn_acc,
            'gnn_advantage': gnn_advantage,
            'within_budget': gnn_advantage <= budget,
            'budget_utilization': gnn_advantage / budget if budget > 0 else 0,
        }

    return results


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    if not OGB_AVAILABLE or not PYG_AVAILABLE:
        print("Required packages not available. Please install:")
        print("  pip install ogb torch-geometric")
        exit(1)

    all_results = {
        'experiment': 'OGB Full Experiments for Information Budget Paper',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'datasets': {}
    }

    # Run experiments
    datasets = ['ogbn-arxiv', 'ogbn-products']

    for dataset_name in datasets:
        try:
            results = run_ogb_experiment(dataset_name, n_runs=3, epochs=500, patience=50)
            all_results['datasets'][dataset_name] = results
        except Exception as e:
            print(f"Error running {dataset_name}: {e}")
            all_results['datasets'][dataset_name] = {'error': str(e)}

    # Save results
    output_file = 'ogb_full_experiments_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")

    # Print summary
    print("\n=== SUMMARY ===")
    for dataset_name, results in all_results['datasets'].items():
        if 'error' in results:
            print(f"\n{dataset_name}: ERROR - {results['error']}")
            continue

        print(f"\n{dataset_name}:")
        print(f"  Homophily: {results['homophily']:.4f}")
        for model_name, model_results in results['models'].items():
            print(f"  {model_name}: {model_results['test_acc_mean']:.4f} ± {model_results['test_acc_std']:.4f}")

        if 'information_budget_analysis' in results:
            iba = results['information_budget_analysis']
            print(f"  Information Budget: {iba['information_budget']:.4f}")
            print(f"  GNN Advantage: {iba['gnn_advantage']:.4f}")
            print(f"  Within Budget: {iba['within_budget']}")
