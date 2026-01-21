"""
OGB Heterophily-Aware Methods Experiments
Runs H2GCN, LINKX, GPR-GNN on ogbn-arxiv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Fix PyTorch 2.6+ weights_only issue
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
print("Patched torch.load for PyTorch 2.6+ compatibility")

try:
    from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
    OGB_AVAILABLE = True
except ImportError:
    print("OGB not installed. Run: pip install ogb")
    OGB_AVAILABLE = False

try:
    from torch_geometric.nn import GCNConv, Linear
    from torch_geometric.utils import add_self_loops, degree
    PYG_AVAILABLE = True
except ImportError:
    print("PyTorch Geometric not installed.")
    PYG_AVAILABLE = False


# ============================================================
# LINKX Model (Luan et al., 2022)
# Separates node features and graph structure
# ============================================================

class LINKX(nn.Module):
    """
    LINKX: Beyond Homophily in Graph Neural Networks
    Separates MLP on features and MLP on adjacency aggregation
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # MLP for node features
        self.mlp_x = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # MLP for adjacency features (simplified: use degree)
        self.mlp_a = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Final MLP
        self.mlp_final = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        self.cached_ax = None

    def compute_ax(self, x, edge_index, num_nodes):
        """Compute A @ X (adjacency aggregation)"""
        if self.cached_ax is not None:
            return self.cached_ax

        row, col = edge_index
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * deg_inv[col]

        # Sparse matrix multiplication
        ax = torch.zeros_like(x)
        for i in range(edge_index.size(1)):
            ax[col[i]] += norm[i] * x[row[i]]

        return ax

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        # Feature branch
        h_x = self.mlp_x(x)

        # Adjacency branch (A @ X)
        ax = self.compute_ax(x, edge_index, num_nodes)
        h_a = self.mlp_a(ax)

        # Combine
        h = torch.cat([h_x, h_a], dim=-1)
        out = self.mlp_final(h)

        return out


# ============================================================
# GPR-GNN Model (Chien et al., 2021)
# Generalized PageRank with learnable weights
# ============================================================

class GPRGNN(nn.Module):
    """
    GPR-GNN: Generalized PageRank Graph Neural Network
    Learns optimal combination of different propagation steps
    """
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.dropout = dropout

        # Feature transformation
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        # Learnable GPR weights (gamma_k)
        self.gamma = nn.Parameter(torch.ones(K + 1) / (K + 1))

    def forward(self, x, edge_index):
        # Transform features
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.lin2(x)

        # Compute propagation matrix (normalized adjacency)
        num_nodes = h.size(0)
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=h.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # GPR propagation: sum_k gamma_k * A^k * h
        out = self.gamma[0] * h
        h_k = h

        for k in range(1, self.K + 1):
            # Propagate: h_k = A @ h_{k-1}
            h_new = torch.zeros_like(h_k)
            for i in range(edge_index.size(1)):
                norm = deg_inv_sqrt[row[i]] * deg_inv_sqrt[col[i]]
                h_new[col[i]] += norm * h_k[row[i]]
            h_k = h_new
            out = out + self.gamma[k] * h_k

        return out


# ============================================================
# H2GCN Model (Zhu et al., 2020)
# Uses 2-hop neighbors and ego-neighbor separation
# ============================================================

class H2GCN(nn.Module):
    """
    H2GCN: Beyond Homophily in Graph Neural Networks (simplified)
    Key ideas: ego-neighbor separation, higher-order neighborhoods
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # Ego embedding
        self.lin_ego = nn.Linear(in_channels, hidden_channels)

        # 1-hop neighbor embedding
        self.lin_n1 = nn.Linear(in_channels, hidden_channels)

        # 2-hop neighbor embedding
        self.lin_n2 = nn.Linear(in_channels, hidden_channels)

        # Combine: ego + n1 + n2
        self.lin_combine = nn.Linear(hidden_channels * 3, hidden_channels)

        # Output
        self.lin_out = nn.Linear(hidden_channels, out_channels)

        self.cached_n1 = None
        self.cached_n2 = None

    def aggregate(self, x, edge_index, num_nodes):
        """Simple mean aggregation"""
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv = 1.0 / deg.clamp(min=1)

        out = torch.zeros_like(x)
        for i in range(edge_index.size(1)):
            out[col[i]] += x[row[i]]

        out = out * deg_inv.unsqueeze(-1)
        return out

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        # Ego features
        h_ego = self.lin_ego(x)

        # 1-hop aggregation
        n1 = self.aggregate(x, edge_index, num_nodes)
        h_n1 = self.lin_n1(n1)

        # 2-hop aggregation (aggregate again)
        n2 = self.aggregate(n1, edge_index, num_nodes)
        h_n2 = self.lin_n2(n2)

        # Combine with ego-neighbor separation
        h = torch.cat([h_ego, h_n1, h_n2], dim=-1)
        h = F.relu(self.lin_combine(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.lin_out(h)

        return out


# ============================================================
# Training Functions
# ============================================================

def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)[train_idx]
    loss = F.cross_entropy(out, data.y[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, split_idx, evaluator):
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
    """Compute edge homophily"""
    edge_index = data.edge_index.cpu().numpy()
    y = data.y.cpu().numpy().flatten()

    same_label = 0
    total = edge_index.shape[1]

    for i in range(total):
        src, dst = edge_index[0, i], edge_index[1, i]
        if y[src] == y[dst]:
            same_label += 1

    return same_label / total


# ============================================================
# Main Experiment
# ============================================================

def run_heterophily_experiment(dataset_name='ogbn-arxiv', n_runs=3, epochs=500, patience=50):
    """Run heterophily-aware methods on OGB dataset"""

    print(f"\n{'='*60}")
    print(f"Heterophily Methods on {dataset_name}")
    print(f"{'='*60}")

    # Load dataset
    dataset = PygNodePropPredDataset(name=dataset_name, root='data/')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name=dataset_name)

    data = data.to(device)
    train_idx = split_idx['train'].to(device)

    n_nodes = data.num_nodes
    n_features = data.x.size(1)
    n_classes = dataset.num_classes

    print(f"Nodes: {n_nodes:,}, Features: {n_features}, Classes: {n_classes}")

    homophily = compute_homophily(data)
    print(f"Homophily: {homophily:.4f}")

    results = {
        'dataset': dataset_name,
        'n_nodes': n_nodes,
        'n_features': n_features,
        'n_classes': n_classes,
        'homophily': homophily,
        'models': {}
    }

    # Model configs
    hidden = 256
    dropout = 0.5
    lr = 0.01

    models_config = {
        'LINKX': lambda: LINKX(n_features, hidden, n_classes, dropout=dropout),
        'GPR-GNN': lambda: GPRGNN(n_features, hidden, n_classes, K=10, dropout=dropout),
        'H2GCN': lambda: H2GCN(n_features, hidden, n_classes, dropout=dropout),
    }

    for model_name, model_fn in models_config.items():
        print(f"\n--- Training {model_name} ---")

        all_test_accs = []
        all_val_accs = []
        all_times = []

        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")

            model = model_fn().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

            best_val = 0
            best_test = 0
            patience_cnt = 0
            start_time = time.time()

            for epoch in range(epochs):
                try:
                    loss = train(model, data, train_idx, optimizer)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"  OOM at epoch {epoch}, stopping")
                        torch.cuda.empty_cache()
                        break
                    raise e

                if (epoch + 1) % 10 == 0:
                    eval_res = evaluate(model, data, split_idx, evaluator)
                    val_acc = eval_res['valid']
                    test_acc = eval_res['test']

                    if val_acc > best_val:
                        best_val = val_acc
                        best_test = test_acc
                        patience_cnt = 0
                    else:
                        patience_cnt += 1

                    if patience_cnt >= patience // 10:
                        print(f"  Early stop at epoch {epoch + 1}")
                        break

            train_time = time.time() - start_time
            all_test_accs.append(best_test)
            all_val_accs.append(best_val)
            all_times.append(train_time)

            print(f"  Val: {best_val:.4f}, Test: {best_test:.4f}, Time: {train_time:.1f}s")

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results['models'][model_name] = {
            'test_acc_mean': float(np.mean(all_test_accs)),
            'test_acc_std': float(np.std(all_test_accs)),
            'val_acc_mean': float(np.mean(all_val_accs)),
            'val_acc_std': float(np.std(all_val_accs)),
            'train_time_mean': float(np.mean(all_times)),
            'all_test_accs': all_test_accs,
        }

        print(f"{model_name}: {np.mean(all_test_accs):.4f} ± {np.std(all_test_accs):.4f}")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    if not OGB_AVAILABLE or not PYG_AVAILABLE:
        print("Required packages not available.")
        exit(1)

    all_results = {
        'experiment': 'OGB Heterophily-Aware Methods',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'datasets': {}
    }

    # Run on ogbn-arxiv (smaller, faster)
    try:
        results = run_heterophily_experiment('ogbn-arxiv', n_runs=3, epochs=300)
        all_results['datasets']['ogbn-arxiv'] = results
    except Exception as e:
        print(f"Error: {e}")
        all_results['datasets']['ogbn-arxiv'] = {'error': str(e)}

    # Save results
    output_file = 'ogb_heterophily_methods_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Summary
    print("\n=== SUMMARY ===")
    for ds, res in all_results['datasets'].items():
        if 'error' in res:
            print(f"{ds}: ERROR")
            continue
        print(f"\n{ds} (h={res['homophily']:.3f}):")
        for m, r in res['models'].items():
            print(f"  {m}: {r['test_acc_mean']:.4f} ± {r['test_acc_std']:.4f}")
