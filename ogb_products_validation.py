"""
OGB Products Large-Scale Validation
====================================

Validate Trust Regions Framework on ogbn-products (2.4M nodes).
Uses mini-batch training due to memory constraints.
"""

import os
import sys

# Auto-confirm OGB downloads by patching input
import builtins
_original_input = builtins.input
def _auto_confirm_input(prompt=''):
    prompt_lower = prompt.lower()
    # Handle download confirmation
    if 'download' in prompt_lower and 'proceed' in prompt_lower:
        print(prompt + "y (auto-confirmed)")
        return 'y'
    # Handle update confirmation
    if 'update' in prompt_lower and '(y/n)' in prompt_lower:
        print(prompt + "n (auto-confirmed - skip update)")
        return 'n'
    return _original_input(prompt)
builtins.input = _auto_confirm_input

import torch
import torch.nn.functional as F
import numpy as np
import json
import time

# Fix PyTorch 2.6 weights_only issue
import torch.serialization
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr])
except:
    pass
try:
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([GlobalStorage])
except:
    pass

from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


class GraphSAGE(torch.nn.Module):
    """GraphSAGE for mini-batch training"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def compute_homophily_sampled(data, sample_size=100000):
    """Compute homophily on a sample of edges (memory efficient)"""
    edge_index = data.edge_index.cpu().numpy()
    labels = data.y.cpu().numpy().squeeze()

    n_edges = edge_index.shape[1]
    if n_edges > sample_size:
        idx = np.random.choice(n_edges, sample_size, replace=False)
        src, dst = edge_index[0, idx], edge_index[1, idx]
    else:
        src, dst = edge_index[0], edge_index[1]

    valid = (labels[src] >= 0) & (labels[dst] >= 0)
    if valid.sum() == 0:
        return 0.5

    return (labels[src][valid] == labels[dst][valid]).mean()


def train_minibatch(model, loader, optimizer):
    """Train one epoch with mini-batches"""
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(out, batch.y[:batch.batch_size].squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_full(model, data, split_idx, evaluator, batch_size=4096):
    """Evaluate on full graph using batched inference"""
    model.eval()

    # For MLP, we can do batched inference easily
    if isinstance(model, MLP):
        y_pred = []
        for i in range(0, data.num_nodes, batch_size):
            end = min(i + batch_size, data.num_nodes)
            x_batch = data.x[i:end].to(device)
            out = model(x_batch)
            y_pred.append(out.argmax(dim=-1).cpu())
        y_pred = torch.cat(y_pred, dim=0).unsqueeze(1)
    else:
        # For GNN, use inference on subgraphs
        from torch_geometric.loader import NeighborLoader
        subgraph_loader = NeighborLoader(
            data,
            num_neighbors=[-1],  # All neighbors for inference
            batch_size=batch_size,
            input_nodes=None,
            shuffle=False
        )

        y_pred = torch.zeros(data.num_nodes, 1, dtype=torch.long)
        for batch in subgraph_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            y_pred[batch.n_id[:batch.batch_size]] = out.argmax(dim=-1).unsqueeze(1).cpu()

    results = {}
    for split in ['train', 'valid', 'test']:
        idx = split_idx[split]
        results[split] = evaluator.eval({
            'y_true': data.y[idx],
            'y_pred': y_pred[idx],
        })['acc']

    return results


def run_products_experiment(n_runs=2, epochs=50):
    """Run experiment on ogbn-products"""
    print("\n" + "=" * 60)
    print("Dataset: ogbn-products")
    print("=" * 60)

    start_time = time.time()

    # Load dataset
    print("Loading dataset...")
    dataset = PygNodePropPredDataset(name='ogbn-products', root='./data')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-products')

    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features: {data.num_features}")
    print(f"  Classes: {dataset.num_classes}")

    # Compute homophily (sampled)
    print("Computing homophily (sampled)...")
    h = compute_homophily_sampled(data, sample_size=500000)
    print(f"  Homophily: {h:.4f}")

    # Create mini-batch loaders
    train_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10, 5],
        batch_size=1024,
        input_nodes=split_idx['train'],
        shuffle=True
    )

    mlp_scores = []
    sage_scores = []

    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}:")

        # MLP (can be trained on full batches)
        print("    Training MLP...")
        mlp = MLP(data.num_features, 256, dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)

        best_val = 0
        best_test = 0
        for epoch in range(epochs):
            mlp.train()
            # Train on random batches of training nodes
            for _ in range(100):  # 100 mini-batches per epoch
                idx = np.random.choice(len(split_idx['train']), 1024, replace=False)
                batch_idx = split_idx['train'][idx]
                x_batch = data.x[batch_idx].to(device)
                y_batch = data.y[batch_idx].squeeze().to(device)

                optimizer.zero_grad()
                out = mlp(x_batch)
                loss = F.cross_entropy(out, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                res = evaluate_full(mlp, data, split_idx, evaluator)
                if res['valid'] > best_val:
                    best_val = res['valid']
                    best_test = res['test']
                print(f"      Epoch {epoch+1}: val={res['valid']:.4f}, test={res['test']:.4f}")

        mlp_scores.append(best_test)
        print(f"    MLP best: {best_test:.4f}")

        # GraphSAGE (mini-batch)
        print("    Training GraphSAGE...")
        sage = GraphSAGE(data.num_features, 256, dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(sage.parameters(), lr=0.01)

        best_val = 0
        best_test = 0
        for epoch in range(epochs):
            train_minibatch(sage, train_loader, optimizer)

            if (epoch + 1) % 10 == 0:
                res = evaluate_full(sage, data, split_idx, evaluator)
                if res['valid'] > best_val:
                    best_val = res['valid']
                    best_test = res['test']
                print(f"      Epoch {epoch+1}: val={res['valid']:.4f}, test={res['test']:.4f}")

        sage_scores.append(best_test)
        print(f"    SAGE best: {best_test:.4f}")

    # Summary
    mlp_mean = np.mean(mlp_scores)
    sage_mean = np.mean(sage_scores)
    delta = sage_mean - mlp_mean

    print(f"\n  Summary:")
    print(f"    MLP: {mlp_mean:.4f} +/- {np.std(mlp_scores):.4f}")
    print(f"    SAGE: {sage_mean:.4f} +/- {np.std(sage_scores):.4f}")
    print(f"    Delta: {delta:+.4f}")

    # SPI prediction
    spi = abs(2 * h - 1)
    print(f"\n  SPI Analysis:")
    print(f"    Homophily h: {h:.4f}")
    print(f"    SPI = |2h-1|: {spi:.4f}")

    if spi > 0.4:
        spi_prediction = "GNN should help"
    else:
        spi_prediction = "Uncertainty Zone"

    winner = 'GNN' if delta > 0.01 else ('MLP' if delta < -0.01 else 'Tie')
    correct = (spi > 0.4 and winner == 'GNN') or (spi <= 0.4 and winner != 'GNN')

    print(f"    SPI prediction: {spi_prediction}")
    print(f"    Actual winner: {winner}")
    print(f"    Prediction correct: {correct}")

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    result = {
        'name': 'ogbn-products',
        'n_nodes': int(data.num_nodes),
        'n_edges': int(data.num_edges),
        'homophily': float(h),
        'spi': float(spi),
        'mlp_mean': float(mlp_mean),
        'mlp_std': float(np.std(mlp_scores)),
        'sage_mean': float(sage_mean),
        'sage_std': float(np.std(sage_scores)),
        'delta': float(delta),
        'spi_prediction': spi_prediction,
        'winner': winner,
        'correct': correct,
        'time_minutes': elapsed / 60
    }

    return result


def main():
    print("=" * 80)
    print("OGBN-PRODUCTS LARGE-SCALE VALIDATION")
    print("Trust Regions of Graph Propagation")
    print("=" * 80)

    result = run_products_experiment(n_runs=2, epochs=50)

    # Save results
    with open('ogb_products_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: ogb_products_results.json")

    return result


if __name__ == '__main__':
    result = main()
