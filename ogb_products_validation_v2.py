"""
OGB Products Large-Scale Validation (V2)
=========================================

Validate Trust Regions Framework on ogbn-products (2.4M nodes).
Uses random node sampling for training without pyg-lib dependency.
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


class SAGE_Simple(torch.nn.Module):
    """Simple GraphSAGE using direct aggregation (no sampling required)"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
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


def sample_subgraph(data, split_idx, batch_size=10000, num_hops=2):
    """Sample a subgraph for training"""
    # Sample training nodes
    train_idx = split_idx['train'].numpy()
    sampled_idx = np.random.choice(train_idx, min(batch_size, len(train_idx)), replace=False)
    sampled_idx = torch.tensor(sampled_idx, dtype=torch.long)

    # Get k-hop neighborhood
    edge_index = data.edge_index.cpu()
    current_nodes = set(sampled_idx.tolist())

    for _ in range(num_hops):
        # Find edges involving current nodes
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        mask_src = np.isin(src, list(current_nodes))
        mask_dst = np.isin(dst, list(current_nodes))
        mask = mask_src | mask_dst

        # Add neighbors
        new_nodes = set(src[mask].tolist()) | set(dst[mask].tolist())
        current_nodes = current_nodes | new_nodes

        # Limit size to avoid memory issues
        if len(current_nodes) > 100000:
            break

    # Create node mapping
    all_nodes = sorted(list(current_nodes))
    node_mapping = {n: i for i, n in enumerate(all_nodes)}

    # Create subgraph
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    mask = np.isin(src, all_nodes) & np.isin(dst, all_nodes)
    new_src = np.array([node_mapping[s] for s in src[mask]])
    new_dst = np.array([node_mapping[d] for d in dst[mask]])

    sub_edge_index = torch.tensor(np.stack([new_src, new_dst]), dtype=torch.long)
    sub_x = data.x[all_nodes]
    sub_y = data.y[all_nodes]

    # Mark which nodes are training nodes in the subgraph
    train_mask = torch.tensor([n in sampled_idx.tolist() for n in all_nodes])

    return sub_x, sub_edge_index, sub_y, train_mask


def train_sage_subgraph(model, data, split_idx, optimizer, batch_size=10000):
    """Train SAGE using subgraph sampling"""
    model.train()

    # Sample subgraph
    sub_x, sub_edge_index, sub_y, train_mask = sample_subgraph(data, split_idx, batch_size)

    sub_x = sub_x.to(device)
    sub_edge_index = sub_edge_index.to(device)
    sub_y = sub_y.to(device)
    train_mask = train_mask.to(device)

    optimizer.zero_grad()
    out = model(sub_x, sub_edge_index)
    loss = F.cross_entropy(out[train_mask], sub_y[train_mask].squeeze())
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_batched(model, data, split_idx, evaluator, batch_size=50000):
    """Evaluate using batched inference for large graphs"""
    model.eval()

    if isinstance(model, MLP):
        # MLP: simple batched inference
        y_pred = []
        for i in range(0, data.num_nodes, batch_size):
            end = min(i + batch_size, data.num_nodes)
            x_batch = data.x[i:end].to(device)
            out = model(x_batch)
            y_pred.append(out.argmax(dim=-1).cpu())
        y_pred = torch.cat(y_pred, dim=0).unsqueeze(1)
    else:
        # SAGE: need to process in chunks with neighborhood
        # For simplicity, we'll use a subset for validation
        # This is approximate but faster
        y_pred = torch.zeros(data.num_nodes, 1, dtype=torch.long)

        # Process validation/test nodes in batches
        for split in ['valid', 'test']:
            idx = split_idx[split].numpy()
            for start in range(0, len(idx), batch_size):
                end = min(start + batch_size, len(idx))
                batch_idx = idx[start:end]

                # Get 1-hop neighborhood for these nodes
                edge_index = data.edge_index.cpu().numpy()
                src, dst = edge_index[0], edge_index[1]

                # Find edges involving batch nodes
                current_nodes = set(batch_idx.tolist())
                mask_src = np.isin(src, batch_idx)
                mask_dst = np.isin(dst, batch_idx)
                mask = mask_src | mask_dst
                neighbor_nodes = set(src[mask].tolist()) | set(dst[mask].tolist())
                all_nodes = sorted(list(current_nodes | neighbor_nodes))

                if len(all_nodes) > 200000:  # Limit subgraph size
                    # Just use MLP-style prediction for this batch
                    x_batch = data.x[batch_idx].to(device)
                    # Use first layer only
                    out = model.convs[0](x_batch, torch.zeros(2, 0, dtype=torch.long).to(device))
                    out = F.relu(out)
                    out = model.convs[-1](out, torch.zeros(2, 0, dtype=torch.long).to(device))
                    y_pred[batch_idx] = out.argmax(dim=-1).unsqueeze(1).cpu()
                else:
                    # Create proper subgraph
                    node_mapping = {n: i for i, n in enumerate(all_nodes)}
                    mask = np.isin(src, all_nodes) & np.isin(dst, all_nodes)
                    new_src = np.array([node_mapping[s] for s in src[mask]])
                    new_dst = np.array([node_mapping[d] for d in dst[mask]])

                    sub_edge_index = torch.tensor(np.stack([new_src, new_dst]), dtype=torch.long).to(device)
                    sub_x = data.x[all_nodes].to(device)

                    out = model(sub_x, sub_edge_index)

                    # Map predictions back
                    for i, n in enumerate(batch_idx):
                        mapped_idx = node_mapping[n]
                        y_pred[n] = out[mapped_idx].argmax().cpu()

    results = {}
    for split in ['train', 'valid', 'test']:
        idx = split_idx[split]
        results[split] = evaluator.eval({
            'y_true': data.y[idx],
            'y_pred': y_pred[idx],
        })['acc']

    return results


def run_products_experiment(n_runs=2, epochs=30, batches_per_epoch=50):
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
            for _ in range(batches_per_epoch):
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
                res = evaluate_batched(mlp, data, split_idx, evaluator)
                if res['valid'] > best_val:
                    best_val = res['valid']
                    best_test = res['test']
                print(f"      Epoch {epoch+1}: val={res['valid']:.4f}, test={res['test']:.4f}")

        mlp_scores.append(best_test)
        print(f"    MLP best: {best_test:.4f}")

        # GraphSAGE (subgraph sampling)
        print("    Training GraphSAGE (subgraph sampling)...")
        sage = SAGE_Simple(data.num_features, 256, dataset.num_classes, num_layers=2).to(device)
        optimizer = torch.optim.Adam(sage.parameters(), lr=0.01)

        best_val = 0
        best_test = 0
        for epoch in range(epochs):
            total_loss = 0
            for _ in range(batches_per_epoch):
                loss = train_sage_subgraph(sage, data, split_idx, optimizer, batch_size=5000)
                total_loss += loss

            if (epoch + 1) % 10 == 0:
                res = evaluate_batched(sage, data, split_idx, evaluator)
                if res['valid'] > best_val:
                    best_val = res['valid']
                    best_test = res['test']
                print(f"      Epoch {epoch+1}: val={res['valid']:.4f}, test={res['test']:.4f} (loss={total_loss/batches_per_epoch:.4f})")

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
    print("OGBN-PRODUCTS LARGE-SCALE VALIDATION (V2)")
    print("Trust Regions of Graph Propagation")
    print("Using subgraph sampling (no pyg-lib required)")
    print("=" * 80)

    result = run_products_experiment(n_runs=2, epochs=30, batches_per_epoch=50)

    # Save results
    with open('ogb_products_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: ogb_products_results.json")

    return result


if __name__ == '__main__':
    result = main()
