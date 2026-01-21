"""
GNN Training Script for IEEE-CIS Dataset

Run 10-seed experiments to validate FSD prediction.

Usage:
    python train_ieee_cis.py --method GCN --seeds 42 123 456 789 1024 2048 3072 4096 5120 6144
    python train_ieee_cis.py --method GAT --seeds 42 123 456 789 1024 2048 3072 4096 5120 6144
    python train_ieee_cis.py --method NAA-GCN --seeds 42 123 456 789 1024 2048 3072 4096 5120 6144

Requirements:
    pip install torch torch_geometric scikit-learn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
import numpy as np
import pickle
import json
import argparse
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import os

# Import DAAA models
from daaa_model import DAAA, DAAAv2, DAAAv3, DAAAv4

# Seeds for reproducibility
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE model."""
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class NAAGCN(nn.Module):
    """NAA-enhanced GCN with numerical similarity and feature importance."""
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5, beta=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout
        self.beta = nn.Parameter(torch.tensor(beta))
        self.feature_importance = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, edge_index):
        # Apply learned feature importance
        x_weighted = x * torch.sigmoid(self.feature_importance)

        x = self.conv1(x_weighted, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class NAAGAT(nn.Module):
    """NAA-enhanced GAT with numerical similarity and feature importance."""
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, heads=4, dropout=0.5, beta=0.1):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        self.beta = nn.Parameter(torch.tensor(beta))
        self.feature_importance = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, edge_index):
        # Apply learned feature importance
        x_weighted = x * torch.sigmoid(self.feature_importance)

        x = self.conv1(x_weighted, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class H2GCN(nn.Module):
    """
    H2GCN: Heterophily-aware GCN with ego-neighbor separation.
    Memory-efficient sparse implementation for large graphs.
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5, num_hops=2):
        super().__init__()
        self.num_hops = num_hops
        self.dropout = dropout

        # Feature transformation for ego
        self.ego_transform = nn.Linear(in_dim, hidden_dim)

        # Feature transformations for each hop
        self.hop_transforms = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim) for _ in range(num_hops)
        ])

        # Final classifier (ego + all hops concatenated)
        self.classifier = nn.Linear(hidden_dim * (1 + num_hops), out_dim)

    def forward(self, x, edge_index):
        from torch_geometric.utils import to_torch_sparse_tensor, degree

        n = x.size(0)

        # Build sparse adjacency
        adj = to_torch_sparse_tensor(edge_index, size=(n, n))

        # Compute degree for normalization
        deg = degree(edge_index[0], n).clamp(min=1)
        deg_inv = 1.0 / deg

        # Ego representation
        h_ego = F.relu(self.ego_transform(x))
        h_ego = F.dropout(h_ego, p=self.dropout, training=self.training)

        # Multi-hop neighbor representations
        hop_features = [h_ego]

        # 1-hop aggregation
        h_1hop = torch.sparse.mm(adj, x)
        h_1hop = h_1hop * deg_inv.unsqueeze(1)  # Normalize
        h_1hop = F.relu(self.hop_transforms[0](h_1hop))
        h_1hop = F.dropout(h_1hop, p=self.dropout, training=self.training)
        hop_features.append(h_1hop)

        if self.num_hops >= 2:
            # 2-hop aggregation (A^2 * x)
            h_2hop = torch.sparse.mm(adj, torch.sparse.mm(adj, x))
            h_2hop = h_2hop * (deg_inv ** 2).unsqueeze(1)
            h_2hop = F.relu(self.hop_transforms[1](h_2hop))
            h_2hop = F.dropout(h_2hop, p=self.dropout, training=self.training)
            hop_features.append(h_2hop)

        # Concatenate all representations
        h_combined = torch.cat(hop_features, dim=1)

        # Final classification
        out = self.classifier(h_combined)
        return out


class FAGCN(nn.Module):
    """
    FAGCN: Frequency Adaptive Graph Convolutional Networks.
    Learns to adaptively combine low-frequency and high-frequency signals.
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5, eps=0.3, num_layers=2):
        super().__init__()
        self.eps = eps
        self.dropout = dropout
        self.num_layers = num_layers

        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, out_dim)

        # Learnable attention for low/high frequency
        self.gates = nn.ModuleList([
            nn.Linear(hidden_dim * 2, 1) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index):
        from torch_geometric.utils import to_torch_sparse_tensor, degree, add_self_loops

        n = x.size(0)

        # Add self-loops and build adjacency
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=n)
        adj = to_torch_sparse_tensor(edge_index_sl, size=(n, n))

        # Degree normalization
        deg = degree(edge_index_sl[0], n).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)

        # Initial transformation
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.lin_in(h)
        h = F.relu(h)
        h_init = h

        for layer in range(self.num_layers):
            # Low-pass (smoothing)
            h_low = torch.sparse.mm(adj, h * deg_inv_sqrt.unsqueeze(1))
            h_low = h_low * deg_inv_sqrt.unsqueeze(1)

            # High-pass (sharpening) = h - h_low
            h_high = h - h_low

            # Adaptive gate
            gate_input = torch.cat([h_low, h_high], dim=1)
            gate = torch.sigmoid(self.gates[layer](gate_input))

            # Combine
            h = gate * h_low + (1 - gate) * h_high
            h = h + self.eps * h_init  # Skip connection
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.lin_out(h)
        return out


class GPRGNN(nn.Module):
    """
    GPR-GNN: Generalized PageRank GNN.
    Learns optimal weights for different propagation steps.
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5, K=10, alpha=0.1):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.dropout = dropout

        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, out_dim)

        # Learnable GPR weights (initialized with PPR-like weights)
        init_weights = torch.tensor([alpha * (1 - alpha) ** k for k in range(K + 1)])
        init_weights = init_weights / init_weights.sum()
        self.gpr_weights = nn.Parameter(init_weights)

    def forward(self, x, edge_index):
        from torch_geometric.utils import to_torch_sparse_tensor, degree, add_self_loops

        n = x.size(0)

        # Build normalized adjacency
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=n)
        adj = to_torch_sparse_tensor(edge_index_sl, size=(n, n))

        deg = degree(edge_index_sl[0], n).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)

        # Initial transformation
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.lin_in(h))

        # GPR propagation
        h_list = [h]
        h_prop = h

        for _ in range(self.K):
            h_prop = torch.sparse.mm(adj, h_prop * deg_inv_sqrt.unsqueeze(1))
            h_prop = h_prop * deg_inv_sqrt.unsqueeze(1)
            h_list.append(h_prop)

        # Weighted combination
        h_stack = torch.stack(h_list, dim=0)  # (K+1, N, H)
        weights = F.softmax(self.gpr_weights, dim=0).view(-1, 1, 1)
        h_final = (weights * h_stack).sum(dim=0)

        h_final = F.dropout(h_final, p=self.dropout, training=self.training)
        out = self.lin_out(h_final)
        return out


class MixHop(nn.Module):
    """
    MixHop: Higher-Order Graph Convolutional Architectures.
    Memory-efficient sparse implementation.
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # Transforms for hop 0, 1, 2
        hidden_per_hop = hidden_dim // 3
        self.hop0_transform = nn.Linear(in_dim, hidden_per_hop)
        self.hop1_transform = nn.Linear(in_dim, hidden_per_hop)
        self.hop2_transform = nn.Linear(in_dim, hidden_dim - 2 * hidden_per_hop)

        # Second layer
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        from torch_geometric.utils import to_torch_sparse_tensor, degree

        n = x.size(0)

        # Build sparse adjacency
        adj = to_torch_sparse_tensor(edge_index, size=(n, n))

        # Degree normalization
        deg = degree(edge_index[0], n).clamp(min=1)
        deg_inv = 1.0 / deg

        # Hop 0: just ego features
        h0 = self.hop0_transform(x)

        # Hop 1: A * x
        h1 = torch.sparse.mm(adj, x) * deg_inv.unsqueeze(1)
        h1 = self.hop1_transform(h1)

        # Hop 2: A^2 * x
        h2 = torch.sparse.mm(adj, torch.sparse.mm(adj, x)) * (deg_inv ** 2).unsqueeze(1)
        h2 = self.hop2_transform(h2)

        # Concatenate
        h = torch.cat([h0, h1, h2], dim=1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.fc(h)
        return out


def load_data(data_path):
    """Load processed IEEE-CIS data."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Convert to PyG Data object
    x = torch.tensor(data['features'], dtype=torch.float32)
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long)
    y = torch.tensor(data['labels'], dtype=torch.long)

    train_mask = torch.tensor(data['train_mask'], dtype=torch.bool)
    val_mask = torch.tensor(data['val_mask'], dtype=torch.bool)
    test_mask = torch.tensor(data['test_mask'], dtype=torch.bool)

    pyg_data = Data(x=x, edge_index=edge_index, y=y)
    pyg_data.train_mask = train_mask
    pyg_data.val_mask = val_mask
    pyg_data.test_mask = test_mask

    return pyg_data


def train_epoch(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()

    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, device):
    model.eval()

    out = model(data.x.to(device), data.edge_index.to(device))
    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    preds = out.argmax(dim=1).cpu().numpy()
    labels = data.y.numpy()

    mask_np = mask.numpy()

    auc = roc_auc_score(labels[mask_np], probs[mask_np])
    f1 = f1_score(labels[mask_np], preds[mask_np])
    prec = precision_score(labels[mask_np], preds[mask_np], zero_division=0)
    rec = recall_score(labels[mask_np], preds[mask_np], zero_division=0)

    return {'auc': auc, 'f1': f1, 'precision': prec, 'recall': rec}


def run_experiment(method, seed, data_path, device='cuda'):
    """Run single experiment with given seed."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load data
    data = load_data(data_path)
    in_dim = data.x.shape[1]

    # Create model
    if method == 'GCN':
        model = GCN(in_dim)
    elif method == 'GAT':
        model = GAT(in_dim)
    elif method == 'GraphSAGE':
        model = GraphSAGE(in_dim)
    elif method == 'NAA-GCN':
        model = NAAGCN(in_dim)
    elif method == 'NAA-GAT':
        model = NAAGAT(in_dim)
    elif method == 'H2GCN':
        model = H2GCN(in_dim)
    elif method == 'MixHop':
        model = MixHop(in_dim)
    elif method == 'FAGCN':
        model = FAGCN(in_dim)
    elif method == 'GPRGNN':
        model = GPRGNN(in_dim)
    elif method == 'DAAA':
        model = DAAA(in_dim, use_learned_gate=True, use_feature_importance=True)
    elif method == 'DAAAv2':
        model = DAAAv2(in_dim, num_hops=2, use_attention=True)
    elif method == 'DAAAv3':
        model = DAAAv3(in_dim, num_hops=2, dilution_threshold=0.5)
    elif method == 'DAAAv4':
        model = DAAAv4(in_dim, num_hops=2)
    else:
        raise ValueError(f"Unknown method: {method}")

    model = model.to(device)

    # Class weights for imbalanced data
    n_pos = data.y[data.train_mask].sum().item()
    n_neg = data.train_mask.sum().item() - n_pos
    weight = torch.tensor([1.0, n_neg / n_pos], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # Training loop
    best_val_auc = 0
    best_test_metrics = None
    patience = 20
    patience_counter = 0

    for epoch in range(200):
        loss = train_epoch(model, data, optimizer, criterion, device)
        val_metrics = evaluate(model, data, data.val_mask, device)

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_test_metrics = evaluate(model, data, data.test_mask, device)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                       choices=['GCN', 'GAT', 'GraphSAGE', 'NAA-GCN', 'NAA-GAT', 'H2GCN', 'MixHop', 'FAGCN', 'GPRGNN', 'DAAA', 'DAAAv2', 'DAAAv3', 'DAAAv4'])
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--data_path', type=str,
                       default='./processed/ieee_cis_graph.pkl')
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    results = {'auc': [], 'f1': [], 'precision': [], 'recall': []}

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        metrics = run_experiment(args.method, seed, args.data_path, device)
        print(f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

        results['auc'].append(metrics['auc'])
        results['f1'].append(metrics['f1'])
        results['precision'].append(metrics['precision'])
        results['recall'].append(metrics['recall'])

    # Summary
    print(f"\n{'='*50}")
    print(f"Method: {args.method}")
    print(f"Seeds: {len(args.seeds)}")
    print(f"AUC: {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")
    print(f"F1:  {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")

    # Save results
    output_file = os.path.join(args.output_dir, f'ieee_cis_{args.method}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
