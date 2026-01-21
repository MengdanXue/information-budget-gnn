"""
Medium Dilution Subgraph Experiment
创建中等δ_agg (5-10) 的子图，测试DAAA是否在该区间有优势

实验设计：
1. 从IEEE-CIS中采样中等度数节点
2. 计算子图的δ_agg
3. 比较DAAA vs H2GCN vs GCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree, subgraph
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.metrics import roc_auc_score, f1_score
import json
import os

# Import models
from daaa_model import DAAA


class H2GCN(nn.Module):
    """
    H2GCN: Heterophily-aware GCN with ego-neighbor separation.
    """
    def __init__(self, in_dim, hidden_dim=64, out_dim=2, dropout=0.5, num_hops=2):
        super().__init__()
        self.num_hops = num_hops
        self.dropout = dropout

        self.ego_transform = nn.Linear(in_dim, hidden_dim)
        self.hop_transforms = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim) for _ in range(num_hops)
        ])
        self.classifier = nn.Linear(hidden_dim * (1 + num_hops), out_dim)

    def forward(self, x, edge_index):
        from torch_geometric.utils import to_torch_sparse_tensor, degree

        n = x.size(0)
        adj = to_torch_sparse_tensor(edge_index, size=(n, n))
        deg = degree(edge_index[0], n).clamp(min=1)
        deg_inv = 1.0 / deg

        h_ego = F.relu(self.ego_transform(x))
        h_ego = F.dropout(h_ego, p=self.dropout, training=self.training)
        hop_features = [h_ego]

        h_1hop = torch.sparse.mm(adj, x)
        h_1hop = h_1hop * deg_inv.unsqueeze(1)
        h_1hop = F.relu(self.hop_transforms[0](h_1hop))
        h_1hop = F.dropout(h_1hop, p=self.dropout, training=self.training)
        hop_features.append(h_1hop)

        if self.num_hops >= 2:
            h_2hop = torch.sparse.mm(adj, torch.sparse.mm(adj, x))
            h_2hop = h_2hop * (deg_inv ** 2).unsqueeze(1)
            h_2hop = F.relu(self.hop_transforms[1](h_2hop))
            h_2hop = F.dropout(h_2hop, p=self.dropout, training=self.training)
            hop_features.append(h_2hop)

        h_combined = torch.cat(hop_features, dim=1)
        return self.classifier(h_combined)


def load_ieee_cis_data():
    """Load IEEE-CIS dataset"""
    import pickle

    data_path = './processed/ieee_cis_graph.pkl'
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        # Create a simple object to hold the data
        class Data:
            pass

        data = Data()
        data.x = torch.tensor(data_dict['features'], dtype=torch.float32)
        data.y = torch.tensor(data_dict['labels'], dtype=torch.long)
        data.edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long)
        return data
    else:
        raise FileNotFoundError(f"Please ensure {data_path} exists")


def compute_delta_agg(x, edge_index, device='cpu'):
    """Compute aggregation dilution for a graph"""
    x = x.to(device)
    edge_index = edge_index.to(device)
    n = x.size(0)

    # Compute degree
    deg = degree(edge_index[0], n).float()

    # Normalize features
    x_norm = F.normalize(x, p=2, dim=1)

    # Compute δ_i = d_i × (1 - avg_sim) for each node
    delta_per_node = torch.zeros(n, device=device)

    for i in range(min(n, 5000)):  # Sample for efficiency
        neighbors = edge_index[1][edge_index[0] == i]
        if len(neighbors) == 0:
            continue
        sims = torch.mm(x_norm[i:i+1], x_norm[neighbors].t()).squeeze()
        avg_sim = sims.mean().item() if sims.numel() > 0 else 0
        delta_per_node[i] = deg[i] * (1 - avg_sim)

    mask = deg[:min(n, 5000)] > 0
    delta_agg = delta_per_node[:min(n, 5000)][mask].mean().item()

    return delta_agg


def create_medium_dilution_subgraph(data, target_delta_range=(5, 10), min_nodes=5000):
    """
    Create a subgraph with medium dilution by selecting nodes with medium degree
    """
    edge_index = data.edge_index
    n = data.x.size(0)

    # Compute node degrees
    deg = degree(edge_index[0], n).float()

    print(f"Original graph: {n} nodes, {edge_index.size(1)} edges")
    print(f"Degree stats: min={deg.min():.0f}, max={deg.max():.0f}, mean={deg.mean():.1f}")

    # Try different degree ranges to achieve target δ_agg
    degree_ranges = [
        (10, 30),   # Low-medium
        (15, 40),   # Medium
        (20, 50),   # Medium-high
        (25, 60),   # Higher medium
        (5, 25),    # Lower medium
    ]

    best_subgraph = None
    best_delta = None
    best_range = None

    for min_deg, max_deg in degree_ranges:
        # Select nodes within degree range
        mask = (deg >= min_deg) & (deg <= max_deg)
        selected_nodes = mask.nonzero().squeeze()

        if selected_nodes.numel() < min_nodes:
            print(f"  Degree [{min_deg}, {max_deg}]: only {selected_nodes.numel()} nodes, skipping")
            continue

        # Sample if too many nodes
        if selected_nodes.numel() > 20000:
            perm = torch.randperm(selected_nodes.numel())[:20000]
            selected_nodes = selected_nodes[perm]

        # Create subgraph
        sub_edge_index, _, edge_mask = subgraph(
            selected_nodes, edge_index, relabel_nodes=True, return_edge_mask=True
        )

        # Get subgraph features and labels
        sub_x = data.x[selected_nodes]
        sub_y = data.y[selected_nodes]

        # Compute δ_agg for subgraph
        delta = compute_delta_agg(sub_x, sub_edge_index)
        sub_deg = degree(sub_edge_index[0], selected_nodes.size(0)).float()

        print(f"  Degree [{min_deg}, {max_deg}]: {selected_nodes.numel()} nodes, "
              f"{sub_edge_index.size(1)} edges, avg_deg={sub_deg.mean():.1f}, δ_agg={delta:.2f}")

        # Check if within target range
        if target_delta_range[0] <= delta <= target_delta_range[1]:
            if best_delta is None or abs(delta - 7.5) < abs(best_delta - 7.5):
                best_subgraph = {
                    'x': sub_x,
                    'y': sub_y,
                    'edge_index': sub_edge_index,
                    'selected_nodes': selected_nodes
                }
                best_delta = delta
                best_range = (min_deg, max_deg)

    if best_subgraph is None:
        print("\nNo subgraph found in target δ_agg range. Using best available.")
        # Use the subgraph with δ_agg closest to 7.5
        for min_deg, max_deg in degree_ranges:
            mask = (deg >= min_deg) & (deg <= max_deg)
            selected_nodes = mask.nonzero().squeeze()
            if selected_nodes.numel() < min_nodes:
                continue

            if selected_nodes.numel() > 20000:
                perm = torch.randperm(selected_nodes.numel())[:20000]
                selected_nodes = selected_nodes[perm]

            sub_edge_index, _, _ = subgraph(
                selected_nodes, edge_index, relabel_nodes=True, return_edge_mask=True
            )
            sub_x = data.x[selected_nodes]
            sub_y = data.y[selected_nodes]
            delta = compute_delta_agg(sub_x, sub_edge_index)

            if best_delta is None or abs(delta - 7.5) < abs(best_delta - 7.5):
                best_subgraph = {
                    'x': sub_x,
                    'y': sub_y,
                    'edge_index': sub_edge_index,
                    'selected_nodes': selected_nodes
                }
                best_delta = delta
                best_range = (min_deg, max_deg)

    print(f"\nSelected subgraph: degree range {best_range}, δ_agg = {best_delta:.2f}")
    return best_subgraph, best_delta


class SimpleGCN(nn.Module):
    """Simple 2-layer GCN"""
    def __init__(self, in_dim, hidden_dim=64, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_and_evaluate(model, data, device, epochs=200, lr=0.01):
    """Train and evaluate a model"""
    x = data['x'].to(device)
    y = data['y'].to(device)
    edge_index = data['edge_index'].to(device)

    n = x.size(0)

    # Create train/val/test split (60/20/20)
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[perm[:int(0.6*n)]] = True
    val_mask[perm[int(0.6*n):int(0.8*n)]] = True
    test_mask[perm[int(0.8*n):]] = True

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Handle class imbalance
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]).to(device))

    best_val_auc = 0
    best_test_auc = 0
    best_test_f1 = 0
    patience = 20
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            prob = F.softmax(out, dim=1)[:, 1]

            val_auc = roc_auc_score(y[val_mask].cpu(), prob[val_mask].cpu())

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                test_auc = roc_auc_score(y[test_mask].cpu(), prob[test_mask].cpu())
                pred = (prob > 0.5).long()
                test_f1 = f1_score(y[test_mask].cpu(), pred[test_mask].cpu())
                best_test_auc = test_auc
                best_test_f1 = test_f1
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

    return best_test_auc, best_test_f1


def run_experiment(seeds=[42, 123, 456]):
    """Run the medium dilution experiment"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*60)
    print("Loading IEEE-CIS data...")
    print("="*60)

    try:
        data = load_ieee_cis_data()
    except FileNotFoundError:
        print("IEEE-CIS data not found. Creating synthetic test...")
        # Create synthetic data for testing
        n = 10000
        data = type('Data', (), {
            'x': torch.randn(n, 394),
            'y': torch.randint(0, 2, (n,)),
            'edge_index': torch.randint(0, n, (2, 50000))
        })()

    # Create medium dilution subgraph
    print("\n" + "="*60)
    print("Creating medium dilution subgraph...")
    print("="*60)

    subgraph_data, delta_agg = create_medium_dilution_subgraph(data)

    if subgraph_data is None:
        print("Failed to create subgraph")
        return

    in_dim = subgraph_data['x'].size(1)

    print(f"\nSubgraph statistics:")
    print(f"  Nodes: {subgraph_data['x'].size(0)}")
    print(f"  Edges: {subgraph_data['edge_index'].size(1)}")
    print(f"  Features: {in_dim}")
    print(f"  δ_agg: {delta_agg:.2f}")
    print(f"  Fraud rate: {subgraph_data['y'].float().mean():.3f}")

    # Initialize results
    results = {
        'delta_agg': delta_agg,
        'GCN': {'auc': [], 'f1': []},
        'H2GCN': {'auc': [], 'f1': []},
        'DAAA': {'auc': [], 'f1': []}
    }

    print("\n" + "="*60)
    print(f"Running experiments (δ_agg = {delta_agg:.2f})...")
    print("="*60)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # GCN
        print("  Training GCN...")
        gcn = SimpleGCN(in_dim)
        auc, f1 = train_and_evaluate(gcn, subgraph_data, device)
        results['GCN']['auc'].append(auc)
        results['GCN']['f1'].append(f1)
        print(f"    GCN: AUC={auc:.4f}, F1={f1:.4f}")

        # H2GCN
        print("  Training H2GCN...")
        h2gcn = H2GCN(in_dim, hidden_dim=64, out_dim=2, num_hops=2)
        auc, f1 = train_and_evaluate(h2gcn, subgraph_data, device)
        results['H2GCN']['auc'].append(auc)
        results['H2GCN']['f1'].append(f1)
        print(f"    H2GCN: AUC={auc:.4f}, F1={f1:.4f}")

        # DAAA
        print("  Training DAAA...")
        daaa = DAAA(in_dim, hidden_dim=64, out_dim=2)
        auc, f1 = train_and_evaluate(daaa, subgraph_data, device)
        results['DAAA']['auc'].append(auc)
        results['DAAA']['f1'].append(f1)
        print(f"    DAAA: AUC={auc:.4f}, F1={f1:.4f}")

    # Summary
    print("\n" + "="*60)
    print(f"RESULTS SUMMARY (δ_agg = {delta_agg:.2f})")
    print("="*60)

    print(f"\n{'Method':<10} {'AUC':<20} {'F1':<20}")
    print("-" * 50)

    for method in ['GCN', 'H2GCN', 'DAAA']:
        auc_mean = np.mean(results[method]['auc'])
        auc_std = np.std(results[method]['auc'])
        f1_mean = np.mean(results[method]['f1'])
        f1_std = np.std(results[method]['f1'])
        print(f"{method:<10} {auc_mean:.4f} +/- {auc_std:.4f}   {f1_mean:.4f} +/- {f1_std:.4f}")

    # FSD prediction analysis
    print("\n" + "="*60)
    print("FSD FRAMEWORK ANALYSIS")
    print("="*60)

    if delta_agg < 5:
        expected = "NAA/GCN (low dilution)"
    elif delta_agg > 10:
        expected = "H2GCN/GraphSAGE (high dilution)"
    else:
        expected = "DAAA (medium dilution - adaptive aggregation)"

    print(f"\nδ_agg = {delta_agg:.2f}")
    print(f"FSD Prediction: {expected}")

    # Determine actual winner
    methods = ['GCN', 'H2GCN', 'DAAA']
    aucs = [np.mean(results[m]['auc']) for m in methods]
    winner = methods[np.argmax(aucs)]

    print(f"Actual Winner: {winner} (AUC = {max(aucs):.4f})")

    # Save results
    os.makedirs('./results', exist_ok=True)
    with open('./results/medium_dilution_experiment.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to ./results/medium_dilution_experiment.json")

    return results


if __name__ == '__main__':
    run_experiment()
