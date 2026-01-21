"""
Dilution-Aware Adaptive Aggregation (DAAA)

A novel GNN architecture that adaptively selects aggregation strategy
based on node-level aggregation dilution δ_agg(i).

Core idea:
- Low δ_agg nodes: use mean aggregation (can leverage NAA attention)
- High δ_agg nodes: preserve ego via concatenation (like H2GCN)
- Learnable gate based on δ_agg determines the mixing

Key insight from FSD theory:
- δ_agg(i) = d_i × (1 - S_feat(x_i, x̄_N(i)))
- When δ_agg is high, mean aggregation causes severe ego information loss
- When δ_agg is low, mean aggregation is effective

Author: FSD Framework
Date: 2024-12-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import degree, to_torch_sparse_tensor


class DilutionComputer(nn.Module):
    """
    Compute node-level aggregation dilution δ_agg(i).

    δ_agg(i) = d_i × (1 - S_feat(x_i, x̄_N(i)))

    where:
    - d_i: degree of node i
    - S_feat: feature similarity (cosine)
    - x̄_N(i): mean neighbor features
    """

    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]

        Returns:
            delta_agg: Node-level dilution [N]
        """
        n = x.size(0)

        # Build sparse adjacency
        adj = to_torch_sparse_tensor(edge_index, size=(n, n))

        # Compute degrees
        deg = degree(edge_index[0], n, dtype=x.dtype)
        deg_safe = deg.clamp(min=1)  # Avoid division by zero

        # Compute mean neighbor features: x̄_N(i) = (1/d_i) * sum_{j in N(i)} x_j
        neighbor_sum = torch.sparse.mm(adj, x)  # [N, F]
        mean_neighbor = neighbor_sum / deg_safe.unsqueeze(1)  # [N, F]

        # Compute cosine similarity S_feat(x_i, x̄_N(i))
        x_norm = F.normalize(x, p=2, dim=1)
        mean_neighbor_norm = F.normalize(mean_neighbor, p=2, dim=1)
        similarity = (x_norm * mean_neighbor_norm).sum(dim=1)  # [N]

        # Handle isolated nodes (deg=0)
        similarity = torch.where(deg > 0, similarity, torch.ones_like(similarity))

        # Compute δ_agg(i) = d_i × (1 - S_feat)
        delta_agg = deg * (1 - similarity)

        if self.normalize:
            # Normalize to [0, 1] range using sigmoid-like transformation
            # This helps with gradient flow
            delta_agg = torch.sigmoid(delta_agg / 10.0 - 0.5)  # threshold around 10

        return delta_agg


class AdaptiveAggregationLayer(nn.Module):
    """
    Adaptive aggregation layer that mixes mean and ego-preserving aggregation
    based on node-level dilution.

    h_i = (1 - g_i) * MeanAgg(N(i)) + g_i * [h_i || MeanAgg(N(i))]

    where g_i = σ(w * δ_agg(i) + b) is a learnable gate.
    """

    def __init__(self, in_dim, out_dim, use_learned_gate=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_learned_gate = use_learned_gate

        # Mean aggregation path
        self.mean_transform = nn.Linear(in_dim, out_dim)

        # Ego-preserving path (concatenation: ego || neighbor)
        self.ego_transform = nn.Linear(in_dim, out_dim // 2)
        self.neighbor_transform = nn.Linear(in_dim, out_dim - out_dim // 2)

        # Learnable gate parameters
        if use_learned_gate:
            self.gate_weight = nn.Parameter(torch.tensor(1.0))
            self.gate_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index, delta_agg):
        """
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            delta_agg: Node-level dilution [N] (normalized to ~[0,1])

        Returns:
            h: Updated node features [N, out_dim]
        """
        n = x.size(0)

        # Build sparse adjacency
        adj = to_torch_sparse_tensor(edge_index, size=(n, n))

        # Compute degrees for normalization
        deg = degree(edge_index[0], n, dtype=x.dtype).clamp(min=1)

        # Compute mean neighbor features
        neighbor_sum = torch.sparse.mm(adj, x)
        mean_neighbor = neighbor_sum / deg.unsqueeze(1)

        # Path 1: Mean aggregation (like GCN)
        # h_mean = transform(mean of ego and neighbors)
        h_mean_input = (x + mean_neighbor) / 2  # Simple mean with ego
        h_mean = self.mean_transform(h_mean_input)

        # Path 2: Ego-preserving concatenation (like H2GCN)
        h_ego = self.ego_transform(x)
        h_neighbor = self.neighbor_transform(mean_neighbor)
        h_concat = torch.cat([h_ego, h_neighbor], dim=1)

        # Compute gate based on dilution
        if self.use_learned_gate:
            # g_i = σ(w * δ_agg(i) + b)
            gate = torch.sigmoid(self.gate_weight * delta_agg + self.gate_bias)
        else:
            # Direct use of normalized dilution as gate
            gate = delta_agg

        gate = gate.unsqueeze(1)  # [N, 1] for broadcasting

        # Adaptive mixing
        # Low dilution (gate ≈ 0): use mean aggregation
        # High dilution (gate ≈ 1): preserve ego via concatenation
        h = (1 - gate) * h_mean + gate * h_concat

        return h


class DAAA(nn.Module):
    """
    Dilution-Aware Adaptive Aggregation (DAAA)

    A GNN that adaptively chooses between mean aggregation and
    ego-preserving concatenation based on node-level dilution δ_agg.

    Architecture:
    1. Compute δ_agg for all nodes (once, based on input features)
    2. Layer 1: Adaptive aggregation based on δ_agg
    3. Layer 2: Standard transformation + classification

    Key properties:
    - Degrades to GCN-like behavior for low-dilution nodes
    - Degrades to H2GCN-like behavior for high-dilution nodes
    - Learns optimal threshold through gate parameters
    """

    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5,
                 use_learned_gate=True, use_feature_importance=True):
        super().__init__()
        self.dropout = dropout
        self.use_feature_importance = use_feature_importance

        # Dilution computer
        self.dilution_computer = DilutionComputer(normalize=True)

        # Optional feature importance (from NAA)
        if use_feature_importance:
            self.feature_importance = nn.Parameter(torch.ones(in_dim))

        # Layer 1: Adaptive aggregation
        self.layer1 = AdaptiveAggregationLayer(in_dim, hidden_dim, use_learned_gate)

        # Layer 2: Standard GCN-style aggregation (dilution is lower after layer 1)
        self.layer2 = GCNConv(hidden_dim, out_dim)

        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]

        Returns:
            out: Class logits [N, out_dim]
        """
        # Apply feature importance weighting
        if self.use_feature_importance:
            x = x * torch.sigmoid(self.feature_importance)

        # Compute node-level dilution
        delta_agg = self.dilution_computer(x, edge_index)

        # Layer 1: Adaptive aggregation
        h = self.layer1(x, edge_index, delta_agg)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2: Standard aggregation
        out = self.layer2(h, edge_index)

        return out

    def get_dilution_stats(self, x, edge_index):
        """Get dilution statistics for analysis."""
        with torch.no_grad():
            delta_agg = self.dilution_computer(x, edge_index)

            if hasattr(self.layer1, 'gate_weight'):
                gate = torch.sigmoid(
                    self.layer1.gate_weight * delta_agg + self.layer1.gate_bias
                )
            else:
                gate = delta_agg

            return {
                'delta_agg_mean': delta_agg.mean().item(),
                'delta_agg_std': delta_agg.std().item(),
                'gate_mean': gate.mean().item(),
                'gate_std': gate.std().item(),
                'pct_high_dilution': (gate > 0.5).float().mean().item(),
            }


class DAAAv2(nn.Module):
    """
    DAAA Version 2: More sophisticated adaptive aggregation.

    Key improvements:
    1. Multi-hop dilution computation
    2. Attention-weighted neighbor aggregation in mean path
    3. Separate transforms for different dilution regimes
    """

    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5,
                 num_hops=2, use_attention=True):
        super().__init__()
        self.dropout = dropout
        self.num_hops = num_hops
        self.use_attention = use_attention

        # Dilution computer
        self.dilution_computer = DilutionComputer(normalize=True)

        # Feature importance
        self.feature_importance = nn.Parameter(torch.ones(in_dim))

        # Low-dilution path: attention-based mean aggregation
        if use_attention:
            from torch_geometric.nn import GATConv
            self.low_dilution_conv = GATConv(in_dim, hidden_dim // 4, heads=4)
        else:
            self.low_dilution_conv = GCNConv(in_dim, hidden_dim)

        # High-dilution path: ego + multi-hop concatenation
        self.ego_transform = nn.Linear(in_dim, hidden_dim // 3)
        self.hop_transforms = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim // 3) for _ in range(num_hops)
        ])
        # Adjust last hop to fill remaining dims
        remaining_dim = hidden_dim - hidden_dim // 3 * num_hops
        if remaining_dim > 0:
            self.hop_transforms[-1] = nn.Linear(in_dim, hidden_dim // 3 + remaining_dim - hidden_dim // 3)

        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Output layers
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        n = x.size(0)

        # Apply feature importance
        x = x * torch.sigmoid(self.feature_importance)

        # Compute dilution
        delta_agg = self.dilution_computer(x, edge_index)

        # Low-dilution path
        h_low = self.low_dilution_conv(x, edge_index)
        if self.use_attention:
            h_low = F.elu(h_low)
        else:
            h_low = F.relu(h_low)

        # High-dilution path
        adj = to_torch_sparse_tensor(edge_index, size=(n, n))
        deg = degree(edge_index[0], n, dtype=x.dtype).clamp(min=1)

        h_ego = F.relu(self.ego_transform(x))
        hop_features = [h_ego]

        h_hop = x
        for i, transform in enumerate(self.hop_transforms):
            h_hop = torch.sparse.mm(adj, h_hop) / deg.unsqueeze(1)
            hop_features.append(F.relu(transform(h_hop)))

        h_high = torch.cat(hop_features, dim=1)

        # Compute gate
        gate = self.gate_net(delta_agg.unsqueeze(1))  # [N, 1]

        # Mix paths
        h = (1 - gate) * h_low + gate * h_high

        # Output
        h = self.bn(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.classifier(h)

        return out


class DAAAv3(nn.Module):
    """
    DAAA Version 3: Hard-gating with stronger H2GCN-style high-dilution path.

    Key insight: For high-dilution nodes, we need to FULLY preserve ego info,
    not just partially. This version uses:
    1. Degree-based hard threshold (not learned) for routing
    2. Full H2GCN-style aggregation for high-dilution nodes
    3. GCN+NAA for low-dilution nodes
    4. Smooth transition zone
    """

    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5,
                 dilution_threshold=0.5, num_hops=2):
        super().__init__()
        self.dropout = dropout
        self.dilution_threshold = dilution_threshold
        self.num_hops = num_hops
        self.hidden_dim = hidden_dim

        # Dilution computer (not normalized, we want raw values)
        self.dilution_computer = DilutionComputer(normalize=True)

        # Feature importance (NAA-style)
        self.feature_importance = nn.Parameter(torch.ones(in_dim))

        # === Low-dilution path (GCN-style with NAA) ===
        self.low_conv1 = GCNConv(in_dim, hidden_dim)
        self.low_conv2 = GCNConv(hidden_dim, hidden_dim)

        # === High-dilution path (H2GCN-style) ===
        # Ego transformation
        self.high_ego = nn.Linear(in_dim, hidden_dim // (num_hops + 1))

        # Hop transformations
        hop_dim = hidden_dim // (num_hops + 1)
        self.high_hops = nn.ModuleList()
        for i in range(num_hops):
            if i == num_hops - 1:
                # Last hop gets remaining dimensions
                self.high_hops.append(nn.Linear(in_dim, hidden_dim - hop_dim * num_hops))
            else:
                self.high_hops.append(nn.Linear(in_dim, hop_dim))

        # === Shared output ===
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

        # Learnable temperature for gating
        self.gate_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index):
        n = x.size(0)

        # Apply feature importance
        x_weighted = x * torch.sigmoid(self.feature_importance)

        # Compute node-level dilution
        delta_agg = self.dilution_computer(x_weighted, edge_index)  # [N], normalized ~[0,1]

        # === Low-dilution path ===
        h_low = self.low_conv1(x_weighted, edge_index)
        h_low = F.relu(h_low)
        h_low = F.dropout(h_low, p=self.dropout, training=self.training)
        h_low = self.low_conv2(h_low, edge_index)

        # === High-dilution path (H2GCN-style) ===
        adj = to_torch_sparse_tensor(edge_index, size=(n, n))
        deg = degree(edge_index[0], n, dtype=x.dtype).clamp(min=1)

        # Ego features (preserved!)
        h_ego = self.high_ego(x_weighted)

        # Multi-hop aggregation
        hop_reps = [h_ego]
        h_hop = x_weighted
        for i, hop_layer in enumerate(self.high_hops):
            # Aggregate from neighbors
            h_hop = torch.sparse.mm(adj, h_hop)
            h_hop = h_hop / deg.unsqueeze(1)
            hop_reps.append(hop_layer(h_hop))

        # Concatenate (key: ego is preserved separately!)
        h_high = torch.cat(hop_reps, dim=1)
        h_high = F.relu(h_high)

        # === Adaptive mixing based on dilution ===
        # Use steep sigmoid for near-hard gating
        gate = torch.sigmoid((delta_agg - self.dilution_threshold) * self.gate_temp * 10)
        gate = gate.unsqueeze(1)  # [N, 1]

        # Mix: low dilution -> h_low, high dilution -> h_high
        h = (1 - gate) * h_low + gate * h_high

        # Output
        h = self.bn(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.classifier(h)

        return out

    def get_routing_stats(self, x, edge_index):
        """Get statistics about routing decisions."""
        with torch.no_grad():
            x_weighted = x * torch.sigmoid(self.feature_importance)
            delta_agg = self.dilution_computer(x_weighted, edge_index)
            gate = torch.sigmoid((delta_agg - self.dilution_threshold) * self.gate_temp * 10)

            return {
                'pct_high_path': (gate > 0.5).float().mean().item(),
                'pct_low_path': (gate <= 0.5).float().mean().item(),
                'gate_mean': gate.mean().item(),
                'gate_std': gate.std().item(),
                'dilution_mean': delta_agg.mean().item(),
                'dilution_std': delta_agg.std().item(),
            }


class DAAAv4(nn.Module):
    """
    DAAA Version 4: Ensemble approach - run both paths and combine.

    Idea: Instead of routing nodes to different paths, run BOTH paths
    for ALL nodes, then combine based on dilution. This ensures:
    1. Both paths see all nodes (better gradient flow)
    2. High-dilution nodes get more weight from H2GCN path
    3. Low-dilution nodes get more weight from GCN path

    This is similar to mixture-of-experts but with dilution-based gating.
    """

    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5, num_hops=2):
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # Dilution computer
        self.dilution_computer = DilutionComputer(normalize=True)

        # Feature importance
        self.feature_importance = nn.Parameter(torch.ones(in_dim))

        # === Expert 1: GCN path (good for low dilution) ===
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # === Expert 2: H2GCN path (good for high dilution) ===
        hop_dim = hidden_dim // (num_hops + 1)
        self.h2gcn_ego = nn.Linear(in_dim, hop_dim)
        self.h2gcn_hops = nn.ModuleList([
            nn.Linear(in_dim, hop_dim if i < num_hops - 1 else hidden_dim - hop_dim * num_hops)
            for i in range(num_hops)
        ])

        # === Expert 3: GraphSAGE path (sampling-based, bounded dilution) ===
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)

        # === Gating network ===
        # Input: dilution value, output: weights for 3 experts
        self.gate_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 experts
        )

        # === Output ===
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        n = x.size(0)

        # Apply feature importance
        x_w = x * torch.sigmoid(self.feature_importance)

        # Compute dilution
        delta_agg = self.dilution_computer(x_w, edge_index)  # [N]

        # === Expert 1: GCN ===
        h_gcn = F.relu(self.gcn1(x_w, edge_index))
        h_gcn = F.dropout(h_gcn, p=self.dropout, training=self.training)
        h_gcn = self.gcn2(h_gcn, edge_index)

        # === Expert 2: H2GCN ===
        adj = to_torch_sparse_tensor(edge_index, size=(n, n))
        deg = degree(edge_index[0], n, dtype=x.dtype).clamp(min=1)

        h2_ego = self.h2gcn_ego(x_w)
        h2_hops = [h2_ego]
        h_hop = x_w
        for hop_layer in self.h2gcn_hops:
            h_hop = torch.sparse.mm(adj, h_hop) / deg.unsqueeze(1)
            h2_hops.append(hop_layer(h_hop))
        h_h2gcn = F.relu(torch.cat(h2_hops, dim=1))

        # === Expert 3: GraphSAGE ===
        h_sage = F.relu(self.sage1(x_w, edge_index))
        h_sage = F.dropout(h_sage, p=self.dropout, training=self.training)
        h_sage = self.sage2(h_sage, edge_index)

        # === Gating: combine experts based on dilution ===
        gate_input = delta_agg.unsqueeze(1)  # [N, 1]
        gate_logits = self.gate_net(gate_input)  # [N, 3]
        gate_weights = F.softmax(gate_logits, dim=1)  # [N, 3]

        # Stack expert outputs: [N, hidden_dim, 3]
        expert_outputs = torch.stack([h_gcn, h_h2gcn, h_sage], dim=2)

        # Weighted combination: [N, hidden_dim]
        h = (expert_outputs * gate_weights.unsqueeze(1)).sum(dim=2)

        # Output
        h = self.bn(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.classifier(h)

        return out

    def get_expert_usage(self, x, edge_index):
        """Analyze which experts are used for which nodes."""
        with torch.no_grad():
            x_w = x * torch.sigmoid(self.feature_importance)
            delta_agg = self.dilution_computer(x_w, edge_index)
            gate_input = delta_agg.unsqueeze(1)
            gate_logits = self.gate_net(gate_input)
            gate_weights = F.softmax(gate_logits, dim=1)

            return {
                'gcn_weight_mean': gate_weights[:, 0].mean().item(),
                'h2gcn_weight_mean': gate_weights[:, 1].mean().item(),
                'sage_weight_mean': gate_weights[:, 2].mean().item(),
                'dilution_mean': delta_agg.mean().item(),
            }


# Test code
if __name__ == '__main__':
    # Create dummy data
    n_nodes = 1000
    n_features = 64
    n_edges = 5000

    x = torch.randn(n_nodes, n_features)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))

    # Test DAAA
    model = DAAA(n_features, hidden_dim=64, out_dim=2)
    out = model(x, edge_index)
    print(f"DAAA output shape: {out.shape}")

    stats = model.get_dilution_stats(x, edge_index)
    print(f"Dilution stats: {stats}")

    # Test DAAAv2
    model_v2 = DAAAv2(n_features, hidden_dim=64, out_dim=2)
    out_v2 = model_v2(x, edge_index)
    print(f"DAAAv2 output shape: {out_v2.shape}")

    print("\nDAAA model created successfully!")
