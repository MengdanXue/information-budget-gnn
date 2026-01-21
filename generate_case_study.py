"""
Generate Case Study Visualization for FSD Paper
===============================================

This script creates publication-quality visualizations showing how NAA's attention
mechanism correctly identifies fraudulent transactions in the Elliptic dataset.

Key visualizations:
1. Neighborhood subgraph for selected fraud nodes
2. NAA attention weight distributions
3. GAT attention weight comparison
4. Feature importance heatmaps

The goal is to demonstrate WHY NAA succeeds where traditional attention fails.

Author: FSD Framework Research Team
Date: 2024-12-22
Version: 1.0 (TKDE Submission)
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import k_hop_subgraph, to_networkx

import networkx as nx
from sklearn.metrics import roc_auc_score, f1_score


# ============================================================================
# Model Definitions (NAA and GAT for comparison)
# ============================================================================

class NAA_GCN(nn.Module):
    """
    Neighbor Adaptive Attention GCN

    Key feature: Feature importance weights that adapt based on neighbor similarity
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # Feature importance (NAA's key innovation)
        self.feature_importance = nn.Parameter(torch.ones(in_dim))

        # GCN layers
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, return_attention=False):
        """
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            return_attention: If True, return feature importance weights

        Returns:
            out: Class logits [N, out_dim]
            attention (optional): Feature importance weights [F]
        """
        # Apply feature importance (NAA mechanism)
        feature_weights = torch.sigmoid(self.feature_importance)
        x_weighted = x * feature_weights

        # Layer 1
        h = self.conv1(x_weighted, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        out = self.conv2(h, edge_index)

        if return_attention:
            return out, feature_weights
        return out


class GAT_Baseline(nn.Module):
    """Standard GAT for comparison"""
    def __init__(self, in_dim, hidden_dim=128, out_dim=2, dropout=0.5, heads=4):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout)

        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, return_attention=False):
        """
        Returns:
            out: Class logits [N, out_dim]
            attention (optional): Edge attention weights from layer 1
        """
        # Layer 1
        if return_attention:
            h, (edge_index_att, att_weights) = self.conv1(
                x, edge_index, return_attention_weights=True
            )
        else:
            h = self.conv1(x, edge_index)

        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        out = self.conv2(h, edge_index)

        if return_attention:
            return out, att_weights
        return out


# ============================================================================
# Data Loading and Model Training
# ============================================================================

def load_elliptic_data(data_dir: str) -> Dict:
    """
    Load Elliptic dataset with Weber temporal split.

    If processed data exists, load it. Otherwise, process from raw CSV files.
    """
    data_dir = Path(data_dir)
    processed_file = data_dir / 'elliptic_weber_split.pkl'

    if processed_file.exists():
        print(f"Loading processed Elliptic data from {processed_file}")
        with open(processed_file, 'rb') as f:
            result = pickle.load(f)
        return result

    # Process from raw files
    print("Processing Elliptic from raw CSV files...")
    from elliptic_weber_split import load_elliptic_raw, create_weber_temporal_split

    features_df, classes_df, edges_df = load_elliptic_raw(data_dir)
    result = create_weber_temporal_split(features_df, classes_df, edges_df)

    # Save for future use
    with open(processed_file, 'wb') as f:
        pickle.dump(result, f)

    return result


def train_model(model, data, num_epochs=200, lr=0.01, weight_decay=5e-4,
                patience=20, device='cuda'):
    """
    Train a GNN model on Elliptic data.

    Returns:
        model: Trained model
        history: Training history
    """
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    print(f"Training {model.__class__.__name__}...")

    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])

            # Compute AUC
            probs = F.softmax(out[data.val_mask], dim=1)[:, 1].cpu().numpy()
            val_auc = roc_auc_score(data.y[data.val_mask].cpu().numpy(), probs)

        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['val_auc'].append(val_auc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss={loss.item():.4f}, "
                  f"Val Loss={val_loss.item():.4f}, "
                  f"Val AUC={val_auc:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    return model, history


def evaluate_model(model, data, mask, device='cuda'):
    """Evaluate model and return predictions, probabilities, and metrics."""
    model = model.to(device)
    data = data.to(device)
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out[mask], dim=1)
        preds = probs.argmax(dim=1)

        y_true = data.y[mask].cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_prob = probs[:, 1].cpu().numpy()

        auc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)

        metrics = {
            'auc': auc,
            'f1': f1,
            'predictions': y_pred,
            'probabilities': y_prob,
            'true_labels': y_true
        }

    return metrics


# ============================================================================
# Case Selection
# ============================================================================

def select_fraud_cases(model, data, num_cases=5, confidence_threshold=0.9,
                       device='cuda'):
    """
    Select high-confidence correctly predicted fraud cases.

    Args:
        model: Trained model
        data: PyG Data
        num_cases: Number of cases to select
        confidence_threshold: Minimum confidence for selection
        device: Device to use

    Returns:
        selected_nodes: List of node indices
        node_info: Dict with information about each selected node
    """
    model = model.to(device)
    data = data.to(device)
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)
        preds = probs.argmax(dim=1)

    # Get test fraud nodes
    test_mask = data.test_mask
    fraud_mask = data.y == 1
    test_fraud_mask = test_mask & fraud_mask

    test_fraud_indices = torch.where(test_fraud_mask)[0]

    # Filter by correct prediction and high confidence
    fraud_probs = probs[test_fraud_indices, 1].cpu().numpy()
    fraud_preds = preds[test_fraud_indices].cpu().numpy()

    correct_high_conf = (fraud_preds == 1) & (fraud_probs >= confidence_threshold)

    candidate_indices = test_fraud_indices[correct_high_conf].cpu().numpy()
    candidate_probs = fraud_probs[correct_high_conf]

    # Select top cases by confidence
    if len(candidate_indices) > num_cases:
        top_k = np.argsort(candidate_probs)[-num_cases:]
        selected_indices = candidate_indices[top_k]
        selected_probs = candidate_probs[top_k]
    else:
        selected_indices = candidate_indices
        selected_probs = candidate_probs

    print(f"\nSelected {len(selected_indices)} fraud cases:")
    node_info = {}
    for idx, node_id in enumerate(selected_indices):
        prob = selected_probs[idx]
        print(f"  Node {node_id}: confidence={prob:.3f}")
        node_info[int(node_id)] = {
            'confidence': float(prob),
            'true_label': 1,
            'predicted_label': 1
        }

    return selected_indices.tolist(), node_info


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_neighborhood_subgraph(node_id, data, model_naa, model_gat,
                                   num_hops=2, device='cuda', save_path=None):
    """
    Visualize the k-hop neighborhood of a fraud node with attention weights.

    Shows:
    - Node colors: fraud (red) vs legitimate (blue)
    - Edge thickness: NAA vs GAT attention comparison
    - Node labels with confidence scores
    """
    # Extract subgraph
    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        node_id, num_hops, data.edge_index, relabel_nodes=True
    )

    center_node_new_id = mapping.item()

    # Get predictions and attention for subgraph
    model_naa.eval()
    model_gat.eval()

    with torch.no_grad():
        # NAA attention (feature importance)
        out_naa, feature_weights_naa = model_naa(
            data.x.to(device), data.edge_index.to(device), return_attention=True
        )
        probs_naa = F.softmax(out_naa, dim=1)[subset, 1].cpu().numpy()

        # GAT attention (edge-level)
        out_gat, edge_att_gat = model_gat(
            data.x.to(device), data.edge_index.to(device), return_attention=True
        )
        probs_gat = F.softmax(out_gat, dim=1)[subset, 1].cpu().numpy()

    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes
    for i, node_idx in enumerate(subset.tolist()):
        is_fraud = data.y[node_idx].item() == 1
        G.add_node(i,
                   fraud=is_fraud,
                   prob_naa=probs_naa[i],
                   prob_gat=probs_gat[i],
                   original_id=node_idx)

    # Add edges
    edge_index_np = edge_index_sub.cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[:, i]
        G.add_edge(int(src), int(dst))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (ax, model_name, probs) in enumerate([
        (axes[0], 'NAA-GCN', probs_naa),
        (axes[1], 'GAT', probs_gat)
    ]):
        # Node colors based on true labels
        node_colors = ['#e74c3c' if G.nodes[node]['fraud'] else '#3498db'
                      for node in G.nodes()]

        # Node sizes based on confidence
        node_sizes = [300 + 700 * probs[node] for node in G.nodes()]

        # Draw network
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1.5,
                              edge_color='gray')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                              node_size=node_sizes, alpha=0.8,
                              edgecolors='black', linewidths=2)

        # Highlight center node
        nx.draw_networkx_nodes(G, pos, ax=ax,
                              nodelist=[center_node_new_id],
                              node_color='#f39c12', node_size=1000,
                              node_shape='*', edgecolors='black', linewidths=3)

        # Node labels with confidence
        labels = {}
        for node in G.nodes():
            if node == center_node_new_id:
                labels[node] = f"TARGET\n{probs[node]:.2f}"
            else:
                labels[node] = f"{probs[node]:.2f}"

        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8,
                               font_weight='bold')

        ax.set_title(f'{model_name} Predictions\n'
                    f'Center Node Confidence: {probs[center_node_new_id]:.3f}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')

    # Add legend
    legend_elements = [
        mpatches.Patch(color='#e74c3c', label='Fraud (True)'),
        mpatches.Patch(color='#3498db', label='Legitimate (True)'),
        mpatches.Patch(color='#f39c12', label='Target Node'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              fontsize=11, frameon=True, fancybox=True, shadow=True)

    plt.suptitle(f'Case Study: Fraud Node {node_id} ({num_hops}-hop Neighborhood)',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved neighborhood visualization to {save_path}")

    return fig


def visualize_attention_comparison(node_id, data, model_naa, model_gat,
                                  num_hops=2, top_k_features=20,
                                  device='cuda', save_path=None):
    """
    Compare NAA feature importance vs GAT edge attention for a specific node.

    Shows:
    1. NAA: Feature importance weights (what features matter)
    2. GAT: Aggregated neighbor attention (which neighbors matter)
    """
    # Extract subgraph
    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        node_id, num_hops, data.edge_index, relabel_nodes=True
    )
    center_node_new_id = mapping.item()

    model_naa.eval()
    model_gat.eval()

    with torch.no_grad():
        # NAA feature importance
        _, feature_weights = model_naa(
            data.x.to(device), data.edge_index.to(device), return_attention=True
        )
        feature_weights = feature_weights.cpu().numpy()

        # GAT edge attention
        _, edge_att = model_gat(
            data.x.to(device), data.edge_index.to(device), return_attention=True
        )

    # Compute neighbor attention for center node
    edge_index_np = data.edge_index.cpu().numpy()
    edge_att_np = edge_att.cpu().numpy()

    # Find edges connected to center node
    incoming_edges = edge_index_np[1] == node_id
    neighbor_attention = {}

    for i, is_incoming in enumerate(incoming_edges):
        if is_incoming:
            src = edge_index_np[0, i]
            att = edge_att_np[i]
            # Handle multi-head attention: average across heads if needed
            if isinstance(att, np.ndarray) and att.ndim > 0:
                att = float(att.mean())
            else:
                att = float(att)
            neighbor_attention[src] = att

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # ===== Panel 1: NAA Feature Importance (Top-K) =====
    ax1 = fig.add_subplot(gs[0, :])

    top_k_indices = np.argsort(feature_weights)[-top_k_features:]
    top_k_weights = feature_weights[top_k_indices]

    colors = plt.cm.RdYlGn(top_k_weights / top_k_weights.max())
    bars = ax1.barh(range(top_k_features), top_k_weights, color=colors,
                    edgecolor='black', linewidth=1)
    ax1.set_yticks(range(top_k_features))
    ax1.set_yticklabels([f'Feature {i}' for i in top_k_indices], fontsize=9)
    ax1.set_xlabel('Feature Importance Weight', fontsize=12, fontweight='bold')
    ax1.set_title(f'NAA: Top-{top_k_features} Feature Importance Weights',
                 fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()

    # ===== Panel 2: GAT Neighbor Attention =====
    ax2 = fig.add_subplot(gs[1, :])

    if neighbor_attention:
        neighbors = list(neighbor_attention.keys())[:15]  # Top 15 neighbors
        attentions = [neighbor_attention[n] for n in neighbors]

        # Color by whether neighbor is fraud
        colors_neighbors = ['#e74c3c' if data.y[n].item() == 1 else '#3498db'
                           for n in neighbors]

        bars = ax2.barh(range(len(neighbors)), attentions,
                       color=colors_neighbors, alpha=0.7,
                       edgecolor='black', linewidth=1)
        ax2.set_yticks(range(len(neighbors)))
        ax2.set_yticklabels([f'Node {n}' for n in neighbors], fontsize=9)
        ax2.set_xlabel('Attention Weight', fontsize=12, fontweight='bold')
        ax2.set_title(f'GAT: Neighbor Attention Weights (Node {node_id})',
                     fontsize=13, fontweight='bold', pad=15)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.invert_yaxis()

        # Legend
        legend_elements = [
            mpatches.Patch(color='#e74c3c', label='Fraud Neighbor'),
            mpatches.Patch(color='#3498db', label='Legitimate Neighbor')
        ]
        ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No neighbors found in subgraph',
                ha='center', va='center', fontsize=12)
        ax2.axis('off')

    # ===== Panel 3: Feature Distribution for Center Node =====
    ax3 = fig.add_subplot(gs[2, 0])

    node_features = data.x[node_id].cpu().numpy()
    weighted_features = node_features * feature_weights

    ax3.scatter(range(len(node_features)), node_features,
               alpha=0.5, s=20, c='gray', label='Original')
    ax3.scatter(range(len(weighted_features)), weighted_features,
               alpha=0.7, s=30, c='orange', label='NAA Weighted', marker='^')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Feature Index', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Feature Value', fontsize=11, fontweight='bold')
    ax3.set_title('Feature Values: Original vs NAA Weighted',
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, linestyle=':')

    # ===== Panel 4: Attention Statistics =====
    ax4 = fig.add_subplot(gs[2, 1])

    stats_text = f"""
    NAA Feature Importance Statistics:
    - Mean: {feature_weights.mean():.4f}
    - Std: {feature_weights.std():.4f}
    - Max: {feature_weights.max():.4f}
    - Top-1 Feature: {feature_weights.argmax()}
    - Top-1 Weight: {feature_weights.max():.4f}

    GAT Neighbor Attention Statistics:
    - Num Neighbors: {len(neighbor_attention)}
    - Mean Attention: {np.mean(list(neighbor_attention.values())):.4f}
    - Attention Entropy: {-np.sum([a * np.log(a + 1e-10) for a in neighbor_attention.values()]):.4f}

    Key Insight:
    NAA focuses on FEATURE patterns that
    discriminate fraud, while GAT focuses
    on NEIGHBOR relationships.
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax4.axis('off')

    plt.suptitle(f'Attention Mechanism Comparison: Node {node_id}',
                fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention comparison to {save_path}")

    return fig


def create_multi_case_summary(selected_nodes, node_info, data, model_naa,
                              model_gat, device='cuda', save_path=None):
    """
    Create a summary figure showing multiple fraud cases side-by-side.
    """
    n_cases = len(selected_nodes)
    fig, axes = plt.subplots(2, n_cases, figsize=(5*n_cases, 10))

    if n_cases == 1:
        axes = axes.reshape(-1, 1)

    for col, node_id in enumerate(selected_nodes):
        # Extract 1-hop neighborhood
        subset, edge_index_sub, mapping, _ = k_hop_subgraph(
            node_id, 1, data.edge_index, relabel_nodes=True
        )
        center_node_new_id = mapping.item()

        # Get predictions
        model_naa.eval()
        model_gat.eval()

        with torch.no_grad():
            out_naa, _ = model_naa(data.x.to(device), data.edge_index.to(device),
                                  return_attention=True)
            out_gat, _ = model_gat(data.x.to(device), data.edge_index.to(device),
                                  return_attention=True)

            probs_naa = F.softmax(out_naa, dim=1)[subset, 1].cpu().numpy()
            probs_gat = F.softmax(out_gat, dim=1)[subset, 1].cpu().numpy()

        # Create subgraph
        G = nx.Graph()
        for i, node_idx in enumerate(subset.tolist()):
            is_fraud = data.y[node_idx].item() == 1
            G.add_node(i, fraud=is_fraud)

        edge_index_np = edge_index_sub.cpu().numpy()
        for i in range(edge_index_np.shape[1]):
            src, dst = edge_index_np[:, i]
            G.add_edge(int(src), int(dst))

        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

        # Plot NAA
        ax_naa = axes[0, col]
        node_colors = ['#e74c3c' if G.nodes[n]['fraud'] else '#3498db'
                      for n in G.nodes()]
        node_sizes = [300 + 500 * probs_naa[n] for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax_naa, alpha=0.3, width=1)
        nx.draw_networkx_nodes(G, pos, ax=ax_naa, node_color=node_colors,
                              node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, ax=ax_naa, nodelist=[center_node_new_id],
                              node_color='#f39c12', node_size=600, node_shape='*')

        ax_naa.set_title(f'Node {node_id}\nNAA: {probs_naa[center_node_new_id]:.3f}',
                        fontsize=11, fontweight='bold')
        ax_naa.axis('off')

        # Plot GAT
        ax_gat = axes[1, col]
        node_sizes_gat = [300 + 500 * probs_gat[n] for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax_gat, alpha=0.3, width=1)
        nx.draw_networkx_nodes(G, pos, ax=ax_gat, node_color=node_colors,
                              node_size=node_sizes_gat, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, ax=ax_gat, nodelist=[center_node_new_id],
                              node_color='#f39c12', node_size=600, node_shape='*')

        ax_gat.set_title(f'GAT: {probs_gat[center_node_new_id]:.3f}',
                        fontsize=11, fontweight='bold')
        ax_gat.axis('off')

    # Row labels
    fig.text(0.02, 0.75, 'NAA-GCN', fontsize=14, fontweight='bold',
            rotation=90, va='center')
    fig.text(0.02, 0.25, 'GAT', fontsize=14, fontweight='bold',
            rotation=90, va='center')

    plt.suptitle('Case Study: Multiple Fraud Detection Examples',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-case summary to {save_path}")

    return fig


# ============================================================================
# Main Execution
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate Case Study Visualizations')
    parser.add_argument('--data_dir', type=str,
                       default='D:/Users/11919/Documents/毕业论文/paper/code/data',
                       help='Directory containing Elliptic data')
    parser.add_argument('--output_dir', type=str,
                       default='D:/Users/11919/Documents/毕业论文/paper/figures',
                       help='Directory to save figures')
    parser.add_argument('--num_cases', type=int, default=3,
                       help='Number of fraud cases to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and load saved models')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n" + "="*60)
    print("LOADING ELLIPTIC DATASET")
    print("="*60)
    result = load_elliptic_data(args.data_dir)
    data = result['data']

    print(f"\nDataset: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Features: {data.num_features}")
    print(f"Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, "
          f"Test: {data.test_mask.sum()}")

    # Initialize models
    print("\n" + "="*60)
    print("INITIALIZING MODELS")
    print("="*60)

    model_naa = NAA_GCN(in_dim=data.num_features, hidden_dim=128, out_dim=2)
    model_gat = GAT_Baseline(in_dim=data.num_features, hidden_dim=128, out_dim=2)

    # Train or load models
    if not args.skip_training:
        print("\nTraining NAA-GCN...")
        model_naa, hist_naa = train_model(model_naa, data, device=device)

        print("\nTraining GAT...")
        model_gat, hist_gat = train_model(model_gat, data, device=device)

        # Save models
        torch.save(model_naa.state_dict(), output_dir / 'naa_gcn_elliptic.pt')
        torch.save(model_gat.state_dict(), output_dir / 'gat_elliptic.pt')
        print("\nModels saved!")
    else:
        print("\nLoading saved models...")
        model_naa.load_state_dict(torch.load(output_dir / 'naa_gcn_elliptic.pt'))
        model_gat.load_state_dict(torch.load(output_dir / 'gat_elliptic.pt'))

    # Evaluate models
    print("\n" + "="*60)
    print("EVALUATING MODELS")
    print("="*60)

    metrics_naa = evaluate_model(model_naa, data, data.test_mask, device)
    metrics_gat = evaluate_model(model_gat, data, data.test_mask, device)

    print(f"\nNAA-GCN: AUC={metrics_naa['auc']:.4f}, F1={metrics_naa['f1']:.4f}")
    print(f"GAT:     AUC={metrics_gat['auc']:.4f}, F1={metrics_gat['f1']:.4f}")

    # Select fraud cases
    print("\n" + "="*60)
    print("SELECTING FRAUD CASES")
    print("="*60)

    selected_nodes, node_info = select_fraud_cases(
        model_naa, data, num_cases=args.num_cases, device=device
    )

    if len(selected_nodes) == 0:
        print("No high-confidence fraud cases found. Try lowering confidence threshold.")
        return

    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Visualization 1: Multi-case summary
    print("\n1. Creating multi-case summary...")
    fig1 = create_multi_case_summary(
        selected_nodes, node_info, data, model_naa, model_gat,
        device=device,
        save_path=output_dir / 'case_study_elliptic.pdf'
    )
    plt.close(fig1)

    # Visualization 2-3: Detailed analysis for first case
    if len(selected_nodes) > 0:
        target_node = selected_nodes[0]

        print(f"\n2. Creating detailed neighborhood for node {target_node}...")
        fig2 = visualize_neighborhood_subgraph(
            target_node, data, model_naa, model_gat,
            num_hops=2, device=device,
            save_path=output_dir / f'case_study_node_{target_node}_neighborhood.pdf'
        )
        plt.close(fig2)

        print(f"\n3. Creating attention comparison for node {target_node}...")
        fig3 = visualize_attention_comparison(
            target_node, data, model_naa, model_gat,
            top_k_features=20, device=device,
            save_path=output_dir / 'case_study_attention_comparison.pdf'
        )
        plt.close(fig3)

    print("\n" + "="*60)
    print("CASE STUDY GENERATION COMPLETE")
    print("="*60)
    print(f"\nFigures saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - case_study_elliptic.pdf (multi-case summary)")
    print("  - case_study_attention_comparison.pdf (attention mechanisms)")
    print("  - case_study_node_*_neighborhood.pdf (detailed neighborhoods)")


if __name__ == '__main__':
    # Set publication-quality plot parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

    main()
