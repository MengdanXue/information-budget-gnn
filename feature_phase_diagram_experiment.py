"""
Feature-Structure Phase Diagram Experiment
===========================================

Purpose: Prove that GNN performance regime depends on feature quality

Experiment Design:
1. Feature Ablation: Drop features at different rates (0.1 to 1.0)
2. H-Sweep: Rewire edges to target homophily (0.1 to 0.9)
3. Compare: GCN vs MLP across (feature_quality, homophily) space
4. Generate: 2D phase diagram showing regime transition

Expected Result:
- Below feature_quality ~0.75: U-shape pattern (weak-feature regime)
- Above feature_quality ~0.75: Monotonic pattern (strong-feature regime)

This is the KEY EXPERIMENT for the revised paper.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, from_networkx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json


# ============================================================
# 1. Edge Rewiring to Target Homophily
# ============================================================

def compute_homophily(edge_index, y):
    """
    Compute edge homophily: fraction of edges connecting same-label nodes
    """
    row, col = edge_index
    same_label = (y[row] == y[col]).float()
    return same_label.mean().item()


def rewire_to_target_homophily(edge_index, y, target_h, max_iters=1000, tolerance=0.02):
    """
    Rewire edges to achieve target homophily using edge swapping

    Algorithm:
    1. If current h < target: swap dissimilar edges with similar candidates
    2. If current h > target: swap similar edges with dissimilar candidates
    3. Repeat until h ≈ target

    Args:
        edge_index: [2, num_edges] tensor
        y: [num_nodes] labels
        target_h: target homophily (0.0 to 1.0)
        max_iters: maximum rewiring iterations
        tolerance: acceptable h deviation

    Returns:
        edge_index_new: rewired edge index
        actual_h: achieved homophily
    """
    num_nodes = y.size(0)
    edge_index = edge_index.clone()

    # Convert to edge list for easier manipulation
    edge_set = set((u.item(), v.item()) for u, v in edge_index.t())

    for iteration in range(max_iters):
        current_h = compute_homophily(edge_index, y)

        # Check convergence
        if abs(current_h - target_h) < tolerance:
            print(f"Converged at iteration {iteration}: h={current_h:.3f}")
            break

        # Sample candidate edges to swap
        edges = list(edge_set)
        if len(edges) < 4:
            break

        # Random edge swapping strategy
        idx1, idx2 = np.random.choice(len(edges), size=2, replace=False)
        (u1, v1), (u2, v2) = edges[idx1], edges[idx2]

        # Propose swap: (u1,v1) + (u2,v2) -> (u1,v2) + (u2,v1)
        if (u1, v2) not in edge_set and (u2, v1) not in edge_set:
            # Evaluate swap
            old_homo = int(y[u1] == y[v1]) + int(y[u2] == y[v2])
            new_homo = int(y[u1] == y[v2]) + int(y[u2] == y[v1])

            # Accept swap if it moves toward target
            if (current_h < target_h and new_homo > old_homo) or \
               (current_h > target_h and new_homo < old_homo):
                edge_set.remove((u1, v1))
                edge_set.remove((u2, v2))
                edge_set.add((u1, v2))
                edge_set.add((u2, v1))

                # Update edge_index
                edge_list = list(edge_set)
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    actual_h = compute_homophily(edge_index, y)
    return edge_index, actual_h


# ============================================================
# 2. Feature Ablation
# ============================================================

def ablate_features(X, keep_ratio, seed=42):
    """
    Randomly drop features to reduce feature quality

    Args:
        X: [num_nodes, num_features] feature matrix
        keep_ratio: fraction of features to keep (0.0 to 1.0)
        seed: random seed for reproducibility

    Returns:
        X_ablated: feature matrix with random features zeroed out
        kept_features: indices of kept features
    """
    torch.manual_seed(seed)
    num_features = X.size(1)
    num_keep = int(num_features * keep_ratio)

    # Random feature selection
    perm = torch.randperm(num_features)
    kept_features = perm[:num_keep]

    X_ablated = X.clone()
    mask = torch.zeros(num_features, dtype=torch.bool)
    mask[kept_features] = True
    X_ablated[:, ~mask] = 0

    return X_ablated, kept_features


# ============================================================
# 3. Model Training
# ============================================================

class SimpleMLP(torch.nn.Module):
    """3-layer MLP baseline"""
    def __init__(self, num_features, num_classes, hidden=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_features, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc3(x)


class SimpleGCN(torch.nn.Module):
    """2-layer GCN"""
    def __init__(self, num_features, num_classes, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_model(model, data, optimizer, use_graph=False):
    """Single training epoch"""
    model.train()
    optimizer.zero_grad()

    if use_graph:
        out = model(data.x, data.edge_index)
    else:
        out = model(data.x)

    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_model(model, data, mask, use_graph=False):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        if use_graph:
            out = model(data.x, data.edge_index)
        else:
            out = model(data.x)

        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        return correct.sum().item() / mask.sum().item()


def train_and_evaluate(model, data, epochs=200, lr=0.01, use_graph=False):
    """Full training loop with early stopping"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val_acc = 0
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_model(model, data, optimizer, use_graph)
        val_acc = evaluate_model(model, data, data.val_mask, use_graph)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    test_acc = evaluate_model(model, data, data.test_mask, use_graph)

    return test_acc


# ============================================================
# 4. Phase Diagram Experiment
# ============================================================

def phase_diagram_experiment(dataset_name='Cora', device='cuda'):
    """
    Main experiment: Generate feature-structure phase diagram

    Args:
        dataset_name: 'Cora' or 'CiteSeer'
        device: 'cuda' or 'cpu'

    Returns:
        results: DataFrame with columns [feature_ratio, h, mlp_acc, gcn_acc, advantage]
    """
    print(f"=" * 60)
    print(f"Phase Diagram Experiment: {dataset_name}")
    print(f"=" * 60)

    # Load dataset
    dataset = Planetoid(root='/tmp/Planetoid', name=dataset_name)
    data = dataset[0].to(device)

    # Experiment parameters
    feature_ratios = [0.1, 0.3, 0.5, 0.7, 1.0]  # Feature quality
    h_targets = [0.1, 0.3, 0.5, 0.7, 0.9]      # Homophily targets
    num_seeds = 3  # Multiple runs for stability

    results = []

    # Nested loop over feature quality and homophily
    for feat_ratio in tqdm(feature_ratios, desc="Feature Ratios"):
        # Ablate features
        X_ablated, _ = ablate_features(data.x, feat_ratio)

        for h_target in tqdm(h_targets, desc=f"Homophily (feat={feat_ratio})", leave=False):
            # Rewire edges to target homophily
            edge_index_rewired, actual_h = rewire_to_target_homophily(
                data.edge_index, data.y, h_target
            )

            # Create modified data object
            data_modified = data.clone()
            data_modified.x = X_ablated
            data_modified.edge_index = edge_index_rewired

            # Train models multiple times for stability
            mlp_accs = []
            gcn_accs = []

            for seed in range(num_seeds):
                torch.manual_seed(seed)

                # Train MLP
                mlp = SimpleMLP(
                    data.num_features,
                    dataset.num_classes
                ).to(device)
                mlp_acc = train_and_evaluate(mlp, data_modified, use_graph=False)
                mlp_accs.append(mlp_acc)

                # Train GCN
                gcn = SimpleGCN(
                    data.num_features,
                    dataset.num_classes
                ).to(device)
                gcn_acc = train_and_evaluate(gcn, data_modified, use_graph=True)
                gcn_accs.append(gcn_acc)

            # Average over seeds
            mlp_mean = np.mean(mlp_accs)
            gcn_mean = np.mean(gcn_accs)

            results.append({
                'dataset': dataset_name,
                'feature_ratio': feat_ratio,
                'target_h': h_target,
                'actual_h': actual_h,
                'mlp_acc': mlp_mean,
                'gcn_acc': gcn_mean,
                'advantage': gcn_mean - mlp_mean,
                'mlp_std': np.std(mlp_accs),
                'gcn_std': np.std(gcn_accs)
            })

            print(f"  feat={feat_ratio:.1f}, h={actual_h:.2f}: "
                  f"MLP={mlp_mean:.3f}, GCN={gcn_mean:.3f}, "
                  f"Adv={gcn_mean-mlp_mean:+.3f}")

    return pd.DataFrame(results)


# ============================================================
# 5. Visualization
# ============================================================

def plot_phase_diagram(results, save_path='phase_diagram.pdf'):
    """
    Generate Figure 1: Feature-Structure Phase Diagram

    Args:
        results: DataFrame from phase_diagram_experiment()
        save_path: output file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Heatmap of GCN advantage
    pivot = results.pivot_table(
        values='advantage',
        index='feature_ratio',
        columns='actual_h',
        aggfunc='mean'
    )

    ax1 = axes[0]
    sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                cbar_kws={'label': 'GCN Advantage (%)'}, ax=ax1)
    ax1.set_xlabel('Homophily (h)', fontsize=12)
    ax1.set_ylabel('Feature Keep Ratio', fontsize=12)
    ax1.set_title('Phase Diagram: GCN vs MLP Advantage', fontsize=14)

    # Add regime annotations
    ax1.text(0.5, 0.8, 'Weak-Feature\nU-Shape',
             transform=ax1.transAxes, fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.text(0.5, 0.2, 'Strong-Feature\nMonotonic',
             transform=ax1.transAxes, fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Subplot 2: Line plots for selected feature ratios
    ax2 = axes[1]

    for feat_ratio in [0.3, 0.5, 0.7, 1.0]:
        subset = results[results['feature_ratio'] == feat_ratio]
        ax2.plot(subset['actual_h'], subset['advantage'],
                marker='o', label=f'Feature={feat_ratio:.1f}')

    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Homophily (h)', fontsize=12)
    ax2.set_ylabel('GCN Advantage over MLP (%)', fontsize=12)
    ax2.set_title('Regime Transition with Feature Quality', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Phase diagram saved to {save_path}")


def analyze_phase_transition(results):
    """
    Analyze where phase transition occurs

    Returns:
        transition_point: estimated feature quality threshold
    """
    print("\n" + "=" * 60)
    print("Phase Transition Analysis")
    print("=" * 60)

    # For each feature ratio, check if U-shape or monotonic
    for feat_ratio in sorted(results['feature_ratio'].unique()):
        subset = results[results['feature_ratio'] == feat_ratio].sort_values('actual_h')

        # Check for U-shape: low-h advantage > mid-h advantage
        low_h_adv = subset[subset['actual_h'] < 0.3]['advantage'].mean()
        mid_h_adv = subset[(subset['actual_h'] > 0.4) & (subset['actual_h'] < 0.6)]['advantage'].mean()
        high_h_adv = subset[subset['actual_h'] > 0.7]['advantage'].mean()

        is_ushape = (low_h_adv > mid_h_adv) and (high_h_adv > mid_h_adv)

        print(f"\nFeature Ratio = {feat_ratio:.1f} (MLP ~{subset['mlp_acc'].mean():.2f})")
        print(f"  Low-h advantage:  {low_h_adv:+.3f}")
        print(f"  Mid-h advantage:  {mid_h_adv:+.3f}")
        print(f"  High-h advantage: {high_h_adv:+.3f}")
        print(f"  Pattern: {'U-SHAPE' if is_ushape else 'MONOTONIC'}")

    print("\n" + "=" * 60)


# ============================================================
# 6. Main Execution
# ============================================================

if __name__ == '__main__':
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Run phase diagram experiment
    datasets = ['Cora', 'CiteSeer']

    all_results = []

    for dataset_name in datasets:
        results_df = phase_diagram_experiment(dataset_name, device)
        all_results.append(results_df)

        # Save results
        results_df.to_csv(f'phase_diagram_{dataset_name.lower()}.csv', index=False)

        # Analyze phase transition
        analyze_phase_transition(results_df)

    # Combined visualization
    combined_results = pd.concat(all_results, ignore_index=True)

    for dataset_name in datasets:
        subset = combined_results[combined_results['dataset'] == dataset_name]
        plot_phase_diagram(subset, f'phase_diagram_{dataset_name.lower()}.pdf')

    # Save all results as JSON
    combined_results.to_json('phase_diagram_complete_results.json',
                             orient='records', indent=2)

    print("\n" + "=" * 60)
    print("Phase Diagram Experiment Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - phase_diagram_cora.csv")
    print("  - phase_diagram_citeseer.csv")
    print("  - phase_diagram_cora.pdf")
    print("  - phase_diagram_citeseer.pdf")
    print("  - phase_diagram_complete_results.json")
    print("\nNext steps:")
    print("  1. Examine phase_diagram_*.pdf for visual confirmation")
    print("  2. Check if phase boundary is around feature_ratio ~0.7-0.8")
    print("  3. Use these figures as Figure 1 in revised paper")
    print("=" * 60)
