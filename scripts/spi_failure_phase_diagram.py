"""
Experiment 2: SPI Failure Phase Diagram
Purpose: Map out the exact boundary where SPI succeeds vs fails

This experiment generates a grid of synthetic graphs varying:
- Homophily (h): 0.0 to 1.0
- Feature SNR (signal-to-noise ratio): 0.1 to 1.0

For each combination, we test if SPI correctly predicts GNN vs MLP winner.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from torch_geometric.nn import GCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("PyTorch Geometric not available")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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


def generate_csbm_graph(num_nodes=1000, h_target=0.5, feature_snr=0.5,
                        num_edges_per_node=10, num_classes=2, feature_dim=32):
    """
    Generate a Contextual Stochastic Block Model (cSBM) graph.

    Args:
        num_nodes: Number of nodes
        h_target: Target homophily (0 = all heterophilic, 1 = all homophilic)
        feature_snr: Feature signal-to-noise ratio (0 = pure noise, 1 = perfect signal)
        num_edges_per_node: Average edges per node
        num_classes: Number of classes
        feature_dim: Feature dimension
    """
    # Generate labels
    labels = np.random.randint(0, num_classes, num_nodes)

    # Generate features with controlled SNR
    # Signal: class-specific mean
    class_means = np.random.randn(num_classes, feature_dim) * 2
    signal = class_means[labels]

    # Noise: random Gaussian
    noise = np.random.randn(num_nodes, feature_dim)

    # Mix signal and noise based on SNR
    features = feature_snr * signal + (1 - feature_snr) * noise

    # Generate edges with controlled homophily
    edges_src = []
    edges_dst = []

    for i in range(num_nodes):
        for _ in range(num_edges_per_node):
            if np.random.random() < h_target:
                # Same class neighbor
                same_class = np.where(labels == labels[i])[0]
                if len(same_class) > 1:
                    j = np.random.choice(same_class)
                    while j == i:
                        j = np.random.choice(same_class)
                else:
                    j = np.random.randint(0, num_nodes)
            else:
                # Different class neighbor
                diff_class = np.where(labels != labels[i])[0]
                if len(diff_class) > 0:
                    j = np.random.choice(diff_class)
                else:
                    j = np.random.randint(0, num_nodes)

            edges_src.append(i)
            edges_dst.append(j)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    return features, edge_index, labels, num_nodes


def compute_homophily(edge_index, labels):
    """Compute actual edge homophily."""
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()


def compute_spi(h):
    """Compute Structural Predictability Index."""
    return abs(2 * h - 1)


def train_and_evaluate(model, features, edge_index, labels, train_mask, test_mask,
                       epochs=200, lr=0.01, weight_decay=5e-4):
    """Train model and return test accuracy."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    features = features.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(features, edge_index)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(features, edge_index)
            pred = out.argmax(dim=1)
            acc = (pred[test_mask] == labels[test_mask]).float().mean().item()

            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

    return best_acc


def run_phase_diagram():
    """Generate the SPI failure phase diagram."""
    print("=" * 70)
    print("Experiment 2: SPI Failure Phase Diagram")
    print("=" * 70)

    if not HAS_PYG:
        print("PyTorch Geometric required for this experiment")
        return None

    # Parameters
    h_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    snr_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    num_runs = 3
    num_nodes = 1000

    results = {
        'experiment': 'spi_failure_phase_diagram',
        'h_values': h_values,
        'snr_values': snr_values,
        'num_runs': num_runs,
        'grid': []
    }

    print(f"\nGenerating phase diagram: {len(h_values)} x {len(snr_values)} = {len(h_values)*len(snr_values)} grid points")
    print(f"Each point: {num_runs} runs\n")

    # Phase diagram header
    print("-" * 90)
    print(f"{'h':<8} {'SNR':<8} {'h_actual':<10} {'SPI':<8} {'MLP':<10} {'GCN':<10} {'Winner':<10} {'SPI_Pred':<10} {'Correct':<8}")
    print("-" * 90)

    for h_target in h_values:
        for snr in snr_values:
            mlp_accs = []
            gcn_accs = []
            h_actuals = []

            for run in range(num_runs):
                set_seed(42 + run)

                # Generate graph
                features, edge_index, labels, n_nodes = generate_csbm_graph(
                    num_nodes=num_nodes,
                    h_target=h_target,
                    feature_snr=snr,
                    num_edges_per_node=15,
                    num_classes=2,
                    feature_dim=32
                )

                # Compute actual homophily
                h_actual = compute_homophily(edge_index, labels)
                h_actuals.append(h_actual)

                # Create train/test split
                n = len(labels)
                perm = torch.randperm(n)
                train_mask = torch.zeros(n, dtype=torch.bool)
                test_mask = torch.zeros(n, dtype=torch.bool)
                train_mask[perm[:int(0.6*n)]] = True
                test_mask[perm[int(0.8*n):]] = True

                # Train MLP
                mlp = MLP(features.shape[1], 64, 2)
                mlp_acc = train_and_evaluate(mlp, features, edge_index, labels,
                                            train_mask, test_mask)
                mlp_accs.append(mlp_acc)

                # Train GCN
                gcn = GCN(features.shape[1], 64, 2)
                gcn_acc = train_and_evaluate(gcn, features, edge_index, labels,
                                            train_mask, test_mask)
                gcn_accs.append(gcn_acc)

            # Average results
            h_actual_avg = np.mean(h_actuals)
            mlp_acc_avg = np.mean(mlp_accs)
            gcn_acc_avg = np.mean(gcn_accs)
            spi = compute_spi(h_actual_avg)

            # Determine winner and SPI prediction
            actual_winner = 'GCN' if gcn_acc_avg > mlp_acc_avg else 'MLP'
            spi_prediction = 'GCN' if spi > 0.4 else 'MLP'  # Using 0.4 threshold (h < 0.3 or h > 0.7)
            correct = actual_winner == spi_prediction

            result = {
                'h_target': h_target,
                'snr': snr,
                'h_actual': float(h_actual_avg),
                'spi': float(spi),
                'mlp_acc': float(mlp_acc_avg),
                'gcn_acc': float(gcn_acc_avg),
                'gcn_advantage': float(gcn_acc_avg - mlp_acc_avg),
                'actual_winner': actual_winner,
                'spi_prediction': spi_prediction,
                'prediction_correct': str(correct)
            }
            results['grid'].append(result)

            correct_str = '[OK]' if correct else '[X]'
            print(f"{h_target:<8.1f} {snr:<8.1f} {h_actual_avg:<10.3f} {spi:<8.3f} "
                  f"{mlp_acc_avg:<10.3f} {gcn_acc_avg:<10.3f} {actual_winner:<10} "
                  f"{spi_prediction:<10} {correct_str:<8}")

    print("-" * 90)

    # Summary by region
    print("\n" + "=" * 70)
    print("Summary: SPI Accuracy by Region")
    print("=" * 70)

    # Group by h region
    low_h = [r for r in results['grid'] if r['h_actual'] < 0.3]
    mid_h = [r for r in results['grid'] if 0.3 <= r['h_actual'] <= 0.7]
    high_h = [r for r in results['grid'] if r['h_actual'] > 0.7]

    for name, region in [('Low h (<0.3)', low_h), ('Mid h (0.3-0.7)', mid_h), ('High h (>0.7)', high_h)]:
        if region:
            correct_count = sum(1 for r in region if r['prediction_correct'] == 'True')
            total = len(region)
            avg_gcn_adv = np.mean([r['gcn_advantage'] for r in region])
            print(f"\n{name}:")
            print(f"  SPI Accuracy: {correct_count}/{total} ({100*correct_count/total:.1f}%)")
            print(f"  Avg GCN Advantage: {avg_gcn_adv:+.3f}")

    # Group by SNR
    print("\n" + "-" * 50)
    print("Summary by Feature SNR:")
    for snr in snr_values:
        snr_group = [r for r in results['grid'] if r['snr'] == snr]
        correct_count = sum(1 for r in snr_group if r['prediction_correct'] == 'True')
        total = len(snr_group)
        print(f"  SNR={snr:.1f}: {correct_count}/{total} ({100*correct_count/total:.1f}%)")

    # Overall
    total_correct = sum(1 for r in results['grid'] if r['prediction_correct'] == 'True')
    total = len(results['grid'])
    results['summary'] = {
        'total_accuracy': f"{total_correct}/{total} ({100*total_correct/total:.1f}%)",
        'low_h_accuracy': f"{sum(1 for r in low_h if r['prediction_correct']=='True')}/{len(low_h)}" if low_h else "N/A",
        'mid_h_accuracy': f"{sum(1 for r in mid_h if r['prediction_correct']=='True')}/{len(mid_h)}" if mid_h else "N/A",
        'high_h_accuracy': f"{sum(1 for r in high_h if r['prediction_correct']=='True')}/{len(high_h)}" if high_h else "N/A"
    }

    print("\n" + "=" * 70)
    print(f"OVERALL SPI ACCURACY: {total_correct}/{total} ({100*total_correct/total:.1f}%)")
    print("=" * 70)

    # Key finding
    print("\nKEY FINDING:")
    if low_h:
        low_h_acc = sum(1 for r in low_h if r['prediction_correct'] == 'True') / len(low_h)
        if low_h_acc < 0.5:
            print(f"  SPI fails in low-h region ({100*low_h_acc:.0f}% accuracy)")
            print("  This confirms the asymmetric behavior observed in real data")
        else:
            print(f"  SPI works in low-h region ({100*low_h_acc:.0f}% accuracy)")
            print("  Synthetic data shows symmetric U-shape")

    # Save results
    output_path = Path(__file__).parent / 'spi_failure_phase_diagram_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    results = run_phase_diagram()
