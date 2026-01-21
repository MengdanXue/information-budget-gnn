"""
Feature Centering Experiment: The "Smoking Gun" Test for FSA Hypothesis

Purpose: Test if centering features (allowing negative values) changes:
1. The neighbor type spectrum (opposite/orthogonal/similar ratios)
2. GNN performance on heterophilic datasets

Hypothesis:
- Original BoW/TF-IDF features are non-negative -> cosine similarity >= 0 -> no "opposite" neighbors
- Centered features can have negative values -> cosine similarity can be < 0 -> "opposite" neighbors possible
- If centering creates "opposite" neighbors AND improves GCN performance, FSA hypothesis is confirmed

This is a key experiment for validating the FSA hypothesis.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

try:
    from torch_geometric.datasets import WebKB, WikipediaNetwork, Planetoid
    from torch_geometric.nn import GCNConv, SAGEConv
    from torch_geometric.utils import to_undirected
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("PyTorch Geometric not available")


class GCN(nn.Module):
    """Simple 2-layer GCN."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MLP(nn.Module):
    """Simple 2-layer MLP (no graph structure)."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class GraphSAGE(nn.Module):
    """Simple 2-layer GraphSAGE with concat aggregation."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def center_features(features, method='global'):
    """
    Center features to allow negative values.

    Args:
        features: Node feature matrix (N x D)
        method: 'global' (subtract global mean) or 'per_dim' (subtract per-dimension mean)

    Returns:
        Centered features
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    if method == 'global':
        # Subtract global mean (single value)
        centered = features - features.mean()
    elif method == 'per_dim':
        # Subtract per-dimension mean (column-wise)
        centered = features - features.mean(axis=0, keepdims=True)
    elif method == 'per_node':
        # Subtract per-node mean (row-wise) - like z-scoring each node
        centered = features - features.mean(axis=1, keepdims=True)
    elif method == 'standardize':
        # Full standardization (zero mean, unit variance per dimension)
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        centered = (features - mean) / std
    else:
        raise ValueError(f"Unknown centering method: {method}")

    return torch.tensor(centered, dtype=torch.float32)


def compute_neighbor_spectrum(features, edge_index, labels):
    """
    Compute the neighbor type spectrum (opposite/orthogonal/similar ratios).

    Returns:
        dict with fractions and statistics
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    src, dst = edge_index[0], edge_index[1]

    # Get different-class edges only
    diff_class_mask = labels[src] != labels[dst]
    src_diff = src[diff_class_mask]
    dst_diff = dst[diff_class_mask]

    if len(src_diff) == 0:
        return {'opposite_frac': 0, 'orthogonal_frac': 0, 'similar_frac': 0,
                'mean_sim': 0, 'min_sim': 0, 'max_sim': 0, 'n_negative': 0}

    # Sample if too many
    if len(src_diff) > 50000:
        idx = np.random.choice(len(src_diff), 50000, replace=False)
        src_diff, dst_diff = src_diff[idx], dst_diff[idx]

    # Normalize features for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features_norm = features / norms

    # Compute cosine similarities for different-class edges
    sims = np.sum(features_norm[src_diff] * features_norm[dst_diff], axis=1)

    # Categorize (using threshold of 0.1)
    opposite_frac = (sims < -0.1).mean()
    orthogonal_frac = (np.abs(sims) <= 0.1).mean()
    similar_frac = (sims > 0.1).mean()
    n_negative = (sims < 0).sum()

    return {
        'opposite_frac': float(opposite_frac),
        'orthogonal_frac': float(orthogonal_frac),
        'similar_frac': float(similar_frac),
        'mean_sim': float(sims.mean()),
        'std_sim': float(sims.std()),
        'min_sim': float(sims.min()),
        'max_sim': float(sims.max()),
        'n_negative': int(n_negative),
        'n_total': len(sims)
    }


def compute_homophily(edge_index, labels):
    """Compute edge homophily."""
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    src, dst = edge_index[0], edge_index[1]
    return float((labels[src] == labels[dst]).mean())


def train_and_evaluate(model, features, edge_index, labels, train_mask, val_mask, test_mask,
                       epochs=200, lr=0.01, weight_decay=5e-4, device='cpu'):
    """Train model and return test accuracy."""
    model = model.to(device)
    features = features.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0
    best_test_acc = 0
    patience = 50
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

            val_acc = (pred[val_mask] == labels[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    return best_test_acc


def run_experiment(dataset_name, data, n_runs=10, device='cpu'):
    """
    Run the feature centering experiment on a single dataset.

    Compares:
    - Original features vs Centered features
    - MLP vs GCN vs GraphSAGE
    """
    features_orig = data.x
    edge_index = data.edge_index
    labels = data.y

    # Ensure undirected
    edge_index = to_undirected(edge_index)

    n_nodes = features_orig.size(0)
    n_features = features_orig.size(1)
    n_classes = int(labels.max().item()) + 1

    # Compute homophily
    h = compute_homophily(edge_index, labels)

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Nodes: {n_nodes}, Features: {n_features}, Classes: {n_classes}")
    print(f"Homophily: {h:.3f}")
    print(f"{'='*70}")

    results = {
        'dataset': dataset_name,
        'n_nodes': n_nodes,
        'n_features': n_features,
        'n_classes': n_classes,
        'homophily': h,
        'experiments': {}
    }

    # Different feature configurations
    feature_configs = {
        'original': features_orig,
        'centered_per_dim': center_features(features_orig, method='per_dim'),
        'standardized': center_features(features_orig, method='standardize'),
    }

    for feat_name, features in feature_configs.items():
        print(f"\n--- Feature Type: {feat_name} ---")

        # Analyze neighbor spectrum
        spectrum = compute_neighbor_spectrum(features, edge_index, labels)

        print(f"  Neighbor Spectrum (different-class edges):")
        print(f"    Opposite (sim < -0.1):   {spectrum['opposite_frac']*100:.1f}%")
        print(f"    Orthogonal (|sim| <= 0.1): {spectrum['orthogonal_frac']*100:.1f}%")
        print(f"    Similar (sim > 0.1):     {spectrum['similar_frac']*100:.1f}%")
        print(f"    Mean similarity:         {spectrum['mean_sim']:.3f}")
        print(f"    Min/Max similarity:      [{spectrum['min_sim']:.3f}, {spectrum['max_sim']:.3f}]")
        print(f"    Negative similarities:   {spectrum['n_negative']}/{spectrum['n_total']}")

        # Train models
        model_results = {}

        for model_name, ModelClass in [('MLP', MLP), ('GCN', GCN), ('GraphSAGE', GraphSAGE)]:
            accs = []

            for run in range(n_runs):
                # Create random splits (60/20/20)
                perm = torch.randperm(n_nodes)
                train_size = int(0.6 * n_nodes)
                val_size = int(0.2 * n_nodes)

                train_mask = torch.zeros(n_nodes, dtype=torch.bool)
                val_mask = torch.zeros(n_nodes, dtype=torch.bool)
                test_mask = torch.zeros(n_nodes, dtype=torch.bool)

                train_mask[perm[:train_size]] = True
                val_mask[perm[train_size:train_size+val_size]] = True
                test_mask[perm[train_size+val_size:]] = True

                # Create and train model
                model = ModelClass(n_features, 64, n_classes)
                acc = train_and_evaluate(model, features, edge_index, labels,
                                        train_mask, val_mask, test_mask, device=device)
                accs.append(acc)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            model_results[model_name] = {'mean': mean_acc, 'std': std_acc, 'runs': accs}
            print(f"  {model_name}: {mean_acc*100:.1f}% +/- {std_acc*100:.1f}%")

        results['experiments'][feat_name] = {
            'spectrum': spectrum,
            'models': model_results
        }

    # Compute deltas (centered - original)
    print(f"\n--- Performance Change (Centered - Original) ---")

    for feat_name in ['centered_per_dim', 'standardized']:
        print(f"\n  {feat_name}:")
        for model_name in ['MLP', 'GCN', 'GraphSAGE']:
            orig_acc = results['experiments']['original']['models'][model_name]['mean']
            new_acc = results['experiments'][feat_name]['models'][model_name]['mean']
            delta = (new_acc - orig_acc) * 100
            direction = '+' if delta > 0 else ''
            print(f"    {model_name}: {direction}{delta:.1f}%")

    # Key question: Does centering help GCN more than MLP?
    print(f"\n--- KEY QUESTION: Does centering help GCN more than MLP? ---")

    for feat_name in ['centered_per_dim', 'standardized']:
        mlp_delta = (results['experiments'][feat_name]['models']['MLP']['mean'] -
                     results['experiments']['original']['models']['MLP']['mean'])
        gcn_delta = (results['experiments'][feat_name]['models']['GCN']['mean'] -
                     results['experiments']['original']['models']['GCN']['mean'])

        relative_gain = gcn_delta - mlp_delta

        print(f"  {feat_name}:")
        print(f"    MLP improvement:  {mlp_delta*100:+.1f}%")
        print(f"    GCN improvement:  {gcn_delta*100:+.1f}%")
        print(f"    GCN relative gain over MLP: {relative_gain*100:+.1f}%")

        if relative_gain > 0.02:
            print(f"    --> POSITIVE: Centering helps GCN more than MLP!")
        elif relative_gain < -0.02:
            print(f"    --> NEGATIVE: Centering helps MLP more than GCN")
        else:
            print(f"    --> NEUTRAL: Similar effect on both")

    return results


def main():
    """Run feature centering experiment on heterophilic datasets."""
    print("=" * 70)
    print("FEATURE CENTERING EXPERIMENT")
    print("Testing FSA Hypothesis: Does centering features change GNN performance?")
    print("=" * 70)

    if not HAS_PYG:
        print("PyTorch Geometric not available. Cannot run experiment.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    all_results = {
        'experiment': 'feature_centering',
        'hypothesis': 'Centering features creates opposite neighbors and improves GCN on low-h graphs',
        'datasets': []
    }

    # Test on heterophilic datasets
    datasets_to_test = []

    # WebKB datasets (low h)
    for name in ['Texas', 'Wisconsin', 'Cornell']:
        try:
            dataset = WebKB(root=str(data_dir), name=name)
            datasets_to_test.append((name, dataset[0]))
        except Exception as e:
            print(f"Could not load {name}: {e}")

    # Wikipedia datasets (low h)
    for name in ['chameleon', 'squirrel']:
        try:
            dataset = WikipediaNetwork(root=str(data_dir), name=name)
            datasets_to_test.append((name, dataset[0]))
        except Exception as e:
            print(f"Could not load {name}: {e}")

    # Also test on a homophilic dataset for comparison
    try:
        dataset = Planetoid(root=str(data_dir), name='Cora')
        datasets_to_test.append(('Cora', dataset[0]))
    except Exception as e:
        print(f"Could not load Cora: {e}")

    # Run experiments
    for name, data in datasets_to_test:
        try:
            result = run_experiment(name, data, n_runs=10, device=device)
            all_results['datasets'].append(result)
        except Exception as e:
            print(f"Error on {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: FSA HYPOTHESIS TEST")
    print("=" * 70)

    print("\n1. Did centering create 'opposite' neighbors?")
    for result in all_results['datasets']:
        name = result['dataset']
        h = result['homophily']

        orig_opposite = result['experiments']['original']['spectrum']['opposite_frac']
        cent_opposite = result['experiments']['centered_per_dim']['spectrum']['opposite_frac']

        print(f"  {name} (h={h:.2f}): {orig_opposite*100:.1f}% -> {cent_opposite*100:.1f}% opposite")

    print("\n2. Did GCN improve more than MLP after centering?")
    for result in all_results['datasets']:
        name = result['dataset']
        h = result['homophily']

        mlp_orig = result['experiments']['original']['models']['MLP']['mean']
        mlp_cent = result['experiments']['centered_per_dim']['models']['MLP']['mean']
        gcn_orig = result['experiments']['original']['models']['GCN']['mean']
        gcn_cent = result['experiments']['centered_per_dim']['models']['GCN']['mean']

        mlp_delta = mlp_cent - mlp_orig
        gcn_delta = gcn_cent - gcn_orig
        relative = gcn_delta - mlp_delta

        verdict = "YES" if relative > 0.02 else ("NO" if relative < -0.02 else "NEUTRAL")
        print(f"  {name} (h={h:.2f}): GCN relative gain = {relative*100:+.1f}% [{verdict}]")

    print("\n3. FSA Hypothesis Verdict:")

    low_h_results = [r for r in all_results['datasets'] if r['homophily'] < 0.5]
    if low_h_results:
        # Check if centering created more opposite neighbors
        opposite_increased = sum(
            1 for r in low_h_results
            if r['experiments']['centered_per_dim']['spectrum']['opposite_frac'] >
               r['experiments']['original']['spectrum']['opposite_frac'] + 0.05
        )

        # Check if GCN improved relatively
        gcn_improved = sum(
            1 for r in low_h_results
            if (r['experiments']['centered_per_dim']['models']['GCN']['mean'] -
                r['experiments']['original']['models']['GCN']['mean']) >
               (r['experiments']['centered_per_dim']['models']['MLP']['mean'] -
                r['experiments']['original']['models']['MLP']['mean']) + 0.02
        )

        print(f"  Opposite neighbors increased: {opposite_increased}/{len(low_h_results)} datasets")
        print(f"  GCN relatively improved: {gcn_improved}/{len(low_h_results)} datasets")

        if opposite_increased >= len(low_h_results) // 2 and gcn_improved >= len(low_h_results) // 2:
            print("\n  HYPOTHESIS SUPPORTED: Centering creates opposite neighbors AND helps GCN")
            all_results['verdict'] = 'SUPPORTED'
        elif opposite_increased >= len(low_h_results) // 2:
            print("\n  PARTIAL: Centering creates opposite neighbors, but GCN doesn't improve")
            print("  This suggests other factors beyond FSA affect GNN performance")
            all_results['verdict'] = 'PARTIAL - opposite created but GCN not improved'
        else:
            print("\n  HYPOTHESIS NOT SUPPORTED: Centering doesn't significantly change neighbor spectrum")
            all_results['verdict'] = 'NOT SUPPORTED'

    # Save results
    output_path = Path(__file__).parent / 'feature_centering_results.json'

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        return obj

    all_results = convert_to_native(all_results)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == '__main__':
    results = main()
