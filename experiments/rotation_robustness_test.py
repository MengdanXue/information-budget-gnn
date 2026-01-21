"""
Random Rotation Robustness Test for NOON Phenomenon
Tests whether NOON finding is robust to random orthogonal rotations of feature space.

Codex suggested: "Random orthogonal rotation test" to validate NOON is not an artifact.
"""

import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import ortho_group
import warnings
warnings.filterwarnings('ignore')

try:
    from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("PyTorch Geometric not available")


def generate_random_rotation(dim, seed=None):
    """Generate a random orthogonal rotation matrix."""
    if seed is not None:
        np.random.seed(seed)
    return ortho_group.rvs(dim)


def apply_rotation(features, rotation_matrix):
    """Apply rotation to features."""
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    return features @ rotation_matrix


def compute_cosine_stats(features, edge_index, labels, sample_size=50000):
    """Compute cosine similarity statistics for different-class neighbors."""
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    src, dst = edge_index[0], edge_index[1]
    diff_class_mask = labels[src] != labels[dst]
    src_diff = src[diff_class_mask]
    dst_diff = dst[diff_class_mask]

    if len(src_diff) == 0:
        return None

    # Sample if too many
    if len(src_diff) > sample_size:
        idx = np.random.choice(len(src_diff), sample_size, replace=False)
        src_diff, dst_diff = src_diff[idx], dst_diff[idx]

    # Normalize and compute similarities
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sims = np.sum(features_norm[src_diff] * features_norm[dst_diff], axis=1)

    return {
        'frac_below_-0.5': float((sims < -0.5).mean()),
        'frac_below_-0.3': float((sims < -0.3).mean()),
        'frac_below_-0.1': float((sims < -0.1).mean()),
        'mean': float(np.mean(sims)),
        'std': float(np.std(sims)),
        'min': float(np.min(sims)),
        'max': float(np.max(sims)),
    }


def load_datasets():
    """Load benchmark datasets."""
    datasets = {}
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    if not HAS_PYG:
        return datasets

    # Load a subset of datasets for efficiency
    try:
        for name in ['Cora', 'CiteSeer']:
            dataset = Planetoid(root=str(data_dir), name=name)
            data = dataset[0]
            datasets[name.lower()] = {
                'features': data.x,
                'edge_index': data.edge_index,
                'labels': data.y,
            }
    except Exception as e:
        print(f"Could not load Planetoid: {e}")

    try:
        for name in ['Texas', 'Wisconsin']:
            dataset = WebKB(root=str(data_dir), name=name)
            data = dataset[0]
            datasets[name.lower()] = {
                'features': data.x,
                'edge_index': data.edge_index,
                'labels': data.y,
            }
    except Exception as e:
        print(f"Could not load WebKB: {e}")

    return datasets


def run_rotation_robustness_test(n_rotations=10):
    """
    Test NOON robustness under random orthogonal rotations.

    Key insight: If NOON is due to non-negative features, random rotation
    (which removes this constraint) should significantly increase negative cosines.
    """
    print("=" * 70)
    print("NOON Robustness Test: Random Orthogonal Rotations")
    print("=" * 70)
    print(f"\nRunning {n_rotations} random rotations per dataset...")

    results = {
        'experiment': 'random_rotation_robustness',
        'n_rotations': n_rotations,
        'datasets': {}
    }

    datasets = load_datasets()

    if not datasets:
        print("No datasets available")
        return None

    for name, data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {name.upper()}")
        print(f"{'='*50}")

        features = data['features']
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        edge_index = data['edge_index']
        labels = data['labels']

        n_features = features.shape[1]
        print(f"Feature dimension: {n_features}")

        # Original stats
        orig_stats = compute_cosine_stats(features, edge_index, labels)
        print(f"\nOriginal features:")
        print(f"  frac < -0.1: {orig_stats['frac_below_-0.1']*100:.2f}%")
        print(f"  frac < -0.3: {orig_stats['frac_below_-0.3']*100:.2f}%")
        print(f"  frac < -0.5: {orig_stats['frac_below_-0.5']*100:.2f}%")

        # Rotated features
        rotated_stats = []

        print(f"\nRunning {n_rotations} random rotations...")

        # For high-dimensional features, use PCA first
        if n_features > 200:
            print(f"  (Using PCA to reduce to 100 dims for rotation)")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(100, features.shape[0]-1))
            features_reduced = pca.fit_transform(features)
            n_features_rot = features_reduced.shape[1]
        else:
            features_reduced = features
            n_features_rot = n_features

        for i in range(n_rotations):
            # Generate random rotation
            R = generate_random_rotation(n_features_rot, seed=42+i)

            # Apply rotation
            features_rotated = apply_rotation(features_reduced, R)

            # Compute stats
            stats = compute_cosine_stats(features_rotated, edge_index, labels)
            rotated_stats.append(stats)

            if i % 5 == 0:
                print(f"  Rotation {i+1}: frac<-0.1 = {stats['frac_below_-0.1']*100:.2f}%")

        # Aggregate rotated stats
        avg_frac_neg01 = np.mean([s['frac_below_-0.1'] for s in rotated_stats])
        std_frac_neg01 = np.std([s['frac_below_-0.1'] for s in rotated_stats])
        max_frac_neg01 = np.max([s['frac_below_-0.1'] for s in rotated_stats])

        avg_frac_neg05 = np.mean([s['frac_below_-0.5'] for s in rotated_stats])
        max_frac_neg05 = np.max([s['frac_below_-0.5'] for s in rotated_stats])

        print(f"\nRotated features summary ({n_rotations} rotations):")
        print(f"  frac < -0.1: avg={avg_frac_neg01*100:.2f}%, std={std_frac_neg01*100:.2f}%, max={max_frac_neg01*100:.2f}%")
        print(f"  frac < -0.5: avg={avg_frac_neg05*100:.2f}%, max={max_frac_neg05*100:.2f}%")

        # Statistical test: Is the increase significant?
        change = avg_frac_neg01 - orig_stats['frac_below_-0.1']
        print(f"\n  Change after rotation: {change*100:+.2f}%")

        if max_frac_neg05 < 0.01:
            verdict = "NOON CONFIRMED: Even after rotation, strongly opposite (<-0.5) neighbors are extremely rare"
        elif avg_frac_neg01 < 0.15:
            verdict = "NOON MOSTLY CONFIRMED: Rotation increases negative cosines slightly, but still low"
        else:
            verdict = "CAUTION: Significant negative cosines after rotation"

        print(f"  Verdict: {verdict}")

        results['datasets'][name] = {
            'n_features': n_features,
            'n_features_rotated': n_features_rot,
            'original': orig_stats,
            'rotated_avg': {
                'frac_below_-0.1': float(avg_frac_neg01),
                'frac_below_-0.1_std': float(std_frac_neg01),
                'frac_below_-0.5': float(avg_frac_neg05),
            },
            'rotated_max': {
                'frac_below_-0.1': float(max_frac_neg01),
                'frac_below_-0.5': float(max_frac_neg05),
            },
            'rotated_all': rotated_stats,
            'verdict': verdict,
        }

    # Overall conclusion
    print("\n" + "=" * 70)
    print("OVERALL CONCLUSION")
    print("=" * 70)

    all_max_neg05 = [
        results['datasets'][name]['rotated_max']['frac_below_-0.5']
        for name in results['datasets']
    ]

    if max(all_max_neg05) < 0.01:
        conclusion = "NOON is ROBUST to random rotation: No dataset shows >1% strongly opposite neighbors"
        print(f"[CONFIRMED] {conclusion}")
    elif max(all_max_neg05) < 0.05:
        conclusion = "NOON is MOSTLY ROBUST: Random rotation slightly increases negatives, but still rare"
        print(f"[CONFIRMED] {conclusion}")
    else:
        conclusion = "CAUTION: Some datasets show significant negatives after rotation"
        print(f"[WARNING] {conclusion}")

    results['conclusion'] = conclusion

    # Save results
    output_path = Path(__file__).parent / 'rotation_robustness_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    results = run_rotation_robustness_test(n_rotations=10)
