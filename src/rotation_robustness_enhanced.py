"""
Enhanced Random Rotation Robustness Test for NOON Phenomenon
Addresses Codex's criticism:
1. Add confidence intervals (Clopper-Pearson)
2. Increase random seeds (20 rotations)
3. Generate error bar plots
4. Report total pair counts
5. Full distribution curves (not just thresholds)

Author: Research Team
Date: 2026-01-17
"""

import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import ortho_group, sem, t
from scipy.stats import binom
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

try:
    from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("PyTorch Geometric not available")


def clopper_pearson_ci(k, n, alpha=0.05):
    """
    Compute Clopper-Pearson exact confidence interval.
    k: number of successes
    n: total trials
    alpha: significance level (default 0.05 for 95% CI)
    Returns: (lower, upper) bounds
    """
    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        lower = 0.0
    else:
        lower = binom.ppf(alpha/2, n, k/n) / n
        # More accurate: use beta distribution
        from scipy.stats import beta
        lower = beta.ppf(alpha/2, k, n-k+1)
    if k == n:
        upper = 1.0
    else:
        from scipy.stats import beta
        upper = beta.ppf(1-alpha/2, k+1, n-k)
    return (float(lower), float(upper))


def generate_random_rotation(dim, seed=None):
    """Generate a random orthogonal rotation matrix from Haar measure."""
    if seed is not None:
        np.random.seed(seed)
    return ortho_group.rvs(dim)


def apply_rotation(features, rotation_matrix):
    """Apply rotation to features."""
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    return features @ rotation_matrix


def compute_cosine_stats_detailed(features, edge_index, labels, sample_size=50000):
    """
    Compute detailed cosine similarity statistics for different-class neighbors.
    Returns full histogram for distribution analysis.
    """
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

    total_pairs = len(src_diff)
    if total_pairs == 0:
        return None

    # Sample if too many
    if total_pairs > sample_size:
        idx = np.random.choice(total_pairs, sample_size, replace=False)
        src_diff, dst_diff = src_diff[idx], dst_diff[idx]
        sampled_pairs = sample_size
    else:
        sampled_pairs = total_pairs

    # Normalize and compute similarities
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sims = np.sum(features_norm[src_diff] * features_norm[dst_diff], axis=1)

    # Counts for CI computation
    n_below_05 = int((sims < -0.5).sum())
    n_below_03 = int((sims < -0.3).sum())
    n_below_01 = int((sims < -0.1).sum())

    # Compute Clopper-Pearson CIs
    ci_05 = clopper_pearson_ci(n_below_05, sampled_pairs)
    ci_03 = clopper_pearson_ci(n_below_03, sampled_pairs)
    ci_01 = clopper_pearson_ci(n_below_01, sampled_pairs)

    # Histogram for full distribution
    hist, bin_edges = np.histogram(sims, bins=50, range=(-1, 1))

    return {
        'total_pairs': total_pairs,
        'sampled_pairs': sampled_pairs,
        'n_below_-0.5': n_below_05,
        'n_below_-0.3': n_below_03,
        'n_below_-0.1': n_below_01,
        'frac_below_-0.5': float(n_below_05 / sampled_pairs),
        'frac_below_-0.3': float(n_below_03 / sampled_pairs),
        'frac_below_-0.1': float(n_below_01 / sampled_pairs),
        'ci_-0.5': ci_05,
        'ci_-0.3': ci_03,
        'ci_-0.1': ci_01,
        'mean': float(np.mean(sims)),
        'std': float(np.std(sims)),
        'min': float(np.min(sims)),
        'max': float(np.max(sims)),
        'percentiles': {
            '1%': float(np.percentile(sims, 1)),
            '5%': float(np.percentile(sims, 5)),
            '10%': float(np.percentile(sims, 10)),
            '25%': float(np.percentile(sims, 25)),
            '50%': float(np.percentile(sims, 50)),
        },
        'histogram': hist.tolist(),
        'bin_edges': bin_edges.tolist(),
    }


def load_datasets_extended():
    """Load extended benchmark datasets including more diverse graphs."""
    datasets = {}
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    if not HAS_PYG:
        return datasets

    # Planetoid (homophilic)
    try:
        for name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root=str(data_dir), name=name)
            data = dataset[0]
            datasets[name.lower()] = {
                'features': data.x,
                'edge_index': data.edge_index,
                'labels': data.y,
                'type': 'homophilic'
            }
    except Exception as e:
        print(f"Could not load Planetoid: {e}")

    # WebKB (heterophilic)
    try:
        for name in ['Texas', 'Wisconsin', 'Cornell']:
            dataset = WebKB(root=str(data_dir), name=name)
            data = dataset[0]
            datasets[name.lower()] = {
                'features': data.x,
                'edge_index': data.edge_index,
                'labels': data.y,
                'type': 'heterophilic'
            }
    except Exception as e:
        print(f"Could not load WebKB: {e}")

    # Wikipedia networks (heterophilic)
    try:
        for name in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(root=str(data_dir), name=name)
            data = dataset[0]
            datasets[name] = {
                'features': data.x,
                'edge_index': data.edge_index,
                'labels': data.y,
                'type': 'heterophilic'
            }
    except Exception as e:
        print(f"Could not load WikipediaNetwork: {e}")

    # Actor
    try:
        dataset = Actor(root=str(data_dir))
        data = dataset[0]
        datasets['actor'] = {
            'features': data.x,
            'edge_index': data.edge_index,
            'labels': data.y,
            'type': 'heterophilic'
        }
    except Exception as e:
        print(f"Could not load Actor: {e}")

    return datasets


def run_enhanced_rotation_test(n_rotations=20, random_seed_base=42):
    """
    Enhanced rotation robustness test with:
    - More rotations (20 by default)
    - Confidence intervals
    - Full distribution analysis
    - Error bar visualization
    """
    print("=" * 70)
    print("ENHANCED NOON Robustness Test: Random Orthogonal Rotations")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Number of rotations: {n_rotations}")
    print(f"  - Random seed base: {random_seed_base}")
    print(f"  - Confidence level: 95% (Clopper-Pearson)")

    results = {
        'experiment': 'enhanced_rotation_robustness',
        'n_rotations': n_rotations,
        'random_seed_base': random_seed_base,
        'confidence_level': 0.95,
        'datasets': {}
    }

    datasets = load_datasets_extended()

    if not datasets:
        print("No datasets available")
        return None

    print(f"\nLoaded {len(datasets)} datasets")

    for name, data in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name.upper()} ({data['type']})")
        print(f"{'='*60}")

        features = data['features']
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        edge_index = data['edge_index']
        labels = data['labels']

        n_nodes = features.shape[0]
        n_features = features.shape[1]
        print(f"Nodes: {n_nodes}, Features: {n_features}")

        # Original stats
        orig_stats = compute_cosine_stats_detailed(features, edge_index, labels)
        if orig_stats is None:
            print("  No different-class edges, skipping...")
            continue

        print(f"\nOriginal features:")
        print(f"  Total different-class pairs: {orig_stats['total_pairs']}")
        print(f"  Sampled pairs: {orig_stats['sampled_pairs']}")
        print(f"  frac < -0.1: {orig_stats['frac_below_-0.1']*100:.4f}% "
              f"(n={orig_stats['n_below_-0.1']}, 95% CI: [{orig_stats['ci_-0.1'][0]*100:.4f}%, {orig_stats['ci_-0.1'][1]*100:.4f}%])")
        print(f"  frac < -0.5: {orig_stats['frac_below_-0.5']*100:.4f}% "
              f"(n={orig_stats['n_below_-0.5']}, 95% CI: [{orig_stats['ci_-0.5'][0]*100:.4f}%, {orig_stats['ci_-0.5'][1]*100:.4f}%])")

        # PCA for high-dimensional features
        if n_features > 200:
            print(f"  (Using PCA to reduce to 100 dims for rotation)")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(100, n_nodes-1))
            features_reduced = pca.fit_transform(features)
            n_features_rot = features_reduced.shape[1]
        else:
            features_reduced = features
            n_features_rot = n_features

        # Run rotations
        rotated_stats = []
        print(f"\nRunning {n_rotations} random rotations...")

        for i in range(n_rotations):
            # Generate random rotation with different seed
            R = generate_random_rotation(n_features_rot, seed=random_seed_base + i)

            # Apply rotation
            features_rotated = apply_rotation(features_reduced, R)

            # Compute stats
            stats = compute_cosine_stats_detailed(features_rotated, edge_index, labels)
            rotated_stats.append(stats)

            if (i + 1) % 5 == 0:
                print(f"  Rotation {i+1}/{n_rotations}: "
                      f"frac<-0.1 = {stats['frac_below_-0.1']*100:.2f}%, "
                      f"frac<-0.5 = {stats['frac_below_-0.5']*100:.2f}%")

        # Aggregate statistics
        fracs_01 = [s['frac_below_-0.1'] for s in rotated_stats]
        fracs_05 = [s['frac_below_-0.5'] for s in rotated_stats]

        avg_01, std_01 = np.mean(fracs_01), np.std(fracs_01, ddof=1)
        avg_05, std_05 = np.mean(fracs_05), np.std(fracs_05, ddof=1)

        # 95% CI using t-distribution
        t_crit = t.ppf(0.975, n_rotations - 1)
        ci_01_low = avg_01 - t_crit * std_01 / np.sqrt(n_rotations)
        ci_01_high = avg_01 + t_crit * std_01 / np.sqrt(n_rotations)
        ci_05_low = avg_05 - t_crit * std_05 / np.sqrt(n_rotations)
        ci_05_high = avg_05 + t_crit * std_05 / np.sqrt(n_rotations)

        print(f"\nRotated features summary ({n_rotations} rotations):")
        print(f"  frac < -0.1: mean={avg_01*100:.2f}% +/- {std_01*100:.2f}% "
              f"(95% CI: [{max(0,ci_01_low)*100:.2f}%, {ci_01_high*100:.2f}%])")
        print(f"  frac < -0.5: mean={avg_05*100:.4f}% +/- {std_05*100:.4f}% "
              f"(95% CI: [{max(0,ci_05_low)*100:.4f}%, {ci_05_high*100:.4f}%])")
        print(f"  max frac < -0.5 across all rotations: {max(fracs_05)*100:.4f}%")

        # Verdict
        if max(fracs_05) < 0.01:
            verdict = "NOON CONFIRMED: Even after rotation, strongly opposite (<-0.5) remains <1%"
        elif avg_05 < 0.05:
            verdict = "NOON MOSTLY CONFIRMED: Rotation slightly increases negatives, but still rare"
        else:
            verdict = "CAUTION: Significant negative cosines after rotation"

        print(f"\n  Verdict: {verdict}")

        results['datasets'][name] = {
            'type': data['type'],
            'n_nodes': n_nodes,
            'n_features': n_features,
            'n_features_rotated': n_features_rot,
            'original': orig_stats,
            'rotated': {
                'n_rotations': n_rotations,
                'frac_-0.1': {
                    'mean': float(avg_01),
                    'std': float(std_01),
                    'ci_95': [float(max(0, ci_01_low)), float(ci_01_high)],
                    'all_values': [float(f) for f in fracs_01],
                },
                'frac_-0.5': {
                    'mean': float(avg_05),
                    'std': float(std_05),
                    'ci_95': [float(max(0, ci_05_low)), float(ci_05_high)],
                    'all_values': [float(f) for f in fracs_05],
                    'max': float(max(fracs_05)),
                },
            },
            'verdict': verdict,
        }

    # Overall conclusion
    print("\n" + "=" * 70)
    print("OVERALL CONCLUSION")
    print("=" * 70)

    all_max_neg05 = [
        results['datasets'][name]['rotated']['frac_-0.5']['max']
        for name in results['datasets']
    ]
    all_avg_neg05 = [
        results['datasets'][name]['rotated']['frac_-0.5']['mean']
        for name in results['datasets']
    ]

    if max(all_max_neg05) < 0.01:
        conclusion = "NOON is ROBUST: No dataset shows >1% strongly opposite (cos<-0.5) even after rotation"
    elif max(all_avg_neg05) < 0.05:
        conclusion = "NOON is MOSTLY ROBUST: Rotation increases negatives slightly, but still rare (<5%)"
    else:
        conclusion = "MIXED RESULTS: Some datasets show notable negatives after rotation"

    print(f"\n{conclusion}")
    print(f"\nStatistics across {len(results['datasets'])} datasets:")
    print(f"  Max of max(frac<-0.5): {max(all_max_neg05)*100:.4f}%")
    print(f"  Max of mean(frac<-0.5): {max(all_avg_neg05)*100:.4f}%")

    results['overall_conclusion'] = conclusion
    results['summary'] = {
        'n_datasets': len(results['datasets']),
        'max_of_max_frac_-0.5': float(max(all_max_neg05)),
        'max_of_mean_frac_-0.5': float(max(all_avg_neg05)),
    }

    # Save results
    output_dir = Path(__file__).parent
    output_path = output_dir / 'rotation_robustness_enhanced_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate visualization
    generate_error_bar_plot(results, output_dir)

    return results


def generate_error_bar_plot(results, output_dir):
    """Generate publication-quality error bar plot."""
    print("\nGenerating error bar visualization...")

    datasets = list(results['datasets'].keys())
    n_datasets = len(datasets)

    # Prepare data
    orig_fracs = []
    rot_means = []
    rot_stds = []
    rot_maxs = []
    types = []

    for name in datasets:
        data = results['datasets'][name]
        orig_fracs.append(data['original']['frac_below_-0.1'] * 100)
        rot_means.append(data['rotated']['frac_-0.1']['mean'] * 100)
        rot_stds.append(data['rotated']['frac_-0.1']['std'] * 100)
        rot_maxs.append(max(data['rotated']['frac_-0.1']['all_values']) * 100)
        types.append(data['type'])

    # Sort by type then by name
    sorted_idx = sorted(range(n_datasets), key=lambda i: (types[i], datasets[i]))
    datasets = [datasets[i] for i in sorted_idx]
    orig_fracs = [orig_fracs[i] for i in sorted_idx]
    rot_means = [rot_means[i] for i in sorted_idx]
    rot_stds = [rot_stds[i] for i in sorted_idx]
    rot_maxs = [rot_maxs[i] for i in sorted_idx]
    types = [types[i] for i in sorted_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(n_datasets)
    width = 0.35

    # Bars for original
    bars1 = ax.bar(x - width/2, orig_fracs, width, label='Original Features',
                   color='steelblue', alpha=0.8)

    # Bars for rotated with error bars
    bars2 = ax.bar(x + width/2, rot_means, width, label='After Rotation (mean)',
                   color='coral', alpha=0.8, yerr=rot_stds, capsize=3)

    # Add max markers
    ax.scatter(x + width/2, rot_maxs, marker='v', color='darkred', s=30,
               label='After Rotation (max)', zorder=5)

    # Formatting
    ax.set_ylabel('Fraction with cos < -0.1 (%)')
    ax.set_xlabel('Dataset')
    ax.set_title('NOON Robustness Test: Random Orthogonal Rotation\n'
                 f'(n={results["n_rotations"]} rotations, error bars = 1 std)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d.capitalize()}\n({t[0].upper()})'
                        for d, t in zip(datasets, types)], rotation=0)
    ax.legend(loc='upper right')

    # Add horizontal line at 10%
    ax.axhline(y=10, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(n_datasets-0.5, 10.5, '10% threshold', fontsize=8, color='gray')

    # Add type separators
    homophilic_end = sum(1 for t in types if t == 'homophilic') - 0.5
    ax.axvline(x=homophilic_end, color='black', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_ylim(0, max(max(rot_maxs), max(rot_means) + max(rot_stds)) * 1.2)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'figures' / 'noon_rotation_error_bars.pdf'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path)
    plt.savefig(output_dir / 'figures' / 'noon_rotation_error_bars.png')
    print(f"Saved: {output_path}")
    plt.close()

    # Also generate strongly opposite plot (cos < -0.5)
    fig, ax = plt.subplots(figsize=(12, 5))

    rot_means_05 = [results['datasets'][d]['rotated']['frac_-0.5']['mean'] * 100 for d in datasets]
    rot_stds_05 = [results['datasets'][d]['rotated']['frac_-0.5']['std'] * 100 for d in datasets]
    rot_maxs_05 = [results['datasets'][d]['rotated']['frac_-0.5']['max'] * 100 for d in datasets]
    orig_fracs_05 = [results['datasets'][d]['original']['frac_below_-0.5'] * 100 for d in datasets]

    bars1 = ax.bar(x - width/2, orig_fracs_05, width, label='Original Features',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, rot_means_05, width, label='After Rotation (mean)',
                   color='coral', alpha=0.8, yerr=rot_stds_05, capsize=3)
    ax.scatter(x + width/2, rot_maxs_05, marker='v', color='darkred', s=30,
               label='After Rotation (max)', zorder=5)

    ax.set_ylabel('Fraction with cos < -0.5 (%)')
    ax.set_xlabel('Dataset')
    ax.set_title('NOON Robustness: Strongly Opposite Neighbors (cos < -0.5)\n'
                 f'(n={results["n_rotations"]} rotations, error bars = 1 std)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d.capitalize()}\n({t[0].upper()})'
                        for d, t in zip(datasets, types)], rotation=0)
    ax.legend(loc='upper right')

    ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(n_datasets-0.5, 1.1, '1% threshold', fontsize=8, color='red')

    ax.axvline(x=homophilic_end, color='black', linestyle=':', linewidth=1, alpha=0.5)

    max_val = max(max(rot_maxs_05) if rot_maxs_05 else 0,
                  max(rot_means_05) + max(rot_stds_05) if rot_means_05 and rot_stds_05 else 0)
    ax.set_ylim(0, max(max_val * 1.5, 2))

    plt.tight_layout()

    output_path = output_dir / 'figures' / 'noon_rotation_strongly_opposite.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'figures' / 'noon_rotation_strongly_opposite.png')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    results = run_enhanced_rotation_test(n_rotations=20, random_seed_base=42)
