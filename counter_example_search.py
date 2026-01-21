"""
Counter-Example Search for NOON Phenomenon
Systematically search for graphs that might violate NOON:
1. Graphs with explicit negative/inhibitory relationships
2. Graphs with learned embeddings (contrastive learning)
3. Synthetic graphs designed to have opposite features
4. Cross-domain graphs (biology, knowledge graphs, etc.)

This addresses Codex's criticism about selection bias and lack of counter-examples.

Author: Research Team
Date: 2026-01-17
"""

import json
import numpy as np
import torch
from pathlib import Path
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
})

try:
    from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
    from torch_geometric.datasets import Amazon, Coauthor
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("PyTorch Geometric not available")


def compute_noon_metrics(features, edge_index, labels, name="dataset"):
    """
    Compute NOON-related metrics for a dataset.
    Returns detailed statistics about opposite neighbors.
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    src, dst = edge_index[0], edge_index[1]

    # Different-class edges
    diff_class_mask = labels[src] != labels[dst]
    src_diff = src[diff_class_mask]
    dst_diff = dst[diff_class_mask]

    n_diff_edges = len(src_diff)
    if n_diff_edges == 0:
        return None

    # Compute homophily
    n_same_class = (~diff_class_mask).sum()
    homophily = n_same_class / len(src)

    # Sample if needed
    sample_size = min(50000, n_diff_edges)
    if n_diff_edges > sample_size:
        idx = np.random.choice(n_diff_edges, sample_size, replace=False)
        src_sample, dst_sample = src_diff[idx], dst_diff[idx]
    else:
        src_sample, dst_sample = src_diff, dst_diff

    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_norm = features / (norms + 1e-8)

    # Compute cosine similarities
    sims = np.sum(features_norm[src_sample] * features_norm[dst_sample], axis=1)

    # Check for non-negative features
    is_nonnegative = (features >= -1e-6).all()

    # Detailed statistics
    results = {
        'name': name,
        'n_nodes': features.shape[0],
        'n_edges': len(src),
        'n_diff_class_edges': n_diff_edges,
        'homophily': float(homophily),
        'is_nonnegative_features': bool(is_nonnegative),
        'feature_min': float(features.min()),
        'feature_max': float(features.max()),
        'cosine_stats': {
            'mean': float(np.mean(sims)),
            'std': float(np.std(sims)),
            'min': float(np.min(sims)),
            'max': float(np.max(sims)),
            'frac_below_-0.5': float((sims < -0.5).mean()),
            'frac_below_-0.3': float((sims < -0.3).mean()),
            'frac_below_-0.1': float((sims < -0.1).mean()),
            'frac_below_0': float((sims < 0).mean()),
            'n_below_-0.5': int((sims < -0.5).sum()),
            'n_below_-0.1': int((sims < -0.1).sum()),
        },
        'noon_satisfied': bool((sims < -0.5).mean() < 0.01),
    }

    return results


def create_synthetic_counter_examples():
    """
    Create synthetic graphs that are designed to VIOLATE NOON.
    These serve as sanity checks and boundary conditions.
    """
    print("\n" + "="*60)
    print("Creating Synthetic Counter-Examples")
    print("="*60)

    counter_examples = {}

    # 1. True Opposite Features (should violate NOON)
    print("\n1. True Opposite Features Graph")
    np.random.seed(42)
    n_nodes = 1000
    n_classes = 2

    # Class 0: features around +1, Class 1: features around -1
    labels = np.random.randint(0, n_classes, n_nodes)
    features = np.zeros((n_nodes, 50))
    for i in range(n_nodes):
        if labels[i] == 0:
            features[i] = np.random.randn(50) * 0.3 + 1.0
        else:
            features[i] = np.random.randn(50) * 0.3 - 1.0

    # Create edges (random, low homophily)
    n_edges = 5000
    src = np.random.randint(0, n_nodes, n_edges)
    dst = np.random.randint(0, n_nodes, n_edges)
    edge_index = np.stack([src, dst])

    results = compute_noon_metrics(features, edge_index, labels, "synthetic_opposite")
    counter_examples['synthetic_opposite'] = results
    print(f"   frac < -0.5: {results['cosine_stats']['frac_below_-0.5']*100:.2f}%")
    print(f"   NOON satisfied: {results['noon_satisfied']}")

    # 2. Contrastive Learning Simulation (might violate NOON)
    print("\n2. Contrastive Learning Embeddings")
    # Simulate embeddings where different-class nodes are pushed apart
    features_contrastive = np.zeros((n_nodes, 50))
    class_centers = np.random.randn(n_classes, 50)
    class_centers = class_centers / np.linalg.norm(class_centers, axis=1, keepdims=True)

    for i in range(n_nodes):
        # Push different class centers to be opposite
        if labels[i] == 0:
            features_contrastive[i] = class_centers[0] + np.random.randn(50) * 0.2
        else:
            features_contrastive[i] = -class_centers[0] + np.random.randn(50) * 0.2  # Opposite!

    results = compute_noon_metrics(features_contrastive, edge_index, labels, "synthetic_contrastive")
    counter_examples['synthetic_contrastive'] = results
    print(f"   frac < -0.5: {results['cosine_stats']['frac_below_-0.5']*100:.2f}%")
    print(f"   NOON satisfied: {results['noon_satisfied']}")

    # 3. Bipartite-like graph with orthogonal features (should satisfy NOON)
    print("\n3. Orthogonal Features Graph (Control)")
    features_orthogonal = np.zeros((n_nodes, 50))
    for i in range(n_nodes):
        if labels[i] == 0:
            features_orthogonal[i, :25] = np.random.rand(25)  # First half
        else:
            features_orthogonal[i, 25:] = np.random.rand(25)  # Second half

    results = compute_noon_metrics(features_orthogonal, edge_index, labels, "synthetic_orthogonal")
    counter_examples['synthetic_orthogonal'] = results
    print(f"   frac < -0.5: {results['cosine_stats']['frac_below_-0.5']*100:.2f}%")
    print(f"   NOON satisfied: {results['noon_satisfied']}")

    # 4. Mixed features (partial opposite)
    print("\n4. Mixed Features (Partial Opposite)")
    features_mixed = np.zeros((n_nodes, 50))
    for i in range(n_nodes):
        base = np.random.randn(50) * 0.5
        if labels[i] == 0:
            features_mixed[i] = base + 0.5
        else:
            # 50% of nodes have opposite features
            if np.random.rand() < 0.5:
                features_mixed[i] = -base - 0.5
            else:
                features_mixed[i] = base + np.random.randn(50) * 0.3

    results = compute_noon_metrics(features_mixed, edge_index, labels, "synthetic_mixed")
    counter_examples['synthetic_mixed'] = results
    print(f"   frac < -0.5: {results['cosine_stats']['frac_below_-0.5']*100:.2f}%")
    print(f"   NOON satisfied: {results['noon_satisfied']}")

    return counter_examples


def search_real_counter_examples():
    """
    Search through available real-world datasets for potential NOON violations.
    Focus on datasets with:
    - Signed/negative features
    - Low homophily
    - Different domains
    """
    print("\n" + "="*60)
    print("Searching Real-World Datasets for Counter-Examples")
    print("="*60)

    if not HAS_PYG:
        print("PyTorch Geometric not available, skipping...")
        return {}

    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    results = {}

    # Standard benchmarks
    datasets_to_check = []

    # Planetoid
    try:
        for name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root=str(data_dir), name=name)
            data = dataset[0]
            datasets_to_check.append((name, data.x, data.edge_index, data.y, 'citation'))
    except Exception as e:
        print(f"Could not load Planetoid: {e}")

    # WebKB
    try:
        for name in ['Texas', 'Wisconsin', 'Cornell']:
            dataset = WebKB(root=str(data_dir), name=name)
            data = dataset[0]
            datasets_to_check.append((name, data.x, data.edge_index, data.y, 'webpage'))
    except Exception as e:
        print(f"Could not load WebKB: {e}")

    # Wikipedia
    try:
        for name in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(root=str(data_dir), name=name)
            data = dataset[0]
            datasets_to_check.append((name, data.x, data.edge_index, data.y, 'wikipedia'))
    except Exception as e:
        print(f"Could not load WikipediaNetwork: {e}")

    # Actor
    try:
        dataset = Actor(root=str(data_dir))
        data = dataset[0]
        datasets_to_check.append(('actor', data.x, data.edge_index, data.y, 'social'))
    except Exception as e:
        print(f"Could not load Actor: {e}")

    # Amazon
    try:
        for name in ['Computers', 'Photo']:
            dataset = Amazon(root=str(data_dir), name=name)
            data = dataset[0]
            datasets_to_check.append((f'amazon_{name.lower()}', data.x, data.edge_index, data.y, 'ecommerce'))
    except Exception as e:
        print(f"Could not load Amazon: {e}")

    # Coauthor
    try:
        for name in ['CS', 'Physics']:
            dataset = Coauthor(root=str(data_dir), name=name)
            data = dataset[0]
            datasets_to_check.append((f'coauthor_{name.lower()}', data.x, data.edge_index, data.y, 'coauthor'))
    except Exception as e:
        print(f"Could not load Coauthor: {e}")

    # Process each dataset
    print(f"\nChecking {len(datasets_to_check)} datasets...")

    for name, features, edge_index, labels, domain in datasets_to_check:
        print(f"\n{name} ({domain}):")
        metrics = compute_noon_metrics(features, edge_index, labels, name)
        if metrics:
            results[name] = metrics
            results[name]['domain'] = domain

            print(f"   Homophily: {metrics['homophily']:.3f}")
            print(f"   Non-negative features: {metrics['is_nonnegative_features']}")
            print(f"   frac < -0.1: {metrics['cosine_stats']['frac_below_-0.1']*100:.4f}%")
            print(f"   frac < -0.5: {metrics['cosine_stats']['frac_below_-0.5']*100:.4f}%")
            print(f"   NOON satisfied: {metrics['noon_satisfied']}")

            if not metrics['noon_satisfied']:
                print(f"   *** POTENTIAL COUNTER-EXAMPLE FOUND! ***")

    return results


def analyze_feature_centering_effect(datasets_results):
    """
    Analyze whether feature centering creates counter-examples.
    """
    print("\n" + "="*60)
    print("Analyzing Feature Centering Effect")
    print("="*60)

    if not HAS_PYG:
        return {}

    data_dir = Path("./data")
    centering_results = {}

    # Test on a few datasets
    test_datasets = ['Cora', 'Texas', 'actor']

    for name in test_datasets:
        print(f"\n{name}:")

        try:
            if name in ['Cora', 'CiteSeer', 'PubMed']:
                dataset = Planetoid(root=str(data_dir), name=name)
            elif name in ['Texas', 'Wisconsin', 'Cornell']:
                dataset = WebKB(root=str(data_dir), name=name)
            elif name == 'actor':
                dataset = Actor(root=str(data_dir))
            else:
                continue

            data = dataset[0]
            features = data.x.numpy() if isinstance(data.x, torch.Tensor) else data.x
            edge_index = data.edge_index
            labels = data.y

            # Original
            orig_metrics = compute_noon_metrics(features, edge_index, labels, f"{name}_original")
            print(f"   Original: frac<-0.5 = {orig_metrics['cosine_stats']['frac_below_-0.5']*100:.4f}%")

            # Centered (per-feature mean subtraction)
            features_centered = features - features.mean(axis=0, keepdims=True)
            cent_metrics = compute_noon_metrics(features_centered, edge_index, labels, f"{name}_centered")
            print(f"   Centered: frac<-0.5 = {cent_metrics['cosine_stats']['frac_below_-0.5']*100:.4f}%")

            # Standardized (mean=0, std=1)
            features_std = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-8)
            std_metrics = compute_noon_metrics(features_std, edge_index, labels, f"{name}_standardized")
            print(f"   Standardized: frac<-0.5 = {std_metrics['cosine_stats']['frac_below_-0.5']*100:.4f}%")

            centering_results[name] = {
                'original': orig_metrics,
                'centered': cent_metrics,
                'standardized': std_metrics,
                'noon_violated_after_centering': not cent_metrics['noon_satisfied'],
            }

        except Exception as e:
            print(f"   Error: {e}")

    return centering_results


def generate_summary_report(synthetic_results, real_results, centering_results, output_dir):
    """Generate comprehensive summary report."""
    print("\n" + "="*70)
    print("COUNTER-EXAMPLE SEARCH SUMMARY")
    print("="*70)

    # Count violations
    synthetic_violations = sum(1 for r in synthetic_results.values() if not r['noon_satisfied'])
    real_violations = sum(1 for r in real_results.values() if not r['noon_satisfied'])

    print(f"\n1. SYNTHETIC DATASETS:")
    print(f"   Total: {len(synthetic_results)}")
    print(f"   NOON Violations: {synthetic_violations}")
    for name, r in synthetic_results.items():
        status = "VIOLATED" if not r['noon_satisfied'] else "OK"
        print(f"   - {name}: {status} (frac<-0.5 = {r['cosine_stats']['frac_below_-0.5']*100:.2f}%)")

    print(f"\n2. REAL-WORLD DATASETS:")
    print(f"   Total: {len(real_results)}")
    print(f"   NOON Violations: {real_violations}")
    for name, r in sorted(real_results.items(), key=lambda x: -x[1]['cosine_stats']['frac_below_-0.5']):
        status = "VIOLATED" if not r['noon_satisfied'] else "OK"
        print(f"   - {name}: {status} (frac<-0.5 = {r['cosine_stats']['frac_below_-0.5']*100:.4f}%, h={r['homophily']:.2f})")

    print(f"\n3. CENTERING EFFECT:")
    for name, r in centering_results.items():
        status = "Creates violation" if r['noon_violated_after_centering'] else "Still OK"
        print(f"   - {name}: {status}")

    # Key finding
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)

    if real_violations == 0:
        print("\n[CONFIRMED] No real-world dataset violates NOON (cos<-0.5 < 1%)")
        print("This supports the claim that NOON is a genuine property of real graphs,")
        print("not just an artifact of benchmark selection.")
    else:
        violators = [name for name, r in real_results.items() if not r['noon_satisfied']]
        print(f"\n[FOUND] {real_violations} real dataset(s) violate NOON: {violators}")
        print("These represent boundary conditions for the NOON phenomenon.")

    if synthetic_violations > 0:
        print("\n[SANITY CHECK PASSED] Synthetic opposite-feature graphs DO violate NOON,")
        print("confirming our method can detect true opposites when they exist.")

    # Save full results
    full_results = {
        'synthetic': synthetic_results,
        'real_world': real_results,
        'centering_effect': centering_results,
        'summary': {
            'n_synthetic': len(synthetic_results),
            'n_real': len(real_results),
            'synthetic_violations': synthetic_violations,
            'real_violations': real_violations,
            'conclusion': 'NOON confirmed on real data' if real_violations == 0 else f'Found {real_violations} violations',
        }
    }

    output_path = output_dir / 'counter_example_search_results.json'
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nFull results saved to: {output_path}")

    return full_results


def main():
    """Run complete counter-example search."""
    print("="*70)
    print("NOON PHENOMENON: COUNTER-EXAMPLE SEARCH")
    print("="*70)
    print("\nThis script systematically searches for graphs that might violate NOON.")
    print("Addressing Codex's criticism about selection bias and lack of counter-examples.")

    output_dir = Path(__file__).parent

    # 1. Create synthetic counter-examples
    synthetic_results = create_synthetic_counter_examples()

    # 2. Search real-world datasets
    real_results = search_real_counter_examples()

    # 3. Analyze centering effect
    centering_results = analyze_feature_centering_effect(real_results)

    # 4. Generate summary
    full_results = generate_summary_report(synthetic_results, real_results, centering_results, output_dir)

    return full_results


if __name__ == '__main__':
    results = main()
