"""
CSBM Deviation Metrics: Quantify how real graphs deviate from CSBM assumptions
Computes: Clustering Coefficient, Degree Gini, Feature Gaussianity
"""

import torch
import numpy as np
import json
from datetime import datetime
from scipy import stats
from collections import defaultdict

try:
    from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Actor
    from torch_geometric.utils import to_networkx, degree
    import networkx as nx
    PYG_AVAILABLE = True
except ImportError:
    print("PyTorch Geometric not installed")
    PYG_AVAILABLE = False


# ============================================================
# CSBM Deviation Metrics
# ============================================================

def compute_clustering_coefficient(data):
    """
    Global clustering coefficient
    CSBM assumption: clustering -> 0 as n -> infinity
    Real graphs often have high clustering
    """
    try:
        G = to_networkx(data, to_undirected=True)
        clustering = nx.average_clustering(G)
        return clustering
    except Exception as e:
        print(f"Error computing clustering: {e}")
        # Manual computation for large graphs
        edge_index = data.edge_index.cpu().numpy()
        n_nodes = data.num_nodes

        # Build adjacency list
        adj = defaultdict(set)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            adj[u].add(v)
            adj[v].add(u)

        # Sample nodes for large graphs
        if n_nodes > 10000:
            sample_nodes = np.random.choice(n_nodes, min(5000, n_nodes), replace=False)
        else:
            sample_nodes = range(n_nodes)

        triangles = 0
        triples = 0

        for node in sample_nodes:
            neighbors = list(adj[node])
            k = len(neighbors)
            if k < 2:
                continue

            triples += k * (k - 1) / 2

            # Count triangles
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        triangles += 1

        if triples == 0:
            return 0.0
        return triangles / triples


def compute_degree_gini(data):
    """
    Gini coefficient of degree distribution
    CSBM assumption: Poisson degree distribution (low Gini ~0.3)
    Real graphs often have power-law (high Gini ~0.6-0.8)
    """
    edge_index = data.edge_index
    degrees = degree(edge_index[1], num_nodes=data.num_nodes).cpu().numpy()
    degrees = degrees[degrees > 0]  # Remove isolated nodes

    if len(degrees) == 0:
        return 0.0

    # Sort degrees
    sorted_degrees = np.sort(degrees)
    n = len(sorted_degrees)

    # Compute Gini coefficient
    cumulative = np.cumsum(sorted_degrees)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_degrees))) / (n * np.sum(sorted_degrees)) - (n + 1) / n

    return gini


def compute_feature_gaussianity(data, n_samples=1000):
    """
    Test if aggregated features follow Gaussian distribution
    CSBM assumption: features are Gaussian
    Uses Kolmogorov-Smirnov test

    Returns: average p-value (higher = more Gaussian-like)
    """
    x = data.x.cpu().numpy()

    # Sample features if too many dimensions
    n_features = x.shape[1]
    if n_features > 100:
        feature_idx = np.random.choice(n_features, 100, replace=False)
        x = x[:, feature_idx]

    # Sample nodes if too many
    n_nodes = x.shape[0]
    if n_nodes > n_samples:
        node_idx = np.random.choice(n_nodes, n_samples, replace=False)
        x = x[node_idx]

    # Test each feature for normality
    p_values = []
    for i in range(x.shape[1]):
        col = x[:, i]
        col = col[~np.isnan(col)]  # Remove NaN

        if len(col) < 10 or np.std(col) < 1e-10:
            continue

        # Standardize
        col = (col - np.mean(col)) / (np.std(col) + 1e-10)

        # K-S test against normal distribution
        try:
            _, p = stats.kstest(col, 'norm')
            p_values.append(p)
        except:
            continue

    if len(p_values) == 0:
        return 0.5

    # Return average p-value (higher = more Gaussian)
    # Also compute fraction of features passing test at alpha=0.05
    avg_p = np.mean(p_values)
    pass_rate = np.mean([p > 0.05 for p in p_values])

    # Combined score (weighted average)
    gaussianity_score = 0.5 * avg_p + 0.5 * pass_rate

    return gaussianity_score


def compute_edge_density_variance(data):
    """
    Variance in edge density across node classes
    CSBM assumption: uniform p_in and p_out for all class pairs
    Real graphs may have varying inter-class connectivity
    """
    y = data.y.cpu().numpy().flatten()
    edge_index = data.edge_index.cpu().numpy()

    classes = np.unique(y)
    n_classes = len(classes)

    if n_classes < 2:
        return 0.0

    # Count edges between each class pair
    class_edge_counts = defaultdict(int)
    class_sizes = {c: np.sum(y == c) for c in classes}

    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        cu, cv = y[u], y[v]
        key = (min(cu, cv), max(cu, cv))
        class_edge_counts[key] += 1

    # Compute edge densities
    densities = []
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i <= j:
                key = (min(ci, cj), max(ci, cj))
                edge_count = class_edge_counts.get(key, 0)

                if i == j:
                    # Within-class: n*(n-1)/2 possible edges
                    possible = class_sizes[ci] * (class_sizes[ci] - 1) / 2
                else:
                    # Between-class: n_i * n_j possible edges
                    possible = class_sizes[ci] * class_sizes[cj]

                if possible > 0:
                    density = edge_count / possible
                    densities.append(density)

    if len(densities) < 2:
        return 0.0

    # Return coefficient of variation (normalized variance)
    return np.std(densities) / (np.mean(densities) + 1e-10)


def compute_homophily(data):
    """Compute edge homophily"""
    edge_index = data.edge_index.cpu().numpy()
    y = data.y.cpu().numpy().flatten()

    same = 0
    total = edge_index.shape[1]

    for i in range(total):
        if y[edge_index[0, i]] == y[edge_index[1, i]]:
            same += 1

    return same / total


def compute_all_csbm_metrics(data, dataset_name="Unknown"):
    """Compute all CSBM deviation metrics for a dataset"""

    print(f"\n--- Computing CSBM deviation metrics for {dataset_name} ---")

    metrics = {
        'dataset': dataset_name,
        'n_nodes': data.num_nodes,
        'n_edges': data.edge_index.size(1),
        'n_features': data.x.size(1) if data.x is not None else 0,
        'n_classes': len(torch.unique(data.y)) if data.y is not None else 0,
    }

    # Homophily
    print("  Computing homophily...")
    metrics['homophily'] = compute_homophily(data)

    # Clustering coefficient
    print("  Computing clustering coefficient...")
    metrics['clustering_coefficient'] = compute_clustering_coefficient(data)

    # Degree Gini
    print("  Computing degree Gini...")
    metrics['degree_gini'] = compute_degree_gini(data)

    # Feature Gaussianity
    print("  Computing feature Gaussianity...")
    metrics['feature_gaussianity'] = compute_feature_gaussianity(data)

    # Edge density variance
    print("  Computing edge density variance...")
    metrics['edge_density_variance'] = compute_edge_density_variance(data)

    # Compute composite CSBM deviation score
    # Lower score = closer to CSBM assumptions
    # Weights based on importance
    deviation_components = {
        'clustering': metrics['clustering_coefficient'] / 0.3,  # CSBM expects ~0
        'degree_gini': max(0, metrics['degree_gini'] - 0.3) / 0.4,  # Poisson has Gini ~0.3
        'feature_non_gaussian': 1 - metrics['feature_gaussianity'],
        'edge_variance': metrics['edge_density_variance'],
    }

    metrics['deviation_components'] = deviation_components
    metrics['csbm_deviation_score'] = np.mean(list(deviation_components.values()))

    # Categorize deviation level
    score = metrics['csbm_deviation_score']
    if score < 0.2:
        metrics['deviation_level'] = 'Very Low'
    elif score < 0.35:
        metrics['deviation_level'] = 'Low'
    elif score < 0.5:
        metrics['deviation_level'] = 'Medium'
    else:
        metrics['deviation_level'] = 'High'

    print(f"  Clustering: {metrics['clustering_coefficient']:.3f}")
    print(f"  Degree Gini: {metrics['degree_gini']:.3f}")
    print(f"  Feature Gaussianity: {metrics['feature_gaussianity']:.3f}")
    print(f"  CSBM Deviation Score: {metrics['csbm_deviation_score']:.3f} ({metrics['deviation_level']})")

    return metrics


# ============================================================
# Load Datasets
# ============================================================

def load_datasets():
    """Load all benchmark datasets"""
    datasets = {}

    # Planetoid (Citation networks)
    print("Loading Planetoid datasets...")
    for name in ['Cora', 'CiteSeer', 'PubMed']:
        try:
            dataset = Planetoid(root='data/', name=name)
            datasets[name] = dataset[0]
        except Exception as e:
            print(f"  Error loading {name}: {e}")

    # WebKB (Web page networks)
    print("Loading WebKB datasets...")
    for name in ['Texas', 'Wisconsin', 'Cornell']:
        try:
            dataset = WebKB(root='data/', name=name)
            datasets[name] = dataset[0]
        except Exception as e:
            print(f"  Error loading {name}: {e}")

    # Wikipedia networks
    print("Loading Wikipedia datasets...")
    for name in ['chameleon', 'squirrel']:
        try:
            dataset = WikipediaNetwork(root='data/', name=name)
            datasets[name.capitalize()] = dataset[0]
        except Exception as e:
            print(f"  Error loading {name}: {e}")

    # Actor
    print("Loading Actor dataset...")
    try:
        dataset = Actor(root='data/')
        datasets['Actor'] = dataset[0]
    except Exception as e:
        print(f"  Error loading Actor: {e}")

    return datasets


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    if not PYG_AVAILABLE:
        print("PyTorch Geometric required")
        exit(1)

    print("="*60)
    print("CSBM Deviation Analysis")
    print("="*60)

    # Load datasets
    datasets = load_datasets()

    all_results = {
        'experiment': 'CSBM Deviation Analysis',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Quantifies how real graphs deviate from idealized CSBM assumptions',
        'metrics_explanation': {
            'clustering_coefficient': 'CSBM predicts 0; high values indicate triangle-rich graphs',
            'degree_gini': 'CSBM (Poisson) predicts ~0.3; higher indicates power-law-like',
            'feature_gaussianity': 'CSBM assumes Gaussian; score from K-S test (higher=more Gaussian)',
            'edge_density_variance': 'CSBM assumes uniform p_in/p_out; high variance violates this',
            'csbm_deviation_score': 'Composite score (lower = closer to CSBM)',
        },
        'datasets': {}
    }

    # Compute metrics for each dataset
    for name, data in datasets.items():
        try:
            metrics = compute_all_csbm_metrics(data, name)
            all_results['datasets'][name] = metrics
        except Exception as e:
            print(f"Error processing {name}: {e}")
            all_results['datasets'][name] = {'error': str(e)}

    # Save results
    output_file = 'csbm_deviation_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")

    # Summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Dataset':<15} {'h':<6} {'Clust.':<7} {'Gini':<6} {'Gauss.':<7} {'Dev.':<6} {'Level':<10}")
    print("-" * 65)

    for name, metrics in sorted(all_results['datasets'].items(), key=lambda x: x[1].get('csbm_deviation_score', 999)):
        if 'error' in metrics:
            print(f"{name:<15} ERROR")
            continue

        print(f"{name:<15} {metrics['homophily']:.3f}  {metrics['clustering_coefficient']:.3f}   "
              f"{metrics['degree_gini']:.3f}  {metrics['feature_gaussianity']:.3f}   "
              f"{metrics['csbm_deviation_score']:.3f}  {metrics['deviation_level']}")

    # Generate LaTeX table
    print("\n=== LATEX TABLE ===")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{Clustering} & \textbf{Degree Gini} & \textbf{Feat. Gauss.} & \textbf{CSBM Dev.} & \textbf{Level} \\")
    print(r"\midrule")

    for name, metrics in sorted(all_results['datasets'].items(), key=lambda x: x[1].get('csbm_deviation_score', 999)):
        if 'error' in metrics:
            continue

        print(f"{name} & {metrics['clustering_coefficient']:.2f} & {metrics['degree_gini']:.2f} & "
              f"{metrics['feature_gaussianity']:.2f} & {metrics['csbm_deviation_score']:.2f} & {metrics['deviation_level']} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
