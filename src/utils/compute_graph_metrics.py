"""
Compute extended graph metrics for FSD analysis.

This script computes:
1. Homophily (edge homophily ratio)
2. Multi-hop rho_FS (1-hop and 2-hop)
3. Degree distribution statistics
"""

import numpy as np
import pickle
from scipy import sparse
from collections import Counter

def load_data(data_path):
    """Load processed graph data."""
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def compute_homophily(edge_index, labels):
    """
    Compute edge homophily: fraction of edges connecting same-class nodes.
    """
    src, dst = edge_index[0], edge_index[1]
    same_label = labels[src] == labels[dst]
    homophily = np.mean(same_label)
    return homophily

def compute_degree_stats(edge_index, n_nodes):
    """Compute degree distribution statistics."""
    degrees = np.zeros(n_nodes)
    np.add.at(degrees, edge_index[0], 1)

    return {
        'mean': np.mean(degrees),
        'std': np.std(degrees),
        'median': np.median(degrees),
        'max': np.max(degrees),
        'min': np.min(degrees),
        'skewness': ((degrees - np.mean(degrees)) ** 3).mean() / (np.std(degrees) ** 3)
    }

def compute_rho_fs_multihop(edge_index, features, labels, n_nodes, max_hop=2, n_samples=50000):
    """
    Compute multi-hop rho_FS.

    rho_FS^(k) = mean_sim(k-hop neighbors) - mean_sim(non-neighbors)
    """
    # Build sparse adjacency matrix
    adj = sparse.csr_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(n_nodes, n_nodes)
    )

    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    features_norm = features / norms

    results = {}

    # Compute for each hop
    adj_power = adj.copy()
    for hop in range(1, max_hop + 1):
        if hop > 1:
            adj_power = adj_power @ adj

        # Sample edges at this hop distance
        rows, cols = adj_power.nonzero()

        # Remove self-loops and lower-hop connections
        if hop > 1:
            # For 2-hop, exclude 1-hop neighbors
            mask = np.array(adj[rows, cols]).flatten() == 0
            rows, cols = rows[mask], cols[mask]

        # Also exclude self-loops
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]

        if len(rows) == 0:
            results[f'rho_fs_{hop}hop'] = None
            continue

        # Sample if too many
        if len(rows) > n_samples:
            idx = np.random.choice(len(rows), n_samples, replace=False)
            rows, cols = rows[idx], cols[idx]

        # Compute similarities
        sims = np.sum(features_norm[rows] * features_norm[cols], axis=1)
        mean_hop_sim = np.mean(sims)

        # Compute same-class ratio at this hop
        same_class = labels[rows] == labels[cols]
        same_class_ratio = np.mean(same_class)

        results[f'rho_fs_{hop}hop'] = {
            'mean_sim': mean_hop_sim,
            'std_sim': np.std(sims),
            'same_class_ratio': same_class_ratio,
            'n_pairs': len(rows)
        }

    # Compute non-edge similarities
    edge_set = set(zip(edge_index[0], edge_index[1]))
    non_edge_sims = []
    attempts = 0

    while len(non_edge_sims) < n_samples and attempts < n_samples * 10:
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)
        if i != j and (i, j) not in edge_set:
            sim = np.dot(features_norm[i], features_norm[j])
            non_edge_sims.append(sim)
        attempts += 1

    non_edge_sims = np.array(non_edge_sims)
    results['non_edge'] = {
        'mean_sim': np.mean(non_edge_sims),
        'std_sim': np.std(non_edge_sims)
    }

    # Compute final rho_FS values
    for hop in range(1, max_hop + 1):
        key = f'rho_fs_{hop}hop'
        if results[key] is not None:
            results[f'rho_fs_{hop}hop_value'] = (
                results[key]['mean_sim'] - results['non_edge']['mean_sim']
            )

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                       default='./processed/ieee_cis_graph.pkl')
    args = parser.parse_args()

    print("=" * 60)
    print("Extended Graph Metrics for FSD Analysis")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    data = load_data(args.data_path)
    features = data['features']
    labels = data['labels']
    edge_index = data['edge_index']
    n_nodes = features.shape[0]

    print(f"Nodes: {n_nodes}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Features: {features.shape[1]}")

    # 1. Compute homophily
    print("\n" + "-" * 40)
    print("1. Homophily Analysis")
    print("-" * 40)
    homophily = compute_homophily(edge_index, labels)
    print(f"Edge Homophily: {homophily:.4f}")
    if homophily < 0.5:
        print("  -> Heterophilic graph (h < 0.5)")
    else:
        print("  -> Homophilic graph (h >= 0.5)")

    # 2. Degree statistics
    print("\n" + "-" * 40)
    print("2. Degree Distribution")
    print("-" * 40)
    deg_stats = compute_degree_stats(edge_index, n_nodes)
    print(f"Mean degree: {deg_stats['mean']:.2f}")
    print(f"Std degree: {deg_stats['std']:.2f}")
    print(f"Median degree: {deg_stats['median']:.2f}")
    print(f"Max degree: {deg_stats['max']:.0f}")
    print(f"Skewness: {deg_stats['skewness']:.2f}")

    # 3. Multi-hop rho_FS
    print("\n" + "-" * 40)
    print("3. Multi-hop rho_FS Analysis")
    print("-" * 40)
    rho_results = compute_rho_fs_multihop(edge_index, features, labels, n_nodes)

    print(f"\nNon-edge mean similarity: {rho_results['non_edge']['mean_sim']:.4f}")

    for hop in [1, 2]:
        key = f'rho_fs_{hop}hop'
        if rho_results[key] is not None:
            print(f"\n{hop}-hop neighbors:")
            print(f"  Mean similarity: {rho_results[key]['mean_sim']:.4f}")
            print(f"  Same-class ratio: {rho_results[key]['same_class_ratio']:.4f}")
            print(f"  rho_FS^({hop}): {rho_results[f'{key}_value']:.4f}")

    # 4. Analysis summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    rho_1 = rho_results.get('rho_fs_1hop_value', 0)
    rho_2 = rho_results.get('rho_fs_2hop_value', 0)

    print(f"\nrho_FS^(1) = {rho_1:.4f}")
    print(f"rho_FS^(2) = {rho_2:.4f}")
    print(f"Homophily h = {homophily:.4f}")

    print("\n** Interpretation **")
    if rho_2 > rho_1:
        print("rho_FS^(2) > rho_FS^(1): 2-hop neighbors are MORE similar than 1-hop")
        print("  -> Multi-hop methods (H2GCN, MixHop) should be beneficial")
    else:
        print("rho_FS^(2) <= rho_FS^(1): 2-hop neighbors are LESS similar than 1-hop")
        print("  -> Standard 1-hop methods might be sufficient")

    if homophily < 0.5:
        print(f"\nLow homophily ({homophily:.4f}): Graph is heterophilic")
        print("  -> H2GCN designed for heterophily should perform well")

    # Save results
    import json
    results = {
        'homophily': homophily,
        'degree_stats': deg_stats,
        'rho_fs_1hop': rho_1,
        'rho_fs_2hop': rho_2,
        'detailed_results': {
            k: v for k, v in rho_results.items()
            if isinstance(v, dict)
        }
    }

    with open('./processed/ieee_cis_extended_metrics.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print("\nResults saved to ./processed/ieee_cis_extended_metrics.json")

if __name__ == '__main__':
    main()
