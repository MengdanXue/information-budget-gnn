"""
Compute Aggregation Dilution metric (δ_agg) for FSD framework extension.

δ_agg measures how much information is "diluted" during neighborhood aggregation,
especially for high-degree nodes.
"""

import numpy as np
import pickle
from scipy import sparse

def load_data(data_path):
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def compute_aggregation_dilution(edge_index, features, n_nodes):
    """
    Compute aggregation dilution metric.

    δ_agg = E[d_i · (1 - cos_sim(x_i, mean(x_neighbors)))]

    High δ_agg indicates that mean aggregation causes significant information loss.
    """
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    features_norm = features / norms

    # Build adjacency list
    adj_list = [[] for _ in range(n_nodes)]
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj_list[src].append(dst)

    # Compute dilution for each node
    dilutions = []
    degrees = []
    sims_to_agg = []

    for i in range(n_nodes):
        neighbors = adj_list[i]
        if len(neighbors) == 0:
            continue

        degree = len(neighbors)
        degrees.append(degree)

        # Mean aggregated neighbor features
        neighbor_features = features_norm[neighbors]
        mean_neighbor = np.mean(neighbor_features, axis=0)
        mean_neighbor_norm = mean_neighbor / (np.linalg.norm(mean_neighbor) + 1e-8)

        # Similarity between node and its aggregated neighbors
        sim = np.dot(features_norm[i], mean_neighbor_norm)
        sims_to_agg.append(sim)

        # Dilution weighted by degree
        dilution = degree * (1 - sim)
        dilutions.append(dilution)

    dilutions = np.array(dilutions)
    degrees = np.array(degrees)
    sims_to_agg = np.array(sims_to_agg)

    # Analysis by degree groups
    degree_thresholds = [10, 50, 100, 200]
    results = {
        'overall': {
            'mean_dilution': np.mean(dilutions),
            'std_dilution': np.std(dilutions),
            'mean_sim_to_agg': np.mean(sims_to_agg),
        }
    }

    prev_thresh = 0
    for thresh in degree_thresholds:
        mask = (degrees > prev_thresh) & (degrees <= thresh)
        if np.sum(mask) > 0:
            results[f'degree_{prev_thresh+1}_{thresh}'] = {
                'count': np.sum(mask),
                'mean_dilution': np.mean(dilutions[mask]),
                'mean_sim_to_agg': np.mean(sims_to_agg[mask]),
            }
        prev_thresh = thresh

    # High degree nodes (> 200)
    mask = degrees > 200
    if np.sum(mask) > 0:
        results['degree_200+'] = {
            'count': np.sum(mask),
            'mean_dilution': np.mean(dilutions[mask]),
            'mean_sim_to_agg': np.mean(sims_to_agg[mask]),
        }

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                       default='./processed/ieee_cis_graph.pkl')
    args = parser.parse_args()

    print("=" * 60)
    print("Aggregation Dilution Analysis (δ_agg)")
    print("=" * 60)

    data = load_data(args.data_path)
    features = data['features']
    edge_index = data['edge_index']
    n_nodes = features.shape[0]

    results = compute_aggregation_dilution(edge_index, features, n_nodes)

    print(f"\nOverall Statistics:")
    print(f"  Mean δ_agg: {results['overall']['mean_dilution']:.4f}")
    print(f"  Mean sim(node, agg_neighbors): {results['overall']['mean_sim_to_agg']:.4f}")

    print(f"\nBy Degree Group:")
    print(f"{'Degree Range':<15} {'Count':>10} {'Mean δ_agg':>12} {'Mean Sim':>10}")
    print("-" * 50)

    for key, val in results.items():
        if key != 'overall':
            print(f"{key:<15} {val['count']:>10} {val['mean_dilution']:>12.2f} {val['mean_sim_to_agg']:>10.4f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if results['overall']['mean_sim_to_agg'] < 0.7:
        print("\nLow similarity to aggregated neighbors detected!")
        print("→ Mean aggregation causes significant information loss")
        print("→ GraphSAGE (sampling) or H2GCN/MixHop (concatenation) recommended")

    # Check if high-degree nodes have worse dilution
    if 'degree_200+' in results:
        high_deg_sim = results['degree_200+']['mean_sim_to_agg']
        low_deg_sim = results.get('degree_1_10', {}).get('mean_sim_to_agg', 0.8)
        if high_deg_sim < low_deg_sim - 0.1:
            print(f"\nHigh-degree nodes show worse aggregation quality!")
            print(f"  Low degree sim: {low_deg_sim:.4f}")
            print(f"  High degree sim: {high_deg_sim:.4f}")
            print("→ This explains why sampling/concatenation methods work better")

if __name__ == '__main__':
    main()
