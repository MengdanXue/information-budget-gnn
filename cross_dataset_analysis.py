"""
Cross-Dataset δ_agg Analysis for Multi-Dimensional FSD Framework

This script computes aggregation dilution (δ_agg) and other metrics
across multiple fraud detection datasets to validate the extended FSD framework.
"""

import numpy as np
import torch
import pickle
import scipy.io as sio
from scipy import sparse
import os

def compute_metrics(edge_index, features, labels, dataset_name):
    """Compute all FSD metrics for a dataset."""
    n_nodes = features.shape[0]
    n_features = features.shape[1]
    n_edges = edge_index.shape[1]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Nodes: {n_nodes:,}")
    print(f"Edges: {n_edges:,}")
    print(f"Features: {n_features}")

    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    features_norm = features / norms

    # 1. Degree statistics
    degrees = np.zeros(n_nodes)
    np.add.at(degrees, edge_index[0], 1)

    print(f"\n--- Degree Statistics ---")
    print(f"Mean degree: {np.mean(degrees):.2f}")
    print(f"Std degree: {np.std(degrees):.2f}")
    print(f"Max degree: {np.max(degrees):.0f}")
    print(f"Degree CV (std/mean): {np.std(degrees)/np.mean(degrees):.3f}")

    # 2. Homophily
    src, dst = edge_index[0], edge_index[1]
    same_label = labels[src] == labels[dst]
    homophily = np.mean(same_label)
    print(f"\n--- Homophily ---")
    print(f"Edge homophily h: {homophily:.4f}")

    # 3. rho_FS (1-hop)
    n_sample = min(100000, n_edges)
    idx = np.random.choice(n_edges, n_sample, replace=False)
    edge_sims = np.sum(features_norm[edge_index[0, idx]] * features_norm[edge_index[1, idx]], axis=1)

    # Non-edge similarities
    edge_set = set(zip(edge_index[0], edge_index[1]))
    non_edge_sims = []
    attempts = 0
    while len(non_edge_sims) < 50000 and attempts < 500000:
        i, j = np.random.randint(0, n_nodes, 2)
        if i != j and (i, j) not in edge_set:
            non_edge_sims.append(np.dot(features_norm[i], features_norm[j]))
        attempts += 1
    non_edge_sims = np.array(non_edge_sims)

    rho_fs = np.mean(edge_sims) - np.mean(non_edge_sims)
    print(f"\n--- Feature-Structure Alignment ---")
    print(f"rho_FS: {rho_fs:.4f}")
    print(f"Mean edge sim: {np.mean(edge_sims):.4f}")
    print(f"Mean non-edge sim: {np.mean(non_edge_sims):.4f}")

    # 4. Aggregation Dilution (δ_agg)
    adj_list = [[] for _ in range(n_nodes)]
    for i in range(n_edges):
        adj_list[edge_index[0, i]].append(edge_index[1, i])

    dilutions = []
    sims_to_agg = []
    node_degrees = []

    for i in range(n_nodes):
        neighbors = adj_list[i]
        if len(neighbors) == 0:
            continue

        degree = len(neighbors)
        node_degrees.append(degree)

        neighbor_features = features_norm[neighbors]
        mean_neighbor = np.mean(neighbor_features, axis=0)
        mean_neighbor_norm = mean_neighbor / (np.linalg.norm(mean_neighbor) + 1e-8)

        sim = np.dot(features_norm[i], mean_neighbor_norm)
        sims_to_agg.append(sim)
        dilutions.append(degree * (1 - sim))

    dilutions = np.array(dilutions)
    node_degrees = np.array(node_degrees)
    sims_to_agg = np.array(sims_to_agg)

    print(f"\n--- Aggregation Dilution (δ_agg) ---")
    print(f"Mean δ_agg: {np.mean(dilutions):.4f}")
    print(f"Mean sim(node, agg_neighbors): {np.mean(sims_to_agg):.4f}")

    # δ_agg by degree groups
    print(f"\nδ_agg by degree group:")
    for low, high in [(1, 10), (11, 50), (51, 100), (100, 500)]:
        mask = (node_degrees >= low) & (node_degrees < high)
        if np.sum(mask) > 0:
            print(f"  Degree {low}-{high}: δ_agg={np.mean(dilutions[mask]):.2f}, "
                  f"sim={np.mean(sims_to_agg[mask]):.4f}, count={np.sum(mask)}")

    # High degree
    mask = node_degrees >= 500
    if np.sum(mask) > 0:
        print(f"  Degree 500+: δ_agg={np.mean(dilutions[mask]):.2f}, "
              f"sim={np.mean(sims_to_agg[mask]):.4f}, count={np.sum(mask)}")

    # Class balance
    fraud_rate = np.mean(labels == 1) if len(np.unique(labels)) == 2 else np.nan
    print(f"\n--- Class Balance ---")
    print(f"Fraud rate: {fraud_rate*100:.2f}%")

    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'n_features': n_features,
        'mean_degree': np.mean(degrees),
        'std_degree': np.std(degrees),
        'degree_cv': np.std(degrees)/np.mean(degrees),
        'homophily': homophily,
        'rho_fs': rho_fs,
        'delta_agg': np.mean(dilutions),
        'sim_to_agg': np.mean(sims_to_agg),
        'fraud_rate': fraud_rate
    }


def load_yelpchi(path):
    """Load YelpChi dataset from .mat file."""
    data = sio.loadmat(path)

    # Features and labels
    features = data['features'].toarray() if sparse.issparse(data['features']) else data['features']
    labels = data['label'].flatten()

    # Graph (use homo graph)
    adj = data['homo']
    if sparse.issparse(adj):
        adj = adj.tocoo()
        edge_index = np.vstack([adj.row, adj.col])
    else:
        edge_index = np.array(np.nonzero(adj))

    return edge_index, features, labels


def load_dgraphfin(path):
    """Load DGraphFin dataset from .npz file."""
    data = np.load(path)

    features = data['x']
    labels = data['y']
    edge_index = data['edge_index']

    # Filter out unlabeled nodes (label == 2 or 3 typically)
    valid_mask = labels < 2

    # Remap node indices
    if not np.all(valid_mask):
        print(f"  Filtering {np.sum(~valid_mask)} unlabeled nodes...")
        node_map = np.cumsum(valid_mask) - 1
        node_map[~valid_mask] = -1

        # Filter edges
        src_valid = valid_mask[edge_index[0]]
        dst_valid = valid_mask[edge_index[1]]
        edge_mask = src_valid & dst_valid

        new_edge_index = np.stack([
            node_map[edge_index[0, edge_mask]],
            node_map[edge_index[1, edge_mask]]
        ])

        features = features[valid_mask]
        labels = labels[valid_mask]
        edge_index = new_edge_index

    return edge_index, features, labels


def load_elliptic(feature_path, edge_path, class_path):
    """Load Elliptic dataset from CSV files."""
    import pandas as pd

    # Load features
    features_df = pd.read_csv(feature_path, header=None)
    node_ids = features_df.iloc[:, 0].values
    features = features_df.iloc[:, 2:].values  # Skip txId and time_step

    # Create node ID to index mapping
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    # Load edges
    edges_df = pd.read_csv(edge_path)
    src = [node_to_idx.get(s, -1) for s in edges_df['txId1'].values]
    dst = [node_to_idx.get(d, -1) for d in edges_df['txId2'].values]

    # Filter invalid edges
    valid = [(s, d) for s, d in zip(src, dst) if s >= 0 and d >= 0]
    edge_index = np.array(valid).T if valid else np.zeros((2, 0), dtype=np.int64)

    # Load labels
    classes_df = pd.read_csv(class_path)
    label_map = {'1': 0, '2': 1, 'unknown': -1}  # 1=licit, 2=illicit
    labels = np.array([label_map.get(str(c), -1) for c in classes_df['class'].values])

    # Align with node order
    class_node_ids = classes_df['txId'].values
    final_labels = np.full(len(node_ids), -1)
    for nid, label in zip(class_node_ids, labels):
        if nid in node_to_idx:
            final_labels[node_to_idx[nid]] = label

    return edge_index, features.astype(np.float32), final_labels


def main():
    print("=" * 70)
    print("Cross-Dataset δ_agg Analysis for Multi-Dimensional FSD Framework")
    print("=" * 70)

    results = {}

    # 1. IEEE-CIS (already analyzed)
    ieee_path = r'D:\Users\11919\Documents\毕业论文\paper\code\processed\ieee_cis_graph.pkl'
    if os.path.exists(ieee_path):
        with open(ieee_path, 'rb') as f:
            data = pickle.load(f)
        results['IEEE-CIS'] = compute_metrics(
            data['edge_index'], data['features'], data['labels'], 'IEEE-CIS'
        )

    # 2. YelpChi
    yelp_path = r'D:\Users\11919\Documents\毕业论文\data\yelpchi\raw\YelpChi.mat'
    if os.path.exists(yelp_path):
        edge_index, features, labels = load_yelpchi(yelp_path)
        results['YelpChi'] = compute_metrics(edge_index, features, labels, 'YelpChi')

    # 3. DGraphFin
    dgraph_path = r'D:\Users\11919\Documents\毕业论文\scrp_gnn\data\dgraphfin\raw\dgraphfin.npz'
    if os.path.exists(dgraph_path):
        edge_index, features, labels = load_dgraphfin(dgraph_path)
        results['DGraphFin'] = compute_metrics(edge_index, features, labels, 'DGraphFin')

    # 4. Elliptic
    elliptic_base = r'D:\Users\11919\Documents\毕业论文\data\elliptic\raw'
    if os.path.exists(elliptic_base):
        edge_index, features, labels = load_elliptic(
            os.path.join(elliptic_base, 'elliptic_txs_features.csv'),
            os.path.join(elliptic_base, 'elliptic_txs_edgelist.csv'),
            os.path.join(elliptic_base, 'elliptic_txs_classes.csv')
        )
        results['Elliptic'] = compute_metrics(edge_index, features, labels, 'Elliptic')

    # Summary table
    print("\n" + "=" * 70)
    print("CROSS-DATASET SUMMARY")
    print("=" * 70)
    print(f"\n{'Dataset':<12} {'Nodes':>10} {'Edges':>12} {'Deg_CV':>8} {'h':>8} {'ρ_FS':>8} {'δ_agg':>10}")
    print("-" * 70)

    for name, r in results.items():
        print(f"{name:<12} {r['n_nodes']:>10,} {r['n_edges']:>12,} "
              f"{r['degree_cv']:>8.3f} {r['homophily']:>8.4f} "
              f"{r['rho_fs']:>8.4f} {r['delta_agg']:>10.2f}")

    # Save results
    import json
    output_path = r'D:\Users\11919\Documents\毕业论文\paper\code\cross_dataset_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()
