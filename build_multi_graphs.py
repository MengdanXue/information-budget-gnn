"""
Multi-Graph Construction for IEEE-CIS Dataset

This script implements different graph construction strategies to analyze
how graph construction affects Feature-Structure Alignment (ρ_FS) and
subsequently, optimal method selection.

Key Insight: The same dataset can yield different ρ_FS values depending
on how the graph is constructed, leading to different FSD predictions.

Graph Construction Strategies:
1. Entity-based (original): Connect transactions sharing entities
2. kNN-based: Connect transactions to their k most similar neighbors
3. Hybrid: Combine entity and kNN edges

Usage:
    python build_multi_graphs.py --data_dir ./ieee_cis_data --output_dir ./multi_graphs
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import os
import argparse
from datetime import datetime
import json


def load_processed_data(data_path: str) -> Dict:
    """Load already processed IEEE-CIS data."""
    with open(data_path, 'rb') as f:
        return pickle.load(f)


def compute_rho_fs(edge_index: np.ndarray, features: np.ndarray, n_samples: int = 100000) -> Dict:
    """
    Compute Feature-Structure Alignment score.
    """
    n = features.shape[0]
    num_edges = edge_index.shape[1]

    # Normalize features for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    features_norm = features / norms

    # Edge similarities
    edge_sims = []
    for i in range(0, num_edges, 10000):
        batch = edge_index[:, i:min(i+10000, num_edges)]
        sims = np.sum(features_norm[batch[0]] * features_norm[batch[1]], axis=1)
        edge_sims.extend(sims)
    edge_sims = np.array(edge_sims)

    # Non-edge similarities (sampling)
    edge_set = set(zip(edge_index[0], edge_index[1]))
    non_edge_sims = []
    attempts = 0
    max_attempts = n_samples * 10

    while len(non_edge_sims) < n_samples and attempts < max_attempts:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i != j and (i, j) not in edge_set:
            sim = np.dot(features_norm[i], features_norm[j])
            non_edge_sims.append(sim)
        attempts += 1
    non_edge_sims = np.array(non_edge_sims)

    rho_fs = np.mean(edge_sims) - np.mean(non_edge_sims)

    return {
        'rho_fs': rho_fs,
        'mean_edge_sim': np.mean(edge_sims),
        'mean_non_edge_sim': np.mean(non_edge_sims),
        'std_edge_sim': np.std(edge_sims),
        'std_non_edge_sim': np.std(non_edge_sims),
        'num_edges': num_edges
    }


def build_knn_graph(features: np.ndarray, k: int = 10, metric: str = 'cosine') -> np.ndarray:
    """
    Build k-NN graph based on feature similarity.

    Args:
        features: Node features (n_nodes, n_features)
        k: Number of nearest neighbors
        metric: Distance metric ('cosine', 'euclidean')

    Returns:
        edge_index: Shape (2, num_edges)
    """
    print(f"Building {k}-NN graph with {metric} metric...")

    n = features.shape[0]

    # Normalize for cosine similarity
    if metric == 'cosine':
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        features_norm = features / norms
        nn_metric = 'cosine'
    else:
        features_norm = features
        nn_metric = 'euclidean'

    # Build kNN index (in batches for memory efficiency)
    batch_size = 10000
    edges = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_features = features_norm[start:end]

        # Find nearest neighbors for this batch
        nn = NearestNeighbors(n_neighbors=min(k+1, n), metric=nn_metric, algorithm='auto')
        nn.fit(features_norm)
        distances, indices = nn.kneighbors(batch_features)

        # Add edges (skip self-loops)
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            src = start + i
            for j, tgt in enumerate(idx[1:k+1]):  # Skip first (self)
                if src != tgt:
                    edges.append((src, tgt))
                    edges.append((tgt, src))  # Undirected

        if start % 50000 == 0:
            print(f"  Processed {end}/{n} nodes...")

    # Remove duplicates
    edges = list(set(edges))
    edge_index = np.array(edges, dtype=np.int64).T

    print(f"  Created {edge_index.shape[1]} edges (avg degree: {edge_index.shape[1]/n:.2f})")
    return edge_index


def build_hybrid_graph(
    entity_edges: np.ndarray,
    features: np.ndarray,
    k: int = 5,
    feature_threshold: float = 0.8
) -> np.ndarray:
    """
    Build hybrid graph: entity edges + kNN edges with feature similarity filter.

    Strategy: Keep entity edges where feature similarity > threshold,
              Add kNN edges for nodes with few entity edges.
    """
    print("Building hybrid graph...")

    n = features.shape[0]

    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    features_norm = features / norms

    # Filter entity edges by feature similarity
    filtered_edges = []
    for i in range(0, entity_edges.shape[1], 10000):
        batch = entity_edges[:, i:min(i+10000, entity_edges.shape[1])]
        sims = np.sum(features_norm[batch[0]] * features_norm[batch[1]], axis=1)
        mask = sims > feature_threshold
        filtered_edges.extend(zip(batch[0][mask], batch[1][mask]))

    print(f"  Entity edges after filtering: {len(filtered_edges)} (threshold={feature_threshold})")

    # Find nodes with few edges
    node_degrees = np.zeros(n)
    for src, tgt in filtered_edges:
        node_degrees[src] += 1

    low_degree_nodes = np.where(node_degrees < k)[0]
    print(f"  Nodes with < {k} edges: {len(low_degree_nodes)}")

    # Add kNN edges for low-degree nodes
    if len(low_degree_nodes) > 0:
        nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        nn.fit(features_norm)

        for node in low_degree_nodes:
            distances, indices = nn.kneighbors(features_norm[node:node+1])
            for tgt in indices[0][1:]:  # Skip self
                filtered_edges.append((node, tgt))
                filtered_edges.append((tgt, node))

    # Remove duplicates
    filtered_edges = list(set(filtered_edges))
    edge_index = np.array(filtered_edges, dtype=np.int64).T

    print(f"  Final edges: {edge_index.shape[1]}")
    return edge_index


def get_fsd_prediction(rho_fs: float, feature_dim: int) -> str:
    """Get FSD prediction based on ρ_FS and feature dimension."""
    if rho_fs > 0.15 and feature_dim > 50:
        return "NAA (Feature-Dominant Regime)"
    elif rho_fs < -0.05:
        return "H2GCN (Heterophily Regime)"
    else:
        return "Standard GCN/GAT (Structure-Dominant or Mixed)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                       default='./processed/ieee_cis_graph.pkl')
    parser.add_argument('--output_dir', type=str, default='./multi_graphs')
    parser.add_argument('--knn_k', type=int, default=10)
    parser.add_argument('--hybrid_threshold', type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("Multi-Graph Construction for FSD Analysis")
    print("="*60)

    # Load original data
    print("\nLoading processed data...")
    data = load_processed_data(args.data_path)
    features = data['features']
    labels = data['labels']
    entity_edges = data['edge_index']
    train_mask = data['train_mask']
    val_mask = data['val_mask']
    test_mask = data['test_mask']

    n = features.shape[0]
    d = features.shape[1]
    print(f"Nodes: {n}, Features: {d}")

    results = {}

    # 1. Entity-based graph (original)
    print("\n" + "-"*40)
    print("Graph 1: Entity-based (Original)")
    print("-"*40)
    rho_fs_entity = compute_rho_fs(entity_edges, features)
    prediction_entity = get_fsd_prediction(rho_fs_entity['rho_fs'], d)
    print(f"ρ_FS: {rho_fs_entity['rho_fs']:.4f}")
    print(f"FSD Prediction: {prediction_entity}")

    results['entity_based'] = {
        'rho_fs': rho_fs_entity,
        'fsd_prediction': prediction_entity,
        'num_edges': entity_edges.shape[1]
    }

    # 2. kNN graph
    print("\n" + "-"*40)
    print(f"Graph 2: {args.knn_k}-NN Graph")
    print("-"*40)
    knn_edges = build_knn_graph(features, k=args.knn_k)
    rho_fs_knn = compute_rho_fs(knn_edges, features)
    prediction_knn = get_fsd_prediction(rho_fs_knn['rho_fs'], d)
    print(f"ρ_FS: {rho_fs_knn['rho_fs']:.4f}")
    print(f"FSD Prediction: {prediction_knn}")

    # Save kNN graph
    knn_data = {
        'edge_index': knn_edges,
        'features': features,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'rho_fs_results': rho_fs_knn,
        'graph_type': 'knn',
        'k': args.knn_k
    }
    with open(os.path.join(args.output_dir, 'ieee_cis_knn_graph.pkl'), 'wb') as f:
        pickle.dump(knn_data, f)

    results['knn_based'] = {
        'rho_fs': rho_fs_knn,
        'fsd_prediction': prediction_knn,
        'num_edges': knn_edges.shape[1],
        'k': args.knn_k
    }

    # 3. Hybrid graph
    print("\n" + "-"*40)
    print(f"Graph 3: Hybrid (Entity + kNN filtered)")
    print("-"*40)
    hybrid_edges = build_hybrid_graph(
        entity_edges, features,
        k=args.knn_k,
        feature_threshold=args.hybrid_threshold
    )
    rho_fs_hybrid = compute_rho_fs(hybrid_edges, features)
    prediction_hybrid = get_fsd_prediction(rho_fs_hybrid['rho_fs'], d)
    print(f"ρ_FS: {rho_fs_hybrid['rho_fs']:.4f}")
    print(f"FSD Prediction: {prediction_hybrid}")

    # Save hybrid graph
    hybrid_data = {
        'edge_index': hybrid_edges,
        'features': features,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'rho_fs_results': rho_fs_hybrid,
        'graph_type': 'hybrid',
        'feature_threshold': args.hybrid_threshold
    }
    with open(os.path.join(args.output_dir, 'ieee_cis_hybrid_graph.pkl'), 'wb') as f:
        pickle.dump(hybrid_data, f)

    results['hybrid'] = {
        'rho_fs': rho_fs_hybrid,
        'fsd_prediction': prediction_hybrid,
        'num_edges': hybrid_edges.shape[1],
        'feature_threshold': args.hybrid_threshold
    }

    # Summary
    print("\n" + "="*60)
    print("MULTI-GRAPH ANALYSIS SUMMARY")
    print("="*60)
    print(f"\n{'Graph Type':<20} {'ρ_FS':>10} {'Edges':>12} {'FSD Prediction':<30}")
    print("-"*72)
    print(f"{'Entity-based':<20} {rho_fs_entity['rho_fs']:>10.4f} {entity_edges.shape[1]:>12} {prediction_entity:<30}")
    print(f"{'kNN (k={})'.format(args.knn_k):<20} {rho_fs_knn['rho_fs']:>10.4f} {knn_edges.shape[1]:>12} {prediction_knn:<30}")
    print(f"{'Hybrid':<20} {rho_fs_hybrid['rho_fs']:>10.4f} {hybrid_edges.shape[1]:>12} {prediction_hybrid:<30}")
    print("-"*72)

    # Key insight
    print("\n** KEY INSIGHT **")
    if prediction_entity != prediction_knn:
        print(f"Graph construction changes FSD prediction!")
        print(f"  Entity-based: {prediction_entity}")
        print(f"  kNN-based: {prediction_knn}")
        print("This demonstrates that practitioners should consider graph construction")
        print("as part of the method selection process.")
    else:
        print("All graph constructions yield the same FSD prediction.")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'IEEE-CIS',
        'num_nodes': n,
        'num_features': d,
        'results': results
    }
    with open(os.path.join(args.output_dir, 'multi_graph_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
