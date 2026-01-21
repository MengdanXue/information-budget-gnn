"""
Feature-Structure Alignment Score (ρ_FS) Computation

This script implements Algorithm 1 from the paper:
"Feature-Structure Disentanglement in Graph Neural Networks:
A Unified Framework for Financial Fraud Detection"

Usage:
    python compute_fs_alignment.py --dataset elliptic --num_samples 100000
"""

import numpy as np
import torch
from typing import Tuple, Optional, Callable
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import argparse
from tqdm import tqdm


def cosine_similarity(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    norm_i = np.linalg.norm(x_i)
    norm_j = np.linalg.norm(x_j)
    if norm_i < 1e-8 or norm_j < 1e-8:
        return 0.0
    return np.dot(x_i, x_j) / (norm_i * norm_j)


def log_cosine_similarity(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """Compute cosine similarity after log-normalization (for financial features)."""
    # Log-normalize: sign(x) * log(1 + |x|)
    x_i_log = np.sign(x_i) * np.log1p(np.abs(x_i))
    x_j_log = np.sign(x_j) * np.log1p(np.abs(x_j))
    return cosine_similarity(x_i_log, x_j_log)


def numerical_similarity(x_i: np.ndarray, x_j: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Compute NAA-style numerical similarity.
    S_feat(i,j) = -1/d * sum_k |x_ik - x_jk| / max(|x_ik|, |x_jk|, epsilon)
    """
    d = len(x_i)
    diffs = np.abs(x_i - x_j)
    maxs = np.maximum(np.abs(x_i), np.abs(x_j))
    maxs = np.maximum(maxs, epsilon)
    return -np.mean(diffs / maxs)


def compute_fs_alignment(
    edge_index: np.ndarray,
    features: np.ndarray,
    num_samples: int = 100000,
    similarity_fn: Callable = cosine_similarity,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[float, float, float]:
    """
    Compute Feature-Structure Alignment Score (ρ_FS).

    Algorithm 1 from the paper.

    Args:
        edge_index: Shape (2, num_edges) - edge list
        features: Shape (num_nodes, num_features) - node features
        num_samples: Number of non-edge samples (M in the paper)
        similarity_fn: Function to compute feature similarity
        seed: Random seed for reproducibility
        verbose: Whether to show progress bar

    Returns:
        rho_fs: Feature-Structure Alignment Score
        mean_edge_sim: Average similarity for edges
        mean_non_edge_sim: Average similarity for non-edges
    """
    np.random.seed(seed)

    num_nodes = features.shape[0]
    num_edges = edge_index.shape[1]

    # Build edge set for efficient lookup
    edge_set = set()
    for i in range(num_edges):
        src, dst = edge_index[0, i], edge_index[1, i]
        edge_set.add((src, dst))
        edge_set.add((dst, src))  # Undirected

    # Step 1: Compute edge similarities
    if verbose:
        print(f"Computing edge similarities for {num_edges} edges...")

    edge_similarities = []
    iterator = tqdm(range(num_edges)) if verbose else range(num_edges)
    for i in iterator:
        src, dst = edge_index[0, i], edge_index[1, i]
        sim = similarity_fn(features[src], features[dst])
        edge_similarities.append(sim)

    mean_edge_sim = np.mean(edge_similarities)

    # Step 2: Sample non-edges and compute similarities
    if verbose:
        print(f"Sampling {num_samples} non-edges...")

    non_edge_similarities = []
    attempts = 0
    max_attempts = num_samples * 100  # Prevent infinite loop

    pbar = tqdm(total=num_samples) if verbose else None
    while len(non_edge_similarities) < num_samples and attempts < max_attempts:
        # Sample random pair
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)

        if i != j and (i, j) not in edge_set:
            sim = similarity_fn(features[i], features[j])
            non_edge_similarities.append(sim)
            if pbar:
                pbar.update(1)

        attempts += 1

    if pbar:
        pbar.close()

    if len(non_edge_similarities) < num_samples:
        print(f"Warning: Only sampled {len(non_edge_similarities)} non-edges "
              f"(requested {num_samples})")

    mean_non_edge_sim = np.mean(non_edge_similarities)

    # Step 3: Compute alignment score
    rho_fs = mean_edge_sim - mean_non_edge_sim

    return rho_fs, mean_edge_sim, mean_non_edge_sim


def compute_fs_alignment_with_std(
    edge_index: np.ndarray,
    features: np.ndarray,
    num_samples: int = 100000,
    similarity_fn: Callable = cosine_similarity,
    num_runs: int = 5,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Compute ρ_FS with standard deviation over multiple runs.

    Returns:
        mean_rho_fs: Mean alignment score
        std_rho_fs: Standard deviation
    """
    rho_values = []
    for run in range(num_runs):
        rho, _, _ = compute_fs_alignment(
            edge_index, features, num_samples, similarity_fn,
            seed=42 + run, verbose=verbose and run == 0
        )
        rho_values.append(rho)
        if verbose:
            print(f"Run {run+1}/{num_runs}: ρ_FS = {rho:.4f}")

    return np.mean(rho_values), np.std(rho_values)


def analyze_sampling_convergence(
    edge_index: np.ndarray,
    features: np.ndarray,
    sample_sizes: list = [1000, 5000, 10000, 50000, 100000],
    similarity_fn: Callable = cosine_similarity,
    seed: int = 42
) -> dict:
    """
    Analyze how ρ_FS converges with different sample sizes.

    Returns:
        Dictionary with sample sizes and corresponding ρ_FS values
    """
    results = {'sample_size': [], 'rho_fs': [], 'edge_sim': [], 'non_edge_sim': []}

    for M in sample_sizes:
        rho, edge_sim, non_edge_sim = compute_fs_alignment(
            edge_index, features, M, similarity_fn, seed, verbose=False
        )
        results['sample_size'].append(M)
        results['rho_fs'].append(rho)
        results['edge_sim'].append(edge_sim)
        results['non_edge_sim'].append(non_edge_sim)
        print(f"M = {M:6d}: ρ_FS = {rho:.4f}")

    return results


# ============== Dataset Loaders ==============

def load_elliptic_data():
    """Load Elliptic Bitcoin dataset."""
    try:
        from torch_geometric.datasets import EllipticBitcoinDataset
        dataset = EllipticBitcoinDataset(root='./data/elliptic')
        data = dataset[0]
        return data.edge_index.numpy(), data.x.numpy()
    except Exception as e:
        print(f"Error loading Elliptic: {e}")
        print("Please install torch_geometric and download the dataset.")
        return None, None


def load_amazon_data():
    """Load Amazon Fraud dataset."""
    try:
        from torch_geometric.datasets import Amazon
        dataset = Amazon(root='./data/amazon', name='Computers')
        data = dataset[0]
        return data.edge_index.numpy(), data.x.numpy()
    except Exception as e:
        print(f"Error loading Amazon: {e}")
        return None, None


def load_yelp_data():
    """Load YelpChi dataset."""
    try:
        from torch_geometric.datasets import Yelp
        dataset = Yelp(root='./data/yelp')
        data = dataset[0]
        return data.edge_index.numpy(), data.x.numpy()
    except Exception as e:
        print(f"Error loading YelpChi: {e}")
        return None, None


def load_dgraphfin_data():
    """Load DGraphFin dataset."""
    try:
        # DGraphFin requires special loading
        import os
        data_path = './data/dgraphfin'
        if os.path.exists(os.path.join(data_path, 'dgraphfin.npz')):
            data = np.load(os.path.join(data_path, 'dgraphfin.npz'))
            edge_index = data['edge_index']
            features = data['x']
            return edge_index, features
        else:
            print("DGraphFin data not found. Please download from official source.")
            return None, None
    except Exception as e:
        print(f"Error loading DGraphFin: {e}")
        return None, None


def create_synthetic_graph(
    num_nodes: int = 1000,
    num_edges: int = 5000,
    num_features: int = 100,
    alignment: str = 'positive'  # 'positive', 'negative', 'random'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic graph with controlled feature-structure alignment.

    Useful for testing and validating the ρ_FS computation.
    """
    np.random.seed(42)

    # Generate features
    features = np.random.randn(num_nodes, num_features)

    if alignment == 'positive':
        # Edges connect similar nodes
        edge_index = []
        for _ in range(num_edges):
            i = np.random.randint(0, num_nodes)
            # Find similar node based on feature distance
            distances = np.linalg.norm(features - features[i], axis=1)
            distances[i] = np.inf  # Exclude self
            # Sample from top 10% similar nodes
            top_k = max(10, num_nodes // 10)
            candidates = np.argsort(distances)[:top_k]
            j = np.random.choice(candidates)
            edge_index.append([i, j])
        edge_index = np.array(edge_index).T

    elif alignment == 'negative':
        # Edges connect dissimilar nodes (heterophily)
        edge_index = []
        for _ in range(num_edges):
            i = np.random.randint(0, num_nodes)
            distances = np.linalg.norm(features - features[i], axis=1)
            distances[i] = -np.inf  # Exclude self
            # Sample from top 10% dissimilar nodes
            top_k = max(10, num_nodes // 10)
            candidates = np.argsort(distances)[-top_k:]
            j = np.random.choice(candidates)
            edge_index.append([i, j])
        edge_index = np.array(edge_index).T

    else:  # random
        # Random edges
        src = np.random.randint(0, num_nodes, num_edges)
        dst = np.random.randint(0, num_nodes, num_edges)
        edge_index = np.stack([src, dst])

    return edge_index, features


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description='Compute Feature-Structure Alignment Score')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['elliptic', 'amazon', 'yelp', 'dgraphfin', 'synthetic'],
                       help='Dataset to use')
    parser.add_argument('--num_samples', type=int, default=100000,
                       help='Number of non-edge samples')
    parser.add_argument('--similarity', type=str, default='cosine',
                       choices=['cosine', 'log_cosine', 'numerical'],
                       help='Similarity function to use')
    parser.add_argument('--convergence', action='store_true',
                       help='Analyze sampling convergence')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of runs for std computation')
    args = parser.parse_args()

    # Select similarity function
    sim_fns = {
        'cosine': cosine_similarity,
        'log_cosine': log_cosine_similarity,
        'numerical': numerical_similarity
    }
    similarity_fn = sim_fns[args.similarity]

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")

    if args.dataset == 'elliptic':
        edge_index, features = load_elliptic_data()
    elif args.dataset == 'amazon':
        edge_index, features = load_amazon_data()
    elif args.dataset == 'yelp':
        edge_index, features = load_yelp_data()
    elif args.dataset == 'dgraphfin':
        edge_index, features = load_dgraphfin_data()
    else:  # synthetic
        print("Creating synthetic graph with positive alignment...")
        edge_index, features = create_synthetic_graph(
            num_nodes=5000, num_edges=20000, alignment='positive'
        )

    if edge_index is None:
        print("Failed to load dataset. Exiting.")
        return

    print(f"Graph: {features.shape[0]} nodes, {edge_index.shape[1]} edges, "
          f"{features.shape[1]} features")

    # Normalize features (z-score)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    if args.convergence:
        print("\n=== Sampling Convergence Analysis ===")
        results = analyze_sampling_convergence(
            edge_index, features,
            sample_sizes=[1000, 5000, 10000, 25000, 50000, 100000],
            similarity_fn=similarity_fn
        )
    else:
        print(f"\n=== Computing ρ_FS with {args.num_runs} runs ===")
        mean_rho, std_rho = compute_fs_alignment_with_std(
            edge_index, features, args.num_samples,
            similarity_fn, args.num_runs
        )

        print(f"\n{'='*50}")
        print(f"Dataset: {args.dataset}")
        print(f"Similarity: {args.similarity}")
        print(f"ρ_FS = {mean_rho:.4f} ± {std_rho:.4f}")
        print(f"{'='*50}")

        # Interpretation
        if mean_rho > 0.2:
            print("Interpretation: ALIGNED - Structural neighbors are feature-similar")
            print("Recommendation: NAA or standard GCN/GAT should work well")
        elif mean_rho < -0.1:
            print("Interpretation: MISALIGNED (Heterophily) - Structural neighbors are feature-dissimilar")
            print("Recommendation: H2GCN or MixHop recommended")
        else:
            print("Interpretation: WEAKLY ALIGNED - Structure and features are mostly independent")
            print("Recommendation: Consider feature dimension and task specifics")


if __name__ == '__main__':
    main()
