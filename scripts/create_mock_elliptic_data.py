"""
Create Mock Elliptic Dataset for Case Study Visualization
=========================================================

This script generates a synthetic Elliptic-like dataset for testing
the case study visualization pipeline when real Elliptic data is unavailable.

The mock dataset mimics the structure of the real Elliptic dataset:
- Bitcoin transaction graph
- Node features (166 dimensions)
- Binary labels (fraud vs legitimate)
- Temporal split (train/val/test)

Author: FSD Framework Research Team
Date: 2024-12-23
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_undirected


def create_mock_elliptic_data(
    num_nodes=4000,
    num_features=166,
    fraud_ratio=0.15,
    edge_prob=0.003,
    seed=42
):
    """
    Create a mock Elliptic dataset with realistic properties.

    Args:
        num_nodes: Total number of transaction nodes
        num_features: Feature dimension (Elliptic has 166 features)
        fraud_ratio: Proportion of fraudulent transactions
        edge_prob: Edge probability for Erdos-Renyi graph
        seed: Random seed

    Returns:
        result: Dict with PyG Data object and metadata
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Creating mock Elliptic dataset with {num_nodes} nodes...")

    # ===== Generate node features =====
    # Real Elliptic features include:
    # - Local features (tx amount, time, etc.)
    # - Aggregate features (neighbor statistics)

    # Base features: normal distribution
    features = torch.randn(num_nodes, num_features)

    # Create fraud labels
    num_fraud = int(num_nodes * fraud_ratio)
    labels = torch.zeros(num_nodes, dtype=torch.long)
    fraud_indices = np.random.choice(num_nodes, num_fraud, replace=False)
    labels[fraud_indices] = 1

    # Add distinguishing patterns for fraud nodes
    # Fraud transactions tend to have different feature distributions
    for idx in fraud_indices:
        # Add specific patterns to fraud nodes
        features[idx, :50] += torch.randn(50) * 0.5  # Amplify certain features
        features[idx, 50:100] *= 1.3  # Scale up some features
        features[idx, 100:] += 0.3  # Shift features

    # Normalize features
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)

    # ===== Generate graph structure =====
    # Use Erdos-Renyi graph as base
    edge_index = erdos_renyi_graph(num_nodes, edge_prob, directed=False)
    edge_index = to_undirected(edge_index)

    # Add preferential attachment for fraud nodes (fraud networks tend to cluster)
    fraud_fraud_edges = []
    for i in range(len(fraud_indices)):
        for j in range(i+1, min(i+10, len(fraud_indices))):
            if np.random.rand() < 0.3:  # Higher connection probability
                fraud_fraud_edges.append([fraud_indices[i], fraud_indices[j]])
                fraud_fraud_edges.append([fraud_indices[j], fraud_indices[i]])

    if fraud_fraud_edges:
        fraud_edges = torch.tensor(fraud_fraud_edges, dtype=torch.long).t()
        edge_index = torch.cat([edge_index, fraud_edges], dim=1)
        edge_index = torch.unique(edge_index, dim=1)  # Remove duplicates

    print(f"Generated graph: {num_nodes} nodes, {edge_index.shape[1]} edges")
    print(f"Fraud nodes: {num_fraud} ({fraud_ratio*100:.1f}%)")

    # ===== Create temporal split (Weber split) =====
    # Elliptic dataset has temporal structure
    # Train: 60%, Val: 20%, Test: 20%

    num_train = int(num_nodes * 0.6)
    num_val = int(num_nodes * 0.2)

    indices = torch.randperm(num_nodes)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train+num_val]
    test_indices = indices[num_train+num_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # ===== Create PyG Data object =====
    data = Data(
        x=features,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    # ===== Package result =====
    result = {
        'data': data,
        'num_nodes': num_nodes,
        'num_features': num_features,
        'num_edges': edge_index.shape[1],
        'num_fraud': num_fraud,
        'fraud_ratio': fraud_ratio,
        'train_size': train_mask.sum().item(),
        'val_size': val_mask.sum().item(),
        'test_size': test_mask.sum().item(),
        'description': 'Mock Elliptic dataset for case study testing'
    }

    print("\nDataset summary:")
    print(f"  Nodes: {result['num_nodes']}")
    print(f"  Edges: {result['num_edges']}")
    print(f"  Features: {result['num_features']}")
    print(f"  Fraud: {result['num_fraud']} ({result['fraud_ratio']*100:.1f}%)")
    print(f"  Train: {result['train_size']}")
    print(f"  Val: {result['val_size']}")
    print(f"  Test: {result['test_size']}")

    return result


def main():
    # Output directory
    data_dir = Path('paper/code/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create mock dataset
    result = create_mock_elliptic_data(
        num_nodes=4000,
        num_features=166,
        fraud_ratio=0.15,
        edge_prob=0.003,
        seed=42
    )

    # Save to pickle file
    output_file = data_dir / 'elliptic_weber_split.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)

    print(f"\nMock Elliptic dataset saved to: {output_file}")
    print("\nYou can now run generate_case_study.py with this mock data!")

    # Verify the data can be loaded
    print("\nVerifying data can be loaded...")
    with open(output_file, 'rb') as f:
        loaded = pickle.load(f)

    print(f"Successfully loaded! Data shape: {loaded['data'].x.shape}")


if __name__ == '__main__':
    main()
