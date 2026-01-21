"""
IEEE-CIS Fraud Detection Dataset - Graph Construction

This script builds a heterogeneous graph from the IEEE-CIS Fraud Detection dataset
for use with Graph Neural Networks. The graph construction does NOT use label information,
ensuring no label leakage.

Graph Construction Strategy:
- Nodes: Transactions
- Edges: Connect transactions that share:
  1. Same card (card1-card6 hash)
  2. Same device (DeviceType + DeviceInfo)
  3. Same email domain (P_emaildomain, R_emaildomain)
  4. Same address (addr1, addr2)

This creates a multi-relational graph suitable for R-GCN or can be simplified
to a homogeneous graph for GCN/GAT.

Download data from: https://www.kaggle.com/c/ieee-fraud-detection/data

Usage:
    python ieee_cis_graph_builder.py --data_dir ./ieee_cis_data --output_dir ./processed
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import argparse
import os
from typing import Tuple, Dict, List, Optional


def load_ieee_cis_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load IEEE-CIS transaction and identity data.

    Args:
        data_dir: Directory containing train_transaction.csv, train_identity.csv

    Returns:
        Merged dataframe with transaction and identity features
    """
    print("Loading IEEE-CIS data...")

    # Load transaction data
    train_trans = pd.read_csv(os.path.join(data_dir, 'train_transaction.csv'))
    print(f"  Transactions: {len(train_trans)} rows, {train_trans.shape[1]} columns")

    # Load identity data (optional, may not exist for all transactions)
    identity_path = os.path.join(data_dir, 'train_identity.csv')
    if os.path.exists(identity_path):
        train_identity = pd.read_csv(identity_path)
        print(f"  Identity: {len(train_identity)} rows, {train_identity.shape[1]} columns")

        # Merge on TransactionID
        df = train_trans.merge(train_identity, on='TransactionID', how='left')
    else:
        df = train_trans

    print(f"  Merged: {len(df)} rows, {df.shape[1]} columns")
    print(f"  Fraud rate: {df['isFraud'].mean()*100:.2f}%")

    return df


def build_entity_graph(
    df: pd.DataFrame,
    entity_columns: List[str],
    max_edges_per_entity: int = 100
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build graph edges by connecting transactions sharing the same entity.

    Args:
        df: Transaction dataframe
        entity_columns: Columns to use for entity matching
        max_edges_per_entity: Cap edges per entity to prevent super-hubs

    Returns:
        edge_index: Shape (2, num_edges)
        edge_stats: Statistics about edge construction
    """
    print(f"\nBuilding graph from entities: {entity_columns}")

    n = len(df)
    edges = set()
    edge_stats = defaultdict(int)

    for col in entity_columns:
        if col not in df.columns:
            print(f"  Warning: Column {col} not found, skipping")
            continue

        print(f"  Processing {col}...")

        # Group transactions by entity value
        entity_groups = df.groupby(col).groups

        for entity_val, indices in entity_groups.items():
            if pd.isna(entity_val):
                continue

            indices = list(indices)

            # Skip singleton entities
            if len(indices) < 2:
                continue

            # Sample if too many connections
            if len(indices) > max_edges_per_entity:
                indices = np.random.choice(indices, max_edges_per_entity, replace=False)

            # Create edges between all pairs in this entity group
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    edges.add((indices[i], indices[j]))
                    edges.add((indices[j], indices[i]))  # Undirected

        edge_stats[col] = len(edges) - sum(edge_stats.values())
        print(f"    Added {edge_stats[col]} edges from {col}")

    # Convert to numpy array
    edges = list(edges)
    if len(edges) == 0:
        print("  Warning: No edges created!")
        return np.zeros((2, 0), dtype=np.int64), edge_stats

    edge_index = np.array(edges, dtype=np.int64).T

    print(f"\nTotal edges: {edge_index.shape[1]}")
    print(f"Average degree: {edge_index.shape[1] / n:.2f}")

    return edge_index, dict(edge_stats)


def process_features(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    max_categories: int = 50
) -> np.ndarray:
    """
    Process features for GNN input.

    - Numeric features: Log-transform + standardize
    - Categorical features: Label encode (top categories only)

    Args:
        df: Transaction dataframe
        numeric_cols: List of numeric feature columns
        categorical_cols: List of categorical feature columns
        max_categories: Max categories per categorical feature

    Returns:
        features: Shape (num_nodes, num_features)
    """
    print("\nProcessing features...")

    # Default numeric columns (transaction amounts and time features)
    if numeric_cols is None:
        numeric_cols = [
            'TransactionAmt', 'TransactionDT',
            'card1', 'card2', 'card3', 'card5',  # Numeric card features
            'addr1', 'addr2',
            'dist1', 'dist2',
        ]
        # Add C features (count features)
        numeric_cols += [f'C{i}' for i in range(1, 15)]
        # Add D features (time delta features)
        numeric_cols += [f'D{i}' for i in range(1, 16)]
        # Add V features (Vesta engineered features)
        numeric_cols += [f'V{i}' for i in range(1, 340)]

    # Filter to existing columns
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    print(f"  Numeric features: {len(numeric_cols)}")

    # Default categorical columns
    if categorical_cols is None:
        categorical_cols = [
            'ProductCD', 'card4', 'card6',
            'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'DeviceType', 'DeviceInfo',
        ]

    categorical_cols = [c for c in categorical_cols if c in df.columns]
    print(f"  Categorical features: {len(categorical_cols)}")

    feature_list = []

    # Process numeric features
    if numeric_cols:
        numeric_data = df[numeric_cols].copy()

        # Fill NaN with median
        numeric_data = numeric_data.fillna(numeric_data.median())

        # Log transform (for transaction amounts, etc.)
        for col in ['TransactionAmt']:
            if col in numeric_data.columns:
                numeric_data[col] = np.log1p(numeric_data[col])

        # Standardize - NOTE: This should ideally fit only on training data
        # For graph construction, we fit on all data as a preprocessing step
        # The actual train/test split happens later in create_data_splits()
        # TODO: For stricter no-leakage, pass train_mask and fit only on train
        scaler = StandardScaler()
        numeric_features = scaler.fit_transform(numeric_data)
        # Store scaler for later use if needed
        self._numeric_scaler = scaler
        feature_list.append(numeric_features)
        print(f"  Numeric shape: {numeric_features.shape}")

    # Process categorical features
    if categorical_cols:
        cat_features = []
        for col in categorical_cols:
            le = LabelEncoder()

            # Get top categories
            top_cats = df[col].value_counts().head(max_categories).index.tolist()

            # Map to category index (others -> 0)
            col_data = df[col].apply(lambda x: x if x in top_cats else 'OTHER')
            col_data = col_data.fillna('MISSING')

            encoded = le.fit_transform(col_data)
            cat_features.append(encoded.reshape(-1, 1))

        categorical_features = np.hstack(cat_features)
        feature_list.append(categorical_features)
        print(f"  Categorical shape: {categorical_features.shape}")

    # Combine all features
    features = np.hstack(feature_list).astype(np.float32)
    print(f"  Final feature shape: {features.shape}")

    return features


def compute_rho_fs(
    edge_index: np.ndarray,
    features: np.ndarray,
    n_samples: int = 100000
) -> Dict[str, float]:
    """
    Compute Feature-Structure Alignment score (rho_FS).

    This is computed BEFORE seeing any labels, making it a true
    prior diagnostic for method selection.

    Args:
        edge_index: Shape (2, num_edges)
        features: Shape (num_nodes, num_features)
        n_samples: Number of non-edges to sample

    Returns:
        Dictionary with rho_FS and component statistics
    """
    print("\nComputing Feature-Structure Alignment (rho_FS)...")

    n = features.shape[0]
    num_edges = edge_index.shape[1]

    # Normalize features for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    features_norm = features / norms

    # Compute similarity for edges
    print("  Computing edge similarities...")
    edge_sims = []
    for i in range(0, num_edges, 10000):
        batch = edge_index[:, i:min(i+10000, num_edges)]
        sims = np.sum(features_norm[batch[0]] * features_norm[batch[1]], axis=1)
        edge_sims.extend(sims)
    edge_sims = np.array(edge_sims)

    # Sample non-edges
    print(f"  Sampling {n_samples} non-edges...")
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

    # Compute rho_FS
    mean_edge_sim = np.mean(edge_sims)
    mean_non_edge_sim = np.mean(non_edge_sims)
    rho_fs = mean_edge_sim - mean_non_edge_sim

    results = {
        'rho_fs': rho_fs,
        'mean_edge_sim': mean_edge_sim,
        'mean_non_edge_sim': mean_non_edge_sim,
        'std_edge_sim': np.std(edge_sims),
        'std_non_edge_sim': np.std(non_edge_sims),
        'n_edges': num_edges,
        'n_non_edge_samples': len(non_edge_sims),
    }

    print(f"\n  Results:")
    print(f"    Mean edge similarity:     {mean_edge_sim:.4f}")
    print(f"    Mean non-edge similarity: {mean_non_edge_sim:.4f}")
    print(f"    rho_FS:                   {rho_fs:.4f}")

    # FSD prediction
    feature_dim = features.shape[1]
    if rho_fs > 0.15 and feature_dim > 50:
        prediction = "NAA (Feature-Dominant Regime)"
    elif rho_fs < -0.05:
        prediction = "H2GCN (Heterophily Regime)"
    else:
        prediction = "Standard GCN/GAT (Structure-Dominant or Mixed)"

    print(f"\n  FSD Prediction: {prediction}")
    results['fsd_prediction'] = prediction

    return results


def create_data_splits(
    df: pd.DataFrame,
    split_type: str = 'temporal'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/val/test splits.

    Args:
        df: Transaction dataframe with TransactionDT
        split_type: 'temporal' for time-based split, 'random' for random

    Returns:
        train_mask, val_mask, test_mask: Boolean arrays
    """
    n = len(df)

    if split_type == 'temporal' and 'TransactionDT' in df.columns:
        print("\nCreating temporal split...")

        # Sort by time
        time_sorted = df['TransactionDT'].values

        # 70% train, 15% val, 15% test (by time)
        train_cutoff = np.percentile(time_sorted, 70)
        val_cutoff = np.percentile(time_sorted, 85)

        train_mask = time_sorted <= train_cutoff
        val_mask = (time_sorted > train_cutoff) & (time_sorted <= val_cutoff)
        test_mask = time_sorted > val_cutoff

    else:
        print("\nCreating random split...")

        indices = np.random.permutation(n)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)

        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True

    print(f"  Train: {train_mask.sum()} ({train_mask.mean()*100:.1f}%)")
    print(f"  Val:   {val_mask.sum()} ({val_mask.mean()*100:.1f}%)")
    print(f"  Test:  {test_mask.sum()} ({test_mask.mean()*100:.1f}%)")

    # Check fraud rates
    labels = df['isFraud'].values
    print(f"  Train fraud rate: {labels[train_mask].mean()*100:.2f}%")
    print(f"  Val fraud rate:   {labels[val_mask].mean()*100:.2f}%")
    print(f"  Test fraud rate:  {labels[test_mask].mean()*100:.2f}%")

    return train_mask, val_mask, test_mask


def main():
    parser = argparse.ArgumentParser(description='Build graph from IEEE-CIS data')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing IEEE-CIS CSV files')
    parser.add_argument('--output_dir', type=str, default='./processed',
                       help='Output directory for processed data')
    parser.add_argument('--max_edges_per_entity', type=int, default=100,
                       help='Max edges per entity to prevent super-hubs')
    parser.add_argument('--subsample', type=int, default=None,
                       help='Subsample to N transactions (for testing)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = load_ieee_cis_data(args.data_dir)

    # Subsample for testing
    if args.subsample:
        print(f"\nSubsampling to {args.subsample} transactions...")
        df = df.sample(n=args.subsample, random_state=42).reset_index(drop=True)

    # Build graph
    entity_columns = ['card1', 'card2', 'addr1', 'P_emaildomain', 'DeviceInfo']
    edge_index, edge_stats = build_entity_graph(
        df, entity_columns, max_edges_per_entity=args.max_edges_per_entity
    )

    # Process features
    features = process_features(df)

    # Compute rho_FS (BEFORE looking at labels - this is the prior prediction)
    rho_fs_results = compute_rho_fs(edge_index, features)

    # Get labels
    labels = df['isFraud'].values

    # Create splits
    train_mask, val_mask, test_mask = create_data_splits(df, split_type='temporal')

    # Save processed data
    print(f"\nSaving to {args.output_dir}...")

    data = {
        'edge_index': edge_index,
        'features': features,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'rho_fs_results': rho_fs_results,
        'edge_stats': edge_stats,
        'num_nodes': len(df),
        'num_features': features.shape[1],
        'fraud_rate': labels.mean(),
    }

    with open(os.path.join(args.output_dir, 'ieee_cis_graph.pkl'), 'wb') as f:
        pickle.dump(data, f)

    # Save summary
    summary = f"""
IEEE-CIS Graph Dataset Summary
==============================
Nodes: {data['num_nodes']}
Edges: {edge_index.shape[1]}
Features: {data['num_features']}
Fraud Rate: {data['fraud_rate']*100:.2f}%

Feature-Structure Alignment (rho_FS): {rho_fs_results['rho_fs']:.4f}
  - Mean edge similarity: {rho_fs_results['mean_edge_sim']:.4f}
  - Mean non-edge similarity: {rho_fs_results['mean_non_edge_sim']:.4f}

FSD Prediction: {rho_fs_results['fsd_prediction']}

Edge Statistics:
{edge_stats}

Data Splits (temporal):
  - Train: {train_mask.sum()} ({labels[train_mask].mean()*100:.2f}% fraud)
  - Val: {val_mask.sum()} ({labels[val_mask].mean()*100:.2f}% fraud)
  - Test: {test_mask.sum()} ({labels[test_mask].mean()*100:.2f}% fraud)
"""

    with open(os.path.join(args.output_dir, 'ieee_cis_summary.txt'), 'w') as f:
        f.write(summary)

    print(summary)
    print(f"\nDone! Data saved to {args.output_dir}/ieee_cis_graph.pkl")


if __name__ == '__main__':
    main()
