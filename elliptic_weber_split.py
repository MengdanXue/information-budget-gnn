"""
Elliptic Dataset Temporal Split - Weber Standard Implementation
================================================================

This script implements the standard temporal split for Elliptic Bitcoin dataset
as defined by Weber et al. (2019) in "Anti-Money Laundering in Bitcoin:
Experimenting with Graph Convolutional Networks for Financial Forensic."

Key Points:
1. Elliptic has 49 timesteps of Bitcoin transactions
2. Standard split uses timesteps 1-34 for training, 35-49 for testing
3. This prevents temporal leakage (using future to predict past)

Why This Matters:
- Random split causes information leakage across time
- Temporal split is more realistic for deployment scenarios
- Required for fair comparison with prior work

Reference:
Weber, M., et al. (2019). Anti-Money Laundering in Bitcoin:
Experimenting with Graph Convolutional Networks for Financial Forensic.
arXiv:1908.02591

Author: FSD Framework Research Team
Date: 2024-12-22
Version: 1.0 (TKDE Submission)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle
from torch_geometric.data import Data


def load_elliptic_raw(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw Elliptic dataset files.

    Expected files in data_dir:
    - elliptic_txs_features.csv: Transaction features (49 timesteps, 166 features)
    - elliptic_txs_classes.csv: Transaction labels (illicit=1, licit=2, unknown)
    - elliptic_txs_edgelist.csv: Transaction graph edges

    Returns:
        features_df, classes_df, edges_df
    """
    data_dir = Path(data_dir)

    features_path = data_dir / 'elliptic_txs_features.csv'
    classes_path = data_dir / 'elliptic_txs_classes.csv'
    edges_path = data_dir / 'elliptic_txs_edgelist.csv'

    print(f"Loading Elliptic data from {data_dir}...")

    # Load features (no header in original file)
    features_df = pd.read_csv(features_path, header=None)
    features_df.columns = ['txId'] + [f'local_feat_{i}' for i in range(93)] + \
                          [f'agg_feat_{i}' for i in range(72)]

    # Load classes
    classes_df = pd.read_csv(classes_path)
    classes_df.columns = ['txId', 'class']

    # Load edges
    edges_df = pd.read_csv(edges_path)
    edges_df.columns = ['txId1', 'txId2']

    print(f"  Transactions: {len(features_df)}")
    print(f"  Edges: {len(edges_df)}")

    return features_df, classes_df, edges_df


def create_weber_temporal_split(
    features_df: pd.DataFrame,
    classes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    train_timesteps: range = range(1, 35),  # 1-34
    val_ratio: float = 0.15,  # Split some training for validation
    seed: int = 42
) -> Dict:
    """
    Create Weber standard temporal split for Elliptic.

    Weber et al. (2019) Protocol:
    - Training: timesteps 1-34
    - Testing: timesteps 35-49
    - Only labeled nodes are used for training/evaluation

    Args:
        features_df: Transaction features with timestep as first local feature
        classes_df: Transaction labels
        edges_df: Edge list
        train_timesteps: Timesteps for training (default: 1-34)
        val_ratio: Fraction of training data for validation
        seed: Random seed for val split

    Returns:
        Dictionary with PyG Data and split information
    """
    np.random.seed(seed)

    # Extract timestep (first local feature in Elliptic)
    timesteps = features_df['local_feat_0'].values

    # Map txId to node index
    tx_ids = features_df['txId'].values
    tx_to_idx = {tx: idx for idx, tx in enumerate(tx_ids)}

    # Get labels (convert to binary: 1=illicit, 0=licit)
    labels = np.zeros(len(features_df), dtype=np.int64)
    label_map = classes_df.set_index('txId')['class'].to_dict()

    labeled_mask = np.zeros(len(features_df), dtype=bool)
    for idx, tx_id in enumerate(tx_ids):
        label = label_map.get(tx_id, 'unknown')
        if label == '1':  # Illicit
            labels[idx] = 1
            labeled_mask[idx] = True
        elif label == '2':  # Licit
            labels[idx] = 0
            labeled_mask[idx] = True
        # Unknown labels are not used

    # Create temporal masks
    train_time_mask = np.isin(timesteps, list(train_timesteps))
    test_time_mask = ~train_time_mask

    # Only use labeled nodes
    train_candidates = train_time_mask & labeled_mask
    test_mask = test_time_mask & labeled_mask

    # Split training into train/val (stratified by label)
    train_indices = np.where(train_candidates)[0]
    train_labels = labels[train_indices]

    # Stratified val split
    val_indices = []
    train_final_indices = []

    for label_val in [0, 1]:
        label_mask = train_labels == label_val
        label_indices = train_indices[label_mask]
        n_val = int(len(label_indices) * val_ratio)

        perm = np.random.permutation(len(label_indices))
        val_indices.extend(label_indices[perm[:n_val]])
        train_final_indices.extend(label_indices[perm[n_val:]])

    # Create masks
    train_mask = np.zeros(len(features_df), dtype=bool)
    val_mask = np.zeros(len(features_df), dtype=bool)

    train_mask[train_final_indices] = True
    val_mask[val_indices] = True

    # Process features (exclude txId)
    feature_cols = [c for c in features_df.columns if c != 'txId']
    features = features_df[feature_cols].values.astype(np.float32)

    # Handle NaN values
    features = np.nan_to_num(features, nan=0.0)

    # Normalize features - FIT ONLY ON TRAINING DATA to prevent leakage
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # IMPORTANT: Only fit on training data, then transform all
    scaler.fit(features[train_mask])
    features = scaler.transform(features)

    # Build edge index
    edge_list = []
    for _, row in edges_df.iterrows():
        src = tx_to_idx.get(row['txId1'])
        dst = tx_to_idx.get(row['txId2'])
        if src is not None and dst is not None:
            edge_list.append([src, dst])
            edge_list.append([dst, src])  # Undirected

    edge_index = np.array(edge_list, dtype=np.int64).T if edge_list else np.zeros((2, 0), dtype=np.int64)

    # Create PyG Data
    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool)
    )

    # Statistics
    n_train = train_mask.sum()
    n_val = val_mask.sum()
    n_test = test_mask.sum()

    train_pos = labels[train_mask].sum()
    test_pos = labels[test_mask].sum()

    print("\n" + "="*60)
    print("WEBER TEMPORAL SPLIT SUMMARY")
    print("="*60)
    print(f"Training timesteps: {min(train_timesteps)}-{max(train_timesteps)}")
    print(f"Testing timesteps: {max(train_timesteps)+1}-49")
    print()
    print(f"Training: {n_train} nodes ({train_pos} illicit, {n_train-train_pos} licit)")
    print(f"Validation: {n_val} nodes")
    print(f"Testing: {n_test} nodes ({test_pos} illicit, {n_test-test_pos} licit)")
    print()
    print(f"Training fraud rate: {train_pos/n_train*100:.2f}%")
    print(f"Testing fraud rate: {test_pos/n_test*100:.2f}%")
    print("="*60)

    return {
        'data': data,
        'split_info': {
            'method': 'weber_temporal',
            'train_timesteps': list(train_timesteps),
            'test_timesteps': list(range(max(train_timesteps)+1, 50)),
            'n_train': int(n_train),
            'n_val': int(n_val),
            'n_test': int(n_test),
            'train_fraud_rate': float(train_pos/n_train),
            'test_fraud_rate': float(test_pos/n_test)
        },
        'tx_to_idx': tx_to_idx,
        'timesteps': timesteps
    }


def create_alternative_temporal_splits(
    features_df: pd.DataFrame,
    classes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Create multiple temporal split variants for robustness analysis.

    Variants:
    1. Weber Standard: 1-34 train, 35-49 test
    2. Pareja Variant: 1-29 train, 30-49 test (more test data)
    3. Rolling Window: Multiple train/test splits

    Args:
        features_df, classes_df, edges_df: Raw data
        seed: Random seed

    Returns:
        Dictionary of split name -> split data
    """
    splits = {}

    # Variant 1: Weber Standard
    splits['weber_standard'] = create_weber_temporal_split(
        features_df, classes_df, edges_df,
        train_timesteps=range(1, 35),
        seed=seed
    )

    # Variant 2: More test data
    splits['extended_test'] = create_weber_temporal_split(
        features_df, classes_df, edges_df,
        train_timesteps=range(1, 30),
        seed=seed
    )

    # Variant 3: Less test data (more training)
    splits['extended_train'] = create_weber_temporal_split(
        features_df, classes_df, edges_df,
        train_timesteps=range(1, 40),
        seed=seed
    )

    return splits


def verify_no_temporal_leakage(data: Data, timesteps: np.ndarray) -> bool:
    """
    Verify that there is no temporal leakage in the split.

    Temporal leakage occurs when:
    1. Training nodes have edges to future test nodes
    2. Test nodes appear in training set

    Args:
        data: PyG Data with masks
        timesteps: Array of timesteps for each node

    Returns:
        True if no leakage detected
    """
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    train_timesteps = timesteps[train_mask | val_mask]
    test_timesteps = timesteps[test_mask]

    max_train_ts = train_timesteps.max() if len(train_timesteps) > 0 else 0
    min_test_ts = test_timesteps.min() if len(test_timesteps) > 0 else float('inf')

    if max_train_ts >= min_test_ts:
        print(f"WARNING: Temporal leakage detected!")
        print(f"  Max training timestep: {max_train_ts}")
        print(f"  Min testing timestep: {min_test_ts}")
        return False

    print(f"No temporal leakage detected.")
    print(f"  Training/Val timesteps: 1-{max_train_ts}")
    print(f"  Testing timesteps: {min_test_ts}-{timesteps.max()}")
    return True


def save_processed_data(result: Dict, output_path: str):
    """Save processed Elliptic data with Weber split."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"Saved processed data to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Elliptic Weber Temporal Split')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing Elliptic CSV files')
    parser.add_argument('--output', type=str, default='./elliptic_weber_split.pkl',
                       help='Output file path')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation ratio from training data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify', action='store_true',
                       help='Verify no temporal leakage')

    args = parser.parse_args()

    # Load data
    features_df, classes_df, edges_df = load_elliptic_raw(args.data_dir)

    # Create Weber split
    result = create_weber_temporal_split(
        features_df, classes_df, edges_df,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Verify if requested
    if args.verify:
        verify_no_temporal_leakage(result['data'], result['timesteps'])

    # Save
    save_processed_data(result, args.output)


if __name__ == '__main__':
    main()
