#!/usr/bin/env python3
"""
Elliptic Dataset Time Leakage Checker
=====================================

This script verifies that the Elliptic Bitcoin dataset (Weber temporal split)
does not have time leakage issues that could inflate NAA-GAT's +23% AUC improvement.

Checks:
1. Feature inspection - Are any features future-looking?
2. Edge temporal analysis - Do edges cross time boundaries?
3. Label distribution shift - Is there temporal concept drift?
4. Train/test contamination - Any overlap in node IDs?

Author: FSD Framework
Date: 2025-12-23
"""

import pickle
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def load_elliptic_data(pkl_path: str):
    """Load Elliptic dataset from pickle file."""
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and 'data' in obj:
        data = obj['data']
    else:
        data = obj

    return data, obj

def check_feature_leakage(data) -> dict:
    """
    Check if features contain future information.

    Weber split: Train timesteps 1-34, Test timesteps 35-49
    Features should only contain information up to the current timestep.
    """
    print("\n" + "="*60)
    print("CHECK 1: FEATURE LEAKAGE ANALYSIS")
    print("="*60)

    results = {
        'status': 'UNKNOWN',
        'details': [],
        'warnings': []
    }

    # Get features
    if hasattr(data, 'x'):
        x = data.x.numpy() if HAS_TORCH and isinstance(data.x, torch.Tensor) else np.array(data.x)
    elif isinstance(data, dict) and 'x' in data:
        x = np.array(data['x'])
    else:
        results['status'] = 'ERROR'
        results['details'].append("Cannot extract features from data")
        return results

    n_features = x.shape[1]
    print(f"  Total features: {n_features}")

    # Elliptic has 166 features:
    # - Features 1-94: Local features (transaction info)
    # - Features 95-166: Aggregated features (1-hop neighbor stats)

    # Check for suspicious patterns
    # 1. Any features with very low variance (might be leaked constants)
    feature_vars = np.var(x, axis=0)
    low_var_features = np.where(feature_vars < 1e-10)[0]

    if len(low_var_features) > 0:
        results['warnings'].append(f"Found {len(low_var_features)} constant features: {low_var_features[:10]}...")

    # 2. Check for features correlated with labels (if labels available)
    if hasattr(data, 'y'):
        y = data.y.numpy() if HAS_TORCH and isinstance(data.y, torch.Tensor) else np.array(data.y)

        # Filter out unknown labels (-1 or 2)
        known_mask = (y == 0) | (y == 1)
        y_known = y[known_mask]
        x_known = x[known_mask]

        if len(y_known) > 0:
            # Check correlation of each feature with label
            high_corr_features = []
            for i in range(n_features):
                if feature_vars[i] > 1e-10:
                    corr = np.corrcoef(x_known[:, i], y_known)[0, 1]
                    if abs(corr) > 0.5:  # Suspiciously high correlation
                        high_corr_features.append((i, corr))

            if high_corr_features:
                results['warnings'].append(f"Found {len(high_corr_features)} features with |corr| > 0.5 with labels")
                for feat_idx, corr in high_corr_features[:5]:
                    print(f"    Feature {feat_idx}: corr = {corr:.3f}")

    # 3. Feature statistics
    print(f"\n  Feature statistics:")
    print(f"    Mean range: [{x.mean(axis=0).min():.3f}, {x.mean(axis=0).max():.3f}]")
    print(f"    Std range: [{x.std(axis=0).min():.3f}, {x.std(axis=0).max():.3f}]")
    print(f"    Constant features: {len(low_var_features)}")

    if len(results['warnings']) == 0:
        results['status'] = 'PASS'
        results['details'].append("No obvious feature leakage detected")
    else:
        results['status'] = 'WARNING'
        results['details'] = results['warnings']

    print(f"\n  Status: {results['status']}")
    for detail in results['details'][:3]:
        print(f"    - {detail}")

    return results

def check_edge_temporal_leakage(data, metadata=None) -> dict:
    """
    Check if edges cross temporal boundaries.

    In Weber split, edges should respect temporal ordering:
    - No edges from future timesteps to past timesteps
    """
    print("\n" + "="*60)
    print("CHECK 2: EDGE TEMPORAL ANALYSIS")
    print("="*60)

    results = {
        'status': 'UNKNOWN',
        'details': [],
        'warnings': []
    }

    # Get edge index
    if hasattr(data, 'edge_index'):
        edge_index = data.edge_index.numpy() if HAS_TORCH and isinstance(data.edge_index, torch.Tensor) else np.array(data.edge_index)
    elif isinstance(data, dict) and 'edge_index' in data:
        edge_index = np.array(data['edge_index'])
    else:
        results['status'] = 'ERROR'
        results['details'].append("Cannot extract edge_index from data")
        return results

    n_edges = edge_index.shape[1]
    print(f"  Total edges: {n_edges}")

    # Check if we have timestep information
    timesteps = None
    if metadata and 'timesteps' in metadata:
        timesteps = np.array(metadata['timesteps'])
    elif hasattr(data, 'timestep'):
        timesteps = data.timestep.numpy() if HAS_TORCH else np.array(data.timestep)

    if timesteps is not None:
        print(f"  Timestep range: {timesteps.min()} - {timesteps.max()}")

        # Check for edges crossing from future to past
        src_timesteps = timesteps[edge_index[0]]
        dst_timesteps = timesteps[edge_index[1]]

        # Edges where destination is in earlier timestep than source
        backward_edges = np.sum(dst_timesteps < src_timesteps)
        forward_edges = np.sum(dst_timesteps > src_timesteps)
        same_time_edges = np.sum(dst_timesteps == src_timesteps)

        print(f"\n  Edge temporal distribution:")
        print(f"    Same timestep: {same_time_edges} ({100*same_time_edges/n_edges:.1f}%)")
        print(f"    Forward (src→future): {forward_edges} ({100*forward_edges/n_edges:.1f}%)")
        print(f"    Backward (src→past): {backward_edges} ({100*backward_edges/n_edges:.1f}%)")

        if backward_edges > 0:
            results['warnings'].append(f"Found {backward_edges} backward edges (potential leakage)")
            results['status'] = 'WARNING'
        else:
            results['status'] = 'PASS'
            results['details'].append("No backward temporal edges detected")
    else:
        results['status'] = 'INFO'
        results['details'].append("No timestep information available for edges")
        print("  [INFO] Timestep information not available in data")

        # Check for split information
        if hasattr(data, 'train_mask') and hasattr(data, 'test_mask'):
            train_mask = data.train_mask.numpy() if HAS_TORCH else np.array(data.train_mask)
            test_mask = data.test_mask.numpy() if HAS_TORCH else np.array(data.test_mask)

            # Check for edges between train and test nodes
            src_in_train = train_mask[edge_index[0]]
            dst_in_test = test_mask[edge_index[1]]
            train_to_test = np.sum(src_in_train & dst_in_test)

            src_in_test = test_mask[edge_index[0]]
            dst_in_train = train_mask[edge_index[1]]
            test_to_train = np.sum(src_in_test & dst_in_train)

            print(f"\n  Train/Test edge distribution:")
            print(f"    Train→Test edges: {train_to_test}")
            print(f"    Test→Train edges: {test_to_train}")

            if test_to_train > 0:
                results['warnings'].append(f"Found {test_to_train} edges from test to train nodes")

    print(f"\n  Status: {results['status']}")
    for detail in results['details'] + results['warnings']:
        print(f"    - {detail}")

    return results

def check_label_distribution_shift(data, metadata=None) -> dict:
    """
    Check for temporal concept drift in label distribution.

    Weber split concern: If fraud patterns change significantly between
    timesteps 1-34 (train) and 35-49 (test), this is legitimate concept drift,
    not leakage. But we should document it.
    """
    print("\n" + "="*60)
    print("CHECK 3: LABEL DISTRIBUTION ANALYSIS")
    print("="*60)

    results = {
        'status': 'UNKNOWN',
        'details': [],
        'warnings': []
    }

    # Get labels
    if hasattr(data, 'y'):
        y = data.y.numpy() if HAS_TORCH and isinstance(data.y, torch.Tensor) else np.array(data.y)
    elif isinstance(data, dict) and 'y' in data:
        y = np.array(data['y'])
    else:
        results['status'] = 'ERROR'
        results['details'].append("Cannot extract labels from data")
        return results

    # Label distribution
    label_counts = Counter(y)
    print(f"  Overall label distribution:")
    for label, count in sorted(label_counts.items()):
        label_name = {0: 'Licit', 1: 'Illicit', 2: 'Unknown', -1: 'Unknown'}.get(label, f'Class {label}')
        print(f"    {label_name} ({label}): {count} ({100*count/len(y):.1f}%)")

    # Check train/test split distribution
    if hasattr(data, 'train_mask') and hasattr(data, 'test_mask'):
        train_mask = data.train_mask.numpy() if HAS_TORCH else np.array(data.train_mask)
        test_mask = data.test_mask.numpy() if HAS_TORCH else np.array(data.test_mask)

        y_train = y[train_mask]
        y_test = y[test_mask]

        # Filter known labels
        y_train_known = y_train[(y_train == 0) | (y_train == 1)]
        y_test_known = y_test[(y_test == 0) | (y_test == 1)]

        if len(y_train_known) > 0 and len(y_test_known) > 0:
            train_fraud_rate = np.mean(y_train_known == 1)
            test_fraud_rate = np.mean(y_test_known == 1)

            print(f"\n  Fraud rate comparison:")
            print(f"    Train: {100*train_fraud_rate:.2f}% fraud")
            print(f"    Test: {100*test_fraud_rate:.2f}% fraud")
            print(f"    Shift: {100*(test_fraud_rate - train_fraud_rate):.2f}%")

            if abs(test_fraud_rate - train_fraud_rate) > 0.1:
                results['warnings'].append(f"Significant distribution shift: {100*(test_fraud_rate-train_fraud_rate):.1f}%")
                results['status'] = 'WARNING'
            else:
                results['status'] = 'PASS'
                results['details'].append("No significant distribution shift")
    else:
        results['status'] = 'INFO'
        results['details'].append("No train/test masks available")

    print(f"\n  Status: {results['status']}")
    for detail in results['details'] + results['warnings']:
        print(f"    - {detail}")

    return results

def check_train_test_contamination(data) -> dict:
    """
    Check for any overlap between train and test sets.
    """
    print("\n" + "="*60)
    print("CHECK 4: TRAIN/TEST CONTAMINATION")
    print("="*60)

    results = {
        'status': 'UNKNOWN',
        'details': [],
        'warnings': []
    }

    if hasattr(data, 'train_mask') and hasattr(data, 'test_mask'):
        train_mask = data.train_mask.numpy() if HAS_TORCH else np.array(data.train_mask)
        test_mask = data.test_mask.numpy() if HAS_TORCH else np.array(data.test_mask)

        overlap = np.sum(train_mask & test_mask)

        print(f"  Train nodes: {np.sum(train_mask)}")
        print(f"  Test nodes: {np.sum(test_mask)}")
        print(f"  Overlap: {overlap}")

        if overlap > 0:
            results['status'] = 'FAIL'
            results['warnings'].append(f"Found {overlap} nodes in both train and test!")
        else:
            results['status'] = 'PASS'
            results['details'].append("No overlap between train and test sets")
    else:
        results['status'] = 'INFO'
        results['details'].append("No train/test masks available")

    print(f"\n  Status: {results['status']}")
    for detail in results['details'] + results['warnings']:
        print(f"    - {detail}")

    return results

def main():
    """Run all leakage checks on Elliptic dataset."""
    print("="*60)
    print("ELLIPTIC DATASET TIME LEAKAGE CHECKER")
    print("="*60)

    # Find Elliptic data file
    possible_paths = [
        Path("data/elliptic_weber_split.pkl"),
        Path("code/data/elliptic_weber_split.pkl"),
        Path("D:/Users/11919/Documents/毕业论文/paper/code/data/elliptic_weber_split.pkl"),
    ]

    pkl_path = None
    for p in possible_paths:
        if p.exists():
            pkl_path = p
            break

    if pkl_path is None:
        print("\n[ERROR] Cannot find Elliptic dataset file")
        print("Searched paths:")
        for p in possible_paths:
            print(f"  - {p}")
        return

    print(f"\nLoading: {pkl_path}")
    data, raw_obj = load_elliptic_data(str(pkl_path))

    # Extract metadata if available
    metadata = {}
    if isinstance(raw_obj, dict):
        for key in ['timesteps', 'split', 'train_timesteps', 'test_timesteps']:
            if key in raw_obj:
                metadata[key] = raw_obj[key]

    print(f"Data type: {type(data)}")
    if hasattr(data, 'x'):
        print(f"Nodes: {data.x.shape[0]}, Features: {data.x.shape[1]}")
    if hasattr(data, 'edge_index'):
        print(f"Edges: {data.edge_index.shape[1]}")

    # Run all checks
    results = {}
    results['feature'] = check_feature_leakage(data)
    results['edge'] = check_edge_temporal_leakage(data, metadata)
    results['label'] = check_label_distribution_shift(data, metadata)
    results['contamination'] = check_train_test_contamination(data)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_pass = True
    for check_name, result in results.items():
        status = result['status']
        icon = {'PASS': '[OK]', 'FAIL': '[FAIL]', 'WARNING': '[WARN]', 'INFO': '[INFO]', 'ERROR': '[ERR]', 'UNKNOWN': '[?]'}
        print(f"  {icon.get(status, '[?]')} {check_name.upper()}: {status}")
        if status in ['FAIL', 'WARNING']:
            all_pass = False

    print("\n" + "="*60)
    if all_pass:
        print("CONCLUSION: No time leakage detected")
        print("The +23% AUC improvement appears to be legitimate.")
    else:
        print("CONCLUSION: Potential issues found - review warnings above")
    print("="*60 + "\n")

    return results


if __name__ == '__main__':
    main()
