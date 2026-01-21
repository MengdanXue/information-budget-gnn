#!/usr/bin/env python3
"""
Dataset Downloader for FSD Framework Validation
================================================

This script downloads and preprocesses 20+ graph datasets for validating
the FSD framework's multi-metric ensemble approach.

Target datasets:
- 8 Financial Fraud datasets
- 6 Heterophilic benchmark datasets
- 4 Social network datasets
- 3 Synthetic cSBM datasets

Usage:
    python download_datasets.py --all          # Download all available
    python download_datasets.py --fraud        # Fraud datasets only
    python download_datasets.py --hetero       # Heterophilic only
    python download_datasets.py --list         # List available datasets

Author: FSD Framework (TKDE Submission)
Date: 2025-12-23
"""

import os
import sys
import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

# Try importing PyG
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.datasets import (
        Planetoid,
        WikipediaNetwork,
        Actor,
        WebKB,
        Reddit,
        Flickr,
    )
    from torch_geometric.utils import to_undirected, degree
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: PyTorch Geometric not installed. Some datasets unavailable.")

# Try importing PyGOD for fraud datasets
try:
    from pygod.utils import load_data as pygod_load
    HAS_PYGOD = True
except ImportError:
    HAS_PYGOD = False
    print("Warning: PyGOD not installed. Fraud datasets unavailable.")
    print("Install with: pip install pygod")

# Try importing DGL
try:
    import dgl
    from dgl.data import FraudDataset, CoraGraphDataset
    HAS_DGL = True
except (ImportError, OSError, FileNotFoundError) as e:
    HAS_DGL = False
    print(f"Warning: DGL not available ({type(e).__name__}). Fraud datasets will use PyGOD instead.")


# =============================================================================
# Dataset Registry
# =============================================================================

DATASET_REGISTRY = {
    # Financial Fraud Datasets
    'yelp': {
        'category': 'fraud',
        'source': 'pygod',
        'description': 'Yelp spam review detection',
        'nodes': '~45k',
        'features': '32',
    },
    'amazon': {
        'category': 'fraud',
        'source': 'pygod',
        'description': 'Amazon product review fraud',
        'nodes': '~11k',
        'features': '25',
    },
    'elliptic': {
        'category': 'fraud',
        'source': 'local',
        'description': 'Bitcoin transaction fraud (already available)',
        'nodes': '~203k',
        'features': '165',
    },
    'ieee-cis': {
        'category': 'fraud',
        'source': 'local',
        'description': 'IEEE-CIS transaction fraud (already available)',
        'nodes': '~590k',
        'features': '394',
    },

    # Heterophilic Benchmark Datasets
    'cornell': {
        'category': 'hetero',
        'source': 'pyg',
        'description': 'WebKB Cornell university pages',
        'nodes': '183',
        'features': '1703',
    },
    'texas': {
        'category': 'hetero',
        'source': 'pyg',
        'description': 'WebKB Texas university pages',
        'nodes': '183',
        'features': '1703',
    },
    'wisconsin': {
        'category': 'hetero',
        'source': 'pyg',
        'description': 'WebKB Wisconsin university pages',
        'nodes': '251',
        'features': '1703',
    },
    'chameleon': {
        'category': 'hetero',
        'source': 'pyg',
        'description': 'Wikipedia chameleon articles',
        'nodes': '~2k',
        'features': '2325',
    },
    'squirrel': {
        'category': 'hetero',
        'source': 'pyg',
        'description': 'Wikipedia squirrel articles',
        'nodes': '~5k',
        'features': '2089',
    },
    'actor': {
        'category': 'hetero',
        'source': 'pyg',
        'description': 'Actor co-occurrence network',
        'nodes': '~7k',
        'features': '931',
    },

    # Homophilic Benchmark Datasets (for comparison)
    'cora': {
        'category': 'homo',
        'source': 'pyg',
        'description': 'Cora citation network',
        'nodes': '2708',
        'features': '1433',
    },
    'citeseer': {
        'category': 'homo',
        'source': 'pyg',
        'description': 'CiteSeer citation network',
        'nodes': '3327',
        'features': '3703',
    },
    'pubmed': {
        'category': 'homo',
        'source': 'pyg',
        'description': 'PubMed citation network',
        'nodes': '19717',
        'features': '500',
    },

    # Social Network Datasets
    'reddit': {
        'category': 'social',
        'source': 'pyg',
        'description': 'Reddit post communities',
        'nodes': '~232k',
        'features': '602',
    },
    'flickr': {
        'category': 'social',
        'source': 'pyg',
        'description': 'Flickr image network',
        'nodes': '~89k',
        'features': '500',
    },
}


# =============================================================================
# Dataset Loaders
# =============================================================================

def load_pyg_dataset(name: str, root: str = './data') -> Optional[Data]:
    """Load a PyTorch Geometric dataset."""
    if not HAS_PYG:
        print(f"  [ERROR] PyG not installed, cannot load {name}")
        return None

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        if name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=str(root), name=name.capitalize())
            return dataset[0]

        elif name in ['cornell', 'texas', 'wisconsin']:
            dataset = WebKB(root=str(root), name=name.capitalize())
            return dataset[0]

        elif name in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(root=str(root), name=name, geom_gcn_preprocess=True)
            return dataset[0]

        elif name == 'actor':
            dataset = Actor(root=str(root))
            return dataset[0]

        elif name == 'reddit':
            dataset = Reddit(root=str(root / 'Reddit'))
            return dataset[0]

        elif name == 'flickr':
            dataset = Flickr(root=str(root / 'Flickr'))
            return dataset[0]

        else:
            print(f"  [ERROR] Unknown PyG dataset: {name}")
            return None

    except Exception as e:
        print(f"  [ERROR] Failed to load {name}: {e}")
        return None


def load_pygod_dataset(name: str, root: str = './data') -> Optional[Data]:
    """Load a PyGOD fraud detection dataset."""
    if not HAS_PYGOD:
        print(f"  [ERROR] PyGOD not installed, cannot load {name}")
        return None

    try:
        # PyGOD dataset names
        pygod_names = {
            'yelp': 'inj_flickr',  # YelpChi needs special handling
            'amazon': 'inj_amazon',
        }

        # Try direct loading
        if name == 'yelp':
            # YelpChi from DGL
            if HAS_DGL:
                dataset = FraudDataset('yelp', raw_dir=root)
                g = dataset[0]
                # Convert DGL to PyG
                data = dgl_to_pyg(g)
                return data
            else:
                print("  [ERROR] DGL required for YelpChi dataset")
                return None

        elif name == 'amazon':
            if HAS_DGL:
                dataset = FraudDataset('amazon', raw_dir=root)
                g = dataset[0]
                data = dgl_to_pyg(g)
                return data
            else:
                print("  [ERROR] DGL required for Amazon dataset")
                return None

        else:
            print(f"  [ERROR] Unknown PyGOD dataset: {name}")
            return None

    except Exception as e:
        print(f"  [ERROR] Failed to load {name}: {e}")
        return None


def dgl_to_pyg(g) -> Data:
    """Convert a DGL graph to PyG Data object."""
    # Get edges
    src, dst = g.edges()
    edge_index = torch.stack([src, dst], dim=0).long()

    # Get node features
    if 'feature' in g.ndata:
        x = g.ndata['feature'].float()
    elif 'feat' in g.ndata:
        x = g.ndata['feat'].float()
    else:
        # Create dummy features
        x = torch.ones(g.num_nodes(), 1)

    # Get labels
    if 'label' in g.ndata:
        y = g.ndata['label'].long()
    else:
        y = torch.zeros(g.num_nodes()).long()

    # Create PyG data
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = g.num_nodes()

    return data


def generate_csbm_dataset(
    n_nodes: int = 1000,
    n_features: int = 100,
    n_classes: int = 2,
    homophily: float = 0.5,
    feature_noise: float = 0.1,
    seed: int = 42
) -> Data:
    """
    Generate a Contextual Stochastic Block Model (cSBM) dataset.

    This allows controlled experiments varying homophily and feature quality.

    Args:
        n_nodes: Number of nodes
        n_features: Feature dimension
        n_classes: Number of classes
        homophily: Edge homophily ratio (0=heterophilic, 1=homophilic)
        feature_noise: Noise level in features
        seed: Random seed

    Returns:
        PyG Data object
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Assign class labels
    y = torch.randint(0, n_classes, (n_nodes,))

    # Generate class-specific features with noise
    class_centers = torch.randn(n_classes, n_features)
    x = class_centers[y] + feature_noise * torch.randn(n_nodes, n_features)

    # Generate edges based on homophily
    edges = []
    avg_degree = 10
    n_edges_target = n_nodes * avg_degree // 2

    for _ in range(n_edges_target):
        i = np.random.randint(n_nodes)

        # With probability=homophily, connect to same class
        if np.random.random() < homophily:
            # Same class edge
            same_class = (y == y[i]).nonzero().squeeze(-1)
            if len(same_class) > 1:
                j = same_class[np.random.randint(len(same_class))].item()
            else:
                j = np.random.randint(n_nodes)
        else:
            # Different class edge
            diff_class = (y != y[i]).nonzero().squeeze(-1)
            if len(diff_class) > 0:
                j = diff_class[np.random.randint(len(diff_class))].item()
            else:
                j = np.random.randint(n_nodes)

        if i != j:
            edges.append([i, j])
            edges.append([j, i])  # Undirected

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)

    data = Data(x=x.float(), edge_index=edge_index, y=y)
    data.num_nodes = n_nodes

    return data


# =============================================================================
# Main Download Function
# =============================================================================

def download_dataset(name: str, output_dir: str = './data') -> bool:
    """Download a single dataset and save as pickle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = DATASET_REGISTRY.get(name)
    if info is None:
        print(f"  [ERROR] Unknown dataset: {name}")
        return False

    print(f"  Downloading {name} ({info['description']})...")

    source = info['source']
    data = None

    if source == 'pyg':
        data = load_pyg_dataset(name, str(output_dir))
    elif source == 'pygod':
        data = load_pygod_dataset(name, str(output_dir))
    elif source == 'local':
        print(f"  [SKIP] {name} already available locally")
        return True
    else:
        print(f"  [ERROR] Unknown source: {source}")
        return False

    if data is None:
        return False

    # Save as pickle
    output_path = output_dir / f"{name}_graph.pkl"
    save_dict = {
        'data': data,
        'name': name,
        'category': info['category'],
        'description': info['description'],
    }

    with open(output_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"  [OK] Saved to {output_path}")
    print(f"       Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}, Features: {data.x.size(1)}")

    return True


def download_all(category: Optional[str] = None, output_dir: str = './data'):
    """Download all datasets in a category or all categories."""
    print("\n" + "="*60)
    print("FSD Framework Dataset Downloader")
    print("="*60 + "\n")

    results = {'success': [], 'failed': [], 'skipped': []}

    for name, info in DATASET_REGISTRY.items():
        if category and info['category'] != category:
            continue

        print(f"\n[{name.upper()}]")

        if info['source'] == 'local':
            results['skipped'].append(name)
            print(f"  [SKIP] Already available locally")
            continue

        success = download_dataset(name, output_dir)

        if success:
            results['success'].append(name)
        else:
            results['failed'].append(name)

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Success: {len(results['success'])} - {', '.join(results['success']) or 'None'}")
    print(f"Failed:  {len(results['failed'])} - {', '.join(results['failed']) or 'None'}")
    print(f"Skipped: {len(results['skipped'])} - {', '.join(results['skipped']) or 'None'}")
    print("="*60 + "\n")

    return results


def generate_synthetic_datasets(output_dir: str = './data'):
    """Generate cSBM synthetic datasets for theory validation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Generating cSBM Synthetic Datasets")
    print("="*60 + "\n")

    configs = [
        {'name': 'csbm_high_homo', 'homophily': 0.9, 'feature_noise': 0.1},
        {'name': 'csbm_mid_homo', 'homophily': 0.5, 'feature_noise': 0.1},
        {'name': 'csbm_low_homo', 'homophily': 0.1, 'feature_noise': 0.1},
        {'name': 'csbm_noisy_feat', 'homophily': 0.5, 'feature_noise': 0.5},
        {'name': 'csbm_clean_feat', 'homophily': 0.5, 'feature_noise': 0.01},
    ]

    for cfg in configs:
        print(f"  Generating {cfg['name']}...")

        data = generate_csbm_dataset(
            n_nodes=2000,
            n_features=100,
            n_classes=2,
            homophily=cfg['homophily'],
            feature_noise=cfg['feature_noise'],
            seed=42
        )

        # Save
        output_path = output_dir / f"{cfg['name']}_graph.pkl"
        save_dict = {
            'data': data,
            'name': cfg['name'],
            'category': 'synthetic',
            'config': cfg,
        }

        with open(output_path, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"  [OK] Saved to {output_path}")
        print(f"       h={cfg['homophily']}, noise={cfg['feature_noise']}")

    print("\n" + "="*60 + "\n")


def list_datasets():
    """List all available datasets."""
    print("\n" + "="*70)
    print("AVAILABLE DATASETS FOR FSD FRAMEWORK")
    print("="*70 + "\n")

    categories = {}
    for name, info in DATASET_REGISTRY.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info))

    for cat, datasets in categories.items():
        cat_names = {
            'fraud': 'Financial Fraud',
            'hetero': 'Heterophilic Benchmark',
            'homo': 'Homophilic Benchmark',
            'social': 'Social Network',
        }
        print(f"\n{cat_names.get(cat, cat).upper()} ({len(datasets)} datasets)")
        print("-" * 70)
        print(f"{'Name':<15} {'Source':<10} {'Nodes':<10} {'Features':<10} {'Description'}")
        print("-" * 70)

        for name, info in datasets:
            print(f"{name:<15} {info['source']:<10} {info['nodes']:<10} {info['features']:<10} {info['description']}")

    print("\n" + "="*70)
    print("Total: {} datasets".format(len(DATASET_REGISTRY)))
    print("="*70 + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download datasets for FSD framework")
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--fraud', action='store_true', help='Download fraud datasets only')
    parser.add_argument('--hetero', action='store_true', help='Download heterophilic datasets only')
    parser.add_argument('--homo', action='store_true', help='Download homophilic datasets only')
    parser.add_argument('--social', action='store_true', help='Download social network datasets')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic cSBM datasets')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--output', type=str, default='./data', help='Output directory')
    parser.add_argument('--dataset', type=str, help='Download specific dataset by name')

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.dataset:
        download_dataset(args.dataset, args.output)
        return

    if args.synthetic:
        generate_synthetic_datasets(args.output)
        return

    if args.fraud:
        download_all('fraud', args.output)
    elif args.hetero:
        download_all('hetero', args.output)
    elif args.homo:
        download_all('homo', args.output)
    elif args.social:
        download_all('social', args.output)
    elif args.all:
        download_all(None, args.output)
        generate_synthetic_datasets(args.output)
    else:
        parser.print_help()
        print("\n[TIP] Use --list to see available datasets")


if __name__ == '__main__':
    main()
