"""
Data Loaders for Baseline Comparison Experiments

Supports:
- IEEE-CIS Fraud Detection Dataset
- YelpChi (Yelp Chicago) Review Dataset
- Amazon Product Review Dataset
- Elliptic Bitcoin Transaction Dataset

Author: FSD-GNN Paper
Date: 2024-12-23
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """Base class for dataset loaders."""

    def __init__(self, root_dir='./data'):
        self.root_dir = root_dir

    def load(self):
        raise NotImplementedError


class IEEECISLoader(DatasetLoader):
    """
    IEEE-CIS Fraud Detection Dataset Loader.

    Expected format:
    - Processed pickle file with keys:
      ['features', 'edge_index', 'labels', 'train_mask', 'val_mask', 'test_mask']
    """

    def __init__(self, root_dir='./data', data_file='ieee_cis_graph.pkl'):
        super().__init__(root_dir)
        self.data_file = data_file

    def load(self):
        """Load IEEE-CIS data."""
        data_path = os.path.join(self.root_dir, self.data_file)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"IEEE-CIS data not found at {data_path}")

        print(f"Loading IEEE-CIS data from {data_path}")

        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        # Convert to PyG Data object
        x = torch.tensor(data_dict['features'], dtype=torch.float32)
        edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long)
        y = torch.tensor(data_dict['labels'], dtype=torch.long)

        train_mask = torch.tensor(data_dict['train_mask'], dtype=torch.bool)
        val_mask = torch.tensor(data_dict['val_mask'], dtype=torch.bool)
        test_mask = torch.tensor(data_dict['test_mask'], dtype=torch.bool)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.num_edges:,}")
        print(f"  Features: {data.num_features}")
        print(f"  Train: {train_mask.sum():,} | Val: {val_mask.sum():,} | Test: {test_mask.sum():,}")
        print(f"  Fraud rate: {(y == 1).float().mean():.4f}")

        return data


class YelpChiLoader(DatasetLoader):
    """
    YelpChi (Yelp Chicago) Review Dataset Loader.

    This dataset is commonly used for fraud detection benchmarks.
    Expected to be downloaded via DGL or manually placed.

    Format:
    - node features: (N, F) - reviewer/review features
    - edge_index: (2, E) - review-product or reviewer-reviewer edges
    - labels: (N,) - fraud (1) or benign (0)
    """

    def __init__(self, root_dir='./data', feature_dim=32):
        super().__init__(root_dir)
        self.feature_dim = feature_dim
        self.dataset_name = 'YelpChi'

    def load(self):
        """Load YelpChi data."""
        # Try to load from processed file
        processed_file = os.path.join(self.root_dir, 'yelpchi_processed.pkl')

        if os.path.exists(processed_file):
            print(f"Loading YelpChi from {processed_file}")
            with open(processed_file, 'rb') as f:
                data_dict = pickle.load(f)

            x = torch.tensor(data_dict['features'], dtype=torch.float32)
            edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long)
            y = torch.tensor(data_dict['labels'], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)

            # Create masks if not provided
            if 'train_mask' in data_dict:
                data.train_mask = torch.tensor(data_dict['train_mask'], dtype=torch.bool)
                data.val_mask = torch.tensor(data_dict['val_mask'], dtype=torch.bool)
                data.test_mask = torch.tensor(data_dict['test_mask'], dtype=torch.bool)
            else:
                data = self._create_splits(data)

        else:
            # Try to download via DGL
            print("Attempting to download YelpChi via DGL...")
            try:
                import dgl
                dataset = dgl.data.FraudDataset('yelp')
                graph = dataset[0]

                # Convert DGL to PyG
                x = graph.ndata['feature']
                edge_index = torch.stack(graph.edges(), dim=0)
                y = graph.ndata['label']

                data = Data(x=x, edge_index=edge_index, y=y)
                data = self._create_splits(data)

                # Save processed version
                self._save_processed(data, processed_file)

            except Exception as e:
                print(f"Failed to load YelpChi: {e}")
                print("Please manually download YelpChi dataset.")
                raise

        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.num_edges:,}")
        print(f"  Features: {data.num_features}")
        print(f"  Fraud rate: {(data.y == 1).float().mean():.4f}")

        return data

    def _create_splits(self, data, train_ratio=0.4, val_ratio=0.1):
        """Create train/val/test splits."""
        n = data.num_nodes
        indices = torch.randperm(n)

        train_size = int(train_ratio * n)
        val_size = int(val_ratio * n)

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data

    def _save_processed(self, data, filepath):
        """Save processed data."""
        data_dict = {
            'features': data.x.numpy(),
            'edge_index': data.edge_index.numpy(),
            'labels': data.y.numpy(),
            'train_mask': data.train_mask.numpy(),
            'val_mask': data.val_mask.numpy(),
            'test_mask': data.test_mask.numpy()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)

        print(f"Saved processed data to {filepath}")


class AmazonLoader(DatasetLoader):
    """
    Amazon Product Review Dataset Loader.

    Similar structure to YelpChi.
    """

    def __init__(self, root_dir='./data'):
        super().__init__(root_dir)
        self.dataset_name = 'Amazon'

    def load(self):
        """Load Amazon data."""
        processed_file = os.path.join(self.root_dir, 'amazon_processed.pkl')

        if os.path.exists(processed_file):
            print(f"Loading Amazon from {processed_file}")
            with open(processed_file, 'rb') as f:
                data_dict = pickle.load(f)

            x = torch.tensor(data_dict['features'], dtype=torch.float32)
            edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long)
            y = torch.tensor(data_dict['labels'], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)

            if 'train_mask' in data_dict:
                data.train_mask = torch.tensor(data_dict['train_mask'], dtype=torch.bool)
                data.val_mask = torch.tensor(data_dict['val_mask'], dtype=torch.bool)
                data.test_mask = torch.tensor(data_dict['test_mask'], dtype=torch.bool)
            else:
                data = YelpChiLoader._create_splits(self, data)

        else:
            # Try to download via DGL
            print("Attempting to download Amazon via DGL...")
            try:
                import dgl
                dataset = dgl.data.FraudDataset('amazon')
                graph = dataset[0]

                # Convert DGL to PyG
                x = graph.ndata['feature']
                edge_index = torch.stack(graph.edges(), dim=0)
                y = graph.ndata['label']

                data = Data(x=x, edge_index=edge_index, y=y)
                data = YelpChiLoader._create_splits(self, data)

                # Save processed version
                YelpChiLoader._save_processed(self, data, processed_file)

            except Exception as e:
                print(f"Failed to load Amazon: {e}")
                print("Please manually download Amazon dataset.")
                raise

        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.num_edges:,}")
        print(f"  Features: {data.num_features}")
        print(f"  Fraud rate: {(data.y == 1).float().mean():.4f}")

        return data


class EllipticLoader(DatasetLoader):
    """
    Elliptic Bitcoin Transaction Dataset Loader.

    Temporal graph with Bitcoin transactions labeled as licit/illicit.
    """

    def __init__(self, root_dir='./data'):
        super().__init__(root_dir)

    def load(self):
        """Load Elliptic data."""
        processed_file = os.path.join(self.root_dir, 'elliptic_weber_split.pkl')

        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Elliptic data not found at {processed_file}")

        print(f"Loading Elliptic data from {processed_file}")

        with open(processed_file, 'rb') as f:
            data_dict = pickle.load(f)

        # Convert to PyG Data object
        x = torch.tensor(data_dict['features'], dtype=torch.float32)
        edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long)
        y = torch.tensor(data_dict['labels'], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)

        if 'train_mask' in data_dict:
            data.train_mask = torch.tensor(data_dict['train_mask'], dtype=torch.bool)
            data.val_mask = torch.tensor(data_dict['val_mask'], dtype=torch.bool)
            data.test_mask = torch.tensor(data_dict['test_mask'], dtype=torch.bool)

        if 'time_step' in data_dict:
            data.time_step = torch.tensor(data_dict['time_step'], dtype=torch.long)

        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.num_edges:,}")
        print(f"  Features: {data.num_features}")
        print(f"  Illicit rate: {(y == 1).float().mean():.4f}")

        return data


# ========================================
# Unified Loader Factory
# ========================================
def load_dataset(dataset_name, root_dir='./data'):
    """
    Load dataset by name.

    Args:
        dataset_name: One of ['ieee-cis', 'yelpchi', 'amazon', 'elliptic']
        root_dir: Root directory for data

    Returns:
        PyG Data object with train/val/test masks
    """

    loaders = {
        'ieee-cis': IEEECISLoader,
        'yelpchi': YelpChiLoader,
        'amazon': AmazonLoader,
        'elliptic': EllipticLoader
    }

    dataset_name = dataset_name.lower()

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(loaders.keys())}")

    loader = loaders[dataset_name](root_dir=root_dir)
    return loader.load()


if __name__ == '__main__':
    # Test data loaders
    print("Testing data loaders...\n")

    # Test IEEE-CIS (if available)
    try:
        print("="*60)
        print("Testing IEEE-CIS Loader")
        print("="*60)
        data = load_dataset('ieee-cis', root_dir='../processed')
        print(f"Successfully loaded IEEE-CIS: {data}\n")
    except Exception as e:
        print(f"IEEE-CIS loading failed: {e}\n")

    # Test YelpChi
    try:
        print("="*60)
        print("Testing YelpChi Loader")
        print("="*60)
        data = load_dataset('yelpchi', root_dir='../data')
        print(f"Successfully loaded YelpChi: {data}\n")
    except Exception as e:
        print(f"YelpChi loading failed: {e}\n")

    # Test Elliptic
    try:
        print("="*60)
        print("Testing Elliptic Loader")
        print("="*60)
        data = load_dataset('elliptic', root_dir='../data')
        print(f"Successfully loaded Elliptic: {data}\n")
    except Exception as e:
        print(f"Elliptic loading failed: {e}\n")

    print("Data loader testing complete!")
