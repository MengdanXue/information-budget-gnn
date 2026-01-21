"""
Quick Test Script for Case Study Generation

This script validates that all dependencies are installed and data is accessible
before running the full case study generation.

Usage:
    python test_case_study_setup.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")

    required_packages = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'networkx': 'NetworkX',
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)

    if missing:
        print(f"\nERROR: Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("\nAll imports successful!")
    return True


def test_data_availability(data_dir='./data'):
    """Test that Elliptic data files are available."""
    print(f"\nChecking data availability in {data_dir}...")

    data_dir = Path(data_dir)

    required_files = [
        'elliptic_txs_features.csv',
        'elliptic_txs_classes.csv',
        'elliptic_txs_edgelist.csv'
    ]

    # Check for processed file
    processed_file = data_dir / 'elliptic_weber_split.pkl'
    if processed_file.exists():
        print(f"  ✓ Processed data found: {processed_file}")
        return True

    # Check for raw files
    missing = []
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {file} - NOT FOUND")
            missing.append(file)

    if missing:
        print(f"\nWARNING: Missing data files: {', '.join(missing)}")
        print("\nDownload Elliptic dataset from:")
        print("  https://www.kaggle.com/ellipticco/elliptic-data-set")
        print(f"\nPlace files in: {data_dir.absolute()}")
        return False

    print("\nAll data files found!")
    return True


def test_device():
    """Test CUDA availability."""
    print("\nChecking device availability...")

    import torch

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("  ! CUDA not available, will use CPU")
        print("    Training will be slower on CPU")

    return device


def test_model_instantiation():
    """Test that models can be instantiated."""
    print("\nTesting model instantiation...")

    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATConv, GCNConv

    try:
        # Simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(10, 16)
                self.conv2 = GCNConv(16, 2)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = self.conv2(x, edge_index)
                return x

        model = TestModel()

        # Test forward pass
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        out = model(x, edge_index)

        assert out.shape == (100, 2), "Unexpected output shape"

        print("  ✓ Model instantiation successful")
        print("  ✓ Forward pass successful")
        return True

    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_visualization():
    """Test that matplotlib can create figures."""
    print("\nTesting visualization capabilities...")

    import matplotlib.pyplot as plt
    import numpy as np

    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('Test Plot')
        plt.close(fig)

        print("  ✓ Matplotlib working")
        return True

    except Exception as e:
        print(f"  ✗ Visualization test failed: {e}")
        return False


def test_graph_processing():
    """Test PyTorch Geometric graph operations."""
    print("\nTesting graph processing...")

    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import k_hop_subgraph

    try:
        # Create small test graph
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2]
        ], dtype=torch.long)

        x = torch.randn(4, 8)
        y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)

        # Test k-hop subgraph
        subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
            0, 2, data.edge_index, relabel_nodes=True
        )

        print(f"  ✓ Graph creation successful")
        print(f"  ✓ k-hop subgraph extraction successful")
        print(f"    Original nodes: {data.num_nodes}, Subgraph nodes: {len(subset)}")

        return True

    except Exception as e:
        print(f"  ✗ Graph processing failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("CASE STUDY SETUP VALIDATION")
    print("="*60)

    results = []

    # Run tests
    results.append(('Imports', test_imports()))
    results.append(('Data', test_data_availability()))
    results.append(('Device', test_device() is not None))
    results.append(('Models', test_model_instantiation()))
    results.append(('Visualization', test_visualization()))
    results.append(('Graph Processing', test_graph_processing()))

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:<20} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All tests passed! Ready to generate case studies.")
        print("\nRun the case study script with:")
        print("  python generate_case_study.py --data_dir ./data --output_dir ../figures")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues before proceeding.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
