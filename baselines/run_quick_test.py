"""
Quick Test Script for Baseline Comparison Framework

This script runs a quick test with 1-3 seeds to verify everything works
before running full experiments.

Usage:
    python run_quick_test.py
"""

import sys
import torch
from baseline_models import create_baseline_model
from data_loaders import load_dataset

BASELINE_MODELS = ['ARC', 'GAGA', 'CARE-GNN', 'PC-GNN', 'VecAug', 'SEFraud']


def test_models():
    """Test all baseline models with dummy data."""
    print("="*80)
    print("Testing Baseline Models")
    print("="*80)

    n_nodes = 1000
    n_features = 64
    n_edges = 5000

    x = torch.randn(n_nodes, n_features)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))

    for model_name in BASELINE_MODELS:
        print(f"\nTesting {model_name}...")
        try:
            model = create_baseline_model(model_name, n_features, hidden_dim=64, out_dim=2)
            model.eval()

            with torch.no_grad():
                out = model(x, edge_index)

            print(f"  ✓ Input: {x.shape}")
            print(f"  ✓ Output: {out.shape}")
            print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

    print("\n" + "="*80)
    print("All models passed!")
    print("="*80)
    return True


def test_data_loaders():
    """Test data loaders."""
    print("\n" + "="*80)
    print("Testing Data Loaders")
    print("="*80)

    # Test IEEE-CIS
    try:
        print("\nTesting IEEE-CIS Loader...")
        data = load_dataset('ieee-cis', root_dir='../processed')
        print(f"  ✓ Loaded IEEE-CIS: {data}")
        print(f"  ✓ Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}, Features: {data.num_features}")
    except Exception as e:
        print(f"  ✗ IEEE-CIS Error (expected if data not available): {e}")

    # Test Elliptic
    try:
        print("\nTesting Elliptic Loader...")
        data = load_dataset('elliptic', root_dir='../data')
        print(f"  ✓ Loaded Elliptic: {data}")
        print(f"  ✓ Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}, Features: {data.num_features}")
    except Exception as e:
        print(f"  ✗ Elliptic Error (expected if data not available): {e}")

    print("\n" + "="*80)
    print("Data loader tests complete!")
    print("="*80)


def test_training_pipeline():
    """Test training pipeline with dummy data."""
    print("\n" + "="*80)
    print("Testing Training Pipeline")
    print("="*80)

    from run_baselines import run_single_experiment
    from torch_geometric.data import Data

    # Create dummy data
    n_nodes = 1000
    n_features = 64
    n_edges = 5000

    x = torch.randn(n_nodes, n_features)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    y = torch.randint(0, 2, (n_nodes,))

    # Create masks
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[:400] = True
    val_mask[400:600] = True
    test_mask[600:] = True

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test one model
    print(f"\nTesting ARC with dummy data on {device}...")
    try:
        metrics, epochs = run_single_experiment('ARC', data, seed=42, device=device)
        print(f"  ✓ Training completed in {epochs} epochs")
        print(f"  ✓ AUC: {metrics['auc']:.4f}")
        print(f"  ✓ F1: {metrics['f1']:.4f}")
    except Exception as e:
        print(f"  ✗ Training Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("Training pipeline test passed!")
    print("="*80)
    return True


def main():
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "QUICK TEST SUITE" + " "*37 + "║")
    print("╚" + "="*78 + "╝")

    # Test 1: Models
    if not test_models():
        print("\n✗ Model tests failed!")
        sys.exit(1)

    # Test 2: Data loaders
    test_data_loaders()

    # Test 3: Training pipeline
    if not test_training_pipeline():
        print("\n✗ Training pipeline test failed!")
        sys.exit(1)

    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*23 + "ALL TESTS PASSED!" + " "*39 + "║")
    print("║" + " "*15 + "Ready to run full experiments with run_baselines.py" + " "*12 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")


if __name__ == '__main__':
    main()
