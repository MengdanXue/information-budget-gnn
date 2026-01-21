#!/usr/bin/env python3
"""
Test script to verify ablation study setup is correct.

This script performs quick sanity checks:
1. Check Python packages are installed
2. Verify data files exist and are readable
3. Test NAA model variants can be instantiated
4. Run a quick mini-experiment (1 epoch) to verify training works

Usage:
    python test_ablation_setup.py
"""

import sys
import os
from pathlib import Path


def check_packages():
    """Check required packages are installed."""
    print("=" * 70)
    print("1. Checking Python Packages")
    print("=" * 70)

    required_packages = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'sklearn': 'scikit-learn',
        'scipy': 'SciPy',
        'numpy': 'NumPy'
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing.append(package)

    if missing:
        print(f"\nERROR: Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install torch torch_geometric scikit-learn scipy numpy")
        return False

    print("\n✓ All required packages installed\n")
    return True


def check_data_files():
    """Check if data files exist."""
    print("=" * 70)
    print("2. Checking Data Files")
    print("=" * 70)

    data_dir = Path("processed")
    expected_files = [
        "elliptic_graph.pkl",
        "ieee_cis_graph.pkl",
        "yelpchi_graph.pkl",
        "amazon_graph.pkl"
    ]

    found_files = []
    for filename in expected_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"✓ {filename} found")
            found_files.append(str(filepath))
        else:
            print(f"✗ {filename} NOT found (optional)")

    if not found_files:
        print("\nWARNING: No data files found in processed/ directory")
        print("You'll need at least one dataset to run ablation study.")
        print("\nExpected location: processed/{dataset}_graph.pkl")
        return None

    print(f"\n✓ Found {len(found_files)} dataset(s)\n")
    return found_files[0]  # Return first file for testing


def check_model_imports():
    """Check NAA model can be imported."""
    print("=" * 70)
    print("3. Checking Model Imports")
    print("=" * 70)

    try:
        from ablation_study import NAA_Full, BaselineGCN
        print("✓ NAA_Full imported")
        print("✓ BaselineGCN imported")
    except ImportError as e:
        print(f"✗ Failed to import models: {e}")
        return False

    print("\n✓ All models can be imported\n")
    return True


def test_model_instantiation():
    """Test that models can be instantiated."""
    print("=" * 70)
    print("4. Testing Model Instantiation")
    print("=" * 70)

    try:
        import torch
        from ablation_study import NAA_Full, BaselineGCN

        in_dim = 100

        # Test baseline
        model_baseline = BaselineGCN(in_dim=in_dim)
        print("✓ BaselineGCN instantiated")

        # Test NAA full
        model_full = NAA_Full(
            in_dim=in_dim,
            use_log_scale=True,
            use_feature_weights=True,
            use_adaptive_gate=True
        )
        print("✓ NAA (Full) instantiated")

        # Test NAA w/o log-scale
        model_no_log = NAA_Full(
            in_dim=in_dim,
            use_log_scale=False,
            use_feature_weights=True,
            use_adaptive_gate=True
        )
        print("✓ NAA w/o log-scale instantiated")

        # Test NAA w/o feature weights
        model_no_weights = NAA_Full(
            in_dim=in_dim,
            use_log_scale=True,
            use_feature_weights=False,
            use_adaptive_gate=True
        )
        print("✓ NAA w/o feature weights instantiated")

        # Test NAA w/o adaptive gate
        model_no_gate = NAA_Full(
            in_dim=in_dim,
            use_log_scale=True,
            use_feature_weights=True,
            use_adaptive_gate=False
        )
        print("✓ NAA w/o adaptive gate instantiated")

        # Test fixed lambda
        model_lambda = NAA_Full(
            in_dim=in_dim,
            use_log_scale=True,
            use_feature_weights=True,
            use_adaptive_gate=True,
            fixed_lambda=0.5
        )
        print("✓ NAA with fixed λ=0.5 instantiated")

        print("\n✓ All model variants instantiate correctly\n")
        return True

    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test that models can perform forward pass."""
    print("=" * 70)
    print("5. Testing Forward Pass")
    print("=" * 70)

    try:
        import torch
        from ablation_study import NAA_Full, BaselineGCN

        # Create dummy data
        n_nodes = 100
        n_features = 50
        n_edges = 500

        x = torch.randn(n_nodes, n_features)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))

        print(f"Created dummy graph: {n_nodes} nodes, {n_features} features, {n_edges} edges")

        # Test baseline
        model_baseline = BaselineGCN(in_dim=n_features)
        out_baseline = model_baseline(x, edge_index)
        print(f"✓ BaselineGCN forward: {out_baseline.shape}")

        # Test NAA
        model_naa = NAA_Full(in_dim=n_features)
        out_naa = model_naa(x, edge_index)
        print(f"✓ NAA forward: {out_naa.shape}")

        # Test lambda value retrieval
        lambda_val = model_naa.get_lambda_value()
        print(f"✓ Lambda value: {lambda_val:.4f}")

        print("\n✓ Forward pass successful\n")
        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training(data_file):
    """Test training for 1 epoch on real data."""
    print("=" * 70)
    print("6. Testing Mini Training Run")
    print("=" * 70)

    if data_file is None:
        print("✗ Skipping (no data file available)")
        return True  # Don't fail if no data

    try:
        import torch
        from ablation_study import NAA_Full, load_data, train_epoch, evaluate
        import torch.nn as nn

        print(f"Loading data from: {data_file}")
        data = load_data(data_file)
        print(f"✓ Data loaded: {data.x.shape[0]} nodes, {data.x.shape[1]} features")

        # Create model
        model = NAA_Full(in_dim=data.x.shape[1])
        print("✓ Model created")

        # Setup training
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✓ Using device: {device}")

        model = model.to(device)

        # Class weights
        n_pos = data.y[data.train_mask].sum().item()
        n_neg = data.train_mask.sum().item() - n_pos
        weight = torch.tensor([1.0, n_neg / n_pos], device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print("✓ Training setup complete")

        # Train for 1 epoch
        print("Running 1 training epoch...", end=' ')
        loss = train_epoch(model, data, optimizer, criterion, device)
        print(f"loss={loss:.4f}")

        # Evaluate
        print("Evaluating...", end=' ')
        val_metrics = evaluate(model, data, data.val_mask, device)
        print(f"val_auc={val_metrics['auc']:.4f}")

        print("\n✓ Mini training run successful\n")
        return True

    except Exception as e:
        print(f"\n✗ Mini training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY SETUP TEST")
    print("=" * 70 + "\n")

    results = []

    # Test 1: Packages
    results.append(("Packages", check_packages()))

    # Test 2: Data files
    data_file = check_data_files()
    results.append(("Data Files", data_file is not None))

    # Test 3: Model imports
    results.append(("Model Imports", check_model_imports()))

    # Test 4: Model instantiation
    results.append(("Model Instantiation", test_model_instantiation()))

    # Test 5: Forward pass
    results.append(("Forward Pass", test_forward_pass()))

    # Test 6: Mini training (only if we have data)
    if data_file:
        results.append(("Mini Training", test_mini_training(data_file)))

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<25} {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("SUCCESS: All tests passed!")
        print("=" * 70)
        print("\nYou're ready to run the ablation study:")
        print("  python ablation_study.py --data_path processed/elliptic_graph.pkl --dataset_name Elliptic")
        print("\nOr run all datasets:")
        print("  run_ablation_study.bat  (Windows)")
        print("  bash run_ablation_study.sh  (Linux/Mac)")
    else:
        print("FAILURE: Some tests failed")
        print("=" * 70)
        print("\nPlease fix the failed tests before running ablation study.")
        print("See error messages above for details.")
        sys.exit(1)

    print()


if __name__ == '__main__':
    main()
