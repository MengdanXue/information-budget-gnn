#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproducibility Utilities for FSD-GNN Experiments

This module provides utilities to ensure experiment reproducibility:
1. Unified seed management across all random sources
2. Environment logging for debugging
3. Configuration saving/loading

Usage:
    from reproducibility import set_all_seeds, log_environment

    set_all_seeds(42)  # Call at the start of every experiment
    log_environment('experiment_env.json')  # Save environment info

Author: FSD Framework
Date: 2024-12-23
Version: 1.0 (TKDE Submission)
"""

import os
import sys
import json
import random
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np


def set_all_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    This function sets seeds for:
    - Python's built-in random module
    - NumPy's random generator
    - PyTorch (if available)
    - CUDA (if available)
    - Environment variables for hash seed

    Args:
        seed: Random seed value
        deterministic: If True, sets CUDA to deterministic mode (slower but reproducible)

    Example:
        >>> set_all_seeds(42)
        >>> # Now all random operations will be reproducible
    """
    # Python built-in
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU

            if deterministic:
                # Make CUDA operations deterministic
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                # PyTorch >= 1.8
                if hasattr(torch, 'use_deterministic_algorithms'):
                    try:
                        torch.use_deterministic_algorithms(True)
                    except Exception:
                        pass  # Some operations don't support deterministic mode

        print(f"[Reproducibility] Seeds set to {seed} (PyTorch detected)")

    except ImportError:
        print(f"[Reproducibility] Seeds set to {seed} (PyTorch not available)")


def get_environment_info() -> Dict[str, Any]:
    """
    Collect environment information for logging.

    Returns:
        Dictionary containing environment details
    """
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': sys.platform,
        'numpy_version': np.__version__,
    }

    # PyTorch info
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = torch.backends.cudnn.version()
            info['gpu_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        info['torch_version'] = 'not installed'

    # Scikit-learn info
    try:
        import sklearn
        info['sklearn_version'] = sklearn.__version__
    except ImportError:
        info['sklearn_version'] = 'not installed'

    # PyTorch Geometric info
    try:
        import torch_geometric
        info['torch_geometric_version'] = torch_geometric.__version__
    except ImportError:
        info['torch_geometric_version'] = 'not installed'

    return info


def log_environment(output_path: str = 'experiment_environment.json') -> None:
    """
    Log environment information to a JSON file.

    Args:
        output_path: Path to save the environment info
    """
    info = get_environment_info()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[Reproducibility] Environment logged to {output_path}")


def compute_data_hash(data: np.ndarray) -> str:
    """
    Compute MD5 hash of numpy array for data integrity verification.

    Args:
        data: NumPy array to hash

    Returns:
        MD5 hash string
    """
    return hashlib.md5(data.tobytes()).hexdigest()


def save_experiment_config(
    config: Dict[str, Any],
    output_path: str,
    include_environment: bool = True
) -> None:
    """
    Save experiment configuration with optional environment info.

    Args:
        config: Dictionary of experiment parameters
        output_path: Path to save configuration
        include_environment: Whether to include environment info
    """
    save_dict = {
        'experiment_config': config,
        'saved_at': datetime.now().isoformat(),
    }

    if include_environment:
        save_dict['environment'] = get_environment_info()

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, indent=2, ensure_ascii=False)

    print(f"[Reproducibility] Config saved to {output_path}")


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class ExperimentContext:
    """
    Context manager for reproducible experiments.

    Usage:
        with ExperimentContext(seed=42, log_dir='./logs') as ctx:
            # Your experiment code here
            ctx.log('accuracy', 0.85)
    """

    def __init__(
        self,
        seed: int = 42,
        log_dir: str = './experiment_logs',
        experiment_name: Optional[str] = None,
        deterministic: bool = True
    ):
        self.seed = seed
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.deterministic = deterministic
        self.metrics: Dict[str, Any] = {}

    def __enter__(self):
        # Create log directory
        self.exp_dir = self.log_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Set seeds
        set_all_seeds(self.seed, self.deterministic)

        # Log environment
        log_environment(str(self.exp_dir / 'environment.json'))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Save metrics
        if self.metrics:
            metrics_path = self.exp_dir / 'metrics.json'
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2)
            print(f"[Reproducibility] Metrics saved to {metrics_path}")

        return False  # Don't suppress exceptions

    def log(self, key: str, value: Any) -> None:
        """Log a metric value."""
        self.metrics[key] = value


# Convenience function for quick setup
def quick_setup(seed: int = 42) -> None:
    """
    Quick setup for reproducibility. Call at the start of any script.

    Args:
        seed: Random seed

    Example:
        from reproducibility import quick_setup
        quick_setup(42)
    """
    set_all_seeds(seed)
    print(f"[Reproducibility] Quick setup complete (seed={seed})")


if __name__ == '__main__':
    # Demo usage
    print("=== Reproducibility Module Demo ===\n")

    # Set seeds
    set_all_seeds(42)

    # Show environment
    env_info = get_environment_info()
    print("\nEnvironment Info:")
    for key, value in env_info.items():
        print(f"  {key}: {value}")

    # Demo random numbers (should be same every run)
    print("\nRandom numbers (should be reproducible):")
    print(f"  Python random: {random.random():.6f}")
    print(f"  NumPy random: {np.random.random():.6f}")

    try:
        import torch
        print(f"  PyTorch random: {torch.rand(1).item():.6f}")
    except ImportError:
        pass

    print("\n=== Demo Complete ===")
