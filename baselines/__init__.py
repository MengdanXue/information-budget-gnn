"""
2024 SOTA Baseline Comparison Framework for FSD-GNN Paper

This package provides implementations and wrappers for comparing FSD-GNN
against state-of-the-art fraud detection baselines from 2024.

Author: FSD-GNN Paper
Date: 2024-12-23
"""

from .baseline_models import (
    ARCInspired,
    GAGA,
    CAREGNN,
    PCGNN,
    VecAugInspired,
    SEFraudInspired,
    create_baseline_model
)

from .data_loaders import (
    IEEECISLoader,
    YelpChiLoader,
    AmazonLoader,
    EllipticLoader,
    load_dataset
)

__all__ = [
    # Models
    'ARCInspired',
    'GAGA',
    'CAREGNN',
    'PCGNN',
    'VecAugInspired',
    'SEFraudInspired',
    'create_baseline_model',

    # Data loaders
    'IEEECISLoader',
    'YelpChiLoader',
    'AmazonLoader',
    'EllipticLoader',
    'load_dataset',
]

__version__ = '1.0.0'
