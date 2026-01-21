"""
Run 2024 SOTA Baseline Comparison Experiments

This script runs experiments comparing FSD-GNN against SOTA baselines:
- ARC (NeurIPS 2024)
- GAGA (WWW 2023)
- CARE-GNN (CIKM 2020)
- PC-GNN (WWW 2021)
- VecAug (KDD 2024) - approximation
- SEFraud (KDD 2024) - approximation

Datasets: IEEE-CIS, YelpChi, Amazon, Elliptic

Usage:
    # Run single baseline on single dataset
    python run_baselines.py --model ARC --dataset ieee-cis --seeds 42 123 456

    # Run all baselines on all datasets
    python run_baselines.py --model all --dataset all

    # Run specific comparison
    python run_baselines.py --model GAGA PC-GNN --dataset yelpchi --seeds 42 123 456 789 1024

Author: FSD-GNN Paper
Date: 2024-12-23
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from tqdm import tqdm

# Import our modules
from baseline_models import create_baseline_model
from data_loaders import load_dataset

# Also import FSD-GNN models for comparison
sys.path.append('..')
from daaa_model import DAAA, DAAAv2, DAAAv3, DAAAv4


# ========================================
# Configuration
# ========================================
DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

BASELINE_MODELS = ['ARC', 'GAGA', 'CARE-GNN', 'PC-GNN', 'VecAug', 'SEFraud']
FSD_MODELS = ['DAAA', 'DAAAv2', 'DAAAv3', 'DAAAv4']
ALL_MODELS = BASELINE_MODELS + FSD_MODELS

DATASETS = ['ieee-cis', 'yelpchi', 'amazon', 'elliptic']

# Hyperparameters
HIDDEN_DIM = 128
DROPOUT = 0.5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 200
PATIENCE = 20


# ========================================
# Training and Evaluation
# ========================================
def train_epoch(model, data, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()

    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, device):
    """Evaluate model on given mask."""
    model.eval()

    out = model(data.x.to(device), data.edge_index.to(device))
    probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    preds = out.argmax(dim=1).cpu().numpy()
    labels = data.y.numpy()

    mask_np = mask.numpy()

    # Compute metrics
    metrics = {
        'auc': roc_auc_score(labels[mask_np], probs[mask_np]),
        'ap': average_precision_score(labels[mask_np], probs[mask_np]),
        'f1': f1_score(labels[mask_np], preds[mask_np], zero_division=0),
        'precision': precision_score(labels[mask_np], preds[mask_np], zero_division=0),
        'recall': recall_score(labels[mask_np], preds[mask_np], zero_division=0)
    }

    return metrics


def run_single_experiment(model_name, data, seed, device='cuda'):
    """Run single experiment with given model and seed."""

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    in_dim = data.x.shape[1]

    # Create model
    if model_name in BASELINE_MODELS:
        model = create_baseline_model(model_name, in_dim, HIDDEN_DIM, out_dim=2, dropout=DROPOUT)
    elif model_name == 'DAAA':
        model = DAAA(in_dim, hidden_dim=HIDDEN_DIM, out_dim=2, dropout=DROPOUT)
    elif model_name == 'DAAAv2':
        model = DAAAv2(in_dim, hidden_dim=HIDDEN_DIM, out_dim=2, dropout=DROPOUT)
    elif model_name == 'DAAAv3':
        model = DAAAv3(in_dim, hidden_dim=HIDDEN_DIM, out_dim=2, dropout=DROPOUT)
    elif model_name == 'DAAAv4':
        model = DAAAv4(in_dim, hidden_dim=HIDDEN_DIM, out_dim=2, dropout=DROPOUT)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)

    # Class weights for imbalanced data
    n_pos = data.y[data.train_mask].sum().item()
    n_neg = data.train_mask.sum().item() - n_pos
    if n_pos > 0:
        weight = torch.tensor([1.0, n_neg / n_pos], device=device)
    else:
        weight = torch.tensor([1.0, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    best_val_auc = 0
    best_test_metrics = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        loss = train_epoch(model, data, optimizer, criterion, device)
        val_metrics = evaluate(model, data, data.val_mask, device)

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_test_metrics = evaluate(model, data, data.test_mask, device)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    return best_test_metrics, epoch + 1


def run_experiments(model_names, dataset_names, seeds, data_root='../data', output_dir='./results'):
    """Run full experimental comparison."""

    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Results storage
    all_results = {}

    for dataset_name in dataset_names:
        print("="*80)
        print(f"Dataset: {dataset_name.upper()}")
        print("="*80)

        try:
            # Load data
            if dataset_name == 'ieee-cis':
                data = load_dataset(dataset_name, root_dir='../processed')
            else:
                data = load_dataset(dataset_name, root_dir=data_root)

            all_results[dataset_name] = {}

            for model_name in model_names:
                print(f"\n{'─'*80}")
                print(f"Model: {model_name}")
                print(f"{'─'*80}")

                model_results = {
                    'auc': [],
                    'ap': [],
                    'f1': [],
                    'precision': [],
                    'recall': [],
                    'epochs': []
                }

                for seed in tqdm(seeds, desc=f"Running {model_name}"):
                    try:
                        metrics, epochs = run_single_experiment(model_name, data, seed, device)

                        model_results['auc'].append(metrics['auc'])
                        model_results['ap'].append(metrics['ap'])
                        model_results['f1'].append(metrics['f1'])
                        model_results['precision'].append(metrics['precision'])
                        model_results['recall'].append(metrics['recall'])
                        model_results['epochs'].append(epochs)

                    except Exception as e:
                        print(f"\n  Error with seed {seed}: {e}")
                        continue

                # Compute statistics
                if len(model_results['auc']) > 0:
                    stats = {
                        'auc_mean': float(np.mean(model_results['auc'])),
                        'auc_std': float(np.std(model_results['auc'])),
                        'ap_mean': float(np.mean(model_results['ap'])),
                        'ap_std': float(np.std(model_results['ap'])),
                        'f1_mean': float(np.mean(model_results['f1'])),
                        'f1_std': float(np.std(model_results['f1'])),
                        'precision_mean': float(np.mean(model_results['precision'])),
                        'recall_mean': float(np.mean(model_results['recall'])),
                        'avg_epochs': float(np.mean(model_results['epochs'])),
                        'raw_results': model_results
                    }

                    all_results[dataset_name][model_name] = stats

                    print(f"\n  Results ({len(model_results['auc'])} seeds):")
                    print(f"    AUC: {stats['auc_mean']:.4f} ± {stats['auc_std']:.4f}")
                    print(f"    AP:  {stats['ap_mean']:.4f} ± {stats['ap_std']:.4f}")
                    print(f"    F1:  {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}")
                    print(f"    Avg Epochs: {stats['avg_epochs']:.1f}")

                else:
                    print(f"  No successful runs for {model_name}")

        except Exception as e:
            print(f"\nFailed to load dataset {dataset_name}: {e}")
            continue

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'baseline_comparison_{timestamp}.json')

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"{'Model':<15} {'AUC':<20} {'AP':<20} {'F1':<20}")
        print("-"*80)

        for model_name, stats in dataset_results.items():
            auc_str = f"{stats['auc_mean']:.4f} ± {stats['auc_std']:.4f}"
            ap_str = f"{stats['ap_mean']:.4f} ± {stats['ap_std']:.4f}"
            f1_str = f"{stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}"
            print(f"{model_name:<15} {auc_str:<20} {ap_str:<20} {f1_str:<20}")

    return all_results


# ========================================
# Main
# ========================================
def main():
    parser = argparse.ArgumentParser(
        description='Run 2024 SOTA Baseline Comparison Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, nargs='+', default=['all'],
                       help=f'Model(s) to run. Options: {ALL_MODELS + ["all"]}')
    parser.add_argument('--dataset', type=str, nargs='+', default=['all'],
                       help=f'Dataset(s) to use. Options: {DATASETS + ["all"]}')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS[:3],
                       help='Random seeds for experiments')
    parser.add_argument('--data_root', type=str, default='../data',
                       help='Root directory for datasets')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')

    args = parser.parse_args()

    # Update global hyperparameters
    global HIDDEN_DIM, DROPOUT
    HIDDEN_DIM = args.hidden_dim
    DROPOUT = args.dropout

    # Parse model names
    if 'all' in args.model:
        model_names = ALL_MODELS
    else:
        model_names = args.model

    # Parse dataset names
    if 'all' in args.dataset:
        dataset_names = DATASETS
    else:
        dataset_names = args.dataset

    print("\n" + "="*80)
    print("2024 SOTA BASELINE COMPARISON EXPERIMENTS")
    print("="*80)
    print(f"Models: {', '.join(model_names)}")
    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"Seeds: {args.seeds}")
    print(f"Hidden dim: {HIDDEN_DIM}, Dropout: {DROPOUT}")
    print("="*80 + "\n")

    # Run experiments
    start_time = time.time()
    results = run_experiments(model_names, dataset_names, args.seeds, args.data_root, args.output_dir)
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
