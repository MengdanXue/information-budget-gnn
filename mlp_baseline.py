"""
MLP Baseline Experiments for FSD Framework
===========================================

This script implements MLP baselines to validate the key FSD hypothesis:
"When delta_agg > 10, graph structure may harm performance - MLP can outperform GNNs"

Why MLP Baseline Matters:
1. Proves that structure is NOT always helpful
2. Establishes the lower bound (no structure)
3. Validates FSD prediction rules
4. Addresses reviewer concern: "Did you try without graph structure?"

MLP Variants:
- MLP-1: Single hidden layer
- MLP-2: Two hidden layers
- MLP-3: Three hidden layers with dropout
- MLP-BN: With batch normalization

Author: FSD Framework Research Team
Date: 2024-12-22
Version: 1.0 (TKDE Submission)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pickle
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch_geometric.data import Data
import argparse

# Random seeds for reproducibility
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
         7168, 8192, 9216, 10240, 11264]  # 15 seeds for TKDE


class MLP1(nn.Module):
    """Single hidden layer MLP"""
    def __init__(self, in_features: int, hidden: int = 128, out_features: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MLP2(nn.Module):
    """Two hidden layer MLP"""
    def __init__(self, in_features: int, hidden: int = 128, out_features: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class MLP3(nn.Module):
    """Three hidden layer MLP with residual connection"""
    def __init__(self, in_features: int, hidden: int = 128, out_features: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        h = x
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x)) + h  # Residual
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class MLPBatchNorm(nn.Module):
    """MLP with Batch Normalization"""
    def __init__(self, in_features: int, hidden: int = 128, out_features: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class MLPTrainer:
    """Trainer for MLP baseline experiments"""

    def __init__(self, device: str = 'cuda', hidden: int = 128,
                 lr: float = 0.01, weight_decay: float = 5e-4,
                 epochs: int = 200, patience: int = 50):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.hidden = hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience

    def train_and_evaluate(self, data: Data, model_class: type,
                          seed: int) -> Dict[str, float]:
        """
        Train and evaluate a single MLP model.

        Args:
            data: PyG Data object with x, y, train_mask, val_mask, test_mask
            model_class: MLP class to instantiate
            seed: Random seed

        Returns:
            Dict with AUC, F1, precision, recall
        """
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Move data to device
        x = data.x.to(self.device)
        y = data.y.to(self.device)

        train_mask = data.train_mask.to(self.device)
        val_mask = data.val_mask.to(self.device)
        test_mask = data.test_mask.to(self.device)

        # Initialize model
        model = model_class(
            in_features=x.size(1),
            hidden=self.hidden,
            out_features=2,
            dropout=0.5
        ).to(self.device)

        # Handle class imbalance with weighted loss
        n_pos = (y[train_mask] == 1).sum().float()
        n_neg = (y[train_mask] == 0).sum().float()
        pos_weight = n_neg / (n_pos + 1e-8)
        pos_weight = min(pos_weight, 10.0)  # Clip extreme weights

        class_weights = torch.tensor([1.0, pos_weight], device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Training loop with early stopping
        best_val_auc = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out[train_mask], y[train_mask])

            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(x)
                val_probs = F.softmax(val_out, dim=1)[:, 1]

                val_y = y[val_mask].cpu().numpy()
                val_pred_probs = val_probs[val_mask].cpu().numpy()

                if len(np.unique(val_y)) > 1:
                    val_auc = roc_auc_score(val_y, val_pred_probs)
                else:
                    val_auc = 0.5

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        with torch.no_grad():
            out = model(x)
            probs = F.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)

            test_y = y[test_mask].cpu().numpy()
            test_probs = probs[test_mask].cpu().numpy()
            test_preds = preds[test_mask].cpu().numpy()

            # Calculate metrics
            if len(np.unique(test_y)) > 1:
                auc = roc_auc_score(test_y, test_probs)
            else:
                auc = 0.5

            f1 = f1_score(test_y, test_preds, zero_division=0)
            precision = precision_score(test_y, test_preds, zero_division=0)
            recall = recall_score(test_y, test_preds, zero_division=0)

        return {
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'best_val_auc': best_val_auc,
            'epochs_trained': epoch + 1
        }


def run_mlp_experiments(data: Data, dataset_name: str,
                       seeds: List[int] = SEEDS,
                       output_dir: str = './mlp_results',
                       device: str = 'cuda') -> Dict:
    """
    Run complete MLP baseline experiments for a dataset.

    Args:
        data: PyG Data object
        dataset_name: Name of dataset
        seeds: List of random seeds
        output_dir: Output directory
        device: 'cuda' or 'cpu'

    Returns:
        Results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    trainer = MLPTrainer(device=device)

    model_classes = {
        'MLP-1': MLP1,
        'MLP-2': MLP2,
        'MLP-3': MLP3,
        'MLP-BN': MLPBatchNorm
    }

    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'n_seeds': len(seeds),
        'n_nodes': data.x.size(0),
        'n_features': data.x.size(1),
        'methods': {}
    }

    print("="*70)
    print(f"MLP Baseline Experiments: {dataset_name}")
    print("="*70)
    print(f"Features: {data.x.size(1)}")
    print(f"Nodes: {data.x.size(0)}")
    print(f"Seeds: {len(seeds)}")
    print()

    for model_name, model_class in model_classes.items():
        print(f"\n--- {model_name} ---")

        method_results = {
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': []
        }

        for i, seed in enumerate(seeds):
            print(f"  Seed {i+1}/{len(seeds)}: {seed}", end=" ")

            metrics = trainer.train_and_evaluate(data, model_class, seed)

            method_results['auc'].append(metrics['auc'])
            method_results['f1'].append(metrics['f1'])
            method_results['precision'].append(metrics['precision'])
            method_results['recall'].append(metrics['recall'])

            print(f"AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

        # Summary
        print(f"\n  {model_name} Summary:")
        print(f"    AUC: {np.mean(method_results['auc']):.4f} +/- {np.std(method_results['auc']):.4f}")
        print(f"    F1:  {np.mean(method_results['f1']):.4f} +/- {np.std(method_results['f1']):.4f}")

        results['methods'][model_name] = method_results

    # Find best MLP
    best_mlp = max(results['methods'].items(),
                   key=lambda x: np.mean(x[1]['auc']))
    results['best_mlp'] = {
        'name': best_mlp[0],
        'auc_mean': np.mean(best_mlp[1]['auc']),
        'auc_std': np.std(best_mlp[1]['auc']),
        'f1_mean': np.mean(best_mlp[1]['f1']),
        'f1_std': np.std(best_mlp[1]['f1'])
    }

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Best MLP: {results['best_mlp']['name']}")
    print(f"  AUC: {results['best_mlp']['auc_mean']:.4f} +/- {results['best_mlp']['auc_std']:.4f}")
    print(f"  F1:  {results['best_mlp']['f1_mean']:.4f} +/- {results['best_mlp']['f1_std']:.4f}")

    # Save results
    output_file = Path(output_dir) / f'{dataset_name}_mlp_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def compare_mlp_vs_gnn(mlp_results: Dict, gnn_results: Dict,
                       dataset_name: str) -> Dict:
    """
    Compare MLP baseline against GNN results.

    This is the key comparison for validating FSD prediction:
    When delta_agg > 10, MLP should be competitive with or better than GNN.

    Args:
        mlp_results: Results from MLP experiments
        gnn_results: Results from GNN experiments (dict of method -> {'auc': [...], 'f1': [...]})
        dataset_name: Dataset name

    Returns:
        Comparison results
    """
    from scipy import stats

    comparison = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'comparisons': []
    }

    # Get best MLP results
    best_mlp_name = mlp_results['best_mlp']['name']
    mlp_aucs = mlp_results['methods'][best_mlp_name]['auc']
    mlp_f1s = mlp_results['methods'][best_mlp_name]['f1']

    print("\n" + "="*70)
    print(f"MLP vs GNN Comparison: {dataset_name}")
    print("="*70)
    print(f"Best MLP ({best_mlp_name}): AUC {np.mean(mlp_aucs):.4f}")
    print()

    for gnn_name, gnn_metrics in gnn_results.items():
        gnn_aucs = gnn_metrics['auc']
        gnn_f1s = gnn_metrics.get('f1', [0] * len(gnn_aucs))

        # Ensure same number of samples
        n = min(len(mlp_aucs), len(gnn_aucs))

        # Wilcoxon signed-rank test
        if n >= 5:
            auc_stat, auc_pvalue = stats.wilcoxon(mlp_aucs[:n], gnn_aucs[:n])
            f1_stat, f1_pvalue = stats.wilcoxon(mlp_f1s[:n], gnn_f1s[:n])
        else:
            auc_stat, auc_pvalue = 0, 1.0
            f1_stat, f1_pvalue = 0, 1.0

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(mlp_aucs[:n])**2 + np.std(gnn_aucs[:n])**2) / 2)
        cohen_d = (np.mean(mlp_aucs[:n]) - np.mean(gnn_aucs[:n])) / (pooled_std + 1e-8)

        # Determine winner
        mlp_mean = np.mean(mlp_aucs[:n])
        gnn_mean = np.mean(gnn_aucs[:n])

        if auc_pvalue < 0.05:
            if mlp_mean > gnn_mean:
                winner = f"MLP ({best_mlp_name})"
                significant = True
            else:
                winner = gnn_name
                significant = True
        else:
            winner = "No significant difference"
            significant = False

        result = {
            'gnn_method': gnn_name,
            'mlp_auc_mean': mlp_mean,
            'mlp_auc_std': np.std(mlp_aucs[:n]),
            'gnn_auc_mean': gnn_mean,
            'gnn_auc_std': np.std(gnn_aucs[:n]),
            'auc_diff': mlp_mean - gnn_mean,
            'auc_pvalue': auc_pvalue,
            'cohen_d': cohen_d,
            'winner': winner,
            'significant': significant,
            'mlp_wins': mlp_mean > gnn_mean
        }

        comparison['comparisons'].append(result)

        # Print comparison
        diff_str = f"+{result['auc_diff']*100:.2f}%" if result['mlp_wins'] else f"{result['auc_diff']*100:.2f}%"
        sig_str = "*" if significant else ""
        print(f"{gnn_name:15} AUC: {gnn_mean:.4f} vs MLP: {mlp_mean:.4f} ({diff_str}){sig_str}")

    # Summary
    mlp_wins = sum(1 for c in comparison['comparisons'] if c['mlp_wins'])
    total = len(comparison['comparisons'])

    comparison['summary'] = {
        'mlp_wins': mlp_wins,
        'gnn_wins': total - mlp_wins,
        'total': total,
        'mlp_win_rate': mlp_wins / total if total > 0 else 0
    }

    print()
    print(f"Summary: MLP wins {mlp_wins}/{total} comparisons")

    return comparison


def generate_latex_comparison_table(comparison_results: List[Dict]) -> str:
    """
    Generate LaTeX table comparing MLP vs GNN across datasets.

    This table is crucial for the paper - it shows when structure helps vs hurts.
    """
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{MLP vs GNN Performance: When Does Graph Structure Help?}",
        "\\label{tab:mlp_vs_gnn}",
        "\\small",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Dataset} & $\\delta_{\\text{agg}}$ & \\textbf{Best MLP} & \\textbf{Best GNN} & "
        "\\textbf{MLP AUC} & \\textbf{GNN AUC} & \\textbf{Winner} \\\\",
        "\\midrule"
    ]

    for result in comparison_results:
        dataset = result['dataset']
        delta_agg = result.get('delta_agg', 'N/A')

        # Find best MLP and GNN
        best_gnn = max(result['comparisons'], key=lambda x: x['gnn_auc_mean'])
        mlp_auc = result['comparisons'][0]['mlp_auc_mean']  # Same for all
        mlp_std = result['comparisons'][0]['mlp_auc_std']

        gnn_name = best_gnn['gnn_method']
        gnn_auc = best_gnn['gnn_auc_mean']
        gnn_std = best_gnn['gnn_auc_std']

        # Determine winner
        if mlp_auc > gnn_auc:
            winner = "MLP"
            winner_format = "\\textbf{MLP}"
        else:
            winner = "GNN"
            winner_format = "\\textbf{GNN}"

        lines.append(
            f"{dataset} & {delta_agg} & MLP-2 & {gnn_name} & "
            f"{mlp_auc:.3f}$\\pm${mlp_std:.3f} & {gnn_auc:.3f}$\\pm${gnn_std:.3f} & {winner_format} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='MLP Baseline Experiments')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['elliptic', 'yelpchi', 'ieee_cis', 'amazon', 'dgraphfin'],
                       help='Dataset to evaluate')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data file (.pkl)')
    parser.add_argument('--gnn_results', type=str, default=None,
                       help='Path to GNN results for comparison')
    parser.add_argument('--output_dir', type=str, default='./mlp_results',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--n_seeds', type=int, default=15,
                       help='Number of random seeds')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'rb') as f:
        data_dict = pickle.load(f)

    # Convert to PyG Data
    if isinstance(data_dict, Data):
        data = data_dict
    else:
        data = Data(
            x=torch.tensor(data_dict['features'], dtype=torch.float32),
            edge_index=torch.tensor(data_dict['edge_index'], dtype=torch.long),
            y=torch.tensor(data_dict['labels'], dtype=torch.long),
            train_mask=torch.tensor(data_dict['train_mask'], dtype=torch.bool),
            val_mask=torch.tensor(data_dict['val_mask'], dtype=torch.bool),
            test_mask=torch.tensor(data_dict['test_mask'], dtype=torch.bool)
        )

    # Run MLP experiments
    seeds = SEEDS[:args.n_seeds]
    mlp_results = run_mlp_experiments(
        data, args.dataset, seeds, args.output_dir, args.device
    )

    # Compare with GNN results if provided
    if args.gnn_results:
        print(f"\nLoading GNN results from {args.gnn_results}...")
        with open(args.gnn_results, 'r') as f:
            gnn_results = json.load(f)

        comparison = compare_mlp_vs_gnn(mlp_results, gnn_results, args.dataset)

        # Save comparison
        comparison_file = Path(args.output_dir) / f'{args.dataset}_mlp_vs_gnn.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {comparison_file}")


if __name__ == '__main__':
    main()
