"""
Run all IEEE-CIS experiments for FSD validation.

This script runs experiments sequentially to verify FSD prediction:
ρ_FS = 0.059 → FSD predicts "No clear winner"

Expected: All methods should show similar performance.
"""

import subprocess
import json
import os
import numpy as np
from datetime import datetime

METHODS = ['GAT', 'GraphSAGE', 'NAA-GCN', 'NAA-GAT', 'H2GCN', 'MixHop']
# GCN already done

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CODE_DIR, 'processed', 'ieee_cis_graph.pkl')
RESULTS_DIR = os.path.join(CODE_DIR, 'results')

def run_experiment(method):
    """Run single method experiment."""
    print(f"\n{'='*60}")
    print(f"Running {method} on IEEE-CIS (10 seeds)")
    print(f"{'='*60}")

    cmd = [
        'python', os.path.join(CODE_DIR, 'train_ieee_cis.py'),
        '--method', method,
        '--data_path', DATA_PATH,
        '--output_dir', RESULTS_DIR
    ]

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def load_results():
    """Load all results and compute summary."""
    results = {}

    # Load GCN (already done)
    gcn_path = os.path.join(RESULTS_DIR, 'ieee_cis_GCN.json')
    if os.path.exists(gcn_path):
        with open(gcn_path, 'r') as f:
            results['GCN'] = json.load(f)

    # Load other methods
    for method in METHODS:
        path = os.path.join(RESULTS_DIR, f'ieee_cis_{method}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[method] = json.load(f)

    return results

def print_summary(results):
    """Print summary table."""
    print("\n" + "="*80)
    print("IEEE-CIS EXPERIMENT SUMMARY")
    print("="*80)
    print(f"ρ_FS = 0.059 | FSD Prediction: No clear winner")
    print("-"*80)
    print(f"{'Method':<12} {'AUC':>12} {'F1':>12} {'Precision':>12} {'Recall':>12}")
    print("-"*80)

    for method, data in sorted(results.items()):
        auc_mean = np.mean(data['auc'])
        auc_std = np.std(data['auc'])
        f1_mean = np.mean(data['f1'])
        f1_std = np.std(data['f1'])
        prec_mean = np.mean(data['precision'])
        rec_mean = np.mean(data['recall'])

        print(f"{method:<12} {auc_mean:.4f}±{auc_std:.4f} {f1_mean:.4f}±{f1_std:.4f} "
              f"{prec_mean:.4f} {rec_mean:.4f}")

    print("-"*80)

    # Find best method
    auc_means = {m: np.mean(d['auc']) for m, d in results.items()}
    best_method = max(auc_means, key=auc_means.get)
    best_auc = auc_means[best_method]

    print(f"\nBest AUC: {best_method} ({best_auc:.4f})")

    # Check if results support FSD prediction
    auc_values = list(auc_means.values())
    auc_range = max(auc_values) - min(auc_values)

    if auc_range < 0.03:
        print("✓ FSD Prediction VERIFIED: Methods show similar performance (range < 0.03)")
    else:
        print(f"? FSD Prediction PARTIAL: AUC range = {auc_range:.4f}")

def main():
    print("="*60)
    print("IEEE-CIS Full Experiment Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run experiments
    for method in METHODS:
        result_path = os.path.join(RESULTS_DIR, f'ieee_cis_{method}.json')
        if os.path.exists(result_path):
            print(f"\n{method}: Already completed, skipping...")
            continue

        success = run_experiment(method)
        if not success:
            print(f"ERROR: {method} failed!")

    # Summary
    results = load_results()
    print_summary(results)

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'IEEE-CIS',
        'rho_fs': 0.059,
        'fsd_prediction': 'No clear winner',
        'results': {m: {
            'auc_mean': float(np.mean(d['auc'])),
            'auc_std': float(np.std(d['auc'])),
            'f1_mean': float(np.mean(d['f1'])),
            'f1_std': float(np.std(d['f1']))
        } for m, d in results.items()}
    }

    with open(os.path.join(RESULTS_DIR, 'ieee_cis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
