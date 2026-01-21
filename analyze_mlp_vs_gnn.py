"""
MLP vs GNN Comparison Analysis for IEEE-CIS Dataset
====================================================
Validates the FSD hypothesis: When delta_agg > 10, MLP can compete with GNN
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_methods(mlp_results, gnn_results_dict, dataset_name="ieee_cis"):
    """Compare MLP against all GNN methods"""

    print("="*80)
    print(f"MLP vs GNN Comparison Analysis: {dataset_name.upper()}")
    print("="*80)
    print()

    # Extract best MLP results
    best_mlp_name = mlp_results['best_mlp']['name']
    mlp_aucs = mlp_results['methods'][best_mlp_name]['auc']
    mlp_f1s = mlp_results['methods'][best_mlp_name]['f1']

    mlp_auc_mean = np.mean(mlp_aucs)
    mlp_auc_std = np.std(mlp_aucs)
    mlp_f1_mean = np.mean(mlp_f1s)
    mlp_f1_std = np.std(mlp_f1s)

    print(f"Best MLP Model: {best_mlp_name}")
    print(f"  AUC: {mlp_auc_mean:.4f} ± {mlp_auc_std:.4f}")
    print(f"  F1:  {mlp_f1_mean:.4f} ± {mlp_f1_std:.4f}")
    print()
    print("-"*80)
    print()

    comparisons = []

    for gnn_name, gnn_results in gnn_results_dict.items():
        gnn_aucs = gnn_results['auc']
        gnn_f1s = gnn_results['f1']

        gnn_auc_mean = np.mean(gnn_aucs)
        gnn_auc_std = np.std(gnn_aucs)
        gnn_f1_mean = np.mean(gnn_f1s)
        gnn_f1_std = np.std(gnn_f1s)

        # Statistical test
        n = min(len(mlp_aucs), len(gnn_aucs))
        if n >= 5:
            try:
                stat, pvalue = stats.wilcoxon(mlp_aucs[:n], gnn_aucs[:n])
            except:
                pvalue = 1.0
        else:
            pvalue = 1.0

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((mlp_auc_std**2 + gnn_auc_std**2) / 2)
        cohen_d = (mlp_auc_mean - gnn_auc_mean) / (pooled_std + 1e-8)

        # Determine winner
        mlp_wins = mlp_auc_mean > gnn_auc_mean
        significant = pvalue < 0.05

        auc_diff = mlp_auc_mean - gnn_auc_mean
        auc_diff_pct = (auc_diff / gnn_auc_mean) * 100

        comparison = {
            'gnn_method': gnn_name,
            'mlp_auc_mean': mlp_auc_mean,
            'mlp_auc_std': mlp_auc_std,
            'gnn_auc_mean': gnn_auc_mean,
            'gnn_auc_std': gnn_auc_std,
            'auc_diff': auc_diff,
            'auc_diff_pct': auc_diff_pct,
            'pvalue': pvalue,
            'cohen_d': cohen_d,
            'mlp_wins': mlp_wins,
            'significant': significant,
            'mlp_f1_mean': mlp_f1_mean,
            'gnn_f1_mean': gnn_f1_mean
        }
        comparisons.append(comparison)

        # Print comparison
        winner_symbol = "WIN" if mlp_wins else "LOSE"
        sig_symbol = "*" if significant else " "

        print(f"{gnn_name:15} | GNN: {gnn_auc_mean:.4f} ± {gnn_auc_std:.4f} | "
              f"MLP: {mlp_auc_mean:.4f} | Diff: {auc_diff:+.4f} ({auc_diff_pct:+.2f}%) {sig_symbol} {winner_symbol}")

    print()
    print("-"*80)
    print()

    # Summary statistics
    mlp_win_count = sum(1 for c in comparisons if c['mlp_wins'])
    total_count = len(comparisons)
    mlp_win_rate = mlp_win_count / total_count * 100

    significant_wins = sum(1 for c in comparisons if c['mlp_wins'] and c['significant'])

    avg_improvement = np.mean([c['auc_diff_pct'] for c in comparisons if c['mlp_wins']])

    print("SUMMARY:")
    print(f"  MLP wins:      {mlp_win_count}/{total_count} ({mlp_win_rate:.1f}%)")
    print(f"  Significant:   {significant_wins}/{mlp_win_count}")
    print(f"  Avg improvement: {avg_improvement:.2f}% (when MLP wins)")
    print()

    # Key insights
    print("="*80)
    print("KEY INSIGHTS:")
    print("="*80)

    # Compare with best GNN
    best_gnn = max(comparisons, key=lambda x: x['gnn_auc_mean'])
    print(f"\n1. Best GNN: {best_gnn['gnn_method']}")
    print(f"   GNN AUC: {best_gnn['gnn_auc_mean']:.4f} ± {best_gnn['gnn_auc_std']:.4f}")
    print(f"   MLP AUC: {best_gnn['mlp_auc_mean']:.4f} ± {best_gnn['mlp_auc_std']:.4f}")
    print(f"   Difference: {best_gnn['auc_diff']:+.4f} ({best_gnn['auc_diff_pct']:+.2f}%)")

    if best_gnn['mlp_wins']:
        print(f"   --> MLP OUTPERFORMS best GNN by {abs(best_gnn['auc_diff_pct']):.2f}%!")
    elif abs(best_gnn['auc_diff_pct']) < 1.0:
        print(f"   --> MLP is COMPETITIVE with best GNN (< 1% difference)")
    else:
        print(f"   --> GNN outperforms MLP by {abs(best_gnn['auc_diff_pct']):.2f}%")

    # Worst GNN comparison
    worst_gnn = min(comparisons, key=lambda x: x['gnn_auc_mean'])
    improvement = ((mlp_auc_mean - worst_gnn['gnn_auc_mean']) / worst_gnn['gnn_auc_mean']) * 100
    print(f"\n2. Worst GNN: {worst_gnn['gnn_method']}")
    print(f"   GNN AUC: {worst_gnn['gnn_auc_mean']:.4f}")
    print(f"   MLP AUC: {mlp_auc_mean:.4f}")
    print(f"   --> MLP outperforms by {improvement:.2f}%")

    print()
    print("-"*80)

    return {
        'comparisons': comparisons,
        'summary': {
            'mlp_wins': mlp_win_count,
            'total': total_count,
            'win_rate': mlp_win_rate,
            'significant_wins': significant_wins,
            'avg_improvement': avg_improvement,
            'best_gnn': best_gnn,
            'worst_gnn': worst_gnn
        }
    }

def main():
    results_dir = Path("results")

    # Load MLP results
    mlp_file = results_dir / "ieee_cis_mlp_results.json"
    mlp_results = load_json(mlp_file)

    # Load all GNN results
    gnn_methods = ['GCN', 'GAT', 'GraphSAGE', 'H2GCN', 'NAA-GCN', 'DAAA']
    gnn_results_dict = {}

    for method in gnn_methods:
        gnn_file = results_dir / f"ieee_cis_{method}.json"
        if gnn_file.exists():
            gnn_results_dict[method] = load_json(gnn_file)

    # Load extended metrics to get delta_agg
    metrics_file = Path("processed") / "ieee_cis_extended_metrics.json"
    if metrics_file.exists():
        extended_metrics = load_json(metrics_file)
        rho_fs_1hop = extended_metrics['rho_fs_1hop']
        homophily = extended_metrics['homophily']
        delta_agg = homophily - rho_fs_1hop

        print("\nFSD METRICS:")
        print(f"  Homophily (H):     {homophily:.4f}")
        print(f"  rho_fs (1-hop):      {rho_fs_1hop:.4f}")
        print(f"  delta_agg = H - rho_fs:  {delta_agg:.4f}")
        print()

        if delta_agg > 0.10:  # 10%
            print(f"[+] HIGH DILUTION DATASET (delta_agg = {delta_agg:.4f} > 0.10)")
            print("  --> FSD prediction: MLP should be COMPETITIVE with GNN")
        else:
            print(f"  Low dilution dataset (delta_agg = {delta_agg:.4f})")
        print()

    # Run comparison
    comparison_results = compare_methods(mlp_results, gnn_results_dict)

    # Save results
    output_file = results_dir / "ieee_cis_mlp_vs_gnn_comparison.json"

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    comparison_results_json = convert_to_native(comparison_results)

    with open(output_file, 'w') as f:
        json.dump(comparison_results_json, f, indent=2)

    print(f"\nComparison saved to: {output_file}")

    # FSD Hypothesis Validation
    print("\n" + "="*80)
    print("FSD HYPOTHESIS VALIDATION")
    print("="*80)

    if metrics_file.exists() and delta_agg > 0.10:
        best_gnn = comparison_results['summary']['best_gnn']

        print(f"Hypothesis: When delta_agg > 0.10, MLP can compete with GNN")
        print(f"Dataset: IEEE-CIS with delta_agg = {delta_agg:.4f}")
        print()

        if best_gnn['mlp_wins'] or abs(best_gnn['auc_diff_pct']) < 2.0:
            print("*** HYPOTHESIS VALIDATED ***")
            print()
            print("MLP is competitive with or better than GNN on this high-dilution dataset.")
            print("This validates the FSD framework prediction that graph structure provides")
            print("minimal benefit when feature similarity is heavily diluted across edges.")
        else:
            print("! Hypothesis not fully validated")
            print(f"GNN still outperforms MLP by {abs(best_gnn['auc_diff_pct']):.2f}%")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
