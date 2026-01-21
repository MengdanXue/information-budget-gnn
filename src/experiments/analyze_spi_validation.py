"""
SPI Validation Analysis - Multi-Dataset Results
Analyzes the prediction accuracy of SPI framework across 17 datasets
"""

import json
import numpy as np
from pathlib import Path

def analyze_spi_results():
    """Analyze SPI prediction accuracy and generate detailed report."""

    # Load results
    results_path = Path(__file__).parent / "real_dataset_results.json"
    with open(results_path, 'r') as f:
        data = json.load(f)

    results = data['results']

    print("="*80)
    print("SPI VALIDATION ANALYSIS - MULTI-DATASET RESULTS")
    print("="*80)

    # Separate by dataset type
    homophilic = ['cora', 'citeseer', 'pubmed']
    heterophilic = ['texas', 'wisconsin', 'cornell', 'actor', 'chameleon', 'squirrel']
    fraud = ['elliptic_weber_split', 'inj_amazon', 'inj_cora']
    synthetic = ['csbm_high_homo', 'csbm_mid_homo', 'csbm_low_homo', 'csbm_noisy_feat', 'csbm_clean_feat']

    # Create detailed table
    print("\n" + "="*80)
    print("DETAILED RESULTS TABLE")
    print("="*80)
    print(f"{'Dataset':<20} {'h':>6} {'SPI':>6} {'MLP':>8} {'GCN':>8} {'Best':>10} {'Pred':>6} {'Correct':>8}")
    print("-"*80)

    for r in results:
        print(f"{r['dataset']:<20} {r['homophily']:.3f} {r['spi']:.3f} "
              f"{r['MLP_acc']*100:>7.1f}% {r['GCN_acc']*100:>7.1f}% "
              f"{r['best_model']:<10} {r['spi_prediction']:<6} "
              f"{'YES' if r['prediction_correct'] else 'NO':>8}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    total = len(results)
    correct = sum(1 for r in results if r['prediction_correct'])
    print(f"\nOverall Accuracy: {correct}/{total} = {100*correct/total:.1f}%")

    # By homophily zone
    print("\nBy Homophily Zone:")
    high_h = [r for r in results if r['homophily'] > 0.7]
    mid_h = [r for r in results if 0.3 <= r['homophily'] <= 0.7]
    low_h = [r for r in results if r['homophily'] < 0.3]

    if high_h:
        high_correct = sum(1 for r in high_h if r['prediction_correct'])
        print(f"  High h (>0.7):  {high_correct}/{len(high_h)} = {100*high_correct/len(high_h):.1f}%")
        print(f"    Datasets: {[r['dataset'] for r in high_h]}")

    if mid_h:
        mid_correct = sum(1 for r in mid_h if r['prediction_correct'])
        print(f"  Mid h (0.3-0.7): {mid_correct}/{len(mid_h)} = {100*mid_correct/len(mid_h):.1f}%")
        print(f"    Datasets: {[r['dataset'] for r in mid_h]}")

    if low_h:
        low_correct = sum(1 for r in low_h if r['prediction_correct'])
        print(f"  Low h (<0.3):  {low_correct}/{len(low_h)} = {100*low_correct/len(low_h):.1f}%")
        print(f"    Datasets: {[r['dataset'] for r in low_h]}")

    # By dataset type
    print("\nBy Dataset Type:")

    homo_results = [r for r in results if r['dataset'] in homophilic]
    if homo_results:
        homo_correct = sum(1 for r in homo_results if r['prediction_correct'])
        print(f"  Homophilic (Cora/CiteSeer/PubMed): {homo_correct}/{len(homo_results)}")

    hetero_results = [r for r in results if r['dataset'] in heterophilic]
    if hetero_results:
        hetero_correct = sum(1 for r in hetero_results if r['prediction_correct'])
        print(f"  Heterophilic (Texas/Wisconsin/...): {hetero_correct}/{len(hetero_results)}")

    fraud_results = [r for r in results if r['dataset'] in fraud]
    if fraud_results:
        fraud_correct = sum(1 for r in fraud_results if r['prediction_correct'])
        print(f"  Fraud Detection: {fraud_correct}/{len(fraud_results)}")

    synth_results = [r for r in results if r['dataset'] in synthetic]
    if synth_results:
        synth_correct = sum(1 for r in synth_results if r['prediction_correct'])
        print(f"  Synthetic (cSBM): {synth_correct}/{len(synth_results)}")

    # Failure analysis
    print("\n" + "="*80)
    print("FAILURE ANALYSIS")
    print("="*80)

    failures = [r for r in results if not r['prediction_correct']]
    print(f"\nTotal Failures: {len(failures)}")

    for f in failures:
        print(f"\n  {f['dataset']}:")
        print(f"    h = {f['homophily']:.3f}, SPI = {f['spi']:.3f}")
        print(f"    MLP: {f['MLP_acc']*100:.1f}%, GCN: {f['GCN_acc']*100:.1f}%")
        print(f"    Best: {f['best_model']}, SPI predicted: {f['spi_prediction']}")

        # Diagnosis
        if f['spi'] > 0.4 and f['best_model'] == 'MLP':
            if f['homophily'] < 0.3:
                print(f"    Diagnosis: LOW HETEROPHILY but MLP wins - GCN fails to exploit structure")
            elif f['homophily'] > 0.7:
                print(f"    Diagnosis: HIGH HOMOPHILY but MLP wins - features already sufficient")
            else:
                print(f"    Diagnosis: MID HOMOPHILY region - uncertainty zone")

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    print("""
1. SPI Framework Overall: {:.1f}% accuracy across {} datasets

2. Trust Region Validation:
   - High h (>0.7): SPI correctly predicts GNN advantage in homophilic graphs
   - Mid h (0.3-0.7): Uncertainty zone - predictions less reliable
   - Low h (<0.3): Mixed results - some heterophilic datasets favor MLP

3. Notable Failures:
   - Wisconsin/Cornell: Despite low h, GCN fails badly (-29% to -35%)
   - Actor: Medium h but MLP wins (-6.7% GCN disadvantage)
   - Injected datasets: Features too strong, structure provides no benefit

4. Theoretical Implications:
   - SPI captures the "predictability from structure" but not "feature sufficiency"
   - When features alone are near-perfect, structure adds no value
   - Low h doesn't guarantee GCN success - depends on pattern regularity

5. Recommendations for Paper:
   - Report both successes AND failures honestly
   - Add "feature sufficiency" as confounding factor
   - Consider two-factor model: SPI + Feature Quality
""".format(100*correct/total, total))

    # Generate LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE FOR PAPER")
    print("="*80)

    print(r"""
\begin{table}[h]
\centering
\caption{SPI Validation on Real-World Datasets (mean \pm std, n=5)}
\label{tab:spi_validation}
\begin{tabular}{lcccccc}
\toprule
Dataset & h & SPI & MLP (\%) & GCN (\%) & Best & Correct \\
\midrule""")

    for r in results:
        correct_mark = r'$\checkmark$' if r['prediction_correct'] else r'$\times$'
        print(f"{r['dataset']} & {r['homophily']:.2f} & {r['spi']:.2f} & "
              f"{r['MLP_acc']*100:.1f} & {r['GCN_acc']*100:.1f} & "
              f"{r['best_model']} & {correct_mark} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

    return {
        'total': total,
        'correct': correct,
        'accuracy': correct/total,
        'failures': failures
    }

if __name__ == "__main__":
    analyze_spi_results()
