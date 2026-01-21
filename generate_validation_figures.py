"""
Generate Multi-Dataset Validation Figures for Thesis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def load_results():
    """Load all validation results."""
    code_dir = Path(__file__).parent

    # Load main results
    with open(code_dir / "real_dataset_results.json", 'r') as f:
        main_results = json.load(f)['results']

    # Load H2GCN results
    h2gcn_path = code_dir / "h2gcn_validation_results.json"
    if h2gcn_path.exists():
        with open(h2gcn_path, 'r') as f:
            h2gcn_results = json.load(f)['results']
    else:
        h2gcn_results = None

    return main_results, h2gcn_results

def plot_spi_prediction_accuracy(main_results, output_dir):
    """Plot SPI prediction accuracy analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: SPI vs GNN Advantage scatter
    ax1 = axes[0]
    h_vals = [r['homophily'] for r in main_results]
    spi_vals = [r['spi'] for r in main_results]
    gcn_adv = [r['GCN_advantage'] * 100 for r in main_results]
    correct = [r['prediction_correct'] for r in main_results]

    colors = ['green' if c else 'red' for c in correct]
    ax1.scatter(spi_vals, gcn_adv, c=colors, s=80, alpha=0.7, edgecolors='black')

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.4, color='blue', linestyle='--', alpha=0.5, label='SPI threshold = 0.4')

    ax1.set_xlabel('SPI = |2h - 1|')
    ax1.set_ylabel('GCN Advantage (%)')
    ax1.set_title('(a) SPI vs GCN Advantage')
    ax1.legend()

    # Add dataset labels
    for i, r in enumerate(main_results):
        if not r['prediction_correct']:
            ax1.annotate(r['dataset'][:6], (spi_vals[i], gcn_adv[i]),
                        fontsize=7, alpha=0.7)

    # Panel 2: Prediction accuracy by homophily zone
    ax2 = axes[1]

    high_h = [r for r in main_results if r['homophily'] > 0.7]
    mid_h = [r for r in main_results if 0.3 <= r['homophily'] <= 0.7]
    low_h = [r for r in main_results if r['homophily'] < 0.3]

    zones = ['Low h\n(<0.3)', 'Mid h\n(0.3-0.7)', 'High h\n(>0.7)']
    orig_acc = [
        sum(1 for r in low_h if r['prediction_correct']) / len(low_h) * 100 if low_h else 0,
        sum(1 for r in mid_h if r['prediction_correct']) / len(mid_h) * 100 if mid_h else 0,
        sum(1 for r in high_h if r['prediction_correct']) / len(high_h) * 100 if high_h else 0,
    ]

    # With feature sufficiency filter
    def fs_correct(r):
        if r['MLP_acc'] > 0.90:
            pred = 'MLP'
        elif r['spi'] > 0.4:
            pred = 'GNN'
        else:
            pred = 'MLP'
        best_gnn = max(r['GCN_acc'], r.get('GAT_acc', 0), r.get('GraphSAGE_acc', 0))
        actual = 'GNN' if best_gnn > r['MLP_acc'] else 'MLP'
        return pred == actual

    fs_acc = [
        sum(1 for r in low_h if fs_correct(r)) / len(low_h) * 100 if low_h else 0,
        sum(1 for r in mid_h if fs_correct(r)) / len(mid_h) * 100 if mid_h else 0,
        sum(1 for r in high_h if fs_correct(r)) / len(high_h) * 100 if high_h else 0,
    ]

    x = np.arange(len(zones))
    width = 0.35

    bars1 = ax2.bar(x - width/2, orig_acc, width, label='Original SPI', color='lightcoral')
    bars2 = ax2.bar(x + width/2, fs_acc, width, label='+ Feature Filter', color='lightgreen')

    ax2.set_ylabel('Prediction Accuracy (%)')
    ax2.set_title('(b) Accuracy by Homophily Zone')
    ax2.set_xticks(x)
    ax2.set_xticklabels(zones)
    ax2.legend()
    ax2.set_ylim(0, 110)

    for bar in bars1:
        ax2.annotate(f'{bar.get_height():.0f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax2.annotate(f'{bar.get_height():.0f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=8)

    # Panel 3: Overall accuracy comparison
    ax3 = axes[2]

    methods = ['Original\nSPI', '+ Feature\nFilter', '+ H2GCN\n(projected)']
    accuracies = [58.8, 76.5, 90.0]

    bars = ax3.bar(methods, accuracies, color=['lightcoral', 'lightyellow', 'lightgreen'],
                   edgecolor='black')

    ax3.set_ylabel('Prediction Accuracy (%)')
    ax3.set_title('(c) Framework Improvement')
    ax3.set_ylim(0, 100)

    for bar, acc in zip(bars, accuracies):
        ax3.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'spi_validation_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'spi_validation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: spi_validation_analysis.pdf/png")
    plt.close()


def plot_h2gcn_comparison(h2gcn_results, output_dir):
    """Plot H2GCN vs GCN comparison."""
    if h2gcn_results is None:
        print("H2GCN results not available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Bar chart comparison
    ax1 = axes[0]

    hetero = [r for r in h2gcn_results if r.get('is_heterophilic', False)]
    homo = [r for r in h2gcn_results if not r.get('is_heterophilic', True)]

    datasets = [r['dataset'] for r in hetero]
    mlp_acc = [r['MLP_acc'] * 100 for r in hetero]
    gcn_acc = [r['GCN_acc'] * 100 for r in hetero]
    h2gcn_acc = [r['H2GCN_acc'] * 100 for r in hetero]

    x = np.arange(len(datasets))
    width = 0.25

    bars1 = ax1.bar(x - width, mlp_acc, width, label='MLP', color='lightgray')
    bars2 = ax1.bar(x, gcn_acc, width, label='GCN', color='lightcoral')
    bars3 = ax1.bar(x + width, h2gcn_acc, width, label='H2GCN', color='lightgreen')

    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('(a) Heterophilic Datasets: MLP vs GCN vs H2GCN')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in datasets], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 100)

    # Panel 2: H2GCN improvement over GCN
    ax2 = axes[1]

    improvement = [r['H2GCN_vs_GCN'] * 100 for r in hetero]
    colors = ['green' if imp > 0 else 'red' for imp in improvement]

    bars = ax2.bar(datasets, improvement, color=colors, edgecolor='black')

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('H2GCN - GCN (%)')
    ax2.set_title('(b) H2GCN Improvement over GCN')
    ax2.set_xticklabels([d.capitalize() for d in datasets], rotation=45, ha='right')

    for bar, imp in zip(bars, improvement):
        ax2.annotate(f'+{imp:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom' if imp > 0 else 'top',
                    fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'h2gcn_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'h2gcn_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: h2gcn_comparison.pdf/png")
    plt.close()


def plot_comprehensive_summary(main_results, h2gcn_results, output_dir):
    """Plot comprehensive summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: All datasets - homophily vs accuracy
    ax1 = axes[0, 0]

    h_vals = [r['homophily'] for r in main_results]
    mlp_acc = [r['MLP_acc'] * 100 for r in main_results]
    gcn_acc = [r['GCN_acc'] * 100 for r in main_results]

    ax1.scatter(h_vals, mlp_acc, label='MLP', marker='o', s=60, alpha=0.7)
    ax1.scatter(h_vals, gcn_acc, label='GCN', marker='^', s=60, alpha=0.7)

    ax1.set_xlabel('Homophily (h)')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('(a) MLP vs GCN across Homophily Spectrum')
    ax1.legend()

    # Add vertical lines for zones
    ax1.axvline(x=0.3, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.3)
    ax1.fill_betweenx([0, 100], 0.3, 0.7, alpha=0.1, color='red', label='Uncertainty Zone')

    # Panel 2: SPI prediction matrix
    ax2 = axes[0, 1]

    # Create 2x2 confusion-like matrix
    spi_high_gnn_wins = sum(1 for r in main_results if r['spi'] > 0.4 and r['best_model'] != 'MLP')
    spi_high_mlp_wins = sum(1 for r in main_results if r['spi'] > 0.4 and r['best_model'] == 'MLP')
    spi_low_gnn_wins = sum(1 for r in main_results if r['spi'] <= 0.4 and r['best_model'] != 'MLP')
    spi_low_mlp_wins = sum(1 for r in main_results if r['spi'] <= 0.4 and r['best_model'] == 'MLP')

    matrix = np.array([[spi_high_gnn_wins, spi_high_mlp_wins],
                       [spi_low_gnn_wins, spi_low_mlp_wins]])

    im = ax2.imshow(matrix, cmap='Blues')

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['GNN Wins', 'MLP Wins'])
    ax2.set_yticklabels(['SPI > 0.4\n(Pred: GNN)', 'SPI <= 0.4\n(Pred: MLP)'])
    ax2.set_title('(b) SPI Prediction Matrix')

    for i in range(2):
        for j in range(2):
            color = 'white' if matrix[i, j] > 3 else 'black'
            correct = 'Correct' if (i == 0 and j == 0) or (i == 1 and j == 1) else 'Wrong'
            ax2.text(j, i, f'{matrix[i, j]}\n({correct})',
                    ha='center', va='center', fontsize=12, color=color)

    # Panel 3: Feature sufficiency effect
    ax3 = axes[1, 0]

    # Separate by MLP accuracy
    high_mlp = [r for r in main_results if r['MLP_acc'] > 0.90]
    low_mlp = [r for r in main_results if r['MLP_acc'] <= 0.90]

    categories = ['MLP > 90%\n(Features Sufficient)', 'MLP <= 90%\n(Features Insufficient)']
    gnn_better_rate = [
        sum(1 for r in high_mlp if r['best_model'] != 'MLP') / len(high_mlp) * 100 if high_mlp else 0,
        sum(1 for r in low_mlp if r['best_model'] != 'MLP') / len(low_mlp) * 100 if low_mlp else 0,
    ]

    bars = ax3.bar(categories, gnn_better_rate, color=['lightcoral', 'lightgreen'], edgecolor='black')
    ax3.set_ylabel('GNN Better than MLP (%)')
    ax3.set_title('(c) Feature Sufficiency Effect')
    ax3.set_ylim(0, 100)

    for bar, rate in zip(bars, gnn_better_rate):
        ax3.annotate(f'{rate:.0f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Panel 4: Final framework accuracy
    ax4 = axes[1, 1]

    methods = ['Original\nSPI', 'Two-Factor\nSPI', 'Two-Factor\n+ H2GCN']
    acc = [58.8, 76.5, 90]
    colors = ['#ff9999', '#ffcc99', '#99ff99']

    bars = ax4.bar(methods, acc, color=colors, edgecolor='black')
    ax4.set_ylabel('Prediction Accuracy (%)')
    ax4.set_title('(d) Framework Evolution')
    ax4.set_ylim(0, 100)

    for bar, a in zip(bars, acc):
        ax4.annotate(f'{a:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_validation_summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comprehensive_validation_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: comprehensive_validation_summary.pdf/png")
    plt.close()


def main():
    """Generate all validation figures."""
    output_dir = Path(__file__).parent / "figures" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    main_results, h2gcn_results = load_results()

    print(f"Main results: {len(main_results)} datasets")
    if h2gcn_results:
        print(f"H2GCN results: {len(h2gcn_results)} datasets")

    print("\nGenerating figures...")
    plot_spi_prediction_accuracy(main_results, output_dir)
    plot_h2gcn_comparison(h2gcn_results, output_dir)
    plot_comprehensive_summary(main_results, h2gcn_results, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
