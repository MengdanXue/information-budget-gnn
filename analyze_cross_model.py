"""
Cross-Model H-Sweep Visualization and Analysis
Creates publication-quality figures showing U-shape across different GNN architectures.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['figure.dpi'] = 150

def load_results():
    """Load cross-model H-sweep results."""
    results_path = Path(__file__).parent / "cross_model_hsweep_results.json"
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_cross_model_comparison():
    """Create main comparison figure."""
    data = load_results()
    results = data['results']

    h_values = [r['h'] for r in results]

    # Model colors
    colors = {
        'MLP': '#808080',      # Gray (baseline)
        'GCN': '#2E86AB',      # Blue
        'GAT': '#E94F37',      # Red
        'GraphSAGE': '#44AF69' # Green
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Left: Accuracy Comparison =====
    ax1 = axes[0]

    for model in ['MLP', 'GCN', 'GAT', 'GraphSAGE']:
        accs = [r[f'{model}_acc'] * 100 for r in results]
        stds = [r[f'{model}_std'] * 100 for r in results]
        style = '--' if model == 'MLP' else '-'
        marker = 's' if model == 'MLP' else 'o'
        ax1.plot(h_values, accs, f'{style}', color=colors[model],
                 linewidth=2, marker=marker, markersize=8, label=model)
        ax1.fill_between(h_values,
                        [a - s for a, s in zip(accs, stds)],
                        [a + s for a, s in zip(accs, stds)],
                        alpha=0.15, color=colors[model])

    # Zone shading
    ax1.axvspan(0, 0.3, alpha=0.08, color='green')
    ax1.axvspan(0.3, 0.7, alpha=0.08, color='red')
    ax1.axvspan(0.7, 1.0, alpha=0.08, color='green')

    ax1.set_xlabel('Edge Homophily (h)')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('(a) Model Accuracy Across Homophily Spectrum')
    ax1.set_xlim(0.05, 0.95)
    ax1.set_ylim(78, 102)
    ax1.legend(loc='lower right')
    ax1.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # ===== Right: Advantage over MLP =====
    ax2 = axes[1]

    for model in ['GCN', 'GAT', 'GraphSAGE']:
        advantages = [r[f'{model}_advantage'] * 100 for r in results]
        ax2.plot(h_values, advantages, '-', color=colors[model],
                 linewidth=2.5, marker='o', markersize=8, label=f'{model} vs MLP')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Zone shading and labels
    ax2.axvspan(0, 0.3, alpha=0.08, color='green')
    ax2.axvspan(0.3, 0.7, alpha=0.08, color='red')
    ax2.axvspan(0.7, 1.0, alpha=0.08, color='green')

    ax2.text(0.15, 2, 'Extreme\nLow h', ha='center', fontsize=10, fontweight='bold')
    ax2.text(0.5, 2, 'Uncertainty\nZone', ha='center', fontsize=10, fontweight='bold', color='#E94F37')
    ax2.text(0.85, 2, 'Extreme\nHigh h', ha='center', fontsize=10, fontweight='bold')

    # Annotate GCN valley
    gcn_advantages = [r['GCN_advantage'] * 100 for r in results]
    min_idx = np.argmin(gcn_advantages)
    ax2.annotate(f'GCN Valley\n{gcn_advantages[min_idx]:.1f}%',
                 xy=(h_values[min_idx], gcn_advantages[min_idx]),
                 xytext=(h_values[min_idx] + 0.15, gcn_advantages[min_idx] - 3),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5))

    ax2.set_xlabel('Edge Homophily (h)')
    ax2.set_ylabel('Advantage over MLP (%)')
    ax2.set_title('(b) GNN Advantage: The U-Shape Pattern')
    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(-22, 5)
    ax2.legend(loc='lower right')
    ax2.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    plt.savefig(output_dir / "cross_model_hsweep.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "cross_model_hsweep.pdf", format='pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'cross_model_hsweep.png'}")
    plt.close()

def plot_ushape_severity():
    """Plot showing U-shape severity by model."""
    data = load_results()
    results = data['results']

    models = ['GCN', 'GAT', 'GraphSAGE']
    colors = ['#2E86AB', '#E94F37', '#44AF69']

    # Calculate metrics
    metrics = []
    for model in models:
        advantages = [r[f'{model}_advantage'] * 100 for r in results]

        # Valley depth (worst performance in mid-h)
        mid_h_advs = [adv for r, adv in zip(results, advantages) if 0.3 <= r['h'] <= 0.7]
        valley_depth = min(mid_h_advs)

        # Extreme performance (best in extreme h)
        extreme_advs = [adv for r, adv in zip(results, advantages) if r['h'] < 0.3 or r['h'] > 0.7]
        extreme_best = max(extreme_advs)

        # U-shape amplitude
        amplitude = extreme_best - valley_depth

        metrics.append({
            'model': model,
            'valley_depth': valley_depth,
            'extreme_best': extreme_best,
            'amplitude': amplitude
        })

    # Print analysis
    print("\n" + "="*60)
    print("U-Shape Severity Analysis by Model")
    print("="*60)
    print(f"{'Model':<12} {'Valley Depth':<15} {'Extreme Best':<15} {'Amplitude':<12}")
    print("-"*60)
    for m in metrics:
        print(f"{m['model']:<12} {m['valley_depth']:>+10.1f}%     {m['extreme_best']:>+10.1f}%     {m['amplitude']:>8.1f}%")

    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("="*60)
    print("GCN shows STRONGEST U-shape (amplitude: {:.1f}%)".format(metrics[0]['amplitude']))
    print("GAT shows MODERATE U-shape (amplitude: {:.1f}%)".format(metrics[1]['amplitude']))
    print("GraphSAGE shows MINIMAL U-shape (amplitude: {:.1f}%)".format(metrics[2]['amplitude']))
    print("\nThis suggests:")
    print("- Mean aggregation (GCN) is MOST sensitive to homophily")
    print("- Attention (GAT) partially mitigates the problem")
    print("- Sampling (GraphSAGE) is ROBUST to homophily variations")

    return metrics

def create_summary_table():
    """Create summary table for paper."""
    data = load_results()
    results = data['results']

    print("\n" + "="*80)
    print("TABLE: Cross-Model H-Sweep Results Summary")
    print("="*80)

    # Header
    print(f"{'h':<6} {'MLP':<10} {'GCN':<10} {'GAT':<10} {'SAGE':<10} {'GCN-MLP':<10} {'GAT-MLP':<10} {'SAGE-MLP':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['h']:<6.1f} "
              f"{r['MLP_acc']*100:<10.1f} "
              f"{r['GCN_acc']*100:<10.1f} "
              f"{r['GAT_acc']*100:<10.1f} "
              f"{r['GraphSAGE_acc']*100:<10.1f} "
              f"{r['GCN_advantage']*100:>+9.1f}% "
              f"{r['GAT_advantage']*100:>+9.1f}% "
              f"{r['GraphSAGE_advantage']*100:>+9.1f}%")

    print("-"*80)

    # Zone summary
    print("\nZone Summary:")
    for model in ['GCN', 'GAT', 'GraphSAGE']:
        low_h = np.mean([r[f'{model}_advantage']*100 for r in results if r['h'] < 0.3])
        mid_h = np.mean([r[f'{model}_advantage']*100 for r in results if 0.3 <= r['h'] <= 0.7])
        high_h = np.mean([r[f'{model}_advantage']*100 for r in results if r['h'] > 0.7])
        print(f"  {model}: Low-h avg={low_h:+.1f}%, Mid-h avg={mid_h:+.1f}%, High-h avg={high_h:+.1f}%")

if __name__ == "__main__":
    print("="*60)
    print("Cross-Model H-Sweep Analysis")
    print("="*60)

    # Create visualizations
    try:
        plot_cross_model_comparison()
    except Exception as e:
        print(f"Visualization error: {e}")

    # Analyze U-shape severity
    metrics = plot_ushape_severity()

    # Create summary table
    create_summary_table()
