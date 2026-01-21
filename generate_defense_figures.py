"""
Generate Professional Defense Figures with Error Bands
For thesis defense presentation

Creates:
1. Cross-model U-Shape with shaded error regions
2. Feature separability sweep comparison
3. SPI correlation visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette
COLORS = {
    'MLP': '#2E86AB',      # Blue
    'GCN': '#E94F37',      # Red
    'GAT': '#F39C12',      # Orange
    'GraphSAGE': '#27AE60', # Green
    'advantage': '#8E44AD', # Purple
}

def load_data():
    """Load experimental results."""
    base_path = Path(__file__).parent

    with open(base_path / 'cross_model_hsweep_results.json', 'r') as f:
        cross_model = json.load(f)

    with open(base_path / 'separability_sweep_results.json', 'r') as f:
        separability = json.load(f)

    return cross_model, separability


def fig1_cross_model_ushape_with_errorbands(data, save_path):
    """
    Figure 1: Cross-Model U-Shape with Error Bands
    Shows GNN advantage over MLP with shaded confidence regions.
    """
    results = data['results']
    h_values = [r['h'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Raw accuracies with error bands
    for model in ['MLP', 'GCN', 'GAT', 'GraphSAGE']:
        means = [r[f'{model}_acc'] * 100 for r in results]
        stds = [r[f'{model}_std'] * 100 for r in results]

        color = COLORS[model]
        ax1.plot(h_values, means, 'o-', color=color, label=model,
                linewidth=2, markersize=6)
        ax1.fill_between(h_values,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=color, alpha=0.2)

    ax1.set_xlabel('Homophily (h)')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('(a) Model Accuracy vs Homophily')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0.05, 0.95)
    ax1.set_ylim(75, 102)

    # Add U-shape annotation
    ax1.axvspan(0.3, 0.7, alpha=0.1, color='red', label='Uncertainty Zone')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate('h=0.5\n(worst)', xy=(0.5, 78), ha='center', fontsize=9, color='gray')

    # Right panel: GCN Advantage with error propagation
    gcn_means = np.array([r['GCN_acc'] for r in results])
    gcn_stds = np.array([r['GCN_std'] for r in results])
    mlp_means = np.array([r['MLP_acc'] for r in results])
    mlp_stds = np.array([r['MLP_std'] for r in results])

    advantage_means = (gcn_means - mlp_means) * 100
    # Error propagation: sqrt(std1^2 + std2^2) for difference
    advantage_stds = np.sqrt(gcn_stds**2 + mlp_stds**2) * 100

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(h_values,
                     advantage_means - advantage_stds,
                     advantage_means + advantage_stds,
                     color=COLORS['advantage'], alpha=0.3)
    ax2.plot(h_values, advantage_means, 'o-', color=COLORS['advantage'],
            linewidth=2.5, markersize=8, label='GCN - MLP')

    ax2.set_xlabel('Homophily (h)')
    ax2.set_ylabel('GCN Advantage (%)')
    ax2.set_title('(b) U-Shape: GCN Advantage over MLP')

    # Add trust region annotations
    ax2.axvspan(0.05, 0.3, alpha=0.15, color='green')
    ax2.axvspan(0.7, 0.95, alpha=0.15, color='green')
    ax2.axvspan(0.3, 0.7, alpha=0.15, color='red')

    ax2.text(0.15, 5, 'Trust\nRegion', ha='center', fontsize=9, color='green')
    ax2.text(0.85, 5, 'Trust\nRegion', ha='center', fontsize=9, color='green')
    ax2.text(0.5, -15, 'Uncertainty\nZone', ha='center', fontsize=9, color='red')

    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(-22, 10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def fig2_separability_comparison(data, save_path):
    """
    Figure 2: U-Shape across different feature separabilities
    Shows robustness of U-shape pattern.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    separabilities = ['0.3', '0.5', '0.7', '1.0']
    titles = ['(a) s=0.3 (Hard)', '(b) s=0.5 (Medium)',
              '(c) s=0.7 (Easy)', '(d) s=1.0 (Very Easy)']

    for idx, (sep, title) in enumerate(zip(separabilities, titles)):
        ax = axes[idx]
        results = data['results'][sep]

        h_values = [r['h'] for r in results]
        gcn_means = np.array([r['gcn_acc'] * 100 for r in results])
        gcn_stds = np.array([r['gcn_std'] * 100 for r in results])
        mlp_means = np.array([r['mlp_acc'] * 100 for r in results])
        mlp_stds = np.array([r['mlp_std'] * 100 for r in results])

        # Plot with error bands
        ax.fill_between(h_values, gcn_means - gcn_stds, gcn_means + gcn_stds,
                       color=COLORS['GCN'], alpha=0.2)
        ax.fill_between(h_values, mlp_means - mlp_stds, mlp_means + mlp_stds,
                       color=COLORS['MLP'], alpha=0.2)

        ax.plot(h_values, gcn_means, 'o-', color=COLORS['GCN'],
               label='GCN', linewidth=2, markersize=5)
        ax.plot(h_values, mlp_means, 's--', color=COLORS['MLP'],
               label='MLP', linewidth=2, markersize=5)

        ax.set_xlabel('Homophily (h)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(70, 102)

        # Mark worst point
        worst_idx = np.argmin(gcn_means - mlp_means)
        ax.scatter([h_values[worst_idx]], [gcn_means[worst_idx]],
                  marker='v', s=100, color='red', zorder=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def fig3_spi_correlation(cross_model_data, save_path):
    """
    Figure 3: SPI = |2h-1| correlation with GCN advantage
    """
    results = cross_model_data['results']

    h_values = np.array([r['h'] for r in results])
    spi_values = np.abs(2 * h_values - 1)
    gcn_advantage = np.array([r['GCN_advantage'] * 100 for r in results])

    # Error in advantage
    gcn_stds = np.array([r['GCN_std'] for r in results])
    mlp_stds = np.array([r['MLP_std'] for r in results])
    advantage_stds = np.sqrt(gcn_stds**2 + mlp_stds**2) * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter with error bars
    ax.errorbar(spi_values, gcn_advantage, yerr=advantage_stds,
               fmt='o', capsize=5, capthick=2, markersize=10,
               color=COLORS['advantage'], ecolor='gray',
               label='Observed', zorder=5)

    # Linear fit
    coeffs = np.polyfit(spi_values, gcn_advantage, 1)
    x_fit = np.linspace(0, 1, 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, '--', color='gray', linewidth=2,
           label=f'Linear fit (slope={coeffs[0]:.1f})')

    # Calculate R^2
    y_pred = np.polyval(coeffs, spi_values)
    ss_res = np.sum((gcn_advantage - y_pred)**2)
    ss_tot = np.sum((gcn_advantage - np.mean(gcn_advantage))**2)
    r_squared = 1 - (ss_res / ss_tot)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0.4, color='green', linestyle='--', alpha=0.5,
              label='Trust threshold (SPI=0.4)')

    ax.set_xlabel('SPI = |2h - 1|')
    ax.set_ylabel('GCN Advantage over MLP (%)')
    ax.set_title(f'SPI Correlation with GNN Performance (R² = {r_squared:.2f})')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.05, 1.05)

    # Add zone labels
    ax.axvspan(-0.05, 0.4, alpha=0.1, color='red')
    ax.axvspan(0.4, 1.05, alpha=0.1, color='green')
    ax.text(0.2, -15, 'Uncertainty\nZone', ha='center', fontsize=10, color='red')
    ax.text(0.7, 5, 'Trust\nRegion', ha='center', fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    return r_squared


def fig4_defense_summary(cross_model_data, save_path):
    """
    Figure 4: Defense Summary - Key findings visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    results = cross_model_data['results']
    h_values = np.array([r['h'] for r in results])

    # Panel 1: Multi-model U-shape
    ax1 = axes[0]
    for model in ['GCN', 'GAT', 'GraphSAGE']:
        advantages = np.array([r[f'{model}_advantage'] * 100 for r in results])
        ax1.plot(h_values, advantages, 'o-', label=model,
                color=COLORS[model], linewidth=2, markersize=6)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvspan(0.3, 0.7, alpha=0.1, color='red')
    ax1.set_xlabel('Homophily (h)')
    ax1.set_ylabel('Advantage over MLP (%)')
    ax1.set_title('(a) U-Shape Across Models')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0.05, 0.95)

    # Panel 2: Trust Region pie chart
    ax2 = axes[1]

    # Count data points in each zone
    trust_high = sum(1 for h in h_values if h > 0.7)
    trust_low = sum(1 for h in h_values if h < 0.3)
    uncertain = sum(1 for h in h_values if 0.3 <= h <= 0.7)

    sizes = [trust_high + trust_low, uncertain]
    labels = [f'Trust Regions\n(h<0.3 or h>0.7)\n{trust_high + trust_low} points',
              f'Uncertainty Zone\n(0.3≤h≤0.7)\n{uncertain} points']
    colors = ['#27AE60', '#E74C3C']
    explode = (0.05, 0)

    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.0f%%', shadow=False, startangle=90,
           textprops={'fontsize': 10})
    ax2.set_title('(b) Data Distribution')

    # Panel 3: SPI formula visualization
    ax3 = axes[2]
    h_range = np.linspace(0, 1, 100)
    spi = np.abs(2 * h_range - 1)

    ax3.plot(h_range, spi, 'b-', linewidth=3, label='SPI = |2h - 1|')
    ax3.fill_between(h_range, 0, spi, where=(spi >= 0.4),
                     color='green', alpha=0.2, label='Trust (SPI ≥ 0.4)')
    ax3.fill_between(h_range, 0, spi, where=(spi < 0.4),
                     color='red', alpha=0.2, label='Uncertain (SPI < 0.4)')

    ax3.axhline(y=0.4, color='gray', linestyle='--', linewidth=1)
    ax3.set_xlabel('Homophily (h)')
    ax3.set_ylabel('SPI')
    ax3.set_title('(c) SPI Definition')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.05)

    # Mark key points
    ax3.annotate('h=0.5\nSPI=0', xy=(0.5, 0), xytext=(0.5, 0.15),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax3.annotate('h=0,1\nSPI=1', xy=(0, 1), xytext=(0.15, 0.85),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """Generate all defense figures."""
    print("="*60)
    print("Generating Defense Figures with Error Bands")
    print("="*60)

    # Load data
    cross_model_data, separability_data = load_data()

    # Output directory
    output_dir = Path(__file__).parent / 'figures' / 'defense'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("\n[1/4] Cross-Model U-Shape with Error Bands...")
    fig1_cross_model_ushape_with_errorbands(
        cross_model_data,
        str(output_dir / 'fig1_cross_model_ushape_errorbands.pdf')
    )

    print("\n[2/4] Separability Comparison...")
    fig2_separability_comparison(
        separability_data,
        str(output_dir / 'fig2_separability_comparison.pdf')
    )

    print("\n[3/4] SPI Correlation...")
    r_squared = fig3_spi_correlation(
        cross_model_data,
        str(output_dir / 'fig3_spi_correlation.pdf')
    )
    print(f"   R^2 = {r_squared:.3f}")

    print("\n[4/4] Defense Summary...")
    fig4_defense_summary(
        cross_model_data,
        str(output_dir / 'fig4_defense_summary.pdf')
    )

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Generate summary
    summary = {
        'figures_generated': [
            'fig1_cross_model_ushape_errorbands.pdf',
            'fig2_separability_comparison.pdf',
            'fig3_spi_correlation.pdf',
            'fig4_defense_summary.pdf'
        ],
        'r_squared_spi': r_squared,
        'key_findings': {
            'u_shape_confirmed': True,
            'worst_h': 0.5,
            'trust_threshold': 0.4,
            'models_tested': ['MLP', 'GCN', 'GAT', 'GraphSAGE']
        }
    }

    with open(output_dir / 'figure_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    main()
