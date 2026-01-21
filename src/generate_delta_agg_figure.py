"""
Generate scatter plot showing relationship between δ_agg and NAA improvement.

This script visualizes how aggregation dilution (δ_agg) correlates with
NAA's relative improvement over baseline methods across different datasets.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Configure matplotlib for high-quality output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.3

# Data points: (δ_agg, NAA improvement %)
data = {
    'Elliptic': {
        'delta_agg': 0.94,
        'improvement': 25.0,  # +25% AUC
        'color': '#0173B2',  # Blue (colorblind-friendly)
        'marker': 'o'
    },
    'Amazon': {
        'delta_agg': 5.0,
        'improvement': 50.0,  # +0.46 F1 (~50%)
        'color': '#029E73',  # Green (colorblind-friendly)
        'marker': 's'
    },
    'YelpChi': {
        'delta_agg': 12.57,
        'improvement': -0.5,  # -0.5% (H2GCN wins)
        'color': '#DE8F05',  # Orange (colorblind-friendly)
        'marker': '^'
    },
    'IEEE-CIS': {
        'delta_agg': 11.25,
        'improvement': -7.0,  # -7% (H2GCN wins)
        'color': '#CC78BC',  # Purple (colorblind-friendly)
        'marker': 'd'
    }
}

def create_delta_agg_figure():
    """Create scatter plot showing δ_agg vs NAA improvement."""

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each dataset
    for dataset_name, dataset_info in data.items():
        ax.scatter(
            dataset_info['delta_agg'],
            dataset_info['improvement'],
            s=200,
            color=dataset_info['color'],
            marker=dataset_info['marker'],
            label=dataset_name,
            edgecolors='black',
            linewidths=1.5,
            alpha=0.85,
            zorder=3
        )

        # Add dataset label near the point
        offset_x = 0.3
        offset_y = 2 if dataset_info['improvement'] > 0 else -2
        ax.annotate(
            dataset_name,
            (dataset_info['delta_agg'], dataset_info['improvement']),
            xytext=(offset_x, offset_y),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            ha='left' if dataset_info['delta_agg'] < 10 else 'right'
        )

    # Add threshold line at δ_agg = 10
    threshold = 10.0
    ax.axvline(
        x=threshold,
        color='red',
        linestyle='--',
        linewidth=2.5,
        alpha=0.7,
        zorder=2,
        label=f'Threshold (δ_agg = {threshold})'
    )

    # Add horizontal line at y=0 (no improvement)
    ax.axhline(
        y=0,
        color='gray',
        linestyle='-',
        linewidth=1.5,
        alpha=0.5,
        zorder=1
    )

    # Add region labels
    # Class A (left side - NAA wins)
    ax.text(
        2.5, 55,
        'Class A\n(NAA wins)',
        fontsize=12,
        fontweight='bold',
        ha='center',
        va='top',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='lightgreen',
            edgecolor='darkgreen',
            alpha=0.3,
            linewidth=2
        )
    )

    # Class B (right side - Sampling wins)
    ax.text(
        12.5, 55,
        'Class B\n(Sampling wins)',
        fontsize=12,
        fontweight='bold',
        ha='center',
        va='top',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='lightcoral',
            edgecolor='darkred',
            alpha=0.3,
            linewidth=2
        )
    )

    # Set axis labels
    ax.set_xlabel(
        r'Aggregation Dilution ($\delta_{agg}$)',
        fontsize=13,
        fontweight='bold'
    )
    ax.set_ylabel(
        'NAA Relative Improvement (%)',
        fontsize=13,
        fontweight='bold'
    )

    # Set axis limits
    ax.set_xlim(-0.5, 15)
    ax.set_ylim(-10, 60)

    # Add grid
    ax.grid(True, linestyle=':', alpha=0.3, zorder=0)

    # Add legend
    ax.legend(
        loc='lower left',
        frameon=True,
        framealpha=0.95,
        edgecolor='black',
        fontsize=10,
        ncol=2
    )

    # Set title
    ax.set_title(
        r'Relationship between $\delta_{agg}$ and NAA Performance',
        fontsize=14,
        fontweight='bold',
        pad=15
    )

    # Tighten layout
    plt.tight_layout()

    return fig

def main():
    """Main function to generate and save the figure."""

    # Create output directory if it doesn't exist
    output_dir = r'D:\Users\11919\Documents\毕业论文\paper\figures'
    os.makedirs(output_dir, exist_ok=True)

    # Generate figure
    print("Generating δ_agg vs NAA improvement scatter plot...")
    fig = create_delta_agg_figure()

    # Save as PDF
    pdf_path = os.path.join(output_dir, 'delta_agg_scatter.pdf')
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved PDF: {pdf_path}")

    # Save as PNG
    png_path = os.path.join(output_dir, 'delta_agg_scatter.png')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved PNG: {png_path}")

    # Print summary
    print("\n" + "="*60)
    print("Summary of Data Points:")
    print("="*60)
    for dataset_name, dataset_info in data.items():
        status = "NAA wins" if dataset_info['improvement'] > 0 else "Sampling wins"
        print(f"{dataset_name:12} | δ_agg={dataset_info['delta_agg']:6.2f} | "
              f"Improvement={dataset_info['improvement']:+6.1f}% | {status}")
    print("="*60)

    print("\nFigure generation complete!")
    print(f"Output directory: {output_dir}")

    # Close figure to free memory
    plt.close(fig)

if __name__ == '__main__':
    main()
