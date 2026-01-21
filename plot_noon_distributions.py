"""
Visualization of NOON (No Opposite Neighbors) Phenomenon
Creates publication-quality figures for the TKDE paper
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

# Use non-interactive backend for server environments
matplotlib.use('Agg')

# Set publication-quality style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_results():
    """Load the cosine distribution analysis results."""
    results_path = Path(__file__).parent / 'cosine_distribution_analysis_results.json'
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_cosine_histograms(results, output_dir):
    """
    Plot cosine similarity histograms for all datasets.
    Shows original, centered, and whitened features side by side.
    """
    datasets = results['datasets']

    # Select representative datasets (3 homophilic + 3 heterophilic)
    homophilic = ['cora', 'citeseer', 'pubmed']
    heterophilic = ['texas', 'wisconsin', 'actor']

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    for idx, (name, ax) in enumerate(zip(homophilic + heterophilic, axes.flatten())):
        if name not in datasets:
            continue

        data = datasets[name]
        h = data['homophily']

        # Get histogram data
        orig_hist = data['original_features']['histogram_bins']
        orig_edges = data['original_features']['histogram_edges']

        # Compute bin centers
        bin_centers = [(orig_edges[i] + orig_edges[i+1])/2 for i in range(len(orig_edges)-1)]

        # Normalize to density
        orig_density = np.array(orig_hist) / (sum(orig_hist) * (orig_edges[1] - orig_edges[0]))

        # Plot
        ax.bar(bin_centers, orig_density, width=0.04, alpha=0.7, color='steelblue',
               edgecolor='none', label='Original')

        # Add centered features if available
        if 'centered_features' in data:
            cent_hist = data['centered_features']['histogram_bins']
            cent_density = np.array(cent_hist) / (sum(cent_hist) * (orig_edges[1] - orig_edges[0]))
            ax.plot(bin_centers, cent_density, 'r-', linewidth=1.5, label='Centered', alpha=0.8)

        # Add vertical line at 0 and -0.1
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(x=-0.1, color='red', linestyle=':', linewidth=1, alpha=0.7)

        # Labels
        ax.set_title(f'{name.capitalize()} (h={h:.2f})', fontweight='bold')
        ax.set_xlim(-0.5, 0.8)
        ax.set_ylim(0, None)

        if idx >= 3:
            ax.set_xlabel('Cosine Similarity')
        if idx % 3 == 0:
            ax.set_ylabel('Density')

        # Only show legend for first plot
        if idx == 0:
            ax.legend(loc='upper right', framealpha=0.9)

    # Add row labels
    fig.text(0.02, 0.75, 'Homophilic', rotation=90, fontsize=12, fontweight='bold', va='center')
    fig.text(0.02, 0.3, 'Heterophilic', rotation=90, fontsize=12, fontweight='bold', va='center')

    plt.tight_layout()
    plt.subplots_adjust(left=0.08)

    output_path = output_dir / 'noon_cosine_distributions.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'noon_cosine_distributions.png')
    print(f"Saved: {output_path}")
    plt.close()


def plot_noon_summary(results, output_dir):
    """
    Plot summary of NOON phenomenon across all datasets.
    Bar chart showing fraction < -0.1 for original, centered, whitened.
    """
    datasets = results['datasets']

    names = []
    orig_fracs = []
    cent_fracs = []
    white_fracs = []
    homophilies = []

    for name, data in datasets.items():
        names.append(name.capitalize())
        homophilies.append(data['homophily'])

        orig_fracs.append(
            data['original_features']['threshold_sensitivity']['frac_below_-0.1'] * 100
        )

        if 'centered_features' in data:
            cent_fracs.append(
                data['centered_features']['threshold_sensitivity']['frac_below_-0.1'] * 100
            )
        else:
            cent_fracs.append(0)

        if 'whitened_features' in data and 'threshold_sensitivity' in data['whitened_features']:
            white_fracs.append(
                data['whitened_features']['threshold_sensitivity']['frac_below_-0.1'] * 100
            )
        else:
            white_fracs.append(0)

    # Sort by homophily
    sorted_idx = np.argsort(homophilies)
    names = [names[i] for i in sorted_idx]
    orig_fracs = [orig_fracs[i] for i in sorted_idx]
    cent_fracs = [cent_fracs[i] for i in sorted_idx]
    white_fracs = [white_fracs[i] for i in sorted_idx]
    homophilies = [homophilies[i] for i in sorted_idx]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))

    bars1 = ax.bar(x - width, orig_fracs, width, label='Original', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, cent_fracs, width, label='Centered', color='coral', alpha=0.8)
    bars3 = ax.bar(x + width, white_fracs, width, label='Whitened', color='seagreen', alpha=0.8)

    ax.set_ylabel('Fraction with cos < -0.1 (%)')
    ax.set_xlabel('Dataset (sorted by homophily)')
    ax.set_title('NOON Phenomenon: Opposite Neighbors Are Extremely Rare', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n(h={h:.2f})' for n, h in zip(names, homophilies)], rotation=0)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 15)

    # Add horizontal line at 5% threshold
    ax.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='5% threshold')

    # Add synthetic sanity check annotation
    ax.annotate('Synthetic "true opposite": 100%',
                xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top',
                fontsize=9, fontstyle='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = output_dir / 'noon_summary_bar.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'noon_summary_bar.png')
    print(f"Saved: {output_path}")
    plt.close()


def plot_fcs_ranges(results, output_dir):
    """
    Plot stacked bar chart showing FCS ranges for each dataset.
    Shows: Very Opposite, Opposite, Orthogonal, Similar, Very Similar
    """
    datasets = results['datasets']

    names = []
    very_opp = []
    opp = []
    orth = []
    sim = []
    very_sim = []
    homophilies = []

    for name, data in datasets.items():
        names.append(name.capitalize())
        homophilies.append(data['homophily'])

        thresh = data['original_features']['threshold_sensitivity']
        very_opp.append(thresh['range_very_opposite_lt_-0.5'] * 100)
        opp.append(thresh['range_opposite_-0.5_to_-0.1'] * 100)
        orth.append(thresh['range_orthogonal_-0.1_to_0.1'] * 100)
        sim.append(thresh['range_similar_0.1_to_0.5'] * 100)
        very_sim.append(thresh['range_very_similar_gt_0.5'] * 100)

    # Sort by homophily
    sorted_idx = np.argsort(homophilies)
    names = [names[i] for i in sorted_idx]
    very_opp = [very_opp[i] for i in sorted_idx]
    opp = [opp[i] for i in sorted_idx]
    orth = [orth[i] for i in sorted_idx]
    sim = [sim[i] for i in sorted_idx]
    very_sim = [very_sim[i] for i in sorted_idx]
    homophilies = [homophilies[i] for i in sorted_idx]

    x = np.arange(len(names))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 5))

    # Stacked bar chart
    ax.bar(x, very_opp, width, label='Very Opposite (< -0.5)', color='darkred', alpha=0.9)
    ax.bar(x, opp, width, bottom=very_opp, label='Opposite (-0.5 to -0.1)', color='coral', alpha=0.9)
    ax.bar(x, orth, width, bottom=np.array(very_opp)+np.array(opp),
           label='Orthogonal (-0.1 to 0.1)', color='lightgray', alpha=0.9)
    ax.bar(x, sim, width, bottom=np.array(very_opp)+np.array(opp)+np.array(orth),
           label='Similar (0.1 to 0.5)', color='lightblue', alpha=0.9)
    ax.bar(x, very_sim, width, bottom=np.array(very_opp)+np.array(opp)+np.array(orth)+np.array(sim),
           label='Very Similar (> 0.5)', color='steelblue', alpha=0.9)

    ax.set_ylabel('Fraction of Different-Class Edges (%)')
    ax.set_xlabel('Dataset (sorted by homophily)')
    ax.set_title('Feature Contrast Score (FCS) Distribution: Real Graphs Have No Opposite Neighbors',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n(h={h:.2f})' for n, h in zip(names, homophilies)])
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(0, 100)

    plt.tight_layout()

    output_path = output_dir / 'fcs_range_distribution.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'fcs_range_distribution.png')
    print(f"Saved: {output_path}")
    plt.close()


def plot_synthetic_comparison(results, output_dir):
    """
    Plot comparison between real data and synthetic "true opposite" data.
    Shows that our method CAN detect opposite neighbors when they exist.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Real data (Texas - most heterophilic)
    ax1 = axes[0]
    texas_data = results['datasets']['texas']['original_features']
    hist = texas_data['histogram_bins']
    edges = texas_data['histogram_edges']
    bin_centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]
    density = np.array(hist) / (sum(hist) * (edges[1] - edges[0]))

    ax1.bar(bin_centers, density, width=0.04, alpha=0.8, color='steelblue', edgecolor='none')
    ax1.axvline(x=-0.1, color='red', linestyle='--', linewidth=1.5)
    ax1.set_title('Real Data: Texas (h=0.11)\n0% with cos < -0.1', fontweight='bold')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Density')
    ax1.set_xlim(-1, 1)
    ax1.text(0.05, 0.95, 'Heterophilic\nbut NOT opposite', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right: Synthetic data
    ax2 = axes[1]
    syn_data = results['synthetic_sanity_check']
    hist = syn_data['histogram_bins']
    edges = syn_data['histogram_edges']
    bin_centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]
    density = np.array(hist) / (max(sum(hist), 1) * (edges[1] - edges[0]))

    ax2.bar(bin_centers, density, width=0.04, alpha=0.8, color='darkred', edgecolor='none')
    ax2.axvline(x=-0.1, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title('Synthetic Data: True Opposites (h=0.0)\n100% with cos < -0.1', fontweight='bold')
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Density')
    ax2.set_xlim(-1, 1)
    ax2.text(0.05, 0.95, 'Designed with\nopposite features', transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Sanity Check: Our Method Detects Opposite Neighbors When They Exist',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'noon_sanity_check.pdf'
    plt.savefig(output_path)
    plt.savefig(output_dir / 'noon_sanity_check.png')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all NOON visualization figures."""
    print("=" * 60)
    print("Generating NOON Phenomenon Visualizations")
    print("=" * 60)

    # Load results
    results = load_results()

    # Output directory
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    # Generate figures
    print("\n1. Cosine similarity histograms...")
    plot_cosine_histograms(results, output_dir)

    print("\n2. NOON summary bar chart...")
    plot_noon_summary(results, output_dir)

    print("\n3. FCS range distribution...")
    plot_fcs_ranges(results, output_dir)

    print("\n4. Synthetic comparison (sanity check)...")
    plot_synthetic_comparison(results, output_dir)

    print("\n" + "=" * 60)
    print("All figures saved to:", output_dir)
    print("=" * 60)


if __name__ == '__main__':
    main()
