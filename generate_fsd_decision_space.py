"""
Generate ρ_FS vs δ_agg Decision Space Visualization

This script creates a 2D decision space plot showing the relationship between
feature set homophily (ρ_FS) and aggregation depth (δ_agg) for different datasets.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set up matplotlib parameters for publication-quality figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Data points
datasets = {
    'Elliptic': {
        'rho_FS': 0.31,
        'delta_agg': 0.94,
        'n_features': 165,
        'winner': 'NAA'
    },
    'Amazon': {
        'rho_FS': 0.18,
        'delta_agg': 5.0,
        'n_features': 767,
        'winner': 'NAA'
    },
    'YelpChi': {
        'rho_FS': -0.12,
        'delta_agg': 12.57,
        'n_features': 32,
        'winner': 'H2GCN'
    },
    'IEEE-CIS': {
        'rho_FS': 0.06,
        'delta_agg': 11.25,
        'n_features': 133,
        'winner': 'H2GCN'
    }
}

# Color mapping
color_map = {
    'NAA': '#1f77b4',      # Blue
    'H2GCN': '#ff7f0e'      # Orange
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Add decision region shading
# Class A (NAA preferred): δ_agg < 10 (light blue)
ax.axhspan(0, 10, facecolor='#1f77b4', alpha=0.1, label='Class A (NAA preferred)')
# Class B (H2GCN preferred): δ_agg >= 10 (light orange)
ax.axhspan(10, 15, facecolor='#ff7f0e', alpha=0.1, label='Class B (H2GCN preferred)')

# Add horizontal decision boundary line
ax.axhline(y=10, color='gray', linestyle='--', linewidth=2,
           label='Decision Boundary (δ_agg=10)', zorder=2)

# Plot data points
for dataset_name, data in datasets.items():
    rho_FS = data['rho_FS']
    delta_agg = data['delta_agg']
    n_features = data['n_features']
    winner = data['winner']

    # Scale point size based on n_features (normalize to reasonable range)
    # Use sqrt scaling for better visual representation
    size = (n_features / 767) * 500  # Scale relative to max features (Amazon)

    # Plot the point
    ax.scatter(rho_FS, delta_agg, s=size, c=color_map[winner],
               alpha=0.7, edgecolors='black', linewidth=1.5, zorder=3)

    # Add dataset name annotation
    # Adjust text position to avoid overlap
    offset_x = 0.01
    offset_y = 0.3
    if dataset_name == 'YelpChi':
        offset_x = -0.01
        offset_y = -0.5
    elif dataset_name == 'IEEE-CIS':
        offset_x = 0.01
        offset_y = 0.3
    elif dataset_name == 'Amazon':
        offset_x = 0.01
        offset_y = -0.5

    ax.annotate(dataset_name,
                xy=(rho_FS, delta_agg),
                xytext=(rho_FS + offset_x, delta_agg + offset_y),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray', alpha=0.8),
                zorder=4)

# Create custom legend for winners
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='NAA Winner',
           markerfacecolor=color_map['NAA'], markersize=10,
           markeredgecolor='black', markeredgewidth=1.5),
    Line2D([0], [0], marker='o', color='w', label='H2GCN Winner',
           markerfacecolor=color_map['H2GCN'], markersize=10,
           markeredgecolor='black', markeredgewidth=1.5),
    Line2D([0], [0], color='gray', linestyle='--', linewidth=2,
           label='Decision Boundary (δ_agg=10)')
]

# Add legend
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95)

# Add note about point size
ax.text(0.02, 0.98, 'Point size ∝ Number of features',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

# Set axis labels and title
ax.set_xlabel('Feature Set Homophily (ρ_FS)', fontweight='bold')
ax.set_ylabel('Aggregation Depth (δ_agg)', fontweight='bold')
ax.set_title('Decision Space: ρ_FS vs δ_agg', fontweight='bold', pad=20)

# Set axis limits
ax.set_xlim(-0.2, 0.4)
ax.set_ylim(0, 15)

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Tight layout
plt.tight_layout()

# Create figures directory if it doesn't exist
figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Save in multiple formats
output_path_pdf = os.path.join(figures_dir, 'fsd_decision_space.pdf')
output_path_png = os.path.join(figures_dir, 'fsd_decision_space.png')

plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')

print(f"Decision space plot saved to:")
print(f"  - {output_path_pdf}")
print(f"  - {output_path_png}")

# Display the plot
plt.show()

print("\nDataset Summary:")
print("-" * 70)
print(f"{'Dataset':<12} {'ρ_FS':<10} {'δ_agg':<10} {'Features':<10} {'Winner':<10}")
print("-" * 70)
for name, data in datasets.items():
    print(f"{name:<12} {data['rho_FS']:<10.2f} {data['delta_agg']:<10.2f} "
          f"{data['n_features']:<10} {data['winner']:<10}")
print("-" * 70)
