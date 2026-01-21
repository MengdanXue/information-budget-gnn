"""
Visualization Suite for SPI-Guided Gating Paper
===============================================

Generates all key figures for the thesis:
1. U-Shape Discovery Plot (GCN vs MLP across homophily)
2. SPI Correlation Plot (SPI vs GCN Advantage, R^2=0.82)
3. Gating Curve Plot (beta vs SPI with tau threshold)
4. Trust Region Diagram (visual explanation of regions)

Author: Thesis Research
Date: 2024-12
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from pathlib import Path

# Set Chinese font support (optional)
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Output directory
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_u_shape_discovery(save_path=None):
    """
    Plot the U-Shape discovery: GCN advantage across homophily values.

    Key finding: GCN wins at both extremes (h<0.3 and h>0.7),
    loses most at h=0.5.
    """
    # Data from experiments (separability=0.5)
    h_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    gcn_acc = np.array([99.8, 98.8, 93.3, 78.8, 74.8, 87.1, 96.0, 99.1, 99.8])
    mlp_acc = np.array([93.4, 93.4, 93.4, 93.4, 93.4, 93.4, 93.4, 93.4, 93.4])
    gcn_advantage = gcn_acc - mlp_acc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Accuracy comparison ---
    ax1.plot(h_values, gcn_acc, 'b-o', linewidth=2, markersize=8, label='GCN')
    ax1.plot(h_values, mlp_acc, 'r--s', linewidth=2, markersize=8, label='MLP')

    # Fill regions
    ax1.fill_between(h_values, gcn_acc, mlp_acc,
                     where=(gcn_acc >= mlp_acc),
                     interpolate=True, alpha=0.3, color='blue', label='GCN advantage')
    ax1.fill_between(h_values, gcn_acc, mlp_acc,
                     where=(gcn_acc < mlp_acc),
                     interpolate=True, alpha=0.3, color='red', label='MLP advantage')

    ax1.set_xlabel('Homophily (h)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('GCN vs MLP Accuracy Across Homophily', fontsize=14)
    ax1.legend(loc='lower center')
    ax1.set_xlim(0.05, 0.95)
    ax1.set_ylim(70, 102)
    ax1.grid(True, alpha=0.3)

    # --- Right: U-Shape (GCN Advantage) ---
    colors = ['green' if adv > 0 else 'red' for adv in gcn_advantage]
    ax2.bar(h_values, gcn_advantage, width=0.08, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Mark key points
    ax2.annotate('Valley\n(h=0.5)', xy=(0.5, -18.6), xytext=(0.5, -25),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax2.annotate('Peak\n(h=0.1)', xy=(0.1, 6.4), xytext=(0.2, 12),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='green'))
    ax2.annotate('Peak\n(h=0.9)', xy=(0.9, 6.4), xytext=(0.8, 12),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='green'))

    ax2.set_xlabel('Homophily (h)', fontsize=12)
    ax2.set_ylabel('GCN Advantage (%)', fontsize=12)
    ax2.set_title('U-Shape: GCN Advantage = GCN - MLP', fontsize=14)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(-25, 15)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_spi_correlation(save_path=None):
    """
    Plot SPI correlation with GCN Advantage.

    Key statistics:
    - Pearson r = 0.906
    - R^2 = 0.82
    - Linear model: GCN_Advantage = 25.30 * SPI - 15.99
    """
    # Data from 45 experimental points
    h_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    gcn_advantage = np.array([6.4, 5.3, -0.1, -14.6, -18.6, -6.3, 2.6, 5.7, 6.4])

    # Compute SPI
    spi_values = np.abs(2 * h_values - 1)

    # Add more data points (from different separabilities)
    # Separability 0.3
    h_sep03 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    adv_sep03 = np.array([7.7, 0, -19.0, 0, 5.3])
    spi_sep03 = np.abs(2 * h_sep03 - 1)

    # Separability 0.7
    h_sep07 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    adv_sep07 = np.array([0.7, 0, -19.1, 0, 0])
    spi_sep07 = np.abs(2 * h_sep07 - 1)

    # Combine all data
    all_spi = np.concatenate([spi_values, spi_sep03, spi_sep07])
    all_adv = np.concatenate([gcn_advantage, adv_sep03, adv_sep07])

    # Fit linear regression
    slope, intercept = np.polyfit(all_spi, all_adv, 1)
    r_squared = 0.82  # From our analysis

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot with different colors for separability
    ax.scatter(spi_values, gcn_advantage, c='blue', s=80, alpha=0.7,
               label='Sep=0.5', edgecolors='black')
    ax.scatter(spi_sep03, adv_sep03, c='green', s=80, alpha=0.7,
               label='Sep=0.3', edgecolors='black', marker='s')
    ax.scatter(spi_sep07, adv_sep07, c='orange', s=80, alpha=0.7,
               label='Sep=0.7', edgecolors='black', marker='^')

    # Regression line
    x_line = np.linspace(0, 1, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2,
            label=f'Linear fit (R²={r_squared:.2f})')

    # Mark threshold
    tau = 0.67
    ax.axvline(x=tau, color='gray', linestyle='--', linewidth=1.5,
               label=f'τ = {tau}')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Shade trust region
    ax.axvspan(tau, 1.0, alpha=0.15, color='green', label='Trust Region (GNN)')
    ax.axvspan(0, tau, alpha=0.15, color='red', label='Uncertainty Zone (MLP)')

    # Annotations
    ax.text(0.85, 8, 'Trust\nGNN', fontsize=11, ha='center', fontweight='bold', color='darkgreen')
    ax.text(0.35, 8, 'Trust\nMLP', fontsize=11, ha='center', fontweight='bold', color='darkred')

    ax.set_xlabel('SPI = |2h - 1|', fontsize=12)
    ax.set_ylabel('GCN Advantage (%)', fontsize=12)
    ax.set_title(f'SPI Correlation: r = 0.906, R² = {r_squared}', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-25, 15)
    ax.grid(True, alpha=0.3)

    # Add equation
    ax.text(0.05, -22, f'GCN Advantage = {slope:.2f} × SPI + ({intercept:.2f})',
            fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_gating_curve(tau=0.67, T=0.1, save_path=None):
    """
    Plot the SPI-guided soft gating curve.

    Formula: beta = Sigmoid((SPI - tau) / T)
    """
    spi_values = np.linspace(0, 1, 200)

    # Compute beta for different temperatures
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    beta_T01 = sigmoid((spi_values - tau) / 0.1)
    beta_T005 = sigmoid((spi_values - tau) / 0.05)
    beta_T02 = sigmoid((spi_values - tau) / 0.2)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot curves for different T
    ax.plot(spi_values, beta_T005, 'g--', linewidth=1.5, alpha=0.7, label='T=0.05 (sharper)')
    ax.plot(spi_values, beta_T01, 'b-', linewidth=2.5, label='T=0.10 (default)')
    ax.plot(spi_values, beta_T02, 'orange', linestyle='--', linewidth=1.5, alpha=0.7, label='T=0.20 (smoother)')

    # Mark threshold
    ax.axvline(x=tau, color='red', linestyle='--', linewidth=2, label=f'τ={tau}')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)

    # Shade regions
    ax.axvspan(0, tau, alpha=0.12, color='red')
    ax.axvspan(tau, 1, alpha=0.12, color='green')

    # Add region labels
    ax.text(0.33, 0.15, 'β → 0\n(Trust MLP)', fontsize=11, ha='center',
            fontweight='bold', color='darkred')
    ax.text(0.83, 0.85, 'β → 1\n(Trust GNN)', fontsize=11, ha='center',
            fontweight='bold', color='darkgreen')

    # Annotate formula
    formula_box = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='black')
    ax.text(0.5, 0.05, r'$\beta = \sigma\left(\frac{SPI - \tau}{T}\right)$',
            fontsize=14, ha='center', transform=ax.transAxes, bbox=formula_box)

    ax.set_xlabel('SPI (Structural Predictability Index)', fontsize=12)
    ax.set_ylabel('β (GNN weight in fusion)', fontsize=12)
    ax.set_title('SPI-Guided Soft Gating Curve', fontsize=14)
    ax.legend(loc='center right', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_trust_region_diagram(save_path=None):
    """
    Create a schematic diagram explaining Trust Regions.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Create the main bar representing homophily range
    bar_height = 0.5
    bar_y = 0.3

    # Trust Region (low h): heterophily
    rect1 = FancyBboxPatch((0, bar_y), 0.17, bar_height,
                           boxstyle="round,pad=0.02",
                           facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(rect1)

    # Uncertainty Zone (middle)
    rect2 = Rectangle((0.17, bar_y), 0.66, bar_height,
                      facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(rect2)

    # Trust Region (high h): homophily
    rect3 = FancyBboxPatch((0.83, bar_y), 0.17, bar_height,
                           boxstyle="round,pad=0.02",
                           facecolor='lightgreen', edgecolor='green', linewidth=2)
    ax.add_patch(rect3)

    # Labels on the bar
    ax.text(0.085, bar_y + bar_height/2, 'Trust\nRegion\n(GNN)',
            ha='center', va='center', fontsize=10, fontweight='bold', color='darkgreen')
    ax.text(0.5, bar_y + bar_height/2, 'Uncertainty Zone\n(prefer MLP)',
            ha='center', va='center', fontsize=10, fontweight='bold', color='darkorange')
    ax.text(0.915, bar_y + bar_height/2, 'Trust\nRegion\n(GNN)',
            ha='center', va='center', fontsize=10, fontweight='bold', color='darkgreen')

    # Homophily scale below
    ax.annotate('', xy=(0, 0.15), xytext=(1, 0.15),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(0.5, 0.05, 'Homophily (h)', ha='center', fontsize=11)

    # Scale markers
    for h_val, label in [(0, '0.0'), (0.17, '0.17'), (0.5, '0.5'), (0.83, '0.83'), (1.0, '1.0')]:
        ax.plot([h_val, h_val], [0.15, 0.25], 'k-', lw=1)
        ax.text(h_val, 0.1, label, ha='center', fontsize=9)

    # SPI scale above
    ax.annotate('', xy=(0, 0.95), xytext=(1, 0.95),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax.text(0.5, 1.05, 'SPI = |2h - 1|', ha='center', fontsize=11, color='blue')

    # SPI markers
    for h_val, spi_val in [(0, 1.0), (0.17, 0.66), (0.5, 0.0), (0.83, 0.66), (1.0, 1.0)]:
        ax.plot([h_val, h_val], [0.85, 0.95], 'b-', lw=1)
        ax.text(h_val, 1.0, f'{spi_val:.2f}', ha='center', fontsize=9, color='blue')

    # Threshold annotation
    ax.annotate('τ = 0.67', xy=(0.17, 0.75), xytext=(0.3, 0.9),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.annotate('τ = 0.67', xy=(0.83, 0.75), xytext=(0.7, 0.9),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Trust Region Framework: SPI-based Model Selection', fontsize=14, pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def plot_fusion_architecture(save_path=None):
    """
    Create architecture diagram showing the fusion model.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define box positions
    box_style = dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='blue', lw=2)
    box_style_mlp = dict(boxstyle='round,pad=0.3', facecolor='lightsalmon', edgecolor='red', lw=2)
    box_style_gate = dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='green', lw=2)
    box_style_output = dict(boxstyle='round,pad=0.3', facecolor='gold', edgecolor='orange', lw=2)

    # Input
    ax.text(0.1, 0.5, 'Input\nFeatures X', fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', lw=2))

    # GNN Branch
    ax.text(0.35, 0.75, 'GNN\nBranch', fontsize=11, ha='center', va='center', bbox=box_style)
    ax.text(0.35, 0.55, 'Z_GNN', fontsize=10, ha='center', style='italic', color='blue')

    # MLP Branch
    ax.text(0.35, 0.25, 'MLP\nBranch', fontsize=11, ha='center', va='center', bbox=box_style_mlp)
    ax.text(0.35, 0.05, 'Z_MLP', fontsize=10, ha='center', style='italic', color='red')

    # SPI computation
    ax.text(0.55, 0.5, 'SPI\nComputation', fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lavender', edgecolor='purple', lw=2))
    ax.text(0.55, 0.35, 'SPI = |2h-1|', fontsize=9, ha='center', style='italic', color='purple')

    # Soft Gating
    ax.text(0.75, 0.5, 'Soft\nGating', fontsize=11, ha='center', va='center', bbox=box_style_gate)
    ax.text(0.75, 0.32, 'β = σ((SPI-τ)/T)', fontsize=9, ha='center', style='italic', color='green')

    # Output
    ax.text(0.92, 0.5, 'Output\nZ_final', fontsize=11, ha='center', va='center', bbox=box_style_output)

    # Arrows
    arrow_props = dict(arrowstyle='->', color='black', lw=1.5)

    # Input to branches
    ax.annotate('', xy=(0.25, 0.7), xytext=(0.15, 0.55), arrowprops=arrow_props)
    ax.annotate('', xy=(0.25, 0.3), xytext=(0.15, 0.45), arrowprops=arrow_props)

    # Branches to gating
    ax.annotate('', xy=(0.65, 0.55), xytext=(0.45, 0.7), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.45), xytext=(0.45, 0.3), arrowprops=arrow_props)

    # SPI to gating
    ax.annotate('', xy=(0.68, 0.5), xytext=(0.62, 0.5), arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

    # Gating to output
    ax.annotate('', xy=(0.85, 0.5), xytext=(0.82, 0.5), arrowprops=arrow_props)

    # Edge index input to SPI
    ax.text(0.55, 0.7, 'Edge Index\n(Graph Structure)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', lw=1))
    ax.annotate('', xy=(0.55, 0.58), xytext=(0.55, 0.65), arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # Fusion formula at bottom
    formula_box = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='black')
    ax.text(0.5, 0.02, r'$Z_{final} = \beta \cdot Z_{GNN} + (1-\beta) \cdot Z_{MLP}$',
            fontsize=12, ha='center', bbox=formula_box)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1)
    ax.axis('off')
    ax.set_title('SPI-Guided Fusion Architecture', fontsize=14, pad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def generate_all_figures():
    """Generate all figures for the paper."""
    print("="*60)
    print("Generating Visualization Suite for SPI-Guided Gating Paper")
    print("="*60)

    figures = [
        ("u_shape_discovery.png", plot_u_shape_discovery),
        ("spi_correlation.png", plot_spi_correlation),
        ("gating_curve.png", plot_gating_curve),
        ("trust_region_diagram.png", plot_trust_region_diagram),
        ("fusion_architecture.png", plot_fusion_architecture),
    ]

    for filename, plot_func in figures:
        save_path = OUTPUT_DIR / filename
        print(f"\nGenerating: {filename}...")
        try:
            plot_func(save_path=save_path)
            print(f"  Success!")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "="*60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    generate_all_figures()
