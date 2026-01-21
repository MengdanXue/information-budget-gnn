"""
Comprehensive Phase Diagram Experiment for Trust Regions Paper

Creates multiple phase diagrams:
1. Homophily vs MLP Accuracy (Feature Sufficiency) scatter with decision boundary
2. Homophily vs GNN Advantage scatter with quadratic fit
3. SPI correlation analysis

This is the definitive visualization for the paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

# Set style for publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# Load unified validation results
results_path = Path(__file__).parent / 'unified_validation_results.json'
with open(results_path, 'r') as f:
    data = json.load(f)

# Extract dataset info
datasets = []
for r in data['results']:
    datasets.append({
        'name': r['dataset'],
        'h': r['homophily'],
        'spi': r['spi'],
        'mlp_acc': r['mlp_mean'],
        'gcn_acc': r['gcn_mean'],
        'gat_acc': r['gat_mean'],
        'sage_acc': r['graphsage_mean'],
        'best_gnn': r['best_gnn'],
        'best_gnn_acc': r['best_gnn_mean'],
        'gcn_adv': r['gcn_advantage'],
        'best_gnn_adv': r['best_gnn_advantage'],
        'winner': r['actual_winner'],
        'prediction': r['spi_prediction'],
        'correct': r['prediction_correct']
    })

# Convert to numpy arrays
names = [d['name'] for d in datasets]
h_values = np.array([d['h'] for d in datasets])
spi_values = np.array([d['spi'] for d in datasets])
mlp_accs = np.array([d['mlp_acc'] for d in datasets])
gcn_accs = np.array([d['gcn_acc'] for d in datasets])
gcn_advs = np.array([d['gcn_adv'] for d in datasets])
best_gnn_advs = np.array([d['best_gnn_adv'] for d in datasets])
winners = [d['winner'] for d in datasets]
corrects = [d['correct'] for d in datasets]

# Color mapping
color_map = {'GNN': '#2ecc71', 'MLP': '#e74c3c', 'Tie': '#3498db'}
colors = [color_map[w] for w in winners]

# Create output directory
output_dir = Path(__file__).parent / 'figures'
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("COMPREHENSIVE PHASE DIAGRAM EXPERIMENT")
print("=" * 70)

# ============================================================
# Figure 1: Homophily vs MLP Accuracy Phase Diagram
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 7))

# Draw decision boundary regions
# High-h region (h > 0.5) - Trust GNN
ax1.add_patch(Rectangle((0.5, 0), 0.5, 1.0, facecolor='#d5f4e6', alpha=0.4, zorder=0))
# Low-h region (h < 0.5) - Default MLP
ax1.add_patch(Rectangle((0, 0), 0.5, 1.0, facecolor='#fadbd8', alpha=0.4, zorder=0))

# Plot scatter points with markers based on winner
for i, d in enumerate(datasets):
    marker = 'o' if d['winner'] == 'GNN' else ('^' if d['winner'] == 'MLP' else 's')
    size = 200 if d['correct'] else 150
    edgecolor = 'black' if d['correct'] else 'red'
    linewidth = 1.5 if d['correct'] else 3
    ax1.scatter(d['h'], d['mlp_acc'], c=color_map[d['winner']], s=size,
                marker=marker, edgecolors=edgecolor, linewidths=linewidth,
                zorder=3, alpha=0.9)

# Add dataset labels
label_offsets = {
    'Cora': (0.02, 0.01),
    'CiteSeer': (0.02, -0.02),
    'PubMed': (-0.08, 0.015),
    'Cornell': (0.015, 0.015),
    'Texas': (0.015, -0.025),
    'Wisconsin': (0.015, 0.015),
    'Actor': (0.015, -0.02),
    'Chameleon': (0.015, 0.015),
    'Squirrel': (-0.08, -0.015),  # Misclassified - highlight
    'Amazon-Computers': (-0.12, -0.02),
    'Amazon-Photo': (0.015, 0.015),
    'Coauthor-CS': (-0.1, -0.025),
}

for d in datasets:
    name = d['name']
    if name in label_offsets:
        offset = label_offsets[name]
        fontweight = 'bold' if not d['correct'] else 'normal'
        ax1.annotate(name, (d['h'], d['mlp_acc']),
                    xytext=(d['h'] + offset[0], d['mlp_acc'] + offset[1]),
                    fontsize=8, alpha=0.85, fontweight=fontweight)

# Draw decision boundary
ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=2.5,
            label='$h = 0.5$ Decision Boundary')

# Add region labels
ax1.text(0.75, 0.95, 'Trust Region\n(Use GNN)', fontsize=12, ha='center',
        fontweight='bold', color='#27ae60',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.text(0.25, 0.95, 'Uncertainty Zone\n(Use MLP/GraphSAGE)', fontsize=12, ha='center',
        fontweight='bold', color='#c0392b',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Axis settings
ax1.set_xlabel('Homophily ($h$)', fontsize=13)
ax1.set_ylabel('MLP Accuracy (Feature Sufficiency)', fontsize=13)
ax1.set_xlim(0, 1)
ax1.set_ylim(0.2, 1.0)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='GNN wins'),
    mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='MLP wins'),
    mpatches.Patch(facecolor='#3498db', edgecolor='black', label='Tie'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=10, label='Correct prediction'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='red', markeredgewidth=2, markersize=10, label='Incorrect prediction'),
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Title
ax1.set_title('Phase Diagram: Homophily vs Feature Sufficiency\n(N=12 datasets, 91.7% accuracy)',
             fontsize=14, fontweight='bold')

# Grid
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

plt.tight_layout()
fig1.savefig(output_dir / 'phase_diagram_h_vs_mlp.pdf', dpi=300, bbox_inches='tight')
fig1.savefig(output_dir / 'phase_diagram_h_vs_mlp.png', dpi=300, bbox_inches='tight')
print("\nFigure 1: Phase Diagram (h vs MLP Accuracy) saved")

# ============================================================
# Figure 2: Homophily vs GCN Advantage with Regression
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Scatter plot
for i, d in enumerate(datasets):
    marker = 'o' if d['winner'] == 'GNN' else ('^' if d['winner'] == 'MLP' else 's')
    ax2.scatter(d['h'], d['gcn_adv'] * 100, c=color_map[d['winner']], s=150,
                marker=marker, edgecolors='black', linewidths=1.5, zorder=3, alpha=0.9)

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(h_values, gcn_advs * 100)
x_line = np.linspace(0, 1, 100)
y_line = slope * x_line + intercept
ax2.plot(x_line, y_line, 'b--', linewidth=2,
         label=f'Linear: $y = {slope:.1f}x {intercept:+.1f}$, $R^2 = {r_value**2:.3f}$')

# Quadratic regression
coeffs = np.polyfit(h_values, gcn_advs * 100, 2)
y_quad = np.polyval(coeffs, x_line)
ax2.plot(x_line, y_quad, 'r-', linewidth=2,
         label=f'Quadratic: $R^2 = {np.corrcoef(gcn_advs*100, np.polyval(coeffs, h_values))[0,1]**2:.3f}$')

# Zero line
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Decision boundary
ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)

# Add dataset labels for extreme points
extreme_labels = ['Cora', 'Wisconsin', 'Texas', 'Squirrel', 'Chameleon']
for d in datasets:
    if d['name'] in extreme_labels:
        offset_y = 2 if d['gcn_adv'] > 0 else -3
        ax2.annotate(d['name'], (d['h'], d['gcn_adv']*100),
                    xytext=(d['h']+0.02, d['gcn_adv']*100 + offset_y),
                    fontsize=8, alpha=0.8)

ax2.set_xlabel('Homophily ($h$)', fontsize=13)
ax2.set_ylabel('GCN Advantage over MLP (%)', fontsize=13)
ax2.set_xlim(0, 1)
ax2.legend(loc='upper left', fontsize=10)
ax2.set_title(f'Homophily vs GCN Advantage\n(Pearson $r = {r_value:.3f}$, $p = {p_value:.4f}$)',
             fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(output_dir / 'phase_diagram_h_vs_gcn_advantage.pdf', dpi=300, bbox_inches='tight')
fig2.savefig(output_dir / 'phase_diagram_h_vs_gcn_advantage.png', dpi=300, bbox_inches='tight')
print("Figure 2: Homophily vs GCN Advantage saved")

# ============================================================
# Figure 3: SPI vs GCN Advantage (Quadratic Relationship)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Scatter plot with SPI on x-axis
for i, d in enumerate(datasets):
    marker = 'o' if d['winner'] == 'GNN' else ('^' if d['winner'] == 'MLP' else 's')
    ax3.scatter(d['spi'], d['gcn_adv'] * 100, c=color_map[d['winner']], s=150,
                marker=marker, edgecolors='black', linewidths=1.5, zorder=3, alpha=0.9)

# Quadratic fit (theoretical: I proportional to SPI^2)
coeffs_spi = np.polyfit(spi_values, gcn_advs * 100, 2)
x_spi = np.linspace(0.4, 0.85, 100)
y_spi_quad = np.polyval(coeffs_spi, x_spi)

# Linear fit for comparison
slope_spi, intercept_spi, r_spi, p_spi, _ = stats.linregress(spi_values, gcn_advs * 100)
y_spi_linear = slope_spi * x_spi + intercept_spi

ax3.plot(x_spi, y_spi_linear, 'b--', linewidth=2,
         label=f'Linear: $R^2 = {r_spi**2:.3f}$')
ax3.plot(x_spi, y_spi_quad, 'r-', linewidth=2,
         label=f'Quadratic: $R^2 = {np.corrcoef(gcn_advs*100, np.polyval(coeffs_spi, spi_values))[0,1]**2:.3f}$')

ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Trust region threshold
ax3.axvline(x=0.4, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='SPI = 0.4 threshold')

ax3.set_xlabel('Structural Predictability Index (SPI = $|2h-1|$)', fontsize=13)
ax3.set_ylabel('GCN Advantage over MLP (%)', fontsize=13)
ax3.set_xlim(0.35, 0.85)
ax3.legend(loc='upper left', fontsize=10)
ax3.set_title('SPI vs GCN Advantage (Testing $I \\propto SPI^2$ Relationship)',
             fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
fig3.savefig(output_dir / 'phase_diagram_spi_vs_gcn_advantage.pdf', dpi=300, bbox_inches='tight')
fig3.savefig(output_dir / 'phase_diagram_spi_vs_gcn_advantage.png', dpi=300, bbox_inches='tight')
print("Figure 3: SPI vs GCN Advantage saved")

# ============================================================
# Figure 4: Combined Publication Figure (2x2 grid)
# ============================================================
fig4, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Phase Diagram
ax_a = axes[0, 0]
ax_a.add_patch(Rectangle((0.5, 0), 0.5, 1.0, facecolor='#d5f4e6', alpha=0.4, zorder=0))
ax_a.add_patch(Rectangle((0, 0), 0.5, 1.0, facecolor='#fadbd8', alpha=0.4, zorder=0))

for i, d in enumerate(datasets):
    marker = 'o' if d['winner'] == 'GNN' else ('^' if d['winner'] == 'MLP' else 's')
    size = 120
    edgecolor = 'black' if d['correct'] else 'red'
    linewidth = 1 if d['correct'] else 2.5
    ax_a.scatter(d['h'], d['mlp_acc'], c=color_map[d['winner']], s=size,
                marker=marker, edgecolors=edgecolor, linewidths=linewidth,
                zorder=3, alpha=0.9)

ax_a.axvline(x=0.5, color='gray', linestyle='--', linewidth=2)
ax_a.text(0.75, 0.95, 'Trust\nRegion', fontsize=10, ha='center', fontweight='bold', color='#27ae60')
ax_a.text(0.25, 0.95, 'Uncertainty\nZone', fontsize=10, ha='center', fontweight='bold', color='#c0392b')
ax_a.set_xlabel('Homophily ($h$)')
ax_a.set_ylabel('MLP Accuracy')
ax_a.set_xlim(0, 1)
ax_a.set_ylim(0.2, 1.0)
ax_a.set_title('(A) Decision Boundary Phase Diagram', fontweight='bold')
ax_a.grid(True, alpha=0.3)

# Panel B: h vs GCN Advantage
ax_b = axes[0, 1]
for i, d in enumerate(datasets):
    marker = 'o' if d['winner'] == 'GNN' else ('^' if d['winner'] == 'MLP' else 's')
    ax_b.scatter(d['h'], d['gcn_adv']*100, c=color_map[d['winner']], s=120,
                marker=marker, edgecolors='black', linewidths=1, zorder=3, alpha=0.9)

ax_b.plot(x_line, slope * x_line + intercept, 'b--', linewidth=2, label=f'$R^2 = {r_value**2:.2f}$')
ax_b.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax_b.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax_b.set_xlabel('Homophily ($h$)')
ax_b.set_ylabel('GCN Advantage (%)')
ax_b.set_xlim(0, 1)
ax_b.set_title(f'(B) Homophily vs GCN Advantage ($r = {r_value:.2f}$)', fontweight='bold')
ax_b.legend(loc='upper left')
ax_b.grid(True, alpha=0.3)

# Panel C: Region-wise accuracy bar chart
ax_c = axes[1, 0]
high_h = [d for d in datasets if d['h'] > 0.5]
low_h = [d for d in datasets if d['h'] <= 0.5]

high_h_correct = sum(1 for d in high_h if d['correct'])
low_h_correct = sum(1 for d in low_h if d['correct'])

categories = ['High-$h$ ($h > 0.5$)', 'Low-$h$ ($h \\leq 0.5$)', 'Overall']
accuracies = [high_h_correct/len(high_h)*100, low_h_correct/len(low_h)*100,
              (high_h_correct + low_h_correct)/len(datasets)*100]
colors_bar = ['#2ecc71', '#e74c3c', '#3498db']

bars = ax_c.bar(categories, accuracies, color=colors_bar, edgecolor='black', linewidth=1.5)
ax_c.axhline(y=100, color='gray', linestyle=':', alpha=0.5)

for bar, acc, n in zip(bars, accuracies, [len(high_h), len(low_h), len(datasets)]):
    height = bar.get_height()
    ax_c.annotate(f'{acc:.1f}%\n(N={n})',
                 xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

ax_c.set_ylabel('Prediction Accuracy (%)')
ax_c.set_ylim(0, 120)
ax_c.set_title('(C) Region-wise SPI Prediction Accuracy', fontweight='bold')

# Panel D: Model comparison on low-h datasets
ax_d = axes[1, 1]
low_h_names = [d['name'] for d in low_h]
gcn_accs_low = [d['gcn_acc'] * 100 for d in low_h]
sage_accs_low = [d['sage_acc'] * 100 for d in low_h]
mlp_accs_low = [d['mlp_acc'] * 100 for d in low_h]

x = np.arange(len(low_h_names))
width = 0.25

bars1 = ax_d.bar(x - width, mlp_accs_low, width, label='MLP', color='#e74c3c', edgecolor='black')
bars2 = ax_d.bar(x, gcn_accs_low, width, label='GCN', color='#9b59b6', edgecolor='black')
bars3 = ax_d.bar(x + width, sage_accs_low, width, label='GraphSAGE', color='#2ecc71', edgecolor='black')

ax_d.set_ylabel('Accuracy (%)')
ax_d.set_xticks(x)
ax_d.set_xticklabels(low_h_names, rotation=45, ha='right', fontsize=9)
ax_d.legend(loc='upper right')
ax_d.set_title('(D) Model Comparison on Low-$h$ Datasets', fontweight='bold')
ax_d.set_ylim(0, 100)

plt.tight_layout()
fig4.savefig(output_dir / 'phase_diagram_combined.pdf', dpi=300, bbox_inches='tight')
fig4.savefig(output_dir / 'phase_diagram_combined.png', dpi=300, bbox_inches='tight')
print("Figure 4: Combined Publication Figure saved")

# ============================================================
# Print Summary Statistics
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(f"\n1. Overall Accuracy: {sum(corrects)}/{len(datasets)} = {100*sum(corrects)/len(datasets):.1f}%")
print(f"2. High-h Accuracy: {high_h_correct}/{len(high_h)} = {100*high_h_correct/len(high_h):.1f}%")
print(f"3. Low-h Accuracy: {low_h_correct}/{len(low_h)} = {100*low_h_correct/len(low_h):.1f}%")

print(f"\n4. Correlation Analysis:")
print(f"   - Homophily vs GCN Advantage: r = {r_value:.3f}, p = {p_value:.4f}")
print(f"   - SPI vs GCN Advantage: r = {r_spi:.3f}, p = {p_spi:.4f}")

# Identify misclassified
misclassified = [d for d in datasets if not d['correct']]
print(f"\n5. Misclassified datasets ({len(misclassified)}):")
for d in misclassified:
    print(f"   - {d['name']}: h={d['h']:.3f}, winner={d['winner']}, predicted={d['prediction']}")
    print(f"     GCN adv: {d['gcn_adv']*100:.1f}%, GraphSAGE won by: {(d['sage_acc']-d['mlp_acc'])*100:.1f}%")

# GraphSAGE analysis on low-h
print(f"\n6. GraphSAGE on Low-h Datasets:")
for d in low_h:
    sage_vs_gcn = (d['sage_acc'] - d['gcn_acc']) * 100
    sage_vs_mlp = (d['sage_acc'] - d['mlp_acc']) * 100
    print(f"   - {d['name']}: SAGE vs GCN: {sage_vs_gcn:+.1f}%, SAGE vs MLP: {sage_vs_mlp:+.1f}%")

print("\n" + "=" * 70)
print("All figures saved to:", output_dir)
print("=" * 70)
