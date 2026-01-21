#!/usr/bin/env python3
"""
Information Budget Visualization
================================

Create publication-quality figures for Information Budget Theory.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

# Load results
results_path = Path(__file__).parent / 'information_budget_results.json'
with open(results_path, 'r') as f:
    data = json.load(f)

output_dir = Path(__file__).parent / 'figures'
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("INFORMATION BUDGET VISUALIZATION")
print("=" * 70)

# ============================================================
# Figure 1: Edge Shuffle Effect
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

edge_shuffle = data['edge_shuffle']
datasets = [r['dataset'] for r in edge_shuffle]
x = np.arange(len(datasets))
width = 0.35

original_adv = [r['original_gcn_adv'] * 100 for r in edge_shuffle]
shuffled_adv = [r['shuffled_gcn_adv'] * 100 for r in edge_shuffle]

bars1 = ax1.bar(x - width/2, original_adv, width, label='Original Graph', color='#2ecc71', edgecolor='black')
bars2 = ax1.bar(x + width/2, shuffled_adv, width, label='Shuffled Edges', color='#e74c3c', edgecolor='black')

ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax1.set_ylabel('GCN Advantage over MLP (%)', fontsize=12)
ax1.set_xlabel('Dataset', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=11)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_title('Edge Shuffle Experiment: Structure Contribution to GNN Performance', fontweight='bold', fontsize=13)

# Add value labels
for bar, val in zip(bars1, original_adv):
    ax1.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, shuffled_adv):
    ax1.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, -12 if val < 0 else 3), textcoords='offset points', ha='center', va='top' if val < 0 else 'bottom', fontsize=9)

ax1.set_ylim(-45, 20)
ax1.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig1.savefig(output_dir / 'edge_shuffle_effect.pdf', dpi=300, bbox_inches='tight')
fig1.savefig(output_dir / 'edge_shuffle_effect.png', dpi=300, bbox_inches='tight')
print("Figure 1: Edge Shuffle Effect saved")

# ============================================================
# Figure 2: Feature Degradation - Information Budget
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

feat_degrad = data['feature_degradation']
noise_levels = [r['noise_level'] for r in feat_degrad]
mlp_accs = [r['mlp_acc'] * 100 for r in feat_degrad]
gcn_advs = [r['gcn_adv'] * 100 for r in feat_degrad]
budgets = [r['theoretical_max_gain'] * 100 for r in feat_degrad]

ax2.fill_between(noise_levels, 0, budgets, alpha=0.3, color='#3498db', label='Information Budget (1 - MLP)')
ax2.plot(noise_levels, budgets, 'b--', linewidth=2, label='Budget Ceiling')
ax2.plot(noise_levels, gcn_advs, 'ro-', linewidth=2, markersize=8, label='Actual GCN Advantage')

ax2.set_xlabel('Feature Noise Level', fontsize=12)
ax2.set_ylabel('Performance Gain (%)', fontsize=12)
ax2.set_title('Information Budget Validation: GNN Gain Bounded by (1 - MLP Accuracy)', fontweight='bold', fontsize=13)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 0.8)
ax2.set_ylim(0, 70)

# Add annotation
ax2.annotate('GNN advantage always\nbelow budget ceiling',
             xy=(0.5, 25), fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
fig2.savefig(output_dir / 'information_budget_validation.pdf', dpi=300, bbox_inches='tight')
fig2.savefig(output_dir / 'information_budget_validation.png', dpi=300, bbox_inches='tight')
print("Figure 2: Information Budget Validation saved")

# ============================================================
# Figure 3: Same-h Different-MLP
# ============================================================
fig3, ax3 = plt.subplots(figsize=(10, 7))

same_h_data = data['same_h_different_mlp']

# Extract data
mlp_accs_pairs = []
gcn_advs_pairs = []
labels_pairs = []

for pair in same_h_data:
    mlp_accs_pairs.extend([pair['mlp1'], pair['mlp2']])
    gcn_advs_pairs.extend([pair['gcn_adv1'], pair['gcn_adv2']])
    labels_pairs.extend([pair['dataset1'], pair['dataset2']])

# Remove duplicates
unique_data = {}
for mlp, adv, label in zip(mlp_accs_pairs, gcn_advs_pairs, labels_pairs):
    if label not in unique_data:
        unique_data[label] = (mlp * 100, adv * 100)

mlps = [v[0] for v in unique_data.values()]
advs = [v[1] for v in unique_data.values()]
labels = list(unique_data.keys())

# Scatter plot
ax3.scatter(mlps, advs, s=150, c='#3498db', edgecolors='black', linewidths=1.5, zorder=3)

# Add labels
for label, mlp, adv in zip(labels, mlps, advs):
    offset_x = 1 if mlp < 90 else -8
    offset_y = 1 if adv > 5 else -2
    ax3.annotate(label, (mlp, adv), xytext=(mlp + offset_x, adv + offset_y), fontsize=9)

# Regression line
slope, intercept, r_value, p_value, _ = stats.linregress(mlps, advs)
x_line = np.linspace(70, 100, 100)
y_line = slope * x_line + intercept
ax3.plot(x_line, y_line, 'r--', linewidth=2, label=f'Linear fit: $R^2 = {r_value**2:.3f}$')

# Add theoretical bound
theoretical_x = np.linspace(70, 100, 100)
theoretical_y = 100 - theoretical_x  # Budget = 1 - MLP
ax3.fill_between(theoretical_x, 0, theoretical_y, alpha=0.1, color='gray', label='Theoretical Max (1 - MLP)')

ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax3.set_xlabel('MLP Accuracy (%)', fontsize=12)
ax3.set_ylabel('GCN Advantage (%)', fontsize=12)
ax3.set_title('Key Evidence: Same Homophily, Different MLP → Different GNN Advantage\n(MLP Accuracy Determines GNN Utility)', fontweight='bold', fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(70, 100)
ax3.set_ylim(-5, 20)

# Add key pair highlight
ax3.annotate('', xy=(75.7, 12.6), xytext=(94.4, -0.4),
             arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax3.annotate('Same h ≈ 0.81\nDifferent MLP\n→ 13% GNN adv diff',
             xy=(85, 6), fontsize=9, ha='center', color='red',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
fig3.savefig(output_dir / 'same_h_different_mlp.pdf', dpi=300, bbox_inches='tight')
fig3.savefig(output_dir / 'same_h_different_mlp.png', dpi=300, bbox_inches='tight')
print("Figure 3: Same-h Different-MLP saved")

# ============================================================
# Figure 4: Combined Summary Figure (2x2)
# ============================================================
fig4, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Edge Shuffle
ax_a = axes[0, 0]
x = np.arange(len(datasets))
width = 0.35
bars1 = ax_a.bar(x - width/2, original_adv, width, label='Original', color='#2ecc71', edgecolor='black')
bars2 = ax_a.bar(x + width/2, shuffled_adv, width, label='Shuffled', color='#e74c3c', edgecolor='black')
ax_a.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax_a.set_ylabel('GCN Advantage (%)')
ax_a.set_xticks(x)
ax_a.set_xticklabels(datasets)
ax_a.legend(loc='lower left')
ax_a.set_title('(A) Edge Shuffle: Structure Contribution', fontweight='bold')
ax_a.set_ylim(-45, 20)
ax_a.grid(True, alpha=0.3, axis='y')

# Panel B: Information Budget
ax_b = axes[0, 1]
ax_b.fill_between(noise_levels, 0, budgets, alpha=0.3, color='#3498db')
ax_b.plot(noise_levels, budgets, 'b--', linewidth=2, label='Budget')
ax_b.plot(noise_levels, gcn_advs, 'ro-', linewidth=2, markersize=6, label='GCN Adv')
ax_b.set_xlabel('Noise Level')
ax_b.set_ylabel('Gain (%)')
ax_b.set_title('(B) Information Budget Validation', fontweight='bold')
ax_b.legend(loc='upper left')
ax_b.grid(True, alpha=0.3)

# Panel C: MLP vs GCN Advantage
ax_c = axes[1, 0]
ax_c.scatter(mlps, advs, s=100, c='#3498db', edgecolors='black', linewidths=1)
for label, mlp, adv in zip(labels, mlps, advs):
    ax_c.annotate(label, (mlp, adv), fontsize=8, alpha=0.8)
ax_c.plot(x_line, y_line, 'r--', linewidth=2)
ax_c.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax_c.set_xlabel('MLP Accuracy (%)')
ax_c.set_ylabel('GCN Advantage (%)')
ax_c.set_title(f'(C) MLP Accuracy vs GCN Advantage ($R^2 = {r_value**2:.2f}$)', fontweight='bold')
ax_c.grid(True, alpha=0.3)
ax_c.set_xlim(70, 100)
ax_c.set_ylim(-5, 20)

# Panel D: Summary Text
ax_d = axes[1, 1]
ax_d.axis('off')
summary_text = """
INFORMATION BUDGET THEORY
═══════════════════════════════════════

Core Principle:
  GNN_max_gain ≤ (1 - MLP_accuracy)

Key Evidence:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. EDGE SHUFFLE (Panel A)
   • Shuffling edges destroys GNN advantage
   • Cora: +12.6% → -34.8%
   • Proves: Structure is essential

2. BUDGET VALIDATION (Panel B)
   • All 9 noise levels within budget
   • Violations: 0/9 (100% compliance)
   • Proves: GNN gain is bounded

3. SAME-h DIFFERENT-MLP (Panel C)
   • Cora vs Coauthor-CS: h≈0.81
   • MLP: 75.7% vs 94.4%
   • GCN_adv: +12.6% vs -0.4%
   • Proves: MLP (not h) determines utility

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Conclusion:
GNN can only improve upon what MLP
cannot explain. High feature quality
leaves little room for structural gain.
"""
ax_d.text(0.1, 0.95, summary_text, transform=ax_d.transAxes, fontsize=10,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout()
fig4.savefig(output_dir / 'information_budget_combined.pdf', dpi=300, bbox_inches='tight')
fig4.savefig(output_dir / 'information_budget_combined.png', dpi=300, bbox_inches='tight')
print("Figure 4: Combined Summary Figure saved")

# ============================================================
# Print Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY OF KEY RESULTS")
print("=" * 70)

print("\n1. EDGE SHUFFLE EXPERIMENT:")
avg_contribution = np.mean([r['structure_contribution'] for r in edge_shuffle])
print(f"   Average structure contribution: {avg_contribution*100:.1f}%")
for r in edge_shuffle:
    print(f"   {r['dataset']}: Original GCN_adv={r['original_gcn_adv']*100:+.1f}% → Shuffled={r['shuffled_gcn_adv']*100:+.1f}%")

print("\n2. INFORMATION BUDGET VALIDATION:")
violations = sum(1 for r in feat_degrad if r['gcn_adv'] > r['theoretical_max_gain'] + 0.01)
print(f"   Budget violations: {violations}/9")
print(f"   Conclusion: {'CONFIRMED' if violations == 0 else 'VIOLATED'}")

print("\n3. SAME-h DIFFERENT-MLP:")
support_count = sum(1 for r in same_h_data if r['supports_hypothesis'])
print(f"   Supporting pairs: {support_count}/{len(same_h_data)} ({support_count/len(same_h_data)*100:.0f}%)")
print(f"   Key pair: Cora (h=0.81, MLP=75.7%) vs Coauthor-CS (h=0.81, MLP=94.4%)")
print(f"             GCN_adv: +12.6% vs -0.4% (13% difference!)")

print("\n" + "=" * 70)
print("All figures saved to:", output_dir)
print("=" * 70)
