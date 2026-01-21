"""
Generate degree-stratified NAA performance figure from IEEE-CIS data.
Shows NAA improvement across different node degree ranges with δ_agg labels.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Data from paper Table (IEEE-CIS degree stratification)
degree_ranges = ['1-10', '11-50', '51-100', '101-200', '200+']
delta_agg = [0.74, 4.82, 11.53, 22.87, 33.59]
naa_improvement = [0.8, 0.3, -2.1, -5.4, -8.2]

# Create figure with larger size for better readability
fig, ax = plt.subplots(figsize=(10, 6))

# Create x positions
x_pos = np.arange(len(degree_ranges))
bar_width = 0.6

# Create color array: blue for positive, red for negative
colors = ['#2E86DE' if val >= 0 else '#EE5A6F' for val in naa_improvement]

# Create bars
bars = ax.bar(x_pos, naa_improvement, bar_width, color=colors,
              edgecolor='black', linewidth=1.2, alpha=0.8)

# Add δ_agg values as labels above/below bars
for i, (bar, delta, improvement) in enumerate(zip(bars, delta_agg, naa_improvement)):
    height = bar.get_height()
    # Position label above bar for positive, below for negative
    y_offset = 0.3 if height >= 0 else -0.8
    label_y = height + y_offset

    # Add δ_agg label
    ax.text(bar.get_x() + bar.get_width()/2., label_y,
            f'δ_agg={delta:.2f}',
            ha='center', va='bottom' if height >= 0 else 'top',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='gray', alpha=0.8))

    # Add improvement percentage on the bar
    text_y = height / 2
    ax.text(bar.get_x() + bar.get_width()/2., text_y,
            f'{improvement:+.1f}%',
            ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

# Customize axes
ax.set_xlabel('Node Degree Range', fontsize=13, fontweight='bold')
ax.set_ylabel('NAA Relative Improvement (%)', fontsize=13, fontweight='bold')
ax.set_title('NAA Performance by Node Degree (IEEE-CIS)',
             fontsize=15, fontweight='bold', pad=20)

# Set x-axis
ax.set_xticks(x_pos)
ax.set_xticklabels(degree_ranges, fontsize=11)

# Set y-axis
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
ax.set_ylim(min(naa_improvement) - 2, max(naa_improvement) + 2)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86DE', edgecolor='black', label='Positive Improvement'),
    Patch(facecolor='#EE5A6F', edgecolor='black', label='Negative Improvement')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Improve layout
plt.tight_layout()

# Create figures directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(output_dir, exist_ok=True)

# Save figure in both formats
pdf_path = os.path.join(output_dir, 'degree_stratified_naa_performance.pdf')
png_path = os.path.join(output_dir, 'degree_stratified_naa_performance.png')

plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

print(f"Figure saved to:")
print(f"  - {os.path.abspath(pdf_path)}")
print(f"  - {os.path.abspath(png_path)}")

# Display summary statistics
print("\n" + "="*60)
print("Degree Stratification Summary (IEEE-CIS)")
print("="*60)
for degree, delta, improvement in zip(degree_ranges, delta_agg, naa_improvement):
    status = "+" if improvement > 0 else "-"
    print(f"{status} Degree {degree:>8} | delta_agg={delta:6.2f} | NAA Improvement={improvement:+5.1f}%")
print("="*60)
print(f"\nKey Finding: NAA performs better on low-degree nodes")
print(f"  - Low degree (1-10):   delta_agg={delta_agg[0]:.2f},  improvement={naa_improvement[0]:+.1f}%")
print(f"  - High degree (200+):  delta_agg={delta_agg[-1]:.2f}, improvement={naa_improvement[-1]:+.1f}%")
print(f"  - Trend: As delta_agg increases, NAA advantage decreases")

plt.show()
