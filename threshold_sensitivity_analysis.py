"""
Threshold Sensitivity Analysis for Trust Regions Paper
Creates threshold sweep curve + bootstrap confidence intervals
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# Dataset data: (name, h, 2-hop recovery ratio R, actual winner)
# winner encoding: 0 = MLP, 1 = H2GCN, 2 = GNN/Tie
datasets = [
    # High homophily (h > 0.5)
    ('Cora', 0.81, 0.90, 2),
    ('CiteSeer', 0.74, 1.01, 2),
    ('PubMed', 0.80, 0.95, 2),
    ('Amazon-Comp.', 0.78, 0.92, 2),
    ('Amazon-Photo', 0.83, 0.94, 2),
    ('Coauthor-CS', 0.81, 0.93, 2),
    ('Coauthor-Phys.', 0.93, 0.96, 2),
    ('DBLP', 0.83, 0.91, 2),
    ('Questions', 0.84, 0.95, 2),
    ('ogbn-arxiv', 0.655, 1.05, 0),  # MLP wins
    ('Tolokers', 0.595, 1.02, 2),    # Tie -> GNN category
    ('Minesweeper', 0.683, 1.08, 2), # Tie

    # Low homophily - WebKB (recoverable)
    ('Texas', 0.108, 5.26, 1),       # H2GCN wins
    ('Wisconsin', 0.196, 2.15, 1),   # H2GCN wins
    ('Cornell', 0.131, 2.99, 1),     # H2GCN wins

    # Low homophily - Wikipedia (irrecoverable)
    ('Actor', 0.219, 0.96, 0),       # MLP wins
    ('Chameleon', 0.235, 0.97, 0),   # MLP wins
    ('Squirrel', 0.224, 0.88, 0),    # MLP wins
    ('Roman-empire', 0.047, 0.85, 0),# MLP wins
    ('Amazon-ratings', 0.38, 1.12, 2), # GNN wins (borderline)
]

names = [d[0] for d in datasets]
h_values = np.array([d[1] for d in datasets])
recovery_ratios = np.array([d[2] for d in datasets])
actual_winners = np.array([d[3] for d in datasets])

def evaluate_two_factor_accuracy(h_threshold, r_threshold, h_vals, r_vals, actual):
    """Evaluate accuracy of two-factor decision rule"""
    correct = 0
    for h, r, actual_w in zip(h_vals, r_vals, actual):
        if h > h_threshold:
            predicted = 2  # GNN
        elif r > r_threshold:
            predicted = 1  # H2GCN
        else:
            predicted = 0  # MLP

        # Check correctness
        if predicted == actual_w:
            correct += 1
        elif predicted == 2 and actual_w == 2:  # GNN predicted, Tie actual
            correct += 1

    return correct / len(actual)

# ============ Part 1: h-threshold sensitivity ============
h_thresholds = np.arange(0.3, 0.8, 0.02)
h_accuracies = []

for h_t in h_thresholds:
    acc = evaluate_two_factor_accuracy(h_t, 1.5, h_values, recovery_ratios, actual_winners)
    h_accuracies.append(acc)

# ============ Part 2: R-threshold sensitivity ============
r_thresholds = np.arange(1.0, 3.0, 0.1)
r_accuracies = []

for r_t in r_thresholds:
    acc = evaluate_two_factor_accuracy(0.5, r_t, h_values, recovery_ratios, actual_winners)
    r_accuracies.append(acc)

# ============ Part 3: Bootstrap confidence intervals ============
def bootstrap_accuracy(h_t, r_t, h_vals, r_vals, actual, n_bootstrap=1000):
    """Compute bootstrap CI for accuracy"""
    n = len(actual)
    bootstrap_accs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        h_boot = h_vals[indices]
        r_boot = r_vals[indices]
        actual_boot = actual[indices]

        acc = evaluate_two_factor_accuracy(h_t, r_t, h_boot, r_boot, actual_boot)
        bootstrap_accs.append(acc)

    return np.percentile(bootstrap_accs, [2.5, 50, 97.5])

# Bootstrap for current thresholds
print("Computing bootstrap confidence intervals...")
current_ci = bootstrap_accuracy(0.5, 1.5, h_values, recovery_ratios, actual_winners)
print(f"Current (h=0.5, R=1.5): Accuracy = {current_ci[1]*100:.1f}% [95% CI: {current_ci[0]*100:.1f}% - {current_ci[2]*100:.1f}%]")

# Bootstrap for alternative thresholds
alt_h_ci = bootstrap_accuracy(0.6, 1.5, h_values, recovery_ratios, actual_winners)
print(f"Alt h=0.6: Accuracy = {alt_h_ci[1]*100:.1f}% [95% CI: {alt_h_ci[0]*100:.1f}% - {alt_h_ci[2]*100:.1f}%]")

alt_r_ci = bootstrap_accuracy(0.5, 2.0, h_values, recovery_ratios, actual_winners)
print(f"Alt R=2.0: Accuracy = {alt_r_ci[1]*100:.1f}% [95% CI: {alt_r_ci[0]*100:.1f}% - {alt_r_ci[2]*100:.1f}%]")

# ============ Create Figure ============
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: h-threshold sensitivity
ax1 = axes[0]
ax1.plot(h_thresholds, h_accuracies, 'b-', linewidth=2, label='Accuracy')
ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Current (h=0.5)')
ax1.fill_between(h_thresholds,
                  [a - 0.05 for a in h_accuracies],
                  [min(a + 0.05, 1.0) for a in h_accuracies],
                  alpha=0.2, color='blue', label='±5% band')

# Mark optimal region
best_h_idx = np.argmax(h_accuracies)
ax1.scatter([h_thresholds[best_h_idx]], [h_accuracies[best_h_idx]],
            color='green', s=100, zorder=5, label=f'Best: h={h_thresholds[best_h_idx]:.2f}')

ax1.set_xlabel('Homophily Threshold (h)')
ax1.set_ylabel('Model Selection Accuracy')
ax1.set_title('(a) Sensitivity to h Threshold (R fixed at 1.5)', fontweight='bold')
ax1.set_ylim(0.6, 1.0)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)

# Add annotation for stability
stable_range = [i for i, acc in enumerate(h_accuracies) if acc >= max(h_accuracies) - 0.05]
if len(stable_range) > 2:
    h_min, h_max = h_thresholds[stable_range[0]], h_thresholds[stable_range[-1]]
    ax1.axvspan(h_min, h_max, alpha=0.15, color='green')
    ax1.text((h_min + h_max)/2, 0.65, f'Stable range\n[{h_min:.2f}, {h_max:.2f}]',
             ha='center', fontsize=9, color='green')

# Panel B: R-threshold sensitivity
ax2 = axes[1]
ax2.plot(r_thresholds, r_accuracies, 'b-', linewidth=2, label='Accuracy')
ax2.axvline(x=1.5, color='red', linestyle='--', linewidth=2, label='Current (R=1.5)')
ax2.fill_between(r_thresholds,
                  [a - 0.05 for a in r_accuracies],
                  [min(a + 0.05, 1.0) for a in r_accuracies],
                  alpha=0.2, color='blue', label='±5% band')

# Mark optimal region
best_r_idx = np.argmax(r_accuracies)
ax2.scatter([r_thresholds[best_r_idx]], [r_accuracies[best_r_idx]],
            color='green', s=100, zorder=5, label=f'Best: R={r_thresholds[best_r_idx]:.1f}')

ax2.set_xlabel('2-Hop Recovery Threshold (R)')
ax2.set_ylabel('Model Selection Accuracy')
ax2.set_title('(b) Sensitivity to R Threshold (h fixed at 0.5)', fontweight='bold')
ax2.set_ylim(0.6, 1.0)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)

# Add annotation for stability
stable_range_r = [i for i, acc in enumerate(r_accuracies) if acc >= max(r_accuracies) - 0.05]
if len(stable_range_r) > 2:
    r_min, r_max = r_thresholds[stable_range_r[0]], r_thresholds[stable_range_r[-1]]
    ax2.axvspan(r_min, r_max, alpha=0.15, color='green')
    ax2.text((r_min + r_max)/2, 0.65, f'Stable range\n[{r_min:.1f}, {r_max:.1f}]',
             ha='center', fontsize=9, color='green')

plt.tight_layout()
plt.savefig('figures/threshold_sensitivity_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/threshold_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
print("\nThreshold Sensitivity Analysis saved to figures/")

# ============ Print Summary ============
print("\n=== Threshold Sensitivity Summary ===")
print(f"h-threshold range tested: [{h_thresholds[0]:.2f}, {h_thresholds[-1]:.2f}]")
print(f"Best h-threshold: {h_thresholds[best_h_idx]:.2f} (Accuracy: {h_accuracies[best_h_idx]*100:.1f}%)")
print(f"Accuracy at h=0.5: {evaluate_two_factor_accuracy(0.5, 1.5, h_values, recovery_ratios, actual_winners)*100:.1f}%")

print(f"\nR-threshold range tested: [{r_thresholds[0]:.1f}, {r_thresholds[-1]:.1f}]")
print(f"Best R-threshold: {r_thresholds[best_r_idx]:.1f} (Accuracy: {r_accuracies[best_r_idx]*100:.1f}%)")
print(f"Accuracy at R=1.5: {evaluate_two_factor_accuracy(0.5, 1.5, h_values, recovery_ratios, actual_winners)*100:.1f}%")

print("\n=== Bootstrap 95% Confidence Intervals ===")
print(f"Current (h=0.5, R=1.5): {current_ci[1]*100:.1f}% [{current_ci[0]*100:.1f}%, {current_ci[2]*100:.1f}%]")
print(f"Alternative (h=0.6, R=1.5): {alt_h_ci[1]*100:.1f}% [{alt_h_ci[0]*100:.1f}%, {alt_h_ci[2]*100:.1f}%]")
print(f"Alternative (h=0.5, R=2.0): {alt_r_ci[1]*100:.1f}% [{alt_r_ci[0]*100:.1f}%, {alt_r_ci[2]*100:.1f}%]")

# Check if current threshold is within 5% of optimal
h_diff = abs(h_accuracies[best_h_idx] - evaluate_two_factor_accuracy(0.5, 1.5, h_values, recovery_ratios, actual_winners))
r_diff = abs(r_accuracies[best_r_idx] - evaluate_two_factor_accuracy(0.5, 1.5, h_values, recovery_ratios, actual_winners))

print(f"\n=== Robustness Check ===")
print(f"h=0.5 is within {h_diff*100:.1f}% of optimal h={h_thresholds[best_h_idx]:.2f}")
print(f"R=1.5 is within {r_diff*100:.1f}% of optimal R={r_thresholds[best_r_idx]:.1f}")

if h_diff <= 0.05 and r_diff <= 0.05:
    print("✓ Current thresholds are ROBUST (within 5% of optimal)")
else:
    print("⚠ Consider adjusting thresholds")
