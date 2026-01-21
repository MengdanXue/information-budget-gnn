"""
Direction vs Homophily Deep Analysis
=====================================

关键问题：Direction是否只是Homophily的变体？

实验设计：
1. 检查Direction是否在Homophily之外提供额外预测力
2. 寻找Direction和Homophily分离的案例
3. 控制Homophily后Direction的残差价值
"""

import numpy as np
import json
from scipy import stats
import matplotlib.pyplot as plt

# 加载增强版结果
with open('pattern_direction_enhanced_results.json', 'r') as f:
    data = json.load(f)

results = data['results']

# 提取数据
datasets = [r['dataset'] for r in results]
homophilies = np.array([r['edge_homophily'] for r in results])
directions_u = np.array([r['direction_uniform'] for r in results])
directions_c = np.array([r['direction_corrected'] for r in results])
match_rates = np.array([r['global_match_rate'] for r in results])

print("=" * 80)
print("DIRECTION vs HOMOPHILY: DEEP ANALYSIS")
print("=" * 80)

# 1. 基本相关性
print("\n1. BASIC CORRELATIONS")
print("-" * 40)

corr_du_h, p_du_h = stats.pearsonr(directions_u, homophilies)
corr_dc_h, p_dc_h = stats.pearsonr(directions_c, homophilies)
corr_mr_h, p_mr_h = stats.pearsonr(match_rates, homophilies)

print(f"Direction(uniform) vs Homophily: r={corr_du_h:.3f}, p={p_du_h:.6f}")
print(f"Direction(corrected) vs Homophily: r={corr_dc_h:.3f}, p={p_dc_h:.6f}")
print(f"Match Rate vs Homophily: r={corr_mr_h:.3f}, p={p_mr_h:.6f}")

# 2. 残差分析：Direction在控制Homophily后的额外信息
print("\n2. RESIDUAL ANALYSIS")
print("-" * 40)

# 用Homophily预测Direction
slope, intercept, r_value, p_value, std_err = stats.linregress(homophilies, directions_c)
predicted_direction = slope * homophilies + intercept
residuals = directions_c - predicted_direction

print(f"Linear model: Direction = {slope:.3f} * h + {intercept:.3f}")
print(f"R^2 = {r_value**2:.3f}")
print(f"\nResiduals (Direction independent of Homophily):")

for i, ds in enumerate(datasets):
    print(f"  {ds:>12}: h={homophilies[i]:.3f}, Dir={directions_c[i]:+.3f}, "
          f"Pred={predicted_direction[i]:+.3f}, Residual={residuals[i]:+.3f}")

# 3. 寻找异常点（Direction和Homophily不一致的案例）
print("\n3. OUTLIER ANALYSIS (Direction-Homophily Mismatch)")
print("-" * 40)

# 标准化
h_z = (homophilies - homophilies.mean()) / homophilies.std()
d_z = (directions_c - directions_c.mean()) / directions_c.std()
mismatch = np.abs(h_z - d_z)

print(f"{'Dataset':>12} {'h':>8} {'Dir':>8} {'h(z)':>8} {'Dir(z)':>8} {'Mismatch':>10}")
print("-" * 60)

for i in np.argsort(mismatch)[::-1]:
    print(f"{datasets[i]:>12} {homophilies[i]:>8.3f} {directions_c[i]:>+8.3f} "
          f"{h_z[i]:>+8.2f} {d_z[i]:>+8.2f} {mismatch[i]:>10.3f}")

# 4. 关键案例分析
print("\n4. KEY CASES ANALYSIS")
print("-" * 40)

# Squirrel vs Texas: 相似的h，不同的Direction？
print("\nSquirrel vs Texas (similar h, different outcome?):")
squirrel_idx = datasets.index('Squirrel')
texas_idx = datasets.index('Texas')

print(f"  Texas:    h={homophilies[texas_idx]:.3f}, Dir(c)={directions_c[texas_idx]:+.3f}")
print(f"  Squirrel: h={homophilies[squirrel_idx]:.3f}, Dir(c)={directions_c[squirrel_idx]:+.3f}")
print(f"  Δh = {homophilies[squirrel_idx] - homophilies[texas_idx]:.3f}")
print(f"  ΔDir = {directions_c[squirrel_idx] - directions_c[texas_idx]:.3f}")

if abs(homophilies[squirrel_idx] - homophilies[texas_idx]) < 0.15:
    if abs(directions_c[squirrel_idx] - directions_c[texas_idx]) > 0.3:
        print("  ==> EVIDENCE: Direction provides info beyond Homophily!")
    else:
        print("  ==> NO EVIDENCE: Direction tracks Homophily closely")

# 5. 理论分析：Direction和Homophily的数学关系
print("\n5. THEORETICAL RELATIONSHIP")
print("-" * 40)

print("""
Homophily (h) = P(same_label | edge)
             = sum_c P(label=c) * P(neighbor_label=c | label=c)

Direction = P(dominant_neighbor_class = node_label) - baseline

For high-homophily graphs:
  - Most neighbors have same label
  - Dominant neighbor class = node label (high probability)
  - Direction ≈ h - baseline ∝ h

This explains r=0.908 correlation!

Key insight: Direction IS essentially homophily measured differently.
The "unified correlation" r=0.864 was illusory - it's just because
Direction proxies for Homophily.
""")

# 6. 结论
print("\n6. CONCLUSION")
print("-" * 40)

print(f"""
CRITICAL FINDING:
- Direction vs Homophily correlation: r = {corr_dc_h:.3f}
- This means Direction explains ~{corr_dc_h**2*100:.1f}% of Homophily's variance
- Direction is NOT a fundamentally new metric

IMPLICATIONS FOR PAPER:
1. Cannot claim Direction is independent of Homophily
2. The "unified" correlation r=0.864 was misleading
3. Need to find what Direction adds BEYOND Homophily

POTENTIAL SALVAGE:
1. Focus on cases where Direction ≠ expected from Homophily
2. Analyze residuals - what causes Direction to deviate from Homophily?
3. Reframe: "When does local structure (Direction) differ from global structure (Homophily)?"
""")

# 7. 可视化
print("\n7. CREATING VISUALIZATION")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Direction vs Homophily
ax1 = axes[0]
colors = ['red' if h < 0.3 else 'green' for h in homophilies]
ax1.scatter(homophilies, directions_c, c=colors, s=100, alpha=0.7)
ax1.plot([0, 1], [slope * 0 + intercept, slope * 1 + intercept], 'k--', alpha=0.5)
for i, ds in enumerate(datasets):
    ax1.annotate(ds, (homophilies[i], directions_c[i]), fontsize=8, alpha=0.7)
ax1.set_xlabel('Edge Homophily (h)')
ax1.set_ylabel('Direction (corrected)')
ax1.set_title(f'Direction vs Homophily\nr={corr_dc_h:.3f}')
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = axes[1]
ax2.bar(range(len(datasets)), residuals, color=['red' if r < 0 else 'green' for r in residuals])
ax2.set_xticks(range(len(datasets)))
ax2.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
ax2.axhline(y=0, color='gray', linestyle='-')
ax2.set_ylabel('Residual (Direction - Predicted from h)')
ax2.set_title('Residuals: Direction\'s Unique Information')
ax2.grid(True, alpha=0.3)

# Plot 3: Standardized comparison
ax3 = axes[2]
x = np.arange(len(datasets))
width = 0.35
ax3.bar(x - width/2, h_z, width, label='Homophily (z-score)', color='blue', alpha=0.7)
ax3.bar(x + width/2, d_z, width, label='Direction (z-score)', color='orange', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
ax3.axhline(y=0, color='gray', linestyle='-')
ax3.legend()
ax3.set_title('Standardized Comparison')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('direction_vs_homophily_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: direction_vs_homophily_analysis.png")

# 8. 保存分析结果
analysis_output = {
    'correlation_direction_homophily': corr_dc_h,
    'r_squared': corr_dc_h ** 2,
    'linear_model': {
        'slope': slope,
        'intercept': intercept
    },
    'residuals': {ds: float(r) for ds, r in zip(datasets, residuals)},
    'conclusion': 'Direction is highly correlated with Homophily (r=0.908). '
                  'It does not provide substantial independent information.'
}

with open('direction_homophily_analysis.json', 'w') as f:
    json.dump(analysis_output, f, indent=2)

print("\nAnalysis saved to: direction_homophily_analysis.json")
