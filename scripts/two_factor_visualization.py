"""
Two-Factor Framework Visualization
====================================

生成论文所需的可视化图表：
1. 四象限散点图 (FS vs h, 颜色表示GCN-MLP)
2. 回归拟合图
3. 决策边界图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import json

# 设置中文字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False

# 数据
datasets = [
    {'name': 'Roman-empire', 'mlp': 0.656, 'h': 0.047, 'gcn_mlp': -0.186},
    {'name': 'Texas', 'mlp': 0.792, 'h': 0.087, 'gcn_mlp': -0.246},
    {'name': 'Wisconsin', 'mlp': 0.839, 'h': 0.192, 'gcn_mlp': -0.314},
    {'name': 'Cornell', 'mlp': 0.730, 'h': 0.128, 'gcn_mlp': -0.254},
    {'name': 'Squirrel', 'mlp': 0.342, 'h': 0.222, 'gcn_mlp': 0.146},
    {'name': 'Chameleon', 'mlp': 0.505, 'h': 0.231, 'gcn_mlp': 0.137},
    {'name': 'Cora', 'mlp': 0.746, 'h': 0.810, 'gcn_mlp': 0.136},
    {'name': 'CiteSeer', 'mlp': 0.738, 'h': 0.736, 'gcn_mlp': 0.038},
    {'name': 'PubMed', 'mlp': 0.881, 'h': 0.802, 'gcn_mlp': -0.002},
    {'name': 'Computers', 'mlp': 0.825, 'h': 0.777, 'gcn_mlp': 0.073},
    {'name': 'Photo', 'mlp': 0.920, 'h': 0.827, 'gcn_mlp': 0.019},
]

# 提取数据
names = [d['name'] for d in datasets]
mlp_accs = np.array([d['mlp'] for d in datasets])
hs = np.array([d['h'] for d in datasets])
gcn_mlps = np.array([d['gcn_mlp'] for d in datasets])

# ========== Figure 1: Four-Quadrant Scatter Plot ==========
fig, ax = plt.subplots(figsize=(10, 8))

# 阈值线
fs_thresh = 0.65
h_thresh = 0.5

# 绘制象限背景
# Q1: High FS, High h - light blue
ax.add_patch(Rectangle((h_thresh, fs_thresh), 1-h_thresh, 1-fs_thresh,
                        facecolor='lightblue', alpha=0.3, label='Q1: GCN may help'))
# Q2: High FS, Low h - light red
ax.add_patch(Rectangle((0, fs_thresh), h_thresh, 1-fs_thresh,
                        facecolor='lightcoral', alpha=0.3, label='Q2: MLP wins'))
# Q3: Low FS, High h - light green
ax.add_patch(Rectangle((h_thresh, 0), 1-h_thresh, fs_thresh,
                        facecolor='lightgreen', alpha=0.3, label='Q3: GCN wins'))
# Q4: Low FS, Low h - light yellow
ax.add_patch(Rectangle((0, 0), h_thresh, fs_thresh,
                        facecolor='lightyellow', alpha=0.3, label='Q4: Uncertain'))

# 创建自定义colormap
colors = ['red', 'white', 'blue']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('gcn_mlp', colors, N=n_bins)

# 归一化GCN-MLP值到[-0.35, 0.35]
vmin, vmax = -0.35, 0.35
norm_gcn_mlps = np.clip(gcn_mlps, vmin, vmax)

# 散点图
scatter = ax.scatter(hs, mlp_accs, c=gcn_mlps, cmap=cmap,
                     s=200, edgecolors='black', linewidths=1.5,
                     vmin=vmin, vmax=vmax, zorder=5)

# 标注数据集名称
for i, name in enumerate(names):
    offset_x = 0.02
    offset_y = 0.015
    # 特殊调整重叠标签
    if name == 'CiteSeer':
        offset_y = -0.03
    elif name == 'Computers':
        offset_y = 0.025
    elif name == 'Photo':
        offset_x = -0.08
    elif name == 'PubMed':
        offset_y = -0.025

    ax.annotate(name, (hs[i] + offset_x, mlp_accs[i] + offset_y),
                fontsize=9, fontweight='bold')

# 阈值线
ax.axhline(y=fs_thresh, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax.axvline(x=h_thresh, color='gray', linestyle='--', linewidth=2, alpha=0.7)

# 象限标签
ax.text(0.25, 0.85, 'Q2: MLP Wins\n(High FS, Low h)',
        ha='center', va='center', fontsize=11, fontweight='bold', color='darkred')
ax.text(0.75, 0.85, 'Q1: GCN May Help\n(High FS, High h)',
        ha='center', va='center', fontsize=11, fontweight='bold', color='darkblue')
ax.text(0.25, 0.35, 'Q4: Uncertain\n(Low FS, Low h)',
        ha='center', va='center', fontsize=11, fontweight='bold', color='darkorange')
ax.text(0.75, 0.35, 'Q3: GCN Wins\n(Low FS, High h)',
        ha='center', va='center', fontsize=11, fontweight='bold', color='darkgreen')

# 颜色条
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('GCN - MLP Accuracy', fontsize=12)

# 标签
ax.set_xlabel('Edge Homophily (h)', fontsize=14)
ax.set_ylabel('Feature Sufficiency (MLP Accuracy)', fontsize=14)
ax.set_title('Two-Factor Framework: Feature Sufficiency vs Structure Quality', fontsize=14, fontweight='bold')

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.25, 1.0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('two_factor_quadrant.png', dpi=300, bbox_inches='tight')
print("Saved: two_factor_quadrant.png")
plt.close()

# ========== Figure 2: Regression Surface ==========
fig, ax = plt.subplots(figsize=(10, 8))

# 回归系数 (from V3 analysis)
b0, b1, b2, b3 = 0.4799, -1.0337, 0.6194, -0.4722

# 创建网格
h_grid = np.linspace(0, 1, 50)
fs_grid = np.linspace(0.3, 1, 50)
H, FS = np.meshgrid(h_grid, fs_grid)

# 预测GCN-MLP
GCN_MLP_pred = b0 + b1 * FS + b2 * H + b3 * (1 - FS) * H

# 等高线图
levels = np.linspace(-0.4, 0.3, 15)
contour = ax.contourf(H, FS, GCN_MLP_pred, levels=levels, cmap=cmap, alpha=0.8)
contour_lines = ax.contour(H, FS, GCN_MLP_pred, levels=[0], colors='black', linewidths=2)
ax.clabel(contour_lines, inline=True, fontsize=10, fmt='GCN=MLP')

# 散点图叠加
scatter = ax.scatter(hs, mlp_accs, c=gcn_mlps, cmap=cmap,
                     s=150, edgecolors='black', linewidths=2,
                     vmin=-0.35, vmax=0.35, zorder=5)

# 标注
for i, name in enumerate(names):
    ax.annotate(name, (hs[i] + 0.02, mlp_accs[i] + 0.01), fontsize=8)

# 颜色条
cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
cbar.set_label('Predicted GCN - MLP', fontsize=12)

ax.set_xlabel('Edge Homophily (h)', fontsize=14)
ax.set_ylabel('Feature Sufficiency (MLP Accuracy)', fontsize=14)
ax.set_title('Regression Model: GCN-MLP = f(FS, h)\n$R^2$ = 0.936', fontsize=14, fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0.3, 1.0)

plt.tight_layout()
plt.savefig('two_factor_regression.png', dpi=300, bbox_inches='tight')
print("Saved: two_factor_regression.png")
plt.close()

# ========== Figure 3: Roman-empire Case Study ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Spectral comparison (simplified representation)
ax1 = axes[0]
datasets_spectral = ['Roman-empire', 'Texas', 'Squirrel']
low_freq = [0.967, 0.606, 0.45]  # Approximate values
high_freq = [0.033, 0.394, 0.55]

x = np.arange(len(datasets_spectral))
width = 0.35

bars1 = ax1.bar(x - width/2, low_freq, width, label='Low Frequency', color='steelblue')
bars2 = ax1.bar(x + width/2, high_freq, width, label='High Frequency', color='coral')

ax1.set_xlabel('Dataset', fontsize=12)
ax1.set_ylabel('Energy Fraction', fontsize=12)
ax1.set_title('Spectral Energy Distribution', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets_spectral)
ax1.legend()
ax1.set_ylim(0, 1.1)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

# Right: Two-factor explanation
ax2 = axes[1]

# 只显示3个关键数据集
key_datasets = [
    {'name': 'Roman-empire', 'mlp': 0.656, 'h': 0.047, 'gcn_mlp': -0.186, 'color': 'red'},
    {'name': 'Texas', 'mlp': 0.792, 'h': 0.087, 'gcn_mlp': -0.246, 'color': 'orange'},
    {'name': 'Squirrel', 'mlp': 0.342, 'h': 0.222, 'gcn_mlp': 0.146, 'color': 'blue'},
]

for d in key_datasets:
    ax2.scatter(d['h'], d['mlp'], s=300, c=d['color'], edgecolors='black', linewidths=2, zorder=5)
    ax2.annotate(f"{d['name']}\nGCN-MLP={d['gcn_mlp']:+.3f}",
                (d['h'] + 0.02, d['mlp']), fontsize=10, fontweight='bold')

# 阈值线
ax2.axhline(y=0.65, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='FS threshold')
ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='h threshold')

# 标注象限
ax2.text(0.25, 0.8, 'Q2: MLP Wins', ha='center', fontsize=12, fontweight='bold', color='darkred')
ax2.text(0.15, 0.45, 'Q4: Uncertain', ha='center', fontsize=12, fontweight='bold', color='darkorange')

ax2.set_xlabel('Edge Homophily (h)', fontsize=12)
ax2.set_ylabel('Feature Sufficiency (MLP Accuracy)', fontsize=12)
ax2.set_title('Two-Factor Explanation', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 0.5)
ax2.set_ylim(0.25, 0.95)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roman_empire_case_study.png', dpi=300, bbox_inches='tight')
print("Saved: roman_empire_case_study.png")
plt.close()

# ========== Figure 4: GCN-MLP vs Two Factors ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: GCN-MLP vs Feature Sufficiency
ax1 = axes[0]
colors_points = ['red' if g < -0.01 else ('blue' if g > 0.01 else 'gray') for g in gcn_mlps]
ax1.scatter(mlp_accs, gcn_mlps, c=colors_points, s=150, edgecolors='black', linewidths=1.5)

for i, name in enumerate(names):
    ax1.annotate(name, (mlp_accs[i] + 0.01, gcn_mlps[i] + 0.01), fontsize=8)

ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
ax1.axvline(x=0.65, color='green', linestyle='--', linewidth=1.5, label='FS threshold')

# 趋势线
z = np.polyfit(mlp_accs, gcn_mlps, 1)
p = np.poly1d(z)
x_line = np.linspace(0.3, 1, 100)
ax1.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Trend (r=-0.38)')

ax1.set_xlabel('Feature Sufficiency (MLP Accuracy)', fontsize=12)
ax1.set_ylabel('GCN - MLP Accuracy', fontsize=12)
ax1.set_title('Factor 1: Feature Sufficiency', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: GCN-MLP vs Homophily
ax2 = axes[1]
ax2.scatter(hs, gcn_mlps, c=colors_points, s=150, edgecolors='black', linewidths=1.5)

for i, name in enumerate(names):
    ax2.annotate(name, (hs[i] + 0.01, gcn_mlps[i] + 0.01), fontsize=8)

ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=1.5, label='h threshold')

# 趋势线
z = np.polyfit(hs, gcn_mlps, 1)
p = np.poly1d(z)
x_line = np.linspace(0, 1, 100)
ax2.plot(x_line, p(x_line), 'b--', alpha=0.7, label=f'Trend (r=0.62)')

ax2.set_xlabel('Edge Homophily (h)', fontsize=12)
ax2.set_ylabel('GCN - MLP Accuracy', fontsize=12)
ax2.set_title('Factor 2: Structure Quality', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('two_factors_separate.png', dpi=300, bbox_inches='tight')
print("Saved: two_factors_separate.png")
plt.close()

print("\n" + "=" * 60)
print("All visualizations generated successfully!")
print("=" * 60)
print("\nGenerated files:")
print("  1. two_factor_quadrant.png - Four-quadrant scatter plot")
print("  2. two_factor_regression.png - Regression surface")
print("  3. roman_empire_case_study.png - Case study comparison")
print("  4. two_factors_separate.png - Two factors separately")
