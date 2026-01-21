"""
Validate Direction_Residual as Predictor
=========================================

使用新收集的GCN性能数据验证Direction_Residual的预测力
"""

import numpy as np
import json
from scipy import stats
import matplotlib.pyplot as plt

# 加载Direction数据
with open('pattern_direction_enhanced_results.json', 'r') as f:
    direction_data = json.load(f)['results']

# 加载GCN性能数据
with open('comprehensive_gcn_results.json', 'r') as f:
    gcn_data = json.load(f)['results']

# 创建数据字典
direction_dict = {r['dataset']: r for r in direction_data}
gcn_dict = {r['dataset']: r for r in gcn_data}

# 计算Direction_Residual的线性模型参数
homophilies = np.array([r['edge_homophily'] for r in direction_data])
directions = np.array([r['direction_corrected'] for r in direction_data])
slope, intercept, _, _, _ = stats.linregress(homophilies, directions)

print("=" * 80)
print("DIRECTION_RESIDUAL VALIDATION")
print("=" * 80)
print(f"\nLinear model: Direction = {slope:.3f} * h + {intercept:.3f}")

# 匹配数据集
matched_data = []
for ds_name in gcn_dict:
    if ds_name in direction_dict:
        d = direction_dict[ds_name]
        g = gcn_dict[ds_name]

        h = d['edge_homophily']
        direction = d['direction_corrected']
        predicted = slope * h + intercept
        residual = direction - predicted

        matched_data.append({
            'dataset': ds_name,
            'homophily': h,
            'direction': direction,
            'direction_residual': residual,
            'gcn_mlp': g['gcn_mlp'],
            'gat_mlp': g['gat_mlp'],
            'sage_mlp': g['sage_mlp'],
            'gcn_acc': g['gcn_mean'],
            'mlp_acc': g['mlp_mean'],
        })

print(f"\nMatched datasets: {len(matched_data)}")

# 分组：异配 vs 同质
hetero_data = [d for d in matched_data if d['homophily'] < 0.4]
homo_data = [d for d in matched_data if d['homophily'] >= 0.4]

print(f"Heterophilic (h<0.4): {len(hetero_data)}")
print(f"Homophilic (h>=0.4): {len(homo_data)}")

# 1. 全局相关性分析
print("\n" + "=" * 80)
print("1. GLOBAL CORRELATION ANALYSIS (All datasets)")
print("=" * 80)

all_h = np.array([d['homophily'] for d in matched_data])
all_dir = np.array([d['direction'] for d in matched_data])
all_res = np.array([d['direction_residual'] for d in matched_data])
all_gcn = np.array([d['gcn_mlp'] for d in matched_data])
all_gat = np.array([d['gat_mlp'] for d in matched_data])
all_sage = np.array([d['sage_mlp'] for d in matched_data])

corr_h_gcn, p_h_gcn = stats.pearsonr(all_h, all_gcn)
corr_dir_gcn, p_dir_gcn = stats.pearsonr(all_dir, all_gcn)
corr_res_gcn, p_res_gcn = stats.pearsonr(all_res, all_gcn)

print(f"\nPredicting GCN-MLP advantage:")
print(f"  Homophily:          r={corr_h_gcn:.3f}, p={p_h_gcn:.4f}")
print(f"  Direction:          r={corr_dir_gcn:.3f}, p={p_dir_gcn:.4f}")
print(f"  Direction_Residual: r={corr_res_gcn:.3f}, p={p_res_gcn:.4f}")

# 2. 异配数据集分析（关键！）
print("\n" + "=" * 80)
print("2. HETEROPHILIC DATASETS (h < 0.4) - KEY ANALYSIS")
print("=" * 80)

if len(hetero_data) >= 3:
    h_h = np.array([d['homophily'] for d in hetero_data])
    h_res = np.array([d['direction_residual'] for d in hetero_data])
    h_gcn = np.array([d['gcn_mlp'] for d in hetero_data])

    corr_h_gcn_h, p_h_gcn_h = stats.pearsonr(h_h, h_gcn)
    corr_res_gcn_h, p_res_gcn_h = stats.pearsonr(h_res, h_gcn)

    print(f"\nPredicting GCN-MLP on heterophilic graphs (n={len(hetero_data)}):")
    print(f"  Homophily:          r={corr_h_gcn_h:.3f}, p={p_h_gcn_h:.4f}")
    print(f"  Direction_Residual: r={corr_res_gcn_h:.3f}, p={p_res_gcn_h:.4f}")

    if abs(corr_res_gcn_h) > abs(corr_h_gcn_h):
        print(f"\n  ==> Direction_Residual is BETTER predictor!")
    else:
        print(f"\n  ==> Homophily is better predictor")

    print(f"\n  Dataset details:")
    print(f"  {'Dataset':>12} {'h':>6} {'Residual':>10} {'GCN-MLP':>10}")
    print(f"  {'-'*45}")
    for d in sorted(hetero_data, key=lambda x: x['direction_residual']):
        print(f"  {d['dataset']:>12} {d['homophily']:>6.3f} "
              f"{d['direction_residual']:>+10.3f} {d['gcn_mlp']:>+10.3f}")

# 3. 跨模型验证
print("\n" + "=" * 80)
print("3. CROSS-MODEL VALIDATION")
print("=" * 80)

if len(hetero_data) >= 3:
    h_gat = np.array([d['gat_mlp'] for d in hetero_data])
    h_sage = np.array([d['sage_mlp'] for d in hetero_data])

    corr_res_gat, p_res_gat = stats.pearsonr(h_res, h_gat)
    corr_res_sage, p_res_sage = stats.pearsonr(h_res, h_sage)

    print(f"\nDirection_Residual predicting GNN-MLP (heterophilic):")
    print(f"  vs GCN-MLP:  r={corr_res_gcn_h:.3f}, p={p_res_gcn_h:.4f}")
    print(f"  vs GAT-MLP:  r={corr_res_gat:.3f}, p={p_res_gat:.4f}")
    print(f"  vs SAGE-MLP: r={corr_res_sage:.3f}, p={p_res_sage:.4f}")

# 4. 可视化
print("\n" + "=" * 80)
print("4. CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Homophily vs GCN-MLP (all)
ax1 = axes[0, 0]
colors = ['red' if d['homophily'] < 0.4 else 'green' for d in matched_data]
ax1.scatter(all_h, all_gcn, c=colors, s=100, alpha=0.7)
for d in matched_data:
    ax1.annotate(d['dataset'], (d['homophily'], d['gcn_mlp']), fontsize=8)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Edge Homophily', fontsize=12)
ax1.set_ylabel('GCN - MLP', fontsize=12)
ax1.set_title(f'Homophily vs GCN Performance\nr={corr_h_gcn:.3f}', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: Direction_Residual vs GCN-MLP (all)
ax2 = axes[0, 1]
ax2.scatter(all_res, all_gcn, c=colors, s=100, alpha=0.7)
for d in matched_data:
    ax2.annotate(d['dataset'], (d['direction_residual'], d['gcn_mlp']), fontsize=8)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Direction Residual', fontsize=12)
ax2.set_ylabel('GCN - MLP', fontsize=12)
ax2.set_title(f'Direction Residual vs GCN Performance\nr={corr_res_gcn:.3f}', fontsize=12)
ax2.grid(True, alpha=0.3)

# Plot 3: Heterophilic only - comparison
ax3 = axes[1, 0]
if len(hetero_data) >= 3:
    ax3.scatter(h_h, h_gcn, c='blue', s=150, alpha=0.7, label=f'h (r={corr_h_gcn_h:.3f})')
    ax3.scatter(h_res, h_gcn, c='orange', s=150, alpha=0.7, marker='^',
                label=f'Residual (r={corr_res_gcn_h:.3f})')
    for d in hetero_data:
        ax3.annotate(d['dataset'], (d['homophily'], d['gcn_mlp']), fontsize=9, color='blue')
        ax3.annotate(d['dataset'], (d['direction_residual'], d['gcn_mlp']), fontsize=9, color='orange')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Predictor Value', fontsize=12)
    ax3.set_ylabel('GCN - MLP', fontsize=12)
    ax3.set_title('Heterophilic Datasets: h vs Residual', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# Plot 4: Cross-model validation
ax4 = axes[1, 1]
if len(hetero_data) >= 3:
    models = ['GCN', 'GAT', 'SAGE']
    correlations = [corr_res_gcn_h, corr_res_gat, corr_res_sage]
    p_values = [p_res_gcn_h, p_res_gat, p_res_sage]

    bars = ax4.bar(models, correlations, color=['blue', 'green', 'orange'], alpha=0.7)
    ax4.axhline(y=0, color='gray', linestyle='-')
    ax4.set_ylabel('Correlation with Direction_Residual', fontsize=12)
    ax4.set_title('Cross-Model Validation\n(Heterophilic Datasets)', fontsize=12)

    # Add p-values
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        sig = '*' if p < 0.05 else ''
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'r={correlations[i]:.2f}{sig}', ha='center', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('direction_residual_validation.png', dpi=300, bbox_inches='tight')
print("Saved: direction_residual_validation.png")

# 5. 关键结果总结
print("\n" + "=" * 80)
print("5. KEY FINDINGS SUMMARY")
print("=" * 80)

print(f"""
GLOBAL ANALYSIS (n={len(matched_data)}):
- Homophily predicts GCN-MLP: r={corr_h_gcn:.3f}
- Direction_Residual predicts GCN-MLP: r={corr_res_gcn:.3f}
""")

if len(hetero_data) >= 3:
    print(f"""
HETEROPHILIC ANALYSIS (n={len(hetero_data)}, h<0.4):
- Homophily predicts GCN-MLP: r={corr_h_gcn_h:.3f}, p={p_h_gcn_h:.4f}
- Direction_Residual predicts GCN-MLP: r={corr_res_gcn_h:.3f}, p={p_res_gcn_h:.4f}

WINNER: {'Direction_Residual' if abs(corr_res_gcn_h) > abs(corr_h_gcn_h) else 'Homophily'}

CROSS-MODEL (Direction_Residual):
- vs GCN: r={corr_res_gcn_h:.3f}
- vs GAT: r={corr_res_gat:.3f}
- vs SAGE: r={corr_res_sage:.3f}
""")

# 6. 保存结果
output = {
    'global': {
        'n_datasets': len(matched_data),
        'corr_h_gcn': corr_h_gcn,
        'corr_dir_gcn': corr_dir_gcn,
        'corr_res_gcn': corr_res_gcn,
    },
    'heterophilic': {
        'n_datasets': len(hetero_data),
        'corr_h_gcn': corr_h_gcn_h if len(hetero_data) >= 3 else None,
        'corr_res_gcn': corr_res_gcn_h if len(hetero_data) >= 3 else None,
        'p_res_gcn': p_res_gcn_h if len(hetero_data) >= 3 else None,
    },
    'cross_model': {
        'corr_res_gcn': corr_res_gcn_h if len(hetero_data) >= 3 else None,
        'corr_res_gat': corr_res_gat if len(hetero_data) >= 3 else None,
        'corr_res_sage': corr_res_sage if len(hetero_data) >= 3 else None,
    },
    'matched_data': matched_data
}

with open('direction_residual_validation.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults saved to: direction_residual_validation.json")
