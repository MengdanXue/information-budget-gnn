# 多维度GNN诊断框架 (Multi-Dimensional GNN Diagnostic Framework)

## 核心贡献

本框架扩展了原始FSD框架，从单一指标(ρ_FS)扩展到多维度诊断，解决了原框架在高度数图上预测失败的问题。

## 诊断指标

### 1. ρ_FS (Feature-Structure Alignment)
- 定义: ρ_FS = E[sim(x_i, x_j) | (i,j)∈E] - E[sim(x_i, x_j) | (i,j)∉E]
- 含义: 相邻节点特征相似度与非相邻节点的差异
- 高ρ_FS (>0.15): 特征与结构高度对齐，NAA可能有效

### 2. δ_agg (Aggregation Dilution) - 新提出
- 定义: δ_agg = E[d_i · (1 - sim(x_i, mean(x_neighbors)))]
- 含义: 邻居聚合导致的信息稀释程度
- 高δ_agg (>10): 聚合损失大，采样/拼接方法更优

### 3. h (Homophily)
- 定义: h = |{(i,j)∈E : y_i = y_j}| / |E|
- 含义: 相邻节点同类别的比例
- 低h (<0.5): 异配图，H2GCN更优

### 4. Degree CV (Coefficient of Variation)
- 定义: CV = σ(degree) / μ(degree)
- 含义: 度分布的离散程度
- 高CV (>1.0): 存在hub节点，采样方法更稳定

## 决策流程

```
输入: 图G, 特征X, 标签Y
       │
       ▼
  计算 δ_agg
       │
       ├─── δ_agg > 10 ────────────────────────┐
       │                                        │
       │    高聚合稀释                          ▼
       │                               GraphSAGE (采样)
       │                               H2GCN (拼接)
       │                               MixHop (多尺度)
       │
       └─── δ_agg ≤ 10 ─────────┐
                                │
                                ▼
                           计算 ρ_FS
                                │
            ┌───────────────────┴───────────────────┐
            │                                        │
     ρ_FS > 0.15                              ρ_FS ≤ 0.15
            │                                        │
            ▼                                        ▼
    特征信息丰富                              混合场景
            │                                        │
            ▼                                        ▼
    基线强? (F1>0.9)                          计算 h
     ├── Yes ──► GCN/GAT                            │
     └── No ───► NAA方法                    ┌───────┴───────┐
                                            │               │
                                         h > 0.5        h ≤ 0.5
                                            │               │
                                            ▼               ▼
                                        GCN/GAT          H2GCN
```

## 跨数据集验证

| 数据集 | δ_agg | ρ_FS | h | 框架预测 | 实际最优 | 验证 |
|--------|-------|------|---|----------|----------|------|
| IEEE-CIS | 11.25 | 0.058 | 0.93 | 采样/拼接 | H2GCN/GraphSAGE | ✓ |
| YelpChi | 12.57 | 0.008 | 0.77 | 采样/拼接 | GraphSAGE | ✓ |
| Elliptic | 0.94 | 0.278 | 0.71 | GCN/GAT(强基线) | GAT | ✓ |

## δ_agg与性能的关系

### IEEE-CIS (δ_agg=11.25)
```
H2GCN:      0.818 ████████████████████
GraphSAGE:  0.816 ████████████████████
MixHop:     0.813 ███████████████████
NAA-GCN:    0.749 ███████████████
GCN:        0.746 ███████████████
GAT:        0.744 ███████████████
```
差距: 7% (显著)

### YelpChi (δ_agg=12.57)
```
GraphSAGE:  0.919 ██████████████████████
NAA-GCN:    0.913 █████████████████████
GAT:        0.908 █████████████████████
GCN:        0.905 ████████████████████
```
GraphSAGE最优，符合预测

### Elliptic (δ_agg=0.94)
```
GAT:        0.957 ████████████████████████
GCN:        0.862 █████████████████████
```
标准方法已很强，NAA无需介入

## 理论解释

### 为什么高δ_agg需要采样/拼接方法?

1. **高度数节点的过度平滑问题**
   - GCN: h' = Σ_j h_j / d_i (均值聚合)
   - 当d_i很大时，个体信息被"平均掉"

2. **GraphSAGE的解决方案**
   - 采样固定数量邻居(如15个)
   - 避免高度节点的信息稀释

3. **H2GCN/MixHop的解决方案**
   - 使用拼接而非均值
   - 保留ego和neighbor的独立表示

### δ_agg的物理意义

δ_agg = degree × (1 - similarity_to_aggregated)

- degree高 → 聚合更多邻居
- similarity低 → 节点与聚合结果差异大
- 两者相乘 → 信息损失的"面积"

## 代码实现

```python
def compute_delta_agg(edge_index, features, n_nodes):
    # 构建邻接表
    adj_list = build_adjacency_list(edge_index, n_nodes)

    # 归一化特征
    features_norm = normalize(features)

    dilutions = []
    for i in range(n_nodes):
        neighbors = adj_list[i]
        if len(neighbors) == 0:
            continue

        # 聚合邻居
        agg_neighbor = mean(features_norm[neighbors])
        agg_neighbor = normalize(agg_neighbor)

        # 计算相似度
        sim = dot(features_norm[i], agg_neighbor)

        # 稀释度 = 度数 × (1 - 相似度)
        dilution = len(neighbors) * (1 - sim)
        dilutions.append(dilution)

    return mean(dilutions)
```

## 论文贡献总结

1. **诊断指标扩展**: 提出δ_agg(聚合稀释度)作为新的诊断维度
2. **多维度框架**: 从单指标到(δ_agg, ρ_FS, h)三维诊断
3. **跨数据集验证**: 在3个金融欺诈数据集上验证框架有效性
4. **理论解释**: 解释了为什么高度数图需要采样/拼接方法

## 文件位置

- 分析代码: `cross_dataset_analysis.py`
- δ_agg计算: `compute_dilution.py`
- 实验结果: `results/*.json`
- 跨数据集指标: `cross_dataset_metrics.json`

---
*最后更新: 2024-12-14*
