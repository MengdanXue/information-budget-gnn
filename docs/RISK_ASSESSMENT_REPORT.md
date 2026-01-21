# 实验风险评估报告
## Trust Regions of Graph Propagation - 审稿人视角审查

---

## 一、发现的关键问题

### 问题1: SPI在低h区域的系统性失败 [严重]

**问题描述**:
论文声称 SPI = |2h - 1| > 0.4 时可以可靠预测GNN vs MLP的胜负。

然而实际数据显示:

| 数据集 | h | SPI | MLP | GCN | 胜者 | SPI预测 | 结果 |
|--------|---|-----|-----|-----|------|---------|------|
| Chameleon | 0.23 | 0.54 | 49.0% | 64.6% | GCN | MLP | **错误** |
| Squirrel | 0.22 | 0.56 | 32.8% | 48.9% | GCN | MLP | **错误** |

**核心矛盾**:
- SPI > 0.4 且 h < 0.3 → 应该预测 MLP 胜
- 但 Chameleon 和 Squirrel 上 GCN 大幅胜出 (+15.6% 和 +16.1%)
- 这直接违反了论文的核心理论预测

**论文如何处理**:
论文在第412行承认: "Low-h region: SPI achieves **14%** accuracy (1/7)"
这意味着在低h区域,SPI预测基本失效。

**审稿人可能的攻击**:
- "你的框架在最需要指导的地方(低h)反而失效"
- "Chameleon和Squirrel是被广泛使用的benchmark,不能简单忽略"

---

### 问题2: 100%准确率的计算方式有误导性 [中等严重]

**问题描述**:
实验分析报告(EXPERIMENT_ANALYSIS_REPORT.md)声称:
- "High-h区域: 100%准确率"
- "Real-World Validation: 100%"

然而实际计算:
- expanded_validation_results.json 显示 19 个数据集
- 其中 4 个被标记为 "Uncertain" (Q4象限)
- 100% 是基于 15/15,不是 19/19
- 如果包含所有数据集,使用纯SPI预测: **87.5%** (14/16 Trust Region内)

**代码中的逻辑**:
```python
# expanded_dataset_validation.py 第290-301行
def classify_quadrant(mlp_acc, h, fs_thresh=0.65, h_thresh=0.5):
    """Classify dataset into quadrant"""
    if mlp_acc >= fs_thresh:
        if h >= h_thresh:
            return 'Q1', 'GCN_maybe'
        else:
            return 'Q2', 'MLP'
    else:
        if h >= h_thresh:
            return 'Q3', 'GCN'
        else:
            return 'Q4', 'Uncertain'  # <-- 这里排除了问题数据集!
```

**审稿人可能的攻击**:
- "你通过把失败案例标记为'Uncertain'来人为提高准确率"
- "如果SPI无法预测这些数据集,说明框架不完整"

---

### 问题3: Chameleon和Squirrel的特殊性未被充分解释 [中等]

**已知事实**:
- Chameleon和Squirrel是Wikipedia网络数据集
- 它们虽然h很低(0.22-0.23),但GCN仍然大胜MLP
- 论文用"2-hop recovery"来解释,但这些数据集的2-hop recovery < 1x

**论文的解释** (第186行):
> "Wikipedia datasets (Actor, Chameleon, Squirrel) show no recovery, suggesting different failure mechanisms (feature-orthogonal noise)."

**矛盾点**:
- 如果是"feature-orthogonal noise",为什么GCN能大胜MLP?
- "no recovery"应该意味着MLP更好,但实际相反

**需要补充的解释**:
Chameleon和Squirrel可能存在:
1. 特征本身质量很差(MLP只有32-49%)
2. GCN的聚合虽然带来noise,但仍提供了额外信息
3. 这是"两害相权取其轻"的情况

---

### 问题4: 论文中存在不一致的表述 [低风险]

**地点1**: 实验部分声称SPI在高h区域100%准确
**地点2**: 讨论部分承认低h区域14%准确

这种不一致可能被审稿人解读为:
- 选择性报告(cherry-picking)
- 框架的实际价值有限

---

## 二、与之前的问题对比

### 之前被推翻的实验
1. **ρ_FS完全失效** (0%准确率) - 已在论文中移除
2. **"Less is More"叙事崩塌** - 已重新定位
3. **硬编码问题** - 已修复

### 当前问题的严重程度
当前发现的问题**没有达到"推翻实验"的程度**,因为:
1. 论文已经明确承认低h区域准确率只有14%
2. 核心贡献(U-shape, Information Budget)仍然成立
3. 高h区域的100%准确率是真实的(9/9)

但是:
- 论文的"SPI作为通用预测指标"的叙事被削弱
- 需要更诚实地呈现框架的局限性

---

## 三、建议的修改

### 必须修改 (P0)

1. **修改准确率报告方式**
   - 不要说"100% overall"
   - 改为: "100% in high-h region (9/9), 14% in low-h region (1/7)"
   - 强调框架的非对称性是设计选择,不是缺陷

2. **增加对Chameleon/Squirrel的专门讨论**
   - 承认这是框架的已知局限
   - 提供可能的解释(特征质量差,聚合仍有价值)
   - 建议未来工作方向

### 建议修改 (P1)

3. **重新定义Trust Region**
   - 当前: SPI > 0.4 是 Trust Region
   - 建议: 只有 h > 0.7 是 Trust Region (对应高h)
   - 低h区域不应被称为"Trust",而是"Needs Further Analysis"

4. **修改Algorithm 1的决策逻辑**
   - 明确说明低h时不应直接信任SPI
   - 添加: "If h < 0.3, consider domain-specific factors"

---

## 四、TKDE投稿风险重新评估

### 修改前
| AI | 概率 |
|----|------|
| Gemini | 82-88% |
| Codex | 70-80% |
| 共识 | 82% |

### 修改后 (如果不修改上述问题)
| 风险 | 概率影响 |
|------|---------|
| 审稿人发现100%不是真的 | -10% |
| 审稿人质疑Chameleon/Squirrel | -5% |
| 审稿人认为框架不完整 | -5% |

**调整后概率: 62-72%**

### 修改后 (如果修改上述问题)
| 改进 | 概率影响 |
|------|---------|
| 诚实报告增加可信度 | +5% |
| 充分解释局限性 | +3% |

**调整后概率: 75-85%**

---

## 五、结论

### 实验没有被"推翻"
核心发现仍然成立:
1. U-shape现象 ✓
2. Information Budget原则 ✓
3. 高h区域的可靠预测 ✓

### 但存在诚信风险
论文的某些表述可能被视为:
1. 过度宣传(overclaiming)
2. 选择性报告(selective reporting)

### 行动建议
1. **立即**: 重新审查论文中所有"100%"的表述
2. **24小时内**: 补充对Chameleon/Squirrel的解释
3. **48小时内**: 修改准确率报告方式

---

**报告生成时间**: 2025-01-17
**审查范围**: 核心实验代码、结果文件、论文sections
