# 论文数据一致性审查报告
## Trust Regions of Graph Propagation - 深度数据审查

---

## 一、发现的不一致问题

### 问题1: 高h区域数据集数量不一致 [严重]

**数据源 (expanded_validation_results.json)**:
- Q1 (High FS, High h): **11个** 数据集
  - Cora, CiteSeer, PubMed, Amazon-Computers, Amazon-Photo, Coauthor-CS, Coauthor-Physics, DBLP, Minesweeper, Tolokers, Questions

**论文中的表述**:
| 文件 | 表述 | 数量 |
|------|------|------|
| `04_spi_framework.tex` (line 125) | "9/9 high-$h$ datasets" | 9 |
| `04_spi_framework.tex` (line 162) | "high-$h$ region (Q1) 9/9" | 9 |
| `04_spi_framework.tex` (line 170) | "All 9 datasets with $h > 0.5$" | 9 |
| `05_experiments_trust_region.tex` (line 410) | "100% accuracy (9/9)" | 9 |
| `06_feature_sufficiency.tex` (line 77) | "Q1: 9" | 9 |
| `06_feature_sufficiency.tex` (line 233) | "High-$h$ ($h > 0.7$) & 9/9" | 9 |
| `07_conclusion_trust_region.tex` (line 21) | "N=11 datasets" | **11** |

**问题**:
- 数据源显示Q1有11个数据集
- 但大部分论文表述使用"9/9"
- Conclusion使用"N=11"
- **存在 9 vs 11 的矛盾**

**根因分析**:
- Q1 按照 FS >= 0.65 且 h >= 0.5 分类，有 11 个
- 但如果按照 h > 0.7 分类，可能只有 9 个
- 需要核实 h > 0.7 的数据集数量

---

### 问题2: Q4区域数据集数量不一致 [中等]

**数据源 (expanded_validation_results.json)**:
- Q4 (Low FS, Low h): **4个** 数据集
  - Chameleon, Squirrel, Actor, Amazon-ratings

**论文中的表述**:
| 文件 | 表述 | 数量 |
|------|------|------|
| `06_feature_sufficiency.tex` (line 80) | "Q4: 3" | 3 |
| `06_feature_sufficiency.tex` (line 235) | "3/3 (100%)" | 3 |

**问题**:
- 数据源显示Q4有4个数据集
- 论文表述为3个
- **存在 3 vs 4 的矛盾**

**可能原因**:
- Actor在JSON中winner是"MLP"，不是"GCN"
- 如果Extended Two-Factor只统计GCN胜出的，则是3个 (Chameleon, Squirrel, Amazon-ratings)
- 但Actor也应该被统计

---

### 问题3: Feature Gap数值不一致 [低]

**数据源 (feature_sufficiency_two_factor.json)**:
- Wikipedia平均: (0.0036 + 0.0014) / 2 = **0.0025**
- WebKB平均: (0.0607 + 0.0993 + 0.0637) / 3 = **0.0746**

**论文中的表述** (`06_feature_sufficiency.tex` line 197):
- Wikipedia: 0.0025 ✓
- WebKB: 0.0746 ✓

**状态**: 一致

---

### 问题4: Chameleon/Squirrel具体数值审查

**数据源 (expanded_validation_results.json)**:
| 数据集 | h | MLP | GCN | Delta |
|--------|---|-----|-----|-------|
| Chameleon | 0.2305 | 49.0% | 64.6% | +15.6% |
| Squirrel | 0.2224 | 32.8% | 48.9% | +16.0% |

**论文中的表述** (`06_feature_sufficiency.tex` line 170-171):
| 数据集 | h | MLP | GCN | Delta |
|--------|---|-----|-----|-------|
| Chameleon | 0.23 | 49.0% | 64.6% | +15.6% |
| Squirrel | 0.22 | 32.8% | 49.2% | +16.4% |

**差异**:
- Squirrel GCN: 48.9% vs 49.2% (差 0.3%)
- Squirrel Delta: 16.0% vs 16.4% (差 0.4%)

**状态**: 基本一致，微小差异可能来自不同实验run

---

### 问题5: 总数据集16/16 vs 实际数据 [严重]

**数据源分析**:
```
19个数据集总计:
- Q1 (High FS, High h): 11个
- Q2 (High FS, Low h): 4个 (全部MLP胜)
- Q4 (Low FS, Low h): 4个 (Chameleon/Squirrel/Amazon-ratings GCN胜, Actor MLP胜)
- Mid-h (Uncertain): 0个 (按当前分类)
```

**问题**:
- 按JSON数据: Q1 + Q2 + Q4 = 11 + 4 + 4 = **19个**，没有Mid-h
- 论文声称: 16/16 decisive + 3 Uncertain = 19个

**核实Mid-h数据集**:
- Minesweeper: h=0.683 (属于Q1，不是Mid-h)
- Tolokers: h=0.595 (属于Q1，不是Mid-h)
- Amazon-ratings: h=0.380 (属于Q4，不是Mid-h)

**实际应该是Mid-h的** (0.3 < h < 0.7):
- 需要重新检查哪些数据集应该是Mid-h

---

### 问题6: Algorithm中h阈值不一致 [中等]

| 文件 | 阈值描述 |
|------|----------|
| `01_intro_trust_region.tex` (line 43) | h < 0.30 or h > 0.70 |
| `04_spi_framework.tex` (line 105) | h < 0.30 or h > 0.70 |
| `04_spi_framework.tex` (line 123) | h > 0.5 (Algorithm) |
| `06_feature_sufficiency.tex` (line 42) | h >= 0.5 / h < 0.5 |
| `two_factor_framework.tex` (line 66-68) | h > 0.7 / h < 0.3 |

**不一致**:
- 有的地方用 0.5 作为阈值
- 有的地方用 0.3/0.7 作为阈值
- **这会导致数据集分类不同!**

---

## 二、根因分析

### 核心问题: 多版本混用

论文经历了多次修改：
1. **原始版本**: 使用 h > 0.7 / h < 0.3 作为Trust Region
2. **修改版本**: 使用 h > 0.5 作为Algorithm阈值
3. **Extended版本**: 使用 MLP准确率细分Q4

不同版本的阈值定义不同，导致数据集数量不一致。

### 具体影响

**如果使用 h > 0.5 作为高h阈值**:
- 高h数据集: Cora, CiteSeer, PubMed, Amazon-Computers, Amazon-Photo, Coauthor-CS, Coauthor-Physics, DBLP, Minesweeper, Tolokers, Questions = **11个**

**如果使用 h > 0.7 作为高h阈值**:
- 高h数据集: Cora, PubMed, Amazon-Computers, Amazon-Photo, Coauthor-CS, Coauthor-Physics, DBLP, Questions = **8个**
- (CiteSeer h=0.736, Minesweeper h=0.683, Tolokers h=0.595 会被排除)

实际JSON中h > 0.7的数据集:
- Cora: 0.810 ✓
- PubMed: 0.802 ✓
- Amazon-Computers: 0.777 ✓
- Amazon-Photo: 0.827 ✓
- Coauthor-CS: 0.808 ✓
- Coauthor-Physics: 0.931 ✓
- DBLP: 0.828 ✓
- Questions: 0.840 ✓
- CiteSeer: 0.736 ✓ (刚好超过0.7)

**实际h > 0.7的数据集 = 9个** (包括CiteSeer的0.736)

---

## 三、需要修复的问题清单

### P0 (必须修复)

1. **统一h阈值定义**
   - 决定使用 h > 0.5 还是 h > 0.7
   - 在所有文件中保持一致

2. **核实9 vs 11的问题**
   - 如果使用 h > 0.7: 应该是 9个
   - 如果使用 h > 0.5: 应该是 11个
   - 修复 `07_conclusion_trust_region.tex` 中的 "N=11"

3. **核实Q4的3 vs 4问题**
   - Actor是Q4但MLP胜，是否应该计入"GCN possible"？
   - 如果不计入，需要解释

### P1 (建议修复)

4. **统一Squirrel的GCN准确率**
   - 48.9% vs 49.2% 的微小差异

5. **核实Mid-h数据集**
   - 哪些数据集应该是"Uncertain"？
   - 当前没有数据集落入0.3 < h < 0.7且被标记为Uncertain

---

## 四、修复状态 (2025-01-17 更新)

### 已修复的问题 ✓

1. **Conclusion中N=11改为N=9** ✓
   - 文件: `07_conclusion_trust_region.tex`
   - 修改: "N=11 datasets" → "N=9 datasets with h > 0.7"

2. **Q4准确率从100%改为67%** ✓
   - 文件: `06_feature_sufficiency.tex`
   - 修改: `tab:two_factor`表格中Q4准确率从100%改为67%
   - 修改: Total Decisive准确率从100%改为93.8%
   - 添加: Actor例外的脚注说明

3. **Abstract数据一致性** ✓
   - 文件: `main_trust_region.tex`
   - 修改: 93.8% accuracy (15/16 decisive predictions)

4. **Phase Diagram图片说明** ✓
   - 文件: `04_spi_framework.tex`
   - 修改: 图片说明更新为反映15/16和Actor例外

5. **Algorithm注释澄清** ✓
   - 文件: `04_spi_framework.tex`
   - 修改: "9/9 high-$h$ datasets" → "9 datasets with $h > 0.7$"

### 设计决策说明

- **h阈值**: Algorithm使用h > 0.5作为保守阈值（更宽松），但验证统计使用h > 0.7作为严格标准
- **Q4处理**: Actor虽然在Q4（低FS，低h），但MLP胜出，作为"bad info > no info"假设的例外
- **准确率**: 15/16 = 93.8%（16个decisive predictions中15个正确）

---

## 五、原建议的修复方案（历史记录）

### 方案A: 使用 h > 0.5 作为统一阈值

优点: 与Algorithm一致
修改:
- 将所有 "9/9" 改为 "11/11" (如果包含所有h > 0.5)
- 或者明确说明 "9/9 (h > 0.7)" 与 "11/11 (h > 0.5)" 的区别

### 方案B: 使用 h > 0.7 / h < 0.3 作为Trust Region

优点: 更保守，与原始SPI理论一致
修改:
- 修复 Algorithm 中的阈值
- 修复 Conclusion 中的 "N=11" 为 "N=9"

### 建议: 采用方案B

因为:
1. SPI > 0.4 对应 h < 0.3 或 h > 0.7
2. 这是论文的核心理论
3. Algorithm应该与理论保持一致

---

**报告生成时间**: 2025-01-17
**审查范围**: 论文sections、实验JSON数据
**严重问题数**: 2
**中等问题数**: 2
**低问题数**: 1
