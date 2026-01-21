# IEEE-CIS 先验预测实验 - 快速开始指南

## 5分钟快速开始

### 前置条件

```bash
# 1. 安装依赖
pip install torch torch_geometric scikit-learn scipy numpy pandas

# 2. 下载IEEE-CIS数据
# 访问 https://www.kaggle.com/c/ieee-fraud-detection/data
# 下载 train_transaction.csv 和 train_identity.csv
# 放到 ./ieee_cis_data/ 目录
```

### 最简单的运行方式

**Windows**:
```bash
run_ieee_cis_full_pipeline.bat
```

**Linux/Mac**:
```bash
chmod +x run_ieee_cis_full_pipeline.sh
./run_ieee_cis_full_pipeline.sh
```

这会自动运行全部4个阶段。

---

## 分步运行（推荐）

### Step 1: 构建图（5-10分钟）

```bash
python ieee_cis_graph_builder.py \
  --data_dir ./ieee_cis_data \
  --output_dir ./processed
```

**输出**：
- `./processed/ieee_cis_graph.pkl` - 图数据
- `./processed/ieee_cis_summary.txt` - 数据摘要

**检查**：
```bash
cat ./processed/ieee_cis_summary.txt
```

应该看到类似：
```
IEEE-CIS Graph Dataset Summary
==============================
Nodes: 590540
Edges: 28147293
Features: 234
Fraud Rate: 3.50%
...
```

---

### Step 2: 做出预测（<1分钟）

```bash
python prior_prediction_experiment.py \
  --phase 1 \
  --data_path ./processed/ieee_cis_graph.pkl \
  --output_dir ./results
```

**关键输出**：
```
PREDICTION COMMITTED AND TIMESTAMPED
======================================================================
Timestamp: 2024-12-21T10:30:15.123456
Predicted method: H2GCN
Reasoning: High δ_agg (11.25 > 10) indicates...
Prediction hash: a3f5c8e9b7d4f6a1c2e8d9f0b3a5c7e9
======================================================================
```

**重要**：此时不要继续！先验证预测已经锁定。

**验证预测完整性**：
```bash
# 查看预测文件
cat ./results/fsd_prediction.json

# 查看哈希
cat ./results/prediction_hash.json

# （可选）提交到Git
cd ./results
git init
git add .
git commit -m "FSD prediction for IEEE-CIS"
```

---

### Step 3: 运行实验（8-16小时）

现在可以运行GNN训练了：

```bash
python prior_prediction_experiment.py \
  --phase 2 \
  --data_path ./processed/ieee_cis_graph.pkl \
  --output_dir ./results \
  --device cuda  # 或 cpu
```

**如果时间不够，可以减少方法/种子**：

```bash
# 只测试核心方法（快速验证）
python prior_prediction_experiment.py \
  --phase 2 \
  --methods GCN GAT H2GCN \
  --seeds 42 123 456  # 只用3个种子
  --device cuda
```

**监控进度**：
```bash
# 实验会打印进度
--- Method: GCN ---
  Seed 1/10: AUC=0.8201, F1=0.6982
  Seed 2/10: AUC=0.8189, F1=0.6945
  ...
```

**中断恢复**：
如果中断，已完成的结果会保存。你可以修改代码只运行未完成的方法。

---

### Step 4: 生成报告（<1分钟）

```bash
python prior_prediction_experiment.py \
  --phase 4 \
  --output_dir ./results
```

**输出**：
- `./results/validation_report.md` - 完整验证报告

**查看报告**：
```bash
cat ./results/validation_report.md

# 或在浏览器中查看（如果安装了Markdown查看器）
markdown ./results/validation_report.md > report.html
```

---

## 常见问题

### Q1: 我没有GPU怎么办？

使用CPU模式，但会很慢（24-48小时）：
```bash
python prior_prediction_experiment.py --phase 2 --device cpu
```

**加速建议**：
- 减少方法：只测试 GCN, GAT, H2GCN
- 减少种子：用5个而不是10个
- 减少数据：在图构建时加 `--subsample 100000`

### Q2: 图构建失败，提示内存不足？

IEEE-CIS很大（~60万节点）。如果内存不足：

```bash
# 使用子采样（测试用）
python ieee_cis_graph_builder.py \
  --data_dir ./ieee_cis_data \
  --output_dir ./processed \
  --subsample 100000  # 只用10万个交易
```

### Q3: 实验运行了一半就崩溃了？

不用担心！结果会保存到 `./results/experimental_results.json`。

**恢复方法**：
1. 打开 `experimental_results.json`，看哪些方法已完成
2. 修改 `--methods` 参数，只跑未完成的方法
3. 继续运行

**示例**：
```bash
# 假设GCN和GAT已完成，只跑剩余的
python prior_prediction_experiment.py \
  --phase 2 \
  --methods H2GCN FAGCN GPRGNN \
  --output_dir ./results
```

### Q4: 如何修改FSD预测规则？

**不建议**在看到实验结果后修改！这会破坏先验预测的可信度。

如果你想测试新的预测规则：
1. 在新目录做新实验：`--output_dir ./results_v2`
2. 记录修改的原因（理论依据，不是"因为结果不好"）
3. 在论文中诚实报告两个版本的结果

### Q5: 预测错了怎么办？

**诚实报告！** 这不是失败，而是科学发现。

在论文中写：
```
FSD predicted NAA (δ_agg=4.2 < 5), but experiments showed
H2GCN performed best. Further analysis revealed that IEEE-CIS
has strong heterophily (h=0.38), which our current decision
rules underweighted. This suggests we should add heterophily
detection to FSD. We propose a refined rule:

  if h < 0.4: predict H2GCN  (regardless of δ_agg)
  elif δ_agg > 10: predict H2GCN/GraphSAGE
  elif δ_agg < 5 and n_features > 100: predict NAA
  ...

This demonstrates FSD's value as an iterative framework.
```

---

## 文件说明

运行后，`./results/` 目录应该包含：

```
results/
├── fsd_prediction.json          # Phase 1输出：FSD预测
├── prediction_hash.json         # Phase 1输出：哈希验证
├── experimental_results.json    # Phase 2输出：实验结果
└── validation_report.md         # Phase 4输出：验证报告
```

### 各文件用途：

| 文件 | 用于 | 包含内容 |
|------|------|----------|
| `fsd_prediction.json` | 论文方法论部分 | FSD指标、预测、时间戳 |
| `prediction_hash.json` | 补充材料 | 防篡改哈希 |
| `experimental_results.json` | 论文结果部分 | 所有方法的性能 |
| `validation_report.md` | 补充材料 | 完整统计分析 |

---

## 时间规划

| 阶段 | 时间 | 可并行? |
|------|------|---------|
| 下载数据 | 10-30分钟 | 否 |
| 构建图 | 5-10分钟 | 否 |
| Phase 1: 预测 | <1分钟 | 否 |
| Phase 2: 实验 | 8-16小时（GPU）<br>24-48小时（CPU） | 可（多GPU） |
| Phase 4: 报告 | <1分钟 | 否 |
| **总计** | **~8-16小时（GPU）** | |

**建议安排**：
- **第1天上午**：下载数据、构建图、做出预测（1小时）
- **第1天下午 - 第2天上午**：运行实验（过夜，8-16小时）
- **第2天上午**：生成报告、分析结果（1小时）

---

## 在论文中使用

### 1. 方法论部分

```latex
We validate FSD's predictive power through a rigorous prior
prediction protocol. We computed $\delta_{\text{agg}}=11.25$
and predicted H2GCN on 2024-12-21 (see Supplementary S1).
This prediction was timestamped (hash: \texttt{a3f5c8e9...})
before any experiments.
```

### 2. 结果部分

```latex
Our prediction was validated: H2GCN achieved AUC=0.8512,
significantly outperforming GCN (0.8199, $p<0.001$) and
GAT (0.8156, $p<0.001$) after Bonferroni correction.
```

### 3. 补充材料

包含：
- `fsd_prediction.json` - 带时间戳的原始预测
- `validation_report.md` - 完整统计分析
- 源代码 - 可复现性

---

## 检查清单

实验前：
- [ ] 下载了IEEE-CIS数据
- [ ] 安装了所有依赖（torch, torch_geometric, scipy等）
- [ ] 确认有足够的硬盘空间（~10GB）

Phase 1后：
- [ ] `fsd_prediction.json` 存在且包含时间戳
- [ ] `prediction_hash.json` 存在
- [ ] （可选）预测已提交到Git或第三方服务

Phase 2后：
- [ ] `experimental_results.json` 存在
- [ ] 所有方法都有10个种子的结果
- [ ] 结果看起来合理（AUC在0.7-0.9之间）

Phase 4后：
- [ ] `validation_report.md` 存在
- [ ] 报告显示了预测vs实际结果的对比
- [ ] 统计检验结果包含在报告中

论文提交前：
- [ ] 补充材料包含所有4个文件
- [ ] 论文引用了预测时间戳
- [ ] 论文讨论了预测的准确性（无论对错）
- [ ] 代码已上传到公开仓库（如GitHub）

---

## 获取帮助

如果遇到问题：

1. **查看日志**：脚本会打印详细错误信息
2. **检查数据**：确保CSV文件格式正确
3. **降低规模**：先在子采样数据上测试（`--subsample 10000`）
4. **提交Issue**：https://github.com/your-repo/fsd-framework/issues

---

## 下一步

完成IEEE-CIS实验后，你可以：

1. **重复其他数据集**：YelpChi, Elliptic, Amazon等
2. **改进FSD规则**：基于IEEE-CIS的发现
3. **发表论文**：用这些结果支持FSD的预测能力
4. **开发工具**：制作FSD在线计算器

祝实验顺利！
