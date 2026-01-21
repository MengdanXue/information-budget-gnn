#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pr_curve_analysis.py

用途：
1) 加载实验结果（predictions 和 labels）
2) 绘制 Precision-Recall 曲线（保存为 PDF）
3) 在所有阈值上搜索最优 F1 对应阈值（best-F1 threshold）
4) 解释“PR-AUC 高但 F1 低”的原因（阈值/工作点选择问题 + 类不平衡常见现象）
5) 生成 LaTeX 表格（保存为 .tex，便于论文直接引用）

典型用法：
  python pr_curve_analysis.py --pred results/pred.npy --label results/label.npy
  python pr_curve_analysis.py --pred results.csv:score --label results.csv:label --outdir results/pr_curve

输入格式支持（pred/label 均支持）：
  - .npy：一维数组
  - .npz：默认取第一个 key；也可用 path:array_key 指定 key
  - .csv/.tsv：默认取第一列；也可用 path:column_name 指定列名
  - .txt：每行一个数值（同样可用 pandas 读取）

注意：
  - predictions 应该是“正类概率/得分”，范围可为 [0,1] 或任意实数（会自动用 sigmoid 映射到 [0,1]，可通过 --no_sigmoid 关闭）
  - labels 必须是 {0,1} 或可被转换为 0/1 的二值标签
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def _parse_path_and_key(spec: str) -> Tuple[str, Optional[str]]:
    """
    支持两种写法：
      - "path/to/file.ext"（不指定 key/列名）
      - "path/to/file.ext:key_or_column"
    说明：Windows 路径中可能包含 "C:\\..."，其中 ":" 会冲突。
    为了兼容 Windows 盘符，本函数采用“从右向左仅切一次”的策略：
      - 如果 spec 形如 "C:\\a\\b.csv:col" => path="C:\\a\\b.csv", key="col"
      - 如果 spec 形如 "C:\\a\\b.csv"     => path="C:\\a\\b.csv", key=None
    """
    if ":" not in spec:
        return spec, None
    path, maybe_key = spec.rsplit(":", 1)
    # 盘符场景：如果 path 只有一个字母（如 "C"），说明 spec 原本是 "C:\..." 没有 key
    if len(path) == 1 and path.isalpha():
        return spec, None
    return path, maybe_key


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # 数值稳定的 sigmoid
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def load_1d_array(spec: str, *, kind: str) -> np.ndarray:
    """
    从文件加载一维数组。
    kind: 仅用于报错信息（"predictions" 或 "labels"）。
    """
    path, key = _parse_path_and_key(spec)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif ext == ".npz":
        data = np.load(path, allow_pickle=False)
        if key is None:
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"{kind} npz has no arrays: {path}")
            arr = data[keys[0]]
        else:
            if key not in data:
                raise KeyError(f"{kind} key '{key}' not in npz: {path} (keys={list(data.keys())})")
            arr = data[key]
    elif ext in {".csv", ".tsv", ".txt"}:
        sep = "\t" if ext == ".tsv" else None
        df = pd.read_csv(path, sep=sep, engine="python")
        if df.shape[1] == 0:
            raise ValueError(f"{kind} file has no columns: {path}")
        if key is None:
            series = df.iloc[:, 0]
        else:
            if key not in df.columns:
                raise KeyError(f"{kind} column '{key}' not in {path} (columns={list(df.columns)})")
            series = df[key]
        arr = series.to_numpy()
    else:
        raise ValueError(
            f"Unsupported {kind} file extension: {ext}. "
            f"Supported: .npy, .npz, .csv, .tsv, .txt"
        )

    arr = np.asarray(arr)
    arr = arr.reshape(-1)
    return arr


def _coerce_labels_to_binary(labels: np.ndarray) -> np.ndarray:
    """
    将 labels 转成 {0,1}。
    允许输入：
      - {0,1}
      - {False,True}
      - 任意数值：>0 视为 1，否则 0（可用于 -1/+1 或计数标签）
      - 字符串：尝试转数值；失败则识别 "true/false/yes/no"
    """
    labels = np.asarray(labels).reshape(-1)

    if labels.dtype == bool:
        return labels.astype(int)

    if np.issubdtype(labels.dtype, np.number):
        return (labels > 0).astype(int)

    # 字符串/对象：尽量解析
    normalized = np.array([str(x).strip().lower() for x in labels], dtype=object)
    mapping = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0, "y": 1, "n": 0}
    out = np.empty_like(normalized, dtype=int)
    for i, v in enumerate(normalized):
        if v in mapping:
            out[i] = mapping[v]
        else:
            try:
                out[i] = 1 if float(v) > 0 else 0
            except Exception as exc:
                raise ValueError(f"Unrecognized label value at index {i}: {labels[i]!r}") from exc
    return out


def _validate_lengths(pred: np.ndarray, label: np.ndarray) -> None:
    if pred.shape[0] != label.shape[0]:
        raise ValueError(f"Length mismatch: predictions={pred.shape[0]} labels={label.shape[0]}")
    if pred.shape[0] == 0:
        raise ValueError("Empty inputs: no samples found.")


def _f1_from_pr(precision: np.ndarray, recall: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (2.0 * precision * recall) / (precision + recall + eps)


@dataclass(frozen=True)
class BestF1Result:
    threshold: float
    f1: float
    precision: float
    recall: float
    idx: int


def find_best_f1_threshold(pred_proba: np.ndarray, labels01: np.ndarray) -> BestF1Result:
    """
    在 precision_recall_curve 输出的所有阈值上搜索最优 F1。

    sklearn.precision_recall_curve 返回：
      precision: length = n_thresholds + 1
      recall:    length = n_thresholds + 1
      thresholds:length = n_thresholds

    其中 precision[i], recall[i] 对应阈值 thresholds[i]（i 从 0 到 n_thresholds-1）。
    precision[-1], recall[-1] 是补点（阈值无对应值），因此这里仅在有效阈值上取最大值。
    """
    precision, recall, thresholds = precision_recall_curve(labels01, pred_proba)
    if thresholds.size == 0:
        # 极端情况：所有预测分数相同，无法形成有效阈值序列
        f1_all = _f1_from_pr(precision, recall)
        best_idx = int(np.nanargmax(f1_all))
        # 这里没有阈值可返回，只能用 0.5 作为默认
        return BestF1Result(
            threshold=0.5,
            f1=float(f1_all[best_idx]),
            precision=float(precision[best_idx]),
            recall=float(recall[best_idx]),
            idx=best_idx,
        )

    precision_t = precision[:-1]
    recall_t = recall[:-1]
    f1 = _f1_from_pr(precision_t, recall_t)
    best_idx = int(np.nanargmax(f1))
    return BestF1Result(
        threshold=float(thresholds[best_idx]),
        f1=float(f1[best_idx]),
        precision=float(precision_t[best_idx]),
        recall=float(recall_t[best_idx]),
        idx=best_idx,
    )


def _metrics_at_threshold(pred_proba: np.ndarray, labels01: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    """
    计算固定阈值下的 precision/recall/f1。
    """
    pred_pos = pred_proba >= threshold
    tp = int(np.sum((pred_pos == 1) & (labels01 == 1)))
    fp = int(np.sum((pred_pos == 1) & (labels01 == 0)))
    fn = int(np.sum((pred_pos == 0) & (labels01 == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _latex_escape(s: str) -> str:
    # 简单 LaTeX 转义，够用于常见表格内容
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def write_latex_table(
    out_tex_path: str,
    *,
    run_name: str,
    n_samples: int,
    pos_rate: float,
    pr_auc: float,
    best: BestF1Result,
    p_at_05: float,
    r_at_05: float,
    f1_at_05: float,
) -> None:
    """
    生成一个可直接 \\input{} 到论文中的 LaTeX 表格（booktabs 风格）。
    """
    os.makedirs(os.path.dirname(out_tex_path) or ".", exist_ok=True)

    run_name_e = _latex_escape(run_name)
    content = r"""\begin{table}[t]
\centering
\small
\caption{Precision--Recall analysis summary.}
\label{tab:pr_analysis_summary}
\begin{tabular}{lrrrrrr}
\toprule
Run & $N$ & PosRate & PR-AUC & Best-$F_1$ & BestThr & $F_1@0.5$ \\
\midrule
%s & %d & %.4f & %.4f & %.4f & %.6f & %.4f \\
\bottomrule
\end{tabular}
\end{table}
""" % (
        run_name_e,
        int(n_samples),
        float(pos_rate),
        float(pr_auc),
        float(best.f1),
        float(best.threshold),
        float(f1_at_05),
    )

    # 同时把 best 点对应的 P/R、以及 0.5 阈值下的 P/R 写到注释中，便于复现和核对
    content += (
        "\n% Details:\n"
        f"% Best point: precision={best.precision:.6f}, recall={best.recall:.6f}, f1={best.f1:.6f}, thr={best.threshold:.6f}\n"
        f"% At thr=0.5: precision={p_at_05:.6f}, recall={r_at_05:.6f}, f1={f1_at_05:.6f}\n"
    )

    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(content)


def plot_pr_curve(
    out_pdf_path: str,
    *,
    labels01: np.ndarray,
    pred_proba: np.ndarray,
    pr_auc: float,
    best: BestF1Result,
    title: str,
) -> None:
    """
    绘制 PR 曲线并保存为 PDF。
    """
    import matplotlib.pyplot as plt

    precision, recall, thresholds = precision_recall_curve(labels01, pred_proba)

    os.makedirs(os.path.dirname(out_pdf_path) or ".", exist_ok=True)
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(recall, precision, linewidth=2.0, label=f"PR curve (AP={pr_auc:.4f})")

    # 标出 best-F1 点（在曲线上对应 idx）
    # precision/recall 的最后一个点没有阈值，因此 best.idx 对应 precision/recall 的同索引（去掉最后一个点的版本）
    best_x = float(recall[:-1][best.idx]) if recall.size > 1 else float(recall[0])
    best_y = float(precision[:-1][best.idx]) if precision.size > 1 else float(precision[0])
    plt.scatter([best_x], [best_y], s=36, zorder=3, label=f"Best $F_1$={best.f1:.4f} @ thr={best.threshold:.4f}")

    # 基线：随机分类器的 precision 水平（等于正类比例）
    pos_rate = float(np.mean(labels01))
    plt.hlines(pos_rate, xmin=0.0, xmax=1.0, linestyles="--", linewidth=1.5, label=f"Baseline (pos rate={pos_rate:.4f})")

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_pdf_path, format="pdf")
    plt.close()


def explain_auc_high_f1_low(*, pr_auc: float, best_f1: float, f1_at_default: float, default_thr: float = 0.5) -> str:
    """
    给出“PR-AUC 高但 F1 低”的常见解释（阈值/工作点问题）。

    重点：
      - PR-AUC / AP 是“阈值无关”的排序质量指标：模型把正样本排到前面的能力越强，AP 越高。
      - F1 是“阈值相关”的工作点指标：你选的阈值决定 precision/recall 权衡；阈值不合适会导致 F1 低。
      - 类不平衡时，0.5 常常不是好阈值（尤其当预测值并非严格校准概率）。
    """
    lines = []
    lines.append("为什么会出现 PR-AUC 高但 F1 低？（典型是阈值/工作点选择问题）")
    lines.append(f"- PR-AUC(AP)={pr_auc:.4f} 是阈值无关指标：衡量“排序”是否把正样本排在前面。")
    lines.append(f"- F1 是阈值相关指标：在某个阈值下的 Precision/Recall 折中。默认阈值 {default_thr:.2f} 不一定合适。")
    lines.append(f"- 当前：Best-F1={best_f1:.4f}，而 F1@{default_thr:.2f}={f1_at_default:.4f}。差距通常来自阈值未调优。")
    lines.append("- 类不平衡时更常见：正类很少，稍微提高阈值就可能召回急剧下降，导致 F1 变低。")
    lines.append("- 若预测分数是“未校准”的 logits/score，0.5 阈值没有概率含义；应使用验证集调阈值或做概率校准（Platt/Isotonic）。")
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precision-Recall curve analysis + best F1 threshold + LaTeX/PDF export.")
    p.add_argument("--pred", required=True, help="Predictions file spec, e.g. path.npy or path.csv:score")
    p.add_argument("--label", required=True, help="Labels file spec, e.g. path.npy or path.csv:label")
    p.add_argument("--run_name", default="Run", help="Name used in plot title and LaTeX table.")
    p.add_argument(
        "--outdir",
        default=os.path.join("results", "pr_curve"),
        help="Output directory for PDF/TeX/text outputs.",
    )
    p.add_argument("--pdf_name", default="pr_curve.pdf", help="Output PR curve PDF filename.")
    p.add_argument("--tex_name", default="pr_curve_summary.tex", help="Output LaTeX table filename.")
    p.add_argument("--explain_name", default="pr_curve_explanation.txt", help="Output explanation text filename.")
    p.add_argument(
        "--no_sigmoid",
        action="store_true",
        help="Do NOT apply sigmoid to predictions. Use this if predictions are already probabilities in [0,1].",
    )
    p.add_argument(
        "--clip_proba",
        action="store_true",
        help="Clip predictions to [0,1] after (optional) sigmoid. Useful for numeric safety / plotting.",
    )
    p.add_argument(
        "--default_threshold",
        type=float,
        default=0.5,
        help="Default threshold used to report F1@thr (often 0.5).",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()

    pred = load_1d_array(args.pred, kind="predictions")
    labels = load_1d_array(args.label, kind="labels")
    labels01 = _coerce_labels_to_binary(labels)
    _validate_lengths(pred, labels01)

    # 将预测转成“概率”用于 PR 曲线与阈值搜索
    pred = pred.astype(float)
    if not args.no_sigmoid:
        pred_proba = _sigmoid(pred)
    else:
        pred_proba = pred

    if args.clip_proba:
        pred_proba = np.clip(pred_proba, 0.0, 1.0)

    # PR-AUC（Average Precision, AP）
    pr_auc = float(average_precision_score(labels01, pred_proba))

    # 最优 F1 阈值
    best = find_best_f1_threshold(pred_proba, labels01)

    # 报告默认阈值（常用 0.5）下的表现，用来对比“阈值未调优”的影响
    p05, r05, f105 = _metrics_at_threshold(pred_proba, labels01, float(args.default_threshold))

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    out_pdf = os.path.join(outdir, args.pdf_name)
    out_tex = os.path.join(outdir, args.tex_name)
    out_explain = os.path.join(outdir, args.explain_name)

    # 画图并导出 PDF
    plot_pr_curve(
        out_pdf,
        labels01=labels01,
        pred_proba=pred_proba,
        pr_auc=pr_auc,
        best=best,
        title=f"{args.run_name}  Precision-Recall",
    )

    # LaTeX 表格
    pos_rate = float(np.mean(labels01))
    write_latex_table(
        out_tex,
        run_name=args.run_name,
        n_samples=int(labels01.shape[0]),
        pos_rate=pos_rate,
        pr_auc=pr_auc,
        best=best,
        p_at_05=float(p05),
        r_at_05=float(r05),
        f1_at_05=float(f105),
    )

    # 解释文本（写入文件 + 同时打印到控制台）
    explanation = explain_auc_high_f1_low(
        pr_auc=pr_auc,
        best_f1=best.f1,
        f1_at_default=float(f105),
        default_thr=float(args.default_threshold),
    )
    with open(out_explain, "w", encoding="utf-8") as f:
        f.write(explanation + "\n")

    # 控制台摘要（便于你直接复制到日志/论文记录）
    print("=== PR Curve Analysis Summary ===")
    print(f"Run name          : {args.run_name}")
    print(f"N samples         : {labels01.shape[0]}")
    print(f"Positive rate     : {pos_rate:.6f}")
    print(f"PR-AUC (AP)       : {pr_auc:.6f}")
    print(f"Best-F1           : {best.f1:.6f}")
    print(f"Best threshold    : {best.threshold:.8f}")
    print(f"Best precision    : {best.precision:.6f}")
    print(f"Best recall       : {best.recall:.6f}")
    print(f"F1@{args.default_threshold:.2f}          : {f105:.6f} (P={p05:.6f}, R={r05:.6f})")
    print(f"Saved PR curve PDF: {out_pdf}")
    print(f"Saved LaTeX table : {out_tex}")
    print(f"Saved explanation : {out_explain}")
    print()
    print(explanation)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
