#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
t-SNE Visualization for GNN Fraud Detection Embeddings

功能:
1. 加载模型embedding (.npy/.pt格式)
2. 加载标签 (fraud/normal)
3. t-SNE降维到2D
4. 绘制散点图，区分欺诈和正常节点
5. 可选：标注预测错误的点
6. 保存为PDF (适合论文)

使用方法:
    python tsne_visualization.py --embeddings emb.npy --labels labels.npy
    python tsne_visualization.py --embeddings emb.pt --labels labels.pt --predictions preds.pt --output figure.pdf

Author: FSD Framework
Date: 2024-12-23
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 尝试导入 torch，用于支持 .pt 文件
try:
    import torch
except ImportError:
    torch = None


def load_tensor(path):
    """加载 .npy 或 .pt 格式的数据"""
    if not os.path.exists(path):
        print(f"Error: 文件不存在: {path}")
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    elif ext == '.pt':
        if torch is None:
            print("Error: 加载 .pt 文件需要安装 PyTorch")
            sys.exit(1)
        data = torch.load(path, map_location='cpu')
        if isinstance(data, torch.Tensor):
            return data.detach().numpy()
        return data
    else:
        print(f"Error: 不支持的文件格式 {ext}，请使用 .npy 或 .pt")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GNN Fraud Detection t-SNE Visualization")

    # 核心参数
    parser.add_argument("--embeddings", required=True, help="Embedding文件路径 (.npy/.pt)")
    parser.add_argument("--labels", required=True, help="标签文件路径 (.npy/.pt)")

    # 可选参数
    parser.add_argument("--output", default="tsne_embedding_vis.pdf", help="输出PDF文件名")
    parser.add_argument("--predictions", help="预测结果文件，用于标记错误样本 (可选)")
    parser.add_argument("--limit", type=int, default=0, help="限制采样点数以加速 (默认0表示全部)")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity 参数 (默认30)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--title", default="t-SNE Visualization of Fraud Detection Embeddings", help="图标题")

    args = parser.parse_args()

    # 1. 加载数据
    print(f"[*] Loading embeddings from {args.embeddings}...")
    emb = load_tensor(args.embeddings)
    labels = load_tensor(args.labels)

    # 确保标签是一维的
    if labels.ndim > 1:
        labels = labels.flatten()

    if len(emb) != len(labels):
        print(f"Error: 数据量不匹配! Embeddings: {len(emb)}, Labels: {len(labels)}")
        sys.exit(1)

    print(f"[*] Loaded {len(emb)} samples, embedding dim = {emb.shape[1]}")
    print(f"[*] Label distribution: Normal={np.sum(labels==0)}, Fraud={np.sum(labels==1)}")

    # 处理预测错误 (如果有)
    error_indices = []
    if args.predictions:
        print(f"[*] Loading predictions from {args.predictions}...")
        preds = load_tensor(args.predictions)
        # 如果是概率矩阵，取argmax；如果是标签，直接用
        pred_labels = np.argmax(preds, axis=1) if preds.ndim > 1 else preds

        # 找出预测错误的索引
        error_indices = np.where(pred_labels != labels)[0]
        print(f"[*] Found {len(error_indices)} misclassified nodes.")

    # 2. 数据采样 (针对大数据集)
    np.random.seed(args.seed)
    if 0 < args.limit < len(emb):
        print(f"[*] Subsampling {args.limit} points for visualization...")
        indices = np.random.choice(len(emb), args.limit, replace=False)
        emb = emb[indices]
        labels = labels[indices]

        # 映射错误索引到新的采样索引
        if args.predictions:
            is_error_mask = np.isin(indices, error_indices)
            error_indices = np.where(is_error_mask)[0]

    # 3. t-SNE 降维
    print(f"[*] Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=args.seed,
        init='pca',
        learning_rate='auto',
        n_iter=1000
    )
    emb_2d = tsne.fit_transform(emb)
    print(f"[*] t-SNE completed.")

    # 4. 绘图
    print("[*] Plotting...")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # 设置样式
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11

    # 定义掩码
    mask_normal = (labels == 0)
    mask_fraud = (labels == 1)

    # 绘制正常节点 (蓝色)
    ax.scatter(
        emb_2d[mask_normal, 0], emb_2d[mask_normal, 1],
        c='#3498db', label=f'Normal (n={mask_normal.sum()})',
        alpha=0.5, s=15, edgecolors='none'
    )

    # 绘制欺诈节点 (红色)
    ax.scatter(
        emb_2d[mask_fraud, 0], emb_2d[mask_fraud, 1],
        c='#e74c3c', label=f'Fraud (n={mask_fraud.sum()})',
        alpha=0.7, s=25, edgecolors='none'
    )

    # (可选) 标记错误预测点
    if len(error_indices) > 0:
        ax.scatter(
            emb_2d[error_indices, 0], emb_2d[error_indices, 1],
            facecolors='none', edgecolors='black', marker='x',
            s=50, linewidth=1.0, label=f'Misclassified (n={len(error_indices)})',
            alpha=0.8
        )

    # 图表装饰
    ax.set_title(args.title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)

    # 移除刻度数字 (论文常用风格)
    ax.set_xticks([])
    ax.set_yticks([])

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 图例
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=10)

    plt.tight_layout()

    # 5. 保存
    plt.savefig(args.output, format='pdf', bbox_inches='tight')
    print(f"[*] Success! Saved to {args.output}")

    # 同时保存PNG版本
    png_path = args.output.replace('.pdf', '.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"[*] PNG version saved to {png_path}")

    plt.close()


if __name__ == "__main__":
    main()
