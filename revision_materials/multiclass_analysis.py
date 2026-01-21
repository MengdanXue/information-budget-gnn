"""
SPI Multi-class Analysis
========================

分析SPI在多分类场景下的表现。
论文中的Theorem 1假设binary classification，需要验证多分类场景的泛化性。
"""

import json
import numpy as np

# 从已有实验中提取多分类数据
datasets_info = {
    # Binary or near-binary
    'Elliptic': {'classes': 2, 'h': 0.79, 'spi_correct': True},

    # Multi-class datasets
    'Texas': {'classes': 5, 'h': 0.09, 'spi_correct': False, 'note': 'Q2 quadrant'},
    'Wisconsin': {'classes': 5, 'h': 0.19, 'spi_correct': False, 'note': 'Q2 quadrant'},
    'Cornell': {'classes': 5, 'h': 0.13, 'spi_correct': False, 'note': 'Q2 quadrant'},
    'Cora': {'classes': 7, 'h': 0.81, 'spi_correct': True},
    'CiteSeer': {'classes': 6, 'h': 0.74, 'spi_correct': True},
    'PubMed': {'classes': 3, 'h': 0.80, 'spi_correct': True},
    'Roman-empire': {'classes': 18, 'h': 0.05, 'spi_correct': False, 'note': 'Q2 quadrant'},
    'Amazon-Computers': {'classes': 10, 'h': 0.78, 'spi_correct': True},
    'Amazon-Photo': {'classes': 8, 'h': 0.83, 'spi_correct': True},
    'Coauthor-CS': {'classes': 15, 'h': 0.81, 'spi_correct': True},
    'Coauthor-Physics': {'classes': 5, 'h': 0.93, 'spi_correct': True},
    'ogbn-arxiv': {'classes': 40, 'h': 0.66, 'spi_correct': True},
}

print("=" * 70)
print("SPI Performance Analysis by Number of Classes")
print("=" * 70)

# Group by class count
binary = [(k, v) for k, v in datasets_info.items() if v['classes'] == 2]
few_class = [(k, v) for k, v in datasets_info.items() if 3 <= v['classes'] <= 7]
many_class = [(k, v) for k, v in datasets_info.items() if v['classes'] > 7]

print("\n1. Binary Classification (2 classes):")
print("-" * 50)
for name, info in binary:
    status = "Y" if info['spi_correct'] else "N"
    print(f"   {name}: h={info['h']:.2f}, SPI correct: {status}")
if binary:
    acc = sum(1 for _, v in binary if v['spi_correct']) / len(binary)
    print(f"   Accuracy: {acc*100:.0f}%")

print("\n2. Few-class Classification (3-7 classes):")
print("-" * 50)
for name, info in few_class:
    status = "Y" if info['spi_correct'] else "N"
    note = f" ({info.get('note', '')})" if info.get('note') else ""
    print(f"   {name}: {info['classes']} classes, h={info['h']:.2f}, SPI correct: {status}{note}")
if few_class:
    acc = sum(1 for _, v in few_class if v['spi_correct']) / len(few_class)
    print(f"   Accuracy: {acc*100:.0f}%")

print("\n3. Many-class Classification (>7 classes):")
print("-" * 50)
for name, info in many_class:
    status = "Y" if info['spi_correct'] else "N"
    note = f" ({info.get('note', '')})" if info.get('note') else ""
    print(f"   {name}: {info['classes']} classes, h={info['h']:.2f}, SPI correct: {status}{note}")
if many_class:
    acc = sum(1 for _, v in many_class if v['spi_correct']) / len(many_class)
    print(f"   Accuracy: {acc*100:.0f}%")

# Overall analysis
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# By class count
all_datasets = list(datasets_info.items())
high_h = [(k, v) for k, v in all_datasets if v['h'] > 0.5]
low_h = [(k, v) for k, v in all_datasets if v['h'] <= 0.5]

print("\nBy Homophily Region:")
print(f"  High-h (h>0.5): {sum(1 for _, v in high_h if v['spi_correct'])}/{len(high_h)} correct")
print(f"  Low-h (h≤0.5): {sum(1 for _, v in low_h if v['spi_correct'])}/{len(low_h)} correct")

print("\nBy Number of Classes:")
if binary:
    print(f"  Binary (2): {sum(1 for _, v in binary if v['spi_correct'])}/{len(binary)} correct")
if few_class:
    print(f"  Few (3-7): {sum(1 for _, v in few_class if v['spi_correct'])}/{len(few_class)} correct")
if many_class:
    print(f"  Many (>7): {sum(1 for _, v in many_class if v['spi_correct'])}/{len(many_class)} correct")

print("\n" + "=" * 70)
print("KEY FINDING")
print("=" * 70)
print("""
SPI performance does NOT degrade with more classes.
The key factor is homophily level, not number of classes:

- High-h datasets: SPI works well regardless of class count
  (ogbn-arxiv with 40 classes still correctly predicted)

- Low-h datasets: SPI fails due to Feature-Pattern Duality,
  not due to multi-class complexity

This suggests Theorem 1's binary assumption is a sufficient
approximation---the mutual information principle extends
naturally to multi-class settings.
""")
