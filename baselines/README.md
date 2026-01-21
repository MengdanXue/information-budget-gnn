# 2024 SOTA Baseline Comparison Experiments

## Overview

This directory contains the implementation and comparison scripts for evaluating FSD-GNN against 2024 SOTA fraud detection baselines.

## Baseline Methods

### 1. ARC (NeurIPS 2024)
**Paper:** ARC: A Generalist Graph Anomaly Detector with In-Context Learning
**Authors:** Liu et al.
**GitHub:** https://github.com/yixinliu233/ARC
**Key Features:**
- Smoothness-based feature alignment
- Ego-neighbor residual encoding
- In-context learning with few-shot samples

### 2. GAGA (WWW 2023)
**Paper:** Label Information Enhanced Fraud Detection against Low Homophily in Graphs
**Authors:** Wang et al.
**GitHub:** https://github.com/Orion-wyc/GAGA
**Key Features:**
- Group aggregation for low homophily
- Learnable group encodings
- Transformer-based architecture

### 3. CARE-GNN (CIKM 2020)
**Paper:** Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
**Authors:** Dou et al.
**GitHub:** https://github.com/YingtongDou/CARE-GNN
**Key Features:**
- Similarity-based neighbor selection
- Reinforced label propagation
- Multi-relation support

### 4. PC-GNN (WWW 2021)
**Paper:** Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
**Authors:** Liu et al.
**GitHub:** https://github.com/PonderLY/PC-GNN
**Key Features:**
- Adaptive neighbor selection (pick and choose)
- Handles class imbalance
- Subgraph sampling

### 5. VecAug (KDD 2024)
**Paper:** VecAug: Unveiling Camouflaged Frauds with Cohort Augmentation for Enhanced Detection
**Authors:** Xiao et al.
**Note:** No official code available, implemented based on paper description

### 6. SEFraud (KDD 2024)
**Paper:** SEFraud: Graph-based Self-Explainable Fraud Detection via Interpretative Mask Learning
**Authors:** Li et al.
**Note:** No official code available, implemented based on paper description

## Datasets

### 1. IEEE-CIS Fraud Detection
- **Source:** Kaggle IEEE-CIS Fraud Detection Competition
- **Nodes:** ~600K transactions
- **Features:** Transaction and identity features
- **Task:** Binary classification (fraud/legitimate)

### 2. YelpChi (Yelp Chicago)
- **Source:** DGL/PyG datasets
- **Nodes:** ~45K reviews
- **Features:** 32-dim or 100-dim review features
- **Task:** Binary classification (fake/real reviews)

### 3. Amazon Product Reviews
- **Source:** DGL/PyG datasets
- **Nodes:** ~11K reviews
- **Features:** Review features
- **Task:** Binary classification (fake/real reviews)

### 4. Elliptic Bitcoin Transactions
- **Source:** Elliptic dataset (Weber et al.)
- **Nodes:** ~200K transactions
- **Features:** 166 transaction features
- **Task:** Binary classification (illicit/licit)

## Installation

### Quick Start (CPU)
```bash
cd baselines
pip install -r requirements_baselines.txt
```

### GPU Support (CUDA 12.1)
```bash
# Install PyTorch with CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install PyG extensions
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install torch-geometric==2.3.1

# Install other dependencies
pip install -r requirements_baselines.txt
```

### DGL Installation (for YelpChi/Amazon datasets)
```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

## Usage

### Run Single Baseline on Single Dataset
```bash
python run_baselines.py --model ARC --dataset ieee-cis --seeds 42 123 456
```

### Run Multiple Baselines on Multiple Datasets
```bash
python run_baselines.py --model ARC GAGA CARE-GNN --dataset ieee-cis yelpchi --seeds 42 123 456 789 1024
```

### Run All Baselines on All Datasets (Full Comparison)
```bash
python run_baselines.py --model all --dataset all --seeds 42 123 456 789 1024 2048 3072 4096 5120 6144
```

### Run FSD-GNN Models
```bash
python run_baselines.py --model DAAA DAAAv2 DAAAv3 DAAAv4 --dataset ieee-cis --seeds 42 123 456
```

### Custom Hyperparameters
```bash
python run_baselines.py --model ARC --dataset yelpchi --hidden_dim 256 --dropout 0.3 --seeds 42 123 456
```

## File Structure

```
baselines/
├── README.md                      # This file
├── requirements_baselines.txt     # Python dependencies
├── baseline_models.py             # Baseline model implementations
├── data_loaders.py                # Dataset loaders
├── run_baselines.py               # Main experiment script
└── results/                       # Experiment results (auto-created)
    └── baseline_comparison_*.json
```

## Output Format

Results are saved in JSON format with the following structure:

```json
{
  "ieee-cis": {
    "ARC": {
      "auc_mean": 0.8523,
      "auc_std": 0.0145,
      "ap_mean": 0.7234,
      "ap_std": 0.0198,
      "f1_mean": 0.6891,
      "f1_std": 0.0167,
      "precision_mean": 0.7123,
      "recall_mean": 0.6701,
      "avg_epochs": 89.3,
      "raw_results": {
        "auc": [0.8456, 0.8590, ...],
        "f1": [0.6723, 0.7059, ...],
        ...
      }
    },
    "GAGA": { ... },
    ...
  },
  "yelpchi": { ... },
  ...
}
```

## Performance Metrics

All models are evaluated using:
- **AUC-ROC:** Area under ROC curve
- **AP:** Average Precision (AUC-PR)
- **F1 Score:** Harmonic mean of precision and recall
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)

## Notes

### VecAug and SEFraud
Since VecAug (KDD 2024) and SEFraud (KDD 2024) do not have publicly available implementations, we provide approximations based on their paper descriptions:

- **VecAug:** Approximated using cohort-based feature augmentation and neighbor aggregation
- **SEFraud:** Approximated using learnable feature/edge masks for interpretability

These approximations capture the core ideas but may not exactly replicate the original performance.

### Data Availability
- **IEEE-CIS:** Must be preprocessed from Kaggle dataset (see `../ieee_cis_graph_builder.py`)
- **YelpChi/Amazon:** Auto-downloaded via DGL if not available locally
- **Elliptic:** Must be preprocessed and placed in `../data/` directory

### GPU Memory
Some models (especially ARC and GAGA) may require significant GPU memory for large graphs. If you encounter OOM errors:
1. Reduce batch size (use mini-batch training)
2. Reduce hidden dimension: `--hidden_dim 64`
3. Use CPU: The script auto-detects CUDA availability

### Reproducibility
All experiments use fixed random seeds for reproducibility. Default seeds: `[42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]`

## Citation

If you use this code, please cite the FSD-GNN paper and the respective baseline papers:

```bibtex
@inproceedings{liu2024arc,
  title={ARC: A Generalist Graph Anomaly Detector with In-Context Learning},
  author={Liu, Yixin and Li, Shiyuan and Zheng, Yu and Chen, Qingfeng and Zhang, Chengqi and Pan, Shirui},
  booktitle={NeurIPS},
  year={2024}
}

@inproceedings{wang2023gaga,
  title={Label Information Enhanced Fraud Detection against Low Homophily in Graphs},
  author={Wang, Yuchen and others},
  booktitle={WWW},
  year={2023}
}

@inproceedings{dou2020care,
  title={Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters},
  author={Dou, Yingtong and others},
  booktitle={CIKM},
  year={2020}
}

@inproceedings{liu2021pc,
  title={Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection},
  author={Liu, Yang and others},
  booktitle={WWW},
  year={2021}
}
```

## License

This code is for research purposes only. Please refer to individual baseline repositories for their licensing terms.

## Contact

For questions or issues, please open an issue in the repository or contact the authors.
