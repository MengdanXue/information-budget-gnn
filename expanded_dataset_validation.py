"""
Expanded Dataset Validation (30+ Datasets)
==========================================

扩展验证Two-Factor Framework到30+数据集，回应三AI审稿人的关键建议。

数据集来源：
1. PyTorch Geometric内置数据集
2. Heterophilous Graph Datasets (Platonov et al. 2023)
3. OGB (Open Graph Benchmark)
4. 其他常用benchmark

目标：
- 扩展到30+数据集
- 每个象限至少8个样本
- 验证Two-Factor Framework的泛化能力
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import (
    Planetoid, WebKB, WikipediaNetwork, Actor,
    Amazon, Coauthor, CitationFull
)
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split

# 尝试导入更多数据集
try:
    from torch_geometric.datasets import HeterophilousGraphDataset
    HAS_HETEROPHILOUS = True
except ImportError:
    HAS_HETEROPHILOUS = False
    print("Warning: HeterophilousGraphDataset not available")

try:
    from ogb.nodeproppred import PygNodePropPredDataset
    HAS_OGB = True
except ImportError:
    HAS_OGB = False
    print("Warning: OGB not available, skipping large-scale datasets")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def compute_homophily(edge_index, labels):
    """Compute edge homophily"""
    src, dst = edge_index.cpu().numpy()
    lab = labels.cpu().numpy()
    same = lab[src] == lab[dst]
    return same.mean()


def train_and_evaluate(model, x, edge_index, labels, train_mask, val_mask, test_mask,
                       lr=0.01, weight_decay=5e-4, epochs=200, patience=50):
    """Train and evaluate model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(dim=1)
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    return best_test_acc


def evaluate_dataset(data, name, n_runs=5):
    """Evaluate MLP and GCN on a dataset"""

    x = data.x.to(device)
    edge_index = to_undirected(data.edge_index).to(device)
    labels = data.y.to(device)

    # Handle multi-dimensional labels
    if labels.dim() > 1:
        labels = labels.squeeze()

    n_nodes = data.num_nodes
    n_features = data.num_features
    n_classes = len(labels.unique())

    # Skip if too few classes or nodes
    if n_classes < 2 or n_nodes < 50:
        return None

    h = compute_homophily(edge_index, labels)

    mlp_scores = []
    gcn_scores = []

    for seed in range(n_runs):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create splits
        indices = np.arange(n_nodes)
        train_idx, temp_idx = train_test_split(indices, train_size=0.6, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=seed)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # MLP
        mlp = MLP(n_features, 64, n_classes).to(device)
        mlp_acc = train_and_evaluate(mlp, x, edge_index, labels, train_mask, val_mask, test_mask)
        mlp_scores.append(mlp_acc)

        # GCN
        gcn = GCN(n_features, 64, n_classes).to(device)
        gcn_acc = train_and_evaluate(gcn, x, edge_index, labels, train_mask, val_mask, test_mask)
        gcn_scores.append(gcn_acc)

    mlp_mean = np.mean(mlp_scores)
    gcn_mean = np.mean(gcn_scores)
    delta = gcn_mean - mlp_mean

    # Statistical test
    t_stat, p_value = stats.ttest_rel(gcn_scores, mlp_scores)

    return {
        'name': name,
        'n_nodes': n_nodes,
        'n_features': n_features,
        'n_classes': n_classes,
        'homophily': h,
        'mlp_mean': mlp_mean,
        'mlp_std': np.std(mlp_scores),
        'gcn_mean': gcn_mean,
        'gcn_std': np.std(gcn_scores),
        'delta': delta,
        't_stat': t_stat,
        'p_value': p_value,
        'winner': 'GCN' if delta > 0.01 else ('MLP' if delta < -0.01 else 'Tie')
    }


def load_all_datasets():
    """Load all available datasets"""
    datasets = []

    # 1. Planetoid datasets
    print("Loading Planetoid datasets...")
    for name in ['Cora', 'CiteSeer', 'PubMed']:
        try:
            dataset = Planetoid(root='./data', name=name)
            datasets.append((name, dataset[0]))
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    # 2. WebKB datasets
    print("Loading WebKB datasets...")
    for name in ['Cornell', 'Texas', 'Wisconsin']:
        try:
            dataset = WebKB(root='./data', name=name)
            datasets.append((name, dataset[0]))
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    # 3. Wikipedia networks
    print("Loading Wikipedia datasets...")
    for name in ['Chameleon', 'Squirrel']:
        try:
            dataset = WikipediaNetwork(root='./data', name=name)
            datasets.append((name, dataset[0]))
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    # 4. Actor dataset
    print("Loading Actor dataset...")
    try:
        dataset = Actor(root='./data')
        datasets.append(('Actor', dataset[0]))
    except Exception as e:
        print(f"  Failed to load Actor: {e}")

    # 5. Amazon datasets
    print("Loading Amazon datasets...")
    for name in ['Computers', 'Photo']:
        try:
            dataset = Amazon(root='./data', name=name)
            datasets.append((f'Amazon-{name}', dataset[0]))
        except Exception as e:
            print(f"  Failed to load Amazon-{name}: {e}")

    # 6. Coauthor datasets
    print("Loading Coauthor datasets...")
    for name in ['CS', 'Physics']:
        try:
            dataset = Coauthor(root='./data', name=name)
            datasets.append((f'Coauthor-{name}', dataset[0]))
        except Exception as e:
            print(f"  Failed to load Coauthor-{name}: {e}")

    # 7. CitationFull datasets
    print("Loading CitationFull datasets...")
    for name in ['Cora_ML', 'DBLP']:
        try:
            dataset = CitationFull(root='./data', name=name)
            datasets.append((name, dataset[0]))
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    # 8. Heterophilous datasets (if available)
    if HAS_HETEROPHILOUS:
        print("Loading Heterophilous datasets...")
        for name in ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions']:
            try:
                dataset = HeterophilousGraphDataset(root='./data', name=name)
                datasets.append((name, dataset[0]))
            except Exception as e:
                print(f"  Failed to load {name}: {e}")

    # 9. OGB datasets (if available) - 大规模图
    if HAS_OGB:
        print("Loading OGB datasets...")
        for name in ['ogbn-arxiv']:
            try:
                dataset = PygNodePropPredDataset(name=name, root='./data')
                data = dataset[0]
                # OGB labels need special handling
                data.y = data.y.squeeze()
                datasets.append((name, data))
            except Exception as e:
                print(f"  Failed to load {name}: {e}")

    print(f"\nTotal datasets loaded: {len(datasets)}")
    return datasets


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
            return 'Q4', 'Uncertain'


def main():
    print("=" * 80)
    print("EXPANDED DATASET VALIDATION (Target: 30+ Datasets)")
    print("=" * 80)

    # Load all datasets
    datasets = load_all_datasets()

    # Evaluate each dataset
    results = []
    print("\n" + "=" * 80)
    print("EVALUATING DATASETS")
    print("=" * 80)

    for name, data in datasets:
        print(f"\nEvaluating {name}...", end=" ", flush=True)
        try:
            result = evaluate_dataset(data, name, n_runs=5)
            if result:
                results.append(result)
                print(f"h={result['homophily']:.3f}, MLP={result['mlp_mean']:.3f}, "
                      f"GCN={result['gcn_mean']:.3f}, Delta={result['delta']:+.3f}")
            else:
                print("Skipped (too few classes/nodes)")
        except Exception as e:
            print(f"Error: {e}")

    # Classification and analysis
    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY ({len(results)} datasets)")
    print("=" * 80)

    # Classify into quadrants
    quadrants = defaultdict(list)
    for r in results:
        quadrant, prediction = classify_quadrant(r['mlp_mean'], r['homophily'])
        r['quadrant'] = quadrant
        r['prediction'] = prediction
        r['correct'] = (
            (prediction == 'GCN_maybe' and r['winner'] in ['GCN', 'Tie']) or
            (prediction == 'MLP' and r['winner'] == 'MLP') or
            (prediction == 'GCN' and r['winner'] == 'GCN') or
            (prediction == 'Uncertain')
        )
        quadrants[quadrant].append(r)

    # Print by quadrant
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        print(f"\n{q} Quadrant ({len(quadrants[q])} datasets):")
        print("-" * 70)
        for r in quadrants[q]:
            status = "OK" if r['correct'] else "WRONG"
            print(f"  {r['name']:20} h={r['homophily']:.3f} MLP={r['mlp_mean']:.3f} "
                  f"Delta={r['delta']:+.3f} Pred={r['prediction']:10} Actual={r['winner']:4} [{status}]")

    # Calculate accuracy
    print("\n" + "=" * 80)
    print("ACCURACY BY QUADRANT")
    print("=" * 80)

    total_correct = 0
    total_decisive = 0

    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_results = quadrants[q]
        if not q_results:
            continue

        if q == 'Q4':  # Uncertain quadrant
            print(f"\n{q}: {len(q_results)} datasets (Uncertain - no prediction)")
        else:
            correct = sum(1 for r in q_results if r['correct'])
            total = len(q_results)
            total_correct += correct
            total_decisive += total
            acc = correct / total * 100 if total > 0 else 0
            print(f"\n{q}: {correct}/{total} = {acc:.1f}%")

    overall_acc = total_correct / total_decisive * 100 if total_decisive > 0 else 0
    print(f"\nOverall (decisive): {total_correct}/{total_decisive} = {overall_acc:.1f}%")

    # SPI analysis
    print("\n" + "=" * 80)
    print("SPI TRUST REGION ANALYSIS")
    print("=" * 80)

    trust_region = [r for r in results if abs(2*r['homophily']-1) > 0.67]
    uncertain_region = [r for r in results if abs(2*r['homophily']-1) <= 0.67]

    print(f"\nTrust Region (SPI > 0.67): {len(trust_region)} datasets")
    trust_correct = sum(1 for r in trust_region if
                       (r['homophily'] > 0.5 and r['winner'] in ['GCN', 'Tie']) or
                       (r['homophily'] < 0.5 and r['winner'] in ['MLP', 'Tie']))
    print(f"  Correct predictions: {trust_correct}/{len(trust_region)}")

    print(f"\nUncertain Region (SPI <= 0.67): {len(uncertain_region)} datasets")

    # Save results
    output = {
        'n_datasets': len(results),
        'results': results,
        'summary': {
            'total_correct': total_correct,
            'total_decisive': total_decisive,
            'overall_accuracy': overall_acc,
            'trust_region_count': len(trust_region),
            'uncertain_region_count': len(uncertain_region)
        },
        'quadrant_counts': {q: len(quadrants[q]) for q in ['Q1', 'Q2', 'Q3', 'Q4']}
    }

    with open('expanded_validation_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\nResults saved to: expanded_validation_results.json")

    # Final summary table
    print("\n" + "=" * 80)
    print("TABLE FOR PAPER")
    print("=" * 80)
    print("\n| Dataset | Nodes | h | MLP | GCN-MLP | Quadrant | Pred | Actual |")
    print("|---------|-------|-----|-----|---------|----------|------|--------|")
    for r in sorted(results, key=lambda x: x['homophily']):
        print(f"| {r['name'][:15]:15} | {r['n_nodes']:5} | {r['homophily']:.2f} | "
              f"{r['mlp_mean']:.2f} | {r['delta']:+.2f} | {r['quadrant']} | "
              f"{r['prediction'][:4]} | {r['winner'][:3]} |")

    return results


if __name__ == '__main__':
    results = main()
