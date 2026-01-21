"""
Extended Validation of Two-Factor Framework
=============================================

在更多数据集上验证两因素框架
包括Platonov异质图数据集
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import (
    Planetoid, Amazon, WebKB, WikipediaNetwork,
    HeterophilousGraphDataset, Actor
)
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


def compute_homophily(data):
    """计算边同质性"""
    edge_index = to_undirected(data.edge_index)
    src, dst = edge_index.cpu().numpy()
    labels = data.y.cpu().numpy()
    same_label = (labels[src] == labels[dst]).mean()
    return same_label


def train_and_evaluate(model, x, edge_index, labels, train_mask, val_mask, test_mask,
                       lr=0.01, weight_decay=5e-4, epochs=200, patience=50):
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


def evaluate_dataset(data, n_runs=5):
    """在数据集上评估GCN和MLP"""
    x = data.x.to(device)
    edge_index = to_undirected(data.edge_index).to(device)
    labels = data.y.to(device)
    n_nodes = data.num_nodes
    n_features = data.num_features
    n_classes = len(labels.unique())

    gcn_results = []
    mlp_results = []

    for seed in range(n_runs):
        torch.manual_seed(seed)
        np.random.seed(seed)

        indices = np.arange(n_nodes)
        train_idx, temp_idx = train_test_split(indices, train_size=0.6, random_state=seed)
        val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=seed)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # GCN
        gcn = GCN(n_features, 64, n_classes).to(device)
        gcn_acc = train_and_evaluate(gcn, x, edge_index, labels, train_mask, val_mask, test_mask)
        gcn_results.append(gcn_acc)

        # MLP
        mlp = MLP(n_features, 64, n_classes).to(device)
        mlp_acc = train_and_evaluate(mlp, x, edge_index, labels, train_mask, val_mask, test_mask)
        mlp_results.append(mlp_acc)

    return {
        'gcn_mean': np.mean(gcn_results),
        'gcn_std': np.std(gcn_results),
        'mlp_mean': np.mean(mlp_results),
        'mlp_std': np.std(mlp_results),
        'gcn_mlp': np.mean(gcn_results) - np.mean(mlp_results)
    }


def main():
    print("=" * 80)
    print("EXTENDED VALIDATION OF TWO-FACTOR FRAMEWORK")
    print("=" * 80)

    # 数据集配置
    datasets_config = [
        # Planetoid
        ('Cora', Planetoid, {'name': 'Cora'}),
        ('CiteSeer', Planetoid, {'name': 'CiteSeer'}),
        ('PubMed', Planetoid, {'name': 'PubMed'}),

        # Amazon
        ('Computers', Amazon, {'name': 'Computers'}),
        ('Photo', Amazon, {'name': 'Photo'}),

        # WebKB
        ('Texas', WebKB, {'name': 'Texas'}),
        ('Wisconsin', WebKB, {'name': 'Wisconsin'}),
        ('Cornell', WebKB, {'name': 'Cornell'}),

        # Wikipedia
        ('Squirrel', WikipediaNetwork, {'name': 'Squirrel'}),
        ('Chameleon', WikipediaNetwork, {'name': 'Chameleon'}),

        # Heterophilous (Platonov)
        ('Roman-empire', HeterophilousGraphDataset, {'name': 'Roman-empire'}),
        ('Amazon-ratings', HeterophilousGraphDataset, {'name': 'Amazon-ratings'}),
        ('Minesweeper', HeterophilousGraphDataset, {'name': 'Minesweeper'}),
        ('Tolokers', HeterophilousGraphDataset, {'name': 'Tolokers'}),
        ('Questions', HeterophilousGraphDataset, {'name': 'Questions'}),

        # Actor
        ('Actor', Actor, {}),
    ]

    all_results = []

    for name, DatasetClass, kwargs in datasets_config:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")

        try:
            dataset = DatasetClass(root='./data', **kwargs)
            data = dataset[0]

            print(f"  Nodes: {data.num_nodes}, Features: {data.num_features}")
            print(f"  Classes: {len(data.y.unique())}")

            # 计算同质性
            h = compute_homophily(data)
            print(f"  Homophily: {h:.4f}")

            # 评估GCN和MLP
            print("  Training models...")
            result = evaluate_dataset(data, n_runs=5)

            print(f"  GCN: {result['gcn_mean']:.3f} +/- {result['gcn_std']:.3f}")
            print(f"  MLP: {result['mlp_mean']:.3f} +/- {result['mlp_std']:.3f}")
            print(f"  GCN-MLP: {result['gcn_mlp']:+.3f}")

            all_results.append({
                'dataset': name,
                'n_nodes': data.num_nodes,
                'n_features': data.num_features,
                'homophily': h,
                'mlp_mean': result['mlp_mean'],
                'gcn_mean': result['gcn_mean'],
                'gcn_mlp': result['gcn_mlp']
            })

        except Exception as e:
            print(f"  Error: {e}")

    # ========== 两因素框架验证 ==========
    print("\n" + "=" * 80)
    print("TWO-FACTOR FRAMEWORK VALIDATION")
    print("=" * 80)

    # 阈值
    fs_thresh = 0.65
    h_thresh = 0.5

    def predict(mlp_acc, h):
        if mlp_acc >= fs_thresh:
            if h >= h_thresh:
                return "GCN_maybe"
            else:
                return "MLP"
        else:
            if h >= h_thresh:
                return "GCN"
            else:
                return "Uncertain"

    correct = 0
    total = 0
    predictions = []

    print(f"\n{'Dataset':>15} {'MLP':>8} {'h':>8} {'Pred':>12} {'Actual':>8} {'Result':>8}")
    print("-" * 65)

    for r in all_results:
        pred = predict(r['mlp_mean'], r['homophily'])

        if r['gcn_mlp'] > 0.01:
            actual = "GCN"
        elif r['gcn_mlp'] < -0.01:
            actual = "MLP"
        else:
            actual = "Tie"

        # 评估预测
        if pred == "MLP":
            is_correct = actual in ["MLP", "Tie"]
            total += 1
            if is_correct:
                correct += 1
            result = "Y" if is_correct else "N"
        elif pred == "GCN":
            is_correct = actual == "GCN"
            total += 1
            if is_correct:
                correct += 1
            result = "Y" if is_correct else "N"
        elif pred == "GCN_maybe":
            is_correct = actual in ["GCN", "Tie"]
            total += 1
            if is_correct:
                correct += 1
            result = "Y" if is_correct else "N"
        else:
            result = "?"

        print(f"{r['dataset']:>15} {r['mlp_mean']:>8.3f} {r['homophily']:>8.3f} {pred:>12} {actual:>8} {result:>8}")

        predictions.append({
            'dataset': r['dataset'],
            'mlp_acc': r['mlp_mean'],
            'homophily': r['homophily'],
            'gcn_mlp': r['gcn_mlp'],
            'prediction': pred,
            'actual': actual,
            'correct': result == "Y"
        })

    print(f"\nOverall Accuracy: {correct}/{total} = {correct/total:.1%}")

    # ========== 四象限分析 ==========
    print("\n" + "=" * 80)
    print("QUADRANT ANALYSIS")
    print("=" * 80)

    q1 = [r for r in all_results if r['mlp_mean'] >= fs_thresh and r['homophily'] >= h_thresh]
    q2 = [r for r in all_results if r['mlp_mean'] >= fs_thresh and r['homophily'] < h_thresh]
    q3 = [r for r in all_results if r['mlp_mean'] < fs_thresh and r['homophily'] >= h_thresh]
    q4 = [r for r in all_results if r['mlp_mean'] < fs_thresh and r['homophily'] < h_thresh]

    print(f"\nQ1 (High FS, High h): {len(q1)} datasets")
    if q1:
        gcn_wins = sum(1 for r in q1 if r['gcn_mlp'] > 0.01)
        print(f"   GCN wins: {gcn_wins}/{len(q1)} = {gcn_wins/len(q1):.0%}")
        for r in q1:
            print(f"   - {r['dataset']}: GCN-MLP={r['gcn_mlp']:+.3f}")

    print(f"\nQ2 (High FS, Low h): {len(q2)} datasets")
    if q2:
        mlp_wins = sum(1 for r in q2 if r['gcn_mlp'] < -0.01)
        print(f"   MLP wins: {mlp_wins}/{len(q2)} = {mlp_wins/len(q2):.0%}")
        for r in q2:
            print(f"   - {r['dataset']}: GCN-MLP={r['gcn_mlp']:+.3f}")

    print(f"\nQ3 (Low FS, High h): {len(q3)} datasets")
    if q3:
        gcn_wins = sum(1 for r in q3 if r['gcn_mlp'] > 0.01)
        print(f"   GCN wins: {gcn_wins}/{len(q3)} = {gcn_wins/len(q3):.0%}")
        for r in q3:
            print(f"   - {r['dataset']}: GCN-MLP={r['gcn_mlp']:+.3f}")

    print(f"\nQ4 (Low FS, Low h): {len(q4)} datasets")
    if q4:
        for r in q4:
            print(f"   - {r['dataset']}: GCN-MLP={r['gcn_mlp']:+.3f}")

    # ========== 回归分析 ==========
    print("\n" + "=" * 80)
    print("REGRESSION ANALYSIS (Extended)")
    print("=" * 80)

    fs = np.array([r['mlp_mean'] for r in all_results])
    h = np.array([r['homophily'] for r in all_results])
    gcn_mlp = np.array([r['gcn_mlp'] for r in all_results])
    interaction = (1 - fs) * h

    X = np.column_stack([np.ones(len(fs)), fs, h, interaction])
    coeffs, residuals, rank, s = np.linalg.lstsq(X, gcn_mlp, rcond=None)

    y_pred = X @ coeffs
    ss_res = np.sum((gcn_mlp - y_pred) ** 2)
    ss_tot = np.sum((gcn_mlp - np.mean(gcn_mlp)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    print(f"\nMultiple Regression: GCN-MLP = b0 + b1*FS + b2*h + b3*(1-FS)*h")
    print(f"  Intercept (b0): {coeffs[0]:+.4f}")
    print(f"  FS coeff (b1):  {coeffs[1]:+.4f}")
    print(f"  h coeff (b2):   {coeffs[2]:+.4f}")
    print(f"  Interaction (b3): {coeffs[3]:+.4f}")
    print(f"\n  R-squared: {r_squared:.4f}")
    print(f"  N datasets: {len(all_results)}")

    # 保存结果
    output = {
        'n_datasets': len(all_results),
        'prediction_accuracy': correct / total if total > 0 else 0,
        'regression': {
            'intercept': float(coeffs[0]),
            'fs_coeff': float(coeffs[1]),
            'h_coeff': float(coeffs[2]),
            'interaction_coeff': float(coeffs[3]),
            'r_squared': float(r_squared)
        },
        'quadrants': {
            'Q1': len(q1),
            'Q2': len(q2),
            'Q3': len(q3),
            'Q4': len(q4)
        },
        'results': all_results,
        'predictions': predictions
    }

    with open('extended_validation_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to: extended_validation_results.json")


if __name__ == '__main__':
    main()
