"""
Spectral Diagnosis for Roman-empire Failure
=============================================

分析为什么Direction_Residual在Roman-empire上失败

核心假设：Roman-empire的标签信号主要在高频，GCN的低通滤波无法利用
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import json
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.datasets import HeterophilousGraphDataset, WebKB, WikipediaNetwork
from torch_geometric.utils import to_undirected, get_laplacian
from torch_geometric.nn import GCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def compute_spectral_energy(data, n_eigenvalues=100):
    """
    计算标签信号在图频谱上的能量分布

    返回：
    - low_freq_ratio: 低频能量占比 (前25%特征值)
    - high_freq_ratio: 高频能量占比 (后25%特征值)
    - mid_freq_ratio: 中频能量占比
    """
    edge_index = to_undirected(data.edge_index)
    n_nodes = data.num_nodes
    n_classes = len(data.y.unique())

    # 限制特征值数量避免内存问题
    n_eig = min(n_eigenvalues, n_nodes - 2)

    # 构建归一化拉普拉斯矩阵
    edge_index_lap, edge_weight = get_laplacian(
        edge_index,
        normalization='sym',
        num_nodes=n_nodes
    )

    # 转为稀疏矩阵
    row = edge_index_lap[0].numpy()
    col = edge_index_lap[1].numpy()
    data_lap = edge_weight.numpy()
    L = sparse.csr_matrix((data_lap, (row, col)), shape=(n_nodes, n_nodes))

    # 计算特征值和特征向量
    try:
        # 计算最小的n_eig个特征值（对应低频）
        eigenvalues_low, eigenvectors_low = eigsh(L, k=min(50, n_eig), which='SM')
        # 计算最大的n_eig个特征值（对应高频）
        eigenvalues_high, eigenvectors_high = eigsh(L, k=min(50, n_eig), which='LM')

        # 合并并排序
        eigenvalues = np.concatenate([eigenvalues_low, eigenvalues_high])
        eigenvectors = np.concatenate([eigenvectors_low, eigenvectors_high], axis=1)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

    except Exception as e:
        print(f"  Warning: Eigenvalue computation failed: {e}")
        return None

    # 将标签转为one-hot
    labels = data.y.numpy()
    label_onehot = np.zeros((n_nodes, n_classes))
    label_onehot[np.arange(n_nodes), labels] = 1

    # 计算标签信号在各频率上的能量
    n_eig_computed = len(eigenvalues)
    energies = []

    for i in range(n_eig_computed):
        # 标签信号在第i个特征向量上的投影
        proj = np.abs(eigenvectors[:, i].T @ label_onehot)
        energy = np.sum(proj ** 2)
        energies.append(energy)

    energies = np.array(energies)
    total_energy = np.sum(energies)

    if total_energy == 0:
        return None

    # 归一化
    energies_normalized = energies / total_energy

    # 划分频段
    n_low = n_eig_computed // 4
    n_high = n_eig_computed // 4

    low_freq_energy = np.sum(energies_normalized[:n_low])
    high_freq_energy = np.sum(energies_normalized[-n_high:])
    mid_freq_energy = np.sum(energies_normalized[n_low:-n_high])

    return {
        'low_freq_ratio': float(low_freq_energy),
        'high_freq_ratio': float(high_freq_energy),
        'mid_freq_ratio': float(mid_freq_energy),
        'n_eigenvalues': n_eig_computed,
        'eigenvalue_range': (float(eigenvalues[0]), float(eigenvalues[-1]))
    }


def k_hop_accuracy(data, max_k=8, n_runs=3):
    """
    测试不同层数GCN的性能
    """
    from torch_geometric.utils import to_undirected
    from sklearn.model_selection import train_test_split

    class MultiLayerGCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, n_layers):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(n_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
            x = self.convs[-1](x, edge_index)
            return x

    x = data.x.to(device)
    edge_index = to_undirected(data.edge_index).to(device)
    labels = data.y.to(device)
    n_nodes = data.num_nodes
    n_features = data.num_features
    n_classes = len(labels.unique())

    results = {}

    for k in range(1, max_k + 1):
        accs = []
        for run in range(n_runs):
            # 创建split
            indices = np.arange(n_nodes)
            train_idx, temp_idx = train_test_split(indices, train_size=0.6, random_state=42+run)
            val_idx, test_idx = train_test_split(temp_idx, train_size=0.5, random_state=42+run)

            train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

            # 训练
            model = MultiLayerGCN(n_features, 64, n_classes, k).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            best_val = 0
            best_test = 0
            patience = 0

            for epoch in range(200):
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

                    if val_acc > best_val:
                        best_val = val_acc
                        best_test = test_acc
                        patience = 0
                    else:
                        patience += 1
                        if patience >= 30:
                            break

            accs.append(best_test)

        results[k] = {
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs))
        }
        print(f"    K={k}: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

    return results


def main():
    print("=" * 80)
    print("SPECTRAL DIAGNOSIS FOR DIRECTION_RESIDUAL FAILURE")
    print("=" * 80)

    datasets_to_analyze = [
        ('Roman-empire', HeterophilousGraphDataset, {'name': 'Roman-empire'}),
        ('Texas', WebKB, {'name': 'Texas'}),
        ('Wisconsin', WebKB, {'name': 'Wisconsin'}),
        ('Squirrel', WikipediaNetwork, {'name': 'Squirrel'}),
        ('Chameleon', WikipediaNetwork, {'name': 'Chameleon'}),
    ]

    all_results = {}

    for name, DatasetClass, kwargs in datasets_to_analyze:
        print(f"\n{'='*60}")
        print(f"Analyzing: {name}")
        print(f"{'='*60}")

        try:
            dataset = DatasetClass(root='./data', **kwargs)
            data = dataset[0]
            print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")

            # 计算同质性
            edge_index = to_undirected(data.edge_index)
            src, dst = edge_index
            h = (data.y[src] == data.y[dst]).float().mean().item()
            print(f"  Homophily: {h:.3f}")

            # 频谱分析
            print(f"\n  Computing spectral energy distribution...")
            spectral = compute_spectral_energy(data)

            if spectral:
                print(f"  Low-freq energy:  {spectral['low_freq_ratio']:.3f}")
                print(f"  Mid-freq energy:  {spectral['mid_freq_ratio']:.3f}")
                print(f"  High-freq energy: {spectral['high_freq_ratio']:.3f}")

                # 判断信号类型
                if spectral['low_freq_ratio'] > 0.5:
                    signal_type = "LOW-PASS (GCN should help)"
                elif spectral['high_freq_ratio'] > 0.3:
                    signal_type = "HIGH-PASS (GCN will hurt)"
                else:
                    signal_type = "MIXED"
                print(f"  Signal type: {signal_type}")
            else:
                spectral = {'error': 'computation failed'}

            # K-hop扫描
            print(f"\n  K-hop accuracy scan...")
            k_results = k_hop_accuracy(data, max_k=6, n_runs=3)

            # 找最优K
            best_k = max(k_results, key=lambda k: k_results[k]['mean'])
            print(f"\n  Best K: {best_k} (acc={k_results[best_k]['mean']:.3f})")

            all_results[name] = {
                'homophily': h,
                'spectral': spectral,
                'k_hop_results': k_results,
                'best_k': best_k
            }

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 总结分析
    print("\n" + "=" * 80)
    print("SPECTRAL DIAGNOSIS SUMMARY")
    print("=" * 80)

    print(f"\n{'Dataset':>15} {'h':>8} {'Low-F':>8} {'High-F':>8} {'Best-K':>8} {'Signal':>15}")
    print("-" * 70)

    for name, result in all_results.items():
        if 'spectral' in result and 'low_freq_ratio' in result['spectral']:
            low_f = result['spectral']['low_freq_ratio']
            high_f = result['spectral']['high_freq_ratio']

            if low_f > 0.5:
                signal = "LOW-PASS"
            elif high_f > 0.3:
                signal = "HIGH-PASS"
            else:
                signal = "MIXED"

            print(f"{name:>15} {result['homophily']:>8.3f} {low_f:>8.3f} {high_f:>8.3f} "
                  f"{result['best_k']:>8} {signal:>15}")

    # 关键发现
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if 'Roman-empire' in all_results and 'Texas' in all_results:
        roman = all_results['Roman-empire']
        texas = all_results['Texas']

        if 'spectral' in roman and 'low_freq_ratio' in roman['spectral']:
            roman_low = roman['spectral']['low_freq_ratio']
            roman_high = roman['spectral']['high_freq_ratio']

            texas_low = texas['spectral']['low_freq_ratio'] if 'spectral' in texas and 'low_freq_ratio' in texas['spectral'] else 'N/A'
            texas_high = texas['spectral']['high_freq_ratio'] if 'spectral' in texas and 'high_freq_ratio' in texas['spectral'] else 'N/A'

            print(f"""
Roman-empire vs Texas Comparison:
---------------------------------
                    Roman-empire    Texas
Homophily:          {roman['homophily']:.3f}           {texas['homophily']:.3f}
Low-freq energy:    {roman_low:.3f}           {texas_low:.3f if isinstance(texas_low, float) else texas_low}
High-freq energy:   {roman_high:.3f}           {texas_high:.3f if isinstance(texas_high, float) else texas_high}
Best K:             {roman['best_k']}               {texas['best_k']}

INTERPRETATION:
""")
            if roman_high > 0.3 and (isinstance(texas_high, float) and texas_high < 0.3):
                print("""
- Roman-empire has HIGH-FREQUENCY label signal
- Texas has LOW/MIXED frequency signal
- This explains why Direction_Residual fails on Roman-empire:
  * The metric detects "structure has information" (positive residual)
  * But GCN's low-pass filter cannot utilize high-frequency information
  * Result: GCN performs worse than MLP despite positive residual

RECOMMENDATION:
- Redefine Direction_Residual as "homophily signal detector"
- Add frequency analysis as complementary diagnostic
- For high-frequency graphs (Roman-empire), recommend H2GCN/GPRGNN instead of GCN
""")
            elif roman_high <= 0.3:
                print("""
- Roman-empire does NOT have predominantly high-frequency signal
- The failure may be due to other factors:
  * Feature quality
  * Training instability
  * Need for longer-range propagation

RECOMMENDATION:
- Investigate feature quality
- Try APPNP with different alpha values
- Consider structural role embeddings
""")

    # 保存结果
    with open('spectral_diagnosis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\nResults saved to: spectral_diagnosis_results.json")


if __name__ == '__main__':
    main()
