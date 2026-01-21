"""
Ablation Study V2: SPI-Guided Fusion with Optimized Threshold
=============================================================

Key insight from V1 results:
- GCN wins when SPI > ~0.4 (not 0.67)
- τ should be lower to better match the actual boundary

This version uses:
- τ = 0.4 (updated based on experimental observation)
- Temperature T = 0.15 (slightly softer transition)

Author: Thesis Research
Date: 2024-12
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# Import our modules
from spi_guided_gating import (
    SPIGuidedGNN,
    StructureOnlyGNN,
    FeatureOnlyMLP,
    NaiveFusionGNN,
    compute_edge_homophily,
    compute_spi,
)

# Set random seeds for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ExperimentConfig:
    """Configuration for ablation experiments."""
    # Graph parameters
    num_nodes: int = 1000
    num_features: int = 20
    num_classes: int = 2
    avg_degree: int = 15

    # Model parameters
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.5

    # Training parameters
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 20

    # Experiment parameters
    num_seeds: int = 5
    h_values: List[float] = None

    # SPI-Guided parameters (UPDATED)
    tau: float = 0.4  # Updated from 0.67 based on V1 results
    T_init: float = 0.15  # Slightly smoother

    def __post_init__(self):
        if self.h_values is None:
            self.h_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def generate_synthetic_graph(
    h_target: float,
    num_nodes: int,
    num_features: int,
    avg_degree: int,
    feature_separability: float = 0.5,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic graph with controlled homophily.
    """
    np.random.seed(seed)

    # Generate balanced labels
    y = np.array([0] * (num_nodes // 2) + [1] * (num_nodes - num_nodes // 2))
    np.random.shuffle(y)
    y = torch.tensor(y, dtype=torch.long)

    # Generate features from class-conditional Gaussians
    x = np.zeros((num_nodes, num_features))
    for c in range(2):
        mask = y.numpy() == c
        center = np.zeros(num_features)
        center[c * (num_features // 2):(c + 1) * (num_features // 2)] = feature_separability
        x[mask] = np.random.randn(mask.sum(), num_features) + center
    x = torch.tensor(x, dtype=torch.float)

    # Generate edges with target homophily
    num_edges = num_nodes * avg_degree // 2
    num_same_class = int(num_edges * h_target)
    num_diff_class = num_edges - num_same_class

    edges = []

    # Same-class edges
    for c in range(2):
        class_nodes = np.where(y.numpy() == c)[0]
        for _ in range(num_same_class // 2):
            i, j = np.random.choice(class_nodes, 2, replace=False)
            edges.append((i, j))
            edges.append((j, i))

    # Different-class edges
    class_0 = np.where(y.numpy() == 0)[0]
    class_1 = np.where(y.numpy() == 1)[0]
    for _ in range(num_diff_class):
        i = np.random.choice(class_0)
        j = np.random.choice(class_1)
        edges.append((i, j))
        edges.append((j, i))

    edges = list(set(edges))
    edges = [(i, j) for i, j in edges if i != j]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return x, edge_index, y


def train_epoch(model, x, edge_index, y, train_mask, optimizer, spi=None):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()

    if hasattr(model, 'gating'):  # SPI-Guided model
        out, _ = model(x, edge_index, spi=spi, labels=y)
    else:
        out = model(x, edge_index)

    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, x, edge_index, y, mask, spi=None):
    """Evaluate model accuracy."""
    model.eval()

    if hasattr(model, 'gating'):
        out, info = model(x, edge_index, spi=spi, labels=y)
    else:
        out = model(x, edge_index)
        info = {}

    pred = out[mask].argmax(dim=1)
    correct = (pred == y[mask]).sum().item()
    acc = correct / mask.sum().item()

    return acc, info


def run_single_experiment(
    h: float,
    config: ExperimentConfig,
    seed: int
) -> Dict:
    """Run single experiment comparing all models at given h."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate data
    x, edge_index, y = generate_synthetic_graph(
        h_target=h,
        num_nodes=config.num_nodes,
        num_features=config.num_features,
        avg_degree=config.avg_degree,
        feature_separability=0.5,
        seed=seed
    )

    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)

    # Compute actual homophily and SPI
    actual_h = compute_edge_homophily(edge_index, y)
    spi = compute_spi(actual_h)
    spi_tensor = torch.tensor(spi, device=device)

    # Train/val/test split (60/20/20)
    num_nodes = x.size(0)
    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    train_mask[perm[:int(0.6 * num_nodes)]] = True
    val_mask[perm[int(0.6 * num_nodes):int(0.8 * num_nodes)]] = True
    test_mask[perm[int(0.8 * num_nodes):]] = True

    results = {'h': h, 'actual_h': actual_h, 'spi': spi, 'seed': seed}

    # Define models - key change: use updated tau
    models = {
        'GCN': StructureOnlyGNN(
            config.num_features, config.hidden_channels, config.num_classes,
            config.num_layers, config.dropout, gnn_type='gcn'
        ),
        'MLP': FeatureOnlyMLP(
            config.num_features, config.hidden_channels, config.num_classes,
            config.num_layers, config.dropout
        ),
        'NaiveFusion': NaiveFusionGNN(
            config.num_features, config.hidden_channels, config.num_classes,
            config.num_layers, config.dropout, fusion_weight=0.5
        ),
        'SPI-Guided': SPIGuidedGNN(
            config.num_features, config.hidden_channels, config.num_classes,
            config.num_layers, config.dropout, gnn_type='gcn',
            tau=config.tau, T_init=config.T_init, learnable_T=True  # Use config values
        )
    }

    # Train each model
    for model_name, model in models.items():
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                      weight_decay=config.weight_decay)

        best_val_acc = 0
        patience_counter = 0
        best_state = None

        for epoch in range(config.epochs):
            if model_name == 'SPI-Guided':
                train_epoch(model, x, edge_index, y, train_mask, optimizer, spi=spi_tensor)
                val_acc, _ = evaluate(model, x, edge_index, y, val_mask, spi=spi_tensor)
            else:
                train_epoch(model, x, edge_index, y, train_mask, optimizer)
                val_acc, _ = evaluate(model, x, edge_index, y, val_mask)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        if model_name == 'SPI-Guided':
            test_acc, info = evaluate(model, x, edge_index, y, test_mask, spi=spi_tensor)
            results[f'{model_name}_beta'] = info.get('beta', 0.5)
            # Also record gating parameters
            if hasattr(model, 'gating'):
                params = model.gating.get_interpretable_params()
                results['learned_tau'] = params['tau']
                results['learned_T'] = params['T']
        else:
            test_acc, _ = evaluate(model, x, edge_index, y, test_mask)

        results[model_name] = test_acc

    # Compute advantages
    results['SPI-Guided_vs_GCN'] = results['SPI-Guided'] - results['GCN']
    results['SPI-Guided_vs_MLP'] = results['SPI-Guided'] - results['MLP']
    results['SPI-Guided_vs_Naive'] = results['SPI-Guided'] - results['NaiveFusion']

    # Also compute oracle (best possible selection)
    results['Oracle'] = max(results['GCN'], results['MLP'])
    results['SPI-Guided_vs_Oracle'] = results['SPI-Guided'] - results['Oracle']

    return results


def run_ablation_study(config: ExperimentConfig) -> List[Dict]:
    """Run complete ablation study across all h values and seeds."""
    all_results = []

    for h in tqdm(config.h_values, desc="H values"):
        for seed in range(config.num_seeds):
            result = run_single_experiment(h, config, seed)
            all_results.append(result)

    return all_results


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results across seeds."""
    aggregated = {}

    h_values = sorted(list(set(r['h'] for r in results)))

    for h in h_values:
        h_results = [r for r in results if r['h'] == h]

        agg = {
            'h': h,
            'spi': np.mean([r['spi'] for r in h_results]),
            'n': len(h_results)
        }

        for model in ['GCN', 'MLP', 'NaiveFusion', 'SPI-Guided', 'Oracle']:
            accs = [r[model] for r in h_results]
            agg[f'{model}_mean'] = np.mean(accs)
            agg[f'{model}_std'] = np.std(accs)

        for adv in ['SPI-Guided_vs_GCN', 'SPI-Guided_vs_MLP', 'SPI-Guided_vs_Naive', 'SPI-Guided_vs_Oracle']:
            advs = [r[adv] for r in h_results]
            agg[f'{adv}_mean'] = np.mean(advs)
            agg[f'{adv}_std'] = np.std(advs)

        # Record learned parameters
        if 'learned_T' in h_results[0]:
            agg['learned_T_mean'] = np.mean([r['learned_T'] for r in h_results])
            agg['learned_beta_mean'] = np.mean([r.get('SPI-Guided_beta', 0.5) for r in h_results])

        aggregated[h] = agg

    return aggregated


def plot_ablation_results(aggregated: Dict, save_dir: Path):
    """Create publication-quality ablation figures."""
    save_dir.mkdir(exist_ok=True)

    h_values = sorted(aggregated.keys())

    # Figure 1: Model Comparison with Oracle
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Accuracy vs H
    ax1 = axes[0]
    for model, color, marker, ls in [
        ('GCN', '#E94F37', 'o', '-'),
        ('MLP', '#3D5A80', 's', '-'),
        ('NaiveFusion', '#F7B801', '^', '--'),
        ('SPI-Guided', '#44AF69', 'D', '-'),
        ('Oracle', '#9B59B6', '*', ':')
    ]:
        means = [aggregated[h][f'{model}_mean'] * 100 for h in h_values]
        stds = [aggregated[h][f'{model}_std'] * 100 for h in h_values]

        ax1.errorbar(h_values, means, yerr=stds, marker=marker, color=color,
                     linestyle=ls, label=model, linewidth=2, markersize=8, capsize=3)

    # Mark zones
    ax1.axvspan(0.3, 0.7, alpha=0.1, color='red')
    ax1.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    ax1.text(0.5, 72, 'Uncertainty\nZone', ha='center', fontsize=9, color='darkred')

    ax1.set_xlabel('Edge Homophily (h)', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('(a) Model Comparison Across Homophily Spectrum', fontsize=14)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(65, 102)

    # Right: Gap to Oracle
    ax2 = axes[1]

    gap_spi = [aggregated[h]['SPI-Guided_vs_Oracle_mean'] * 100 for h in h_values]
    gap_naive = [(aggregated[h]['NaiveFusion_mean'] - aggregated[h]['Oracle_mean']) * 100 for h in h_values]

    ax2.bar(np.array(h_values) - 0.02, gap_spi, width=0.04, label='SPI-Guided', color='#44AF69', alpha=0.8)
    ax2.bar(np.array(h_values) + 0.02, gap_naive, width=0.04, label='NaiveFusion', color='#F7B801', alpha=0.8)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axvspan(0.3, 0.7, alpha=0.1, color='red')

    ax2.set_xlabel('Edge Homophily (h)', fontsize=12)
    ax2.set_ylabel('Gap to Oracle (%)', fontsize=12)
    ax2.set_title('(b) Gap to Oracle (Best of GCN/MLP)', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'ablation_v2_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'ablation_v2_comparison.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    print(f"Figures saved to {save_dir}")


def generate_latex_table(aggregated: Dict) -> str:
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation Study V2: SPI-Guided Fusion ($\tau=0.4$)}",
        r"\label{tab:ablation_v2}",
        r"\small",
        r"\begin{tabular}{@{}cccccccc@{}}",
        r"\toprule",
        r"$h$ & SPI & GCN & MLP & Oracle & \textbf{Ours} & $\Delta_\text{Oracle}$ & Region \\",
        r"\midrule"
    ]

    for h in sorted(aggregated.keys()):
        agg = aggregated[h]
        delta = agg['SPI-Guided_vs_Oracle_mean'] * 100
        sign = '+' if delta >= 0 else ''

        region = 'Trust' if agg['spi'] >= 0.4 else 'Uncertain'

        line = (f"{h:.1f} & {agg['spi']:.2f} & "
                f"{agg['GCN_mean']*100:.1f} & {agg['MLP_mean']*100:.1f} & "
                f"{agg['Oracle_mean']*100:.1f} & "
                f"\\textbf{{{agg['SPI-Guided_mean']*100:.1f}}} & "
                f"{sign}{delta:.1f} & {region} \\\\")
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    return "\n".join(lines)


def main():
    """Run complete ablation study V2."""
    print("="*60)
    print("SPI-Guided Fusion Ablation Study V2")
    print("Key change: tau = 0.4 (based on V1 observation)")
    print("="*60)

    # Configuration with updated tau
    config = ExperimentConfig(
        num_nodes=1000,
        num_features=20,
        hidden_channels=64,
        num_layers=2,
        epochs=200,
        num_seeds=5,
        h_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        tau=0.4,  # Updated!
        T_init=0.15
    )

    print(f"\nConfig: tau={config.tau}, T_init={config.T_init}")

    # Run experiments
    print("\nRunning experiments...")
    results = run_ablation_study(config)

    # Aggregate results
    aggregated = aggregate_results(results)

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY (V2)")
    print("="*70)
    print(f"{'h':<5} {'SPI':<6} {'GCN':<8} {'MLP':<8} {'Oracle':<8} {'Ours':<8} {'vs Oracle':<10} {'Region'}")
    print("-"*70)

    for h in sorted(aggregated.keys()):
        agg = aggregated[h]
        delta = agg['SPI-Guided_vs_Oracle_mean'] * 100
        sign = '+' if delta >= 0 else ''
        region = 'Trust' if agg['spi'] >= 0.4 else 'Uncertain'

        print(f"{h:<5.1f} {agg['spi']:<6.2f} "
              f"{agg['GCN_mean']*100:<8.1f} {agg['MLP_mean']*100:<8.1f} "
              f"{agg['Oracle_mean']*100:<8.1f} {agg['SPI-Guided_mean']*100:<8.1f} "
              f"{sign}{delta:<9.1f} {region}")

    # Save results
    output_dir = Path(__file__).parent
    results_path = output_dir / "ablation_spi_fusion_v2_results.json"

    with open(results_path, 'w') as f:
        json.dump({
            'raw_results': results,
            'aggregated': {str(k): v for k, v in aggregated.items()},
            'config': config.__dict__
        }, f, indent=2, default=str)

    print(f"\nResults saved: {results_path}")

    # Generate figures
    plot_ablation_results(aggregated, output_dir / "figures")

    # Generate LaTeX table
    latex = generate_latex_table(aggregated)
    latex_path = output_dir / "ablation_table_v2.tex"
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"LaTeX table saved: {latex_path}")

    # Key findings
    print("\n" + "="*60)
    print("KEY FINDINGS (V2)")
    print("="*60)

    # Calculate metrics
    spi_guided_wins_vs_gcn = sum(1 for h in aggregated.keys()
                                  if aggregated[h]['SPI-Guided_vs_GCN_mean'] >= 0)
    spi_guided_wins_vs_mlp = sum(1 for h in aggregated.keys()
                                  if aggregated[h]['SPI-Guided_vs_MLP_mean'] >= 0)
    spi_guided_close_to_oracle = sum(1 for h in aggregated.keys()
                                      if aggregated[h]['SPI-Guided_vs_Oracle_mean'] >= -0.02)

    print(f"1. SPI-Guided >= GCN in {spi_guided_wins_vs_gcn}/{len(aggregated)} h values")
    print(f"2. SPI-Guided >= MLP in {spi_guided_wins_vs_mlp}/{len(aggregated)} h values")
    print(f"3. SPI-Guided within 2% of Oracle in {spi_guided_close_to_oracle}/{len(aggregated)} h values")

    # Average gaps
    avg_gap_oracle = np.mean([aggregated[h]['SPI-Guided_vs_Oracle_mean'] * 100 for h in aggregated.keys()])
    print(f"4. Average gap to Oracle: {avg_gap_oracle:+.2f}%")

    # Zone analysis
    trust_h = [h for h in aggregated.keys() if aggregated[h]['spi'] >= 0.4]
    uncertain_h = [h for h in aggregated.keys() if aggregated[h]['spi'] < 0.4]

    trust_gap = np.mean([aggregated[h]['SPI-Guided_vs_Oracle_mean'] * 100 for h in trust_h])
    uncertain_gap = np.mean([aggregated[h]['SPI-Guided_vs_Oracle_mean'] * 100 for h in uncertain_h])

    print(f"5. Trust zone (SPI>=0.4) avg gap: {trust_gap:+.2f}%")
    print(f"6. Uncertainty zone (SPI<0.4) avg gap: {uncertain_gap:+.2f}%")


if __name__ == "__main__":
    main()
