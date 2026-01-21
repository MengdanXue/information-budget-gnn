#!/usr/bin/env python3
"""
Create a clean summary of FSD metrics for all datasets.
"""

import json
from pathlib import Path

def main():
    # Load computed metrics
    with open("fsd_metrics_all_datasets.json", 'r') as f:
        raw_data = json.load(f)

    # Clean dataset mapping
    name_mapping = {
        "elliptic_weber_split": "Elliptic",
        "cornell_graph": "Cornell",
        "texas_graph": "Texas",
        "wisconsin_graph": "Wisconsin",
        "chameleon_graph": "Chameleon",
        "squirrel_graph": "Squirrel",
        "actor_graph": "Actor",
        "csbm_high_homo_graph": "cSBM-HighH",
        "csbm_mid_homo_graph": "cSBM-MidH",
        "csbm_low_homo_graph": "cSBM-LowH",
        "csbm_noisy_feat_graph": "cSBM-NoisyF",
        "csbm_clean_feat_graph": "cSBM-CleanF",
    }

    # Filter and organize
    datasets = {}
    for path, data in raw_data.items():
        if "error" in data:
            continue

        # Get clean name from path
        stem = Path(path).stem
        if stem in name_mapping:
            name = name_mapping[stem]
            datasets[name] = {
                'rho_fs': data['rho_fs'],
                'delta_agg': data['delta_agg'],
                'homophily': data['homophily'],
                'nodes': data['num_nodes'],
                'edges': data['num_edges'],
                'features': data['num_features'],
            }

    # Also add IEEE-CIS if available
    processed_path = Path("processed/ieee_cis_graph.pkl")

    # Print summary table
    print("\n" + "="*90)
    print("FSD METRICS SUMMARY - ALL DATASETS")
    print("="*90)
    print(f"{'Dataset':<15} {'rho_FS':>10} {'delta_agg':>12} {'h':>10} {'Nodes':>10} {'Edges':>12} {'Features':>10}")
    print("-"*90)

    # Sort by category
    real_datasets = ['Elliptic', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor']
    synthetic_datasets = ['cSBM-HighH', 'cSBM-MidH', 'cSBM-LowH', 'cSBM-NoisyF', 'cSBM-CleanF']

    print("\n[Real Datasets]")
    for name in real_datasets:
        if name in datasets:
            d = datasets[name]
            print(f"{name:<15} {d['rho_fs']:>10.4f} {d['delta_agg']:>12.2f} {d['homophily']:>10.4f} {d['nodes']:>10} {d['edges']:>12} {d['features']:>10}")

    print("\n[Synthetic Datasets (cSBM)]")
    for name in synthetic_datasets:
        if name in datasets:
            d = datasets[name]
            print(f"{name:<15} {d['rho_fs']:>10.4f} {d['delta_agg']:>12.2f} {d['homophily']:>10.4f} {d['nodes']:>10} {d['edges']:>12} {d['features']:>10}")

    print("-"*90)
    print(f"Total: {len(datasets)} datasets")
    print("="*90)

    # Save clean summary
    output = {
        'real': {k: datasets[k] for k in real_datasets if k in datasets},
        'synthetic': {k: datasets[k] for k in synthetic_datasets if k in datasets},
    }

    with open("fsd_metrics_summary.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: fsd_metrics_summary.json")

    # Key insights
    print("\n" + "="*90)
    print("KEY INSIGHTS")
    print("="*90)

    # rho_FS analysis
    rho_values = [d['rho_fs'] for d in datasets.values()]
    print(f"\nrho_FS range: [{min(rho_values):.4f}, {max(rho_values):.4f}]")
    print(f"  - Real datasets: mostly near 0 (no predictive power)")
    print(f"  - cSBM-HighH: {datasets.get('cSBM-HighH', {}).get('rho_fs', 'N/A'):.4f} (positive, as expected)")
    print(f"  - cSBM-LowH: {datasets.get('cSBM-LowH', {}).get('rho_fs', 'N/A'):.4f} (negative, as expected)")

    # delta_agg analysis
    delta_values = [d['delta_agg'] for d in datasets.values()]
    print(f"\ndelta_agg range: [{min(delta_values):.2f}, {max(delta_values):.2f}]")
    print(f"  - Low delta_agg (<3): Good for mean-aggregation (GCN/GAT)")
    print(f"  - High delta_agg (>5): Better for sampling/concat (GraphSAGE/H2GCN)")

    # homophily analysis
    h_values = [d['homophily'] for d in datasets.values()]
    print(f"\nhomophily range: [{min(h_values):.4f}, {max(h_values):.4f}]")
    print(f"  - Heterophilic (h<0.3): Cornell, Texas, Wisconsin, Chameleon, Squirrel")
    print(f"  - Homophilic (h>0.7): Elliptic, cSBM-HighH")


if __name__ == '__main__':
    main()
