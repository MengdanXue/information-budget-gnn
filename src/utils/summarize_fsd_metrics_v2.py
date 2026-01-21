#!/usr/bin/env python3
"""
Create a clean summary of FSD metrics for all datasets (v2 - expanded).
"""

import json
from pathlib import Path

def main():
    # Load computed metrics
    with open("fsd_metrics_all_v2.json", 'r') as f:
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
        "cora_graph": "Cora",
        "citeseer_graph": "CiteSeer",
        "pubmed_graph": "PubMed",
        "inj_cora_graph": "Inj-Cora",
        "inj_amazon_graph": "Inj-Amazon",
        "inj_flickr_graph": "Inj-Flickr",
        "reddit_graph": "Reddit",
        "flickr_graph": "Flickr",
        "csbm_high_homo_graph": "cSBM-HighH",
        "csbm_mid_homo_graph": "cSBM-MidH",
        "csbm_low_homo_graph": "cSBM-LowH",
        "csbm_noisy_feat_graph": "cSBM-NoisyF",
        "csbm_clean_feat_graph": "cSBM-CleanF",
    }

    # Categories
    categories = {
        'fraud': ['Elliptic', 'Inj-Amazon', 'Inj-Flickr', 'Inj-Cora'],
        'hetero': ['Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor'],
        'homo': ['Cora', 'CiteSeer', 'PubMed'],
        'social': ['Reddit', 'Flickr'],
        'synthetic': ['cSBM-HighH', 'cSBM-MidH', 'cSBM-LowH', 'cSBM-NoisyF', 'cSBM-CleanF'],
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

    # Print summary table
    print("\n" + "="*100)
    print("FSD METRICS SUMMARY - ALL DATASETS (v2)")
    print("="*100)
    print(f"{'Dataset':<15} {'rho_FS':>10} {'delta_agg':>12} {'h':>10} {'Nodes':>12} {'Edges':>14} {'Features':>10}")
    print("-"*100)

    total_count = 0
    for cat_name, cat_datasets in categories.items():
        cat_display = {
            'fraud': 'Fraud Detection',
            'hetero': 'Heterophilic',
            'homo': 'Homophilic',
            'social': 'Social Networks',
            'synthetic': 'Synthetic (cSBM)'
        }
        print(f"\n[{cat_display[cat_name]}]")
        for name in cat_datasets:
            if name in datasets:
                d = datasets[name]
                print(f"{name:<15} {d['rho_fs']:>10.4f} {d['delta_agg']:>12.2f} {d['homophily']:>10.4f} {d['nodes']:>12,} {d['edges']:>14,} {d['features']:>10}")
                total_count += 1

    print("-"*100)
    print(f"Total: {total_count} datasets")
    print("="*100)

    # Save clean summary
    output = {}
    for cat_name, cat_datasets in categories.items():
        output[cat_name] = {k: datasets[k] for k in cat_datasets if k in datasets}

    with open("fsd_metrics_summary_v2.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: fsd_metrics_summary_v2.json")

    # Key insights
    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)

    # rho_FS analysis
    rho_values = [d['rho_fs'] for d in datasets.values()]
    print(f"\nrho_FS range: [{min(rho_values):.4f}, {max(rho_values):.4f}]")

    # delta_agg analysis
    delta_values = [d['delta_agg'] for d in datasets.values()]
    print(f"delta_agg range: [{min(delta_values):.2f}, {max(delta_values):.2f}]")

    # homophily analysis
    h_values = [d['homophily'] for d in datasets.values()]
    print(f"homophily range: [{min(h_values):.4f}, {max(h_values):.4f}]")

    # Categorize by homophily
    low_h = [n for n, d in datasets.items() if d['homophily'] < 0.3]
    mid_h = [n for n, d in datasets.items() if 0.3 <= d['homophily'] < 0.7]
    high_h = [n for n, d in datasets.items() if d['homophily'] >= 0.7]

    print(f"\nHomophily distribution:")
    print(f"  Low (h<0.3): {len(low_h)} datasets - {', '.join(low_h[:5])}{'...' if len(low_h) > 5 else ''}")
    print(f"  Mid (0.3<=h<0.7): {len(mid_h)} datasets - {', '.join(mid_h[:5])}{'...' if len(mid_h) > 5 else ''}")
    print(f"  High (h>=0.7): {len(high_h)} datasets - {', '.join(high_h[:5])}{'...' if len(high_h) > 5 else ''}")


if __name__ == '__main__':
    main()
