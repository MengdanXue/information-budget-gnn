#!/usr/bin/env python3
"""
Investigate rho_FS as Tiebreaker in Uncertainty Zone
====================================================

The mid-homophily zone (0.3 < h < 0.7) has 4 datasets:
- Flickr: h=0.32, correctly predicted by delta_agg
- cSBM-MidH: h=0.51, fails both metrics
- cSBM-NoisyF: h=0.51, fails both metrics
- cSBM-CleanF: h=0.51, fails both metrics

Question: Can rho_FS help distinguish these cases?

Hypothesis:
- In uncertain zones, feature-structure alignment (rho_FS) might provide
  additional signal about whether to trust the graph structure.
- Higher rho_FS = features align with structure = more likely Class A
- Lower rho_FS = features don't align = Class B might be better

Author: FSD Framework
Date: 2025-12-23
"""

import json
from pathlib import Path
from typing import Dict, List

def load_metrics():
    """Load all metrics from JSON"""
    json_path = Path(__file__).parent / "fsd_metrics_summary_v2.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Flatten categories
    all_data = {}
    for cat in data.values():
        if isinstance(cat, dict):
            all_data.update(cat)
    return all_data


def analyze_rho_fs_in_uncertainty_zone():
    """Analyze rho_FS values in the uncertainty zone"""

    metrics = load_metrics()

    # Ground truth
    ground_truth = {
        "Elliptic": "A", "Inj-Amazon": "A", "Inj-Flickr": "A", "Inj-Cora": "A",
        "Cornell": "B", "Texas": "B", "Wisconsin": "B",
        "Chameleon": "B", "Squirrel": "B", "Actor": "B",
        "Cora": "A", "CiteSeer": "A", "PubMed": "A",
        "Flickr": "B",
        "cSBM-HighH": "A", "cSBM-MidH": "B", "cSBM-LowH": "B",
        "cSBM-NoisyF": "B", "cSBM-CleanF": "B",
    }

    print("=" * 100)
    print("RHO_FS ANALYSIS IN UNCERTAINTY ZONE")
    print("=" * 100)

    # 1. Identify mid-h datasets
    print("\n1. MID-H DATASETS (0.3 <= h < 0.7)")
    print("-" * 80)
    print(f"{'Dataset':<15} {'h':>8} {'delta_agg':>10} {'rho_FS':>10} {'GT':>6}")
    print("-" * 80)

    mid_h_datasets = []
    for name, m in metrics.items():
        h = m['homophily']
        if 0.3 <= h < 0.7:
            mid_h_datasets.append({
                'name': name,
                'h': h,
                'delta_agg': m['delta_agg'],
                'rho_fs': m['rho_fs'],
                'gt': ground_truth.get(name, '?')
            })

    for d in sorted(mid_h_datasets, key=lambda x: x['h']):
        print(f"{d['name']:<15} {d['h']:>8.3f} {d['delta_agg']:>10.2f} {d['rho_fs']:>10.4f} {d['gt']:>6}")

    # 2. Analyze rho_FS patterns
    print("\n2. RHO_FS PATTERN ANALYSIS")
    print("-" * 80)

    class_a_rho = [d['rho_fs'] for d in mid_h_datasets if d['gt'] == 'A']
    class_b_rho = [d['rho_fs'] for d in mid_h_datasets if d['gt'] == 'B']

    print(f"Class A datasets: {[d['name'] for d in mid_h_datasets if d['gt'] == 'A']}")
    print(f"  rho_FS values: {class_a_rho}")
    if class_a_rho:
        print(f"  rho_FS mean: {sum(class_a_rho)/len(class_a_rho):.4f}")
    else:
        print(f"  rho_FS mean: N/A")

    print(f"\nClass B datasets: {[d['name'] for d in mid_h_datasets if d['gt'] == 'B']}")
    print(f"  rho_FS values: {class_b_rho}")
    if class_b_rho:
        print(f"  rho_FS mean: {sum(class_b_rho)/len(class_b_rho):.4f}")
    else:
        print(f"  rho_FS mean: N/A")

    # 3. Check if rho_FS can separate
    print("\n3. CAN RHO_FS SEPARATE CLASSES IN UNCERTAINTY ZONE?")
    print("-" * 80)

    if class_a_rho and class_b_rho:
        min_a = min(class_a_rho)
        max_b = max(class_b_rho)
        if min_a > max_b:
            print(f"YES! All Class A rho_FS ({min_a:.4f}) > All Class B rho_FS ({max_b:.4f})")
            print(f"Threshold: rho_FS > {(min_a + max_b) / 2:.4f} -> Class A")
        else:
            print(f"NO. Class ranges overlap.")
            print(f"  Class A range: [{min(class_a_rho):.4f}, {max(class_a_rho):.4f}]")
            print(f"  Class B range: [{min(class_b_rho):.4f}, {max(class_b_rho):.4f}]")
    elif not class_a_rho:
        print("No Class A datasets in mid-h zone - cannot evaluate separation")
    elif not class_b_rho:
        print("No Class B datasets in mid-h zone - cannot evaluate separation")

    # 4. Full rho_FS analysis across all zones
    print("\n4. RHO_FS ACROSS ALL HOMOPHILY ZONES")
    print("-" * 80)

    zones = {
        "Low-h (h<0.3)": [],
        "Mid-h (0.3<=h<0.7)": [],
        "High-h (h>=0.7)": []
    }

    for name, m in metrics.items():
        h = m['homophily']
        entry = {
            'name': name,
            'rho_fs': m['rho_fs'],
            'gt': ground_truth.get(name, '?')
        }
        if h < 0.3:
            zones["Low-h (h<0.3)"].append(entry)
        elif h < 0.7:
            zones["Mid-h (0.3<=h<0.7)"].append(entry)
        else:
            zones["High-h (h>=0.7)"].append(entry)

    for zone_name, entries in zones.items():
        class_a = [e for e in entries if e['gt'] == 'A']
        class_b = [e for e in entries if e['gt'] == 'B']

        print(f"\n{zone_name}:")
        print(f"  Class A ({len(class_a)}): ", end="")
        if class_a:
            rho_vals = [e['rho_fs'] for e in class_a]
            print(f"rho_FS range [{min(rho_vals):.4f}, {max(rho_vals):.4f}], mean={sum(rho_vals)/len(rho_vals):.4f}")
        else:
            print("N/A")

        print(f"  Class B ({len(class_b)}): ", end="")
        if class_b:
            rho_vals = [e['rho_fs'] for e in class_b]
            print(f"rho_FS range [{min(rho_vals):.4f}, {max(rho_vals):.4f}], mean={sum(rho_vals)/len(rho_vals):.4f}")
        else:
            print("N/A")

    # 5. Special analysis of the 3 failing datasets
    print("\n5. SPECIAL ANALYSIS: THE 3 FAILING MID-H DATASETS")
    print("-" * 80)

    failing = ['cSBM-MidH', 'cSBM-NoisyF', 'cSBM-CleanF']
    print(f"{'Dataset':<15} {'h':>8} {'delta_agg':>10} {'rho_FS':>10} {'GT':>6}")
    print("-" * 60)

    for name in failing:
        m = metrics[name]
        gt = ground_truth[name]
        print(f"{name:<15} {m['homophily']:>8.3f} {m['delta_agg']:>10.2f} {m['rho_fs']:>10.4f} {gt:>6}")

    print("""
Observation:
- All 3 datasets have nearly identical metrics: h=0.506, delta_agg=4.92, rho_FS~0.002
- They are synthetic cSBM datasets with controlled properties
- rho_FS is essentially 0 (~0.002) - no feature-structure alignment
- This makes sense: cSBM generates features independently of structure
- In real-world fraud datasets, rho_FS might be more informative
""")

    # 6. Compare with the one successful mid-h dataset
    print("\n6. COMPARE WITH FLICKR (SUCCESSFUL MID-H)")
    print("-" * 80)

    flickr = metrics['Flickr']
    print(f"Flickr: h={flickr['homophily']:.3f}, delta_agg={flickr['delta_agg']:.2f}, rho_FS={flickr['rho_fs']:.4f}")
    print(f"Ground truth: Class B (correctly predicted by delta_agg > 5 rule)")

    print("""
Why Flickr works:
- h=0.32 is at the LOW END of mid-h zone (close to low-h boundary)
- delta_agg=6.86 is ABOVE the threshold of 5.0
- delta_agg rule correctly predicts Class B

Why cSBM-MidH/NoisyF/CleanF fail:
- h=0.506 is EXACTLY in the middle
- delta_agg=4.92 is JUST BELOW the threshold of 5.0
- Both rules predict Class A, but ground truth is Class B
- These are adversarial cases right at the decision boundary
""")

    # 7. Final recommendation
    print("\n7. FINAL RECOMMENDATION FOR RHO_FS")
    print("=" * 80)
    print("""
CONCLUSION: rho_FS is NOT helpful in the Uncertainty Zone

Reasons:
1. The 3 failing mid-h datasets all have rho_FS ~ 0.002 (essentially zero)
2. This is because they are synthetic (cSBM) - features are random
3. The one successful mid-h dataset (Flickr) also has low rho_FS (0.006)
4. rho_FS does NOT distinguish Class A from Class B in mid-h zone

The real issue:
- The 3 failing datasets are ADVERSARIAL CASES at exact decision boundaries
- h = 0.506 ~ 0.5 (right at the h=0.5 boundary)
- delta_agg = 4.92 ~ 5.0 (right at the delta_agg=5 boundary)
- No threshold-based rule can handle these edge cases

What might help:
1. Use confidence scores instead of hard predictions
2. Ensemble multiple GNN architectures for mid-h datasets
3. Acknowledge mid-h as "fundamentally uncertain" in the paper
4. Consider alternative metrics specific to financial fraud
""")

    return {
        "mid_h_datasets": mid_h_datasets,
        "rho_fs_helpful": False,
        "reason": "rho_FS is essentially zero for all mid-h synthetic datasets"
    }


if __name__ == '__main__':
    results = analyze_rho_fs_in_uncertainty_zone()

    # Save results
    output_path = Path(__file__).parent / "rho_fs_analysis_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
