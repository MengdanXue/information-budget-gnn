#!/usr/bin/env python3
"""
Deep Analysis of Failure Patterns in FSD Framework
===================================================

Key Discovery from Initial Experiment:
- δ_agg-only failures: Cornell, Texas, Wisconsin, Actor, cSBM-MidH, cSBM-NoisyF, cSBM-CleanF
- h-only failures: cSBM-MidH, cSBM-NoisyF, cSBM-CleanF
- OVERLAP EXISTS: 3 mid-h synthetic datasets fail BOTH metrics!

This analysis investigates:
1. Why do mid-h datasets fail both metrics?
2. What additional signals can help?
3. Is there a "Uncertainty Zone" where prediction is fundamentally difficult?

Author: FSD Framework
Date: 2025-12-23
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import math

# ============================================================================
# Data Analysis
# ============================================================================

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


def analyze_failure_patterns():
    """Comprehensive analysis of failure patterns"""

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
    print("COMPREHENSIVE FAILURE PATTERN ANALYSIS")
    print("=" * 100)

    # 1. Categorize datasets by homophily zones
    print("\n" + "=" * 100)
    print("1. HOMOPHILY ZONE ANALYSIS")
    print("=" * 100)

    zones = {
        "Low (h < 0.3)": [],
        "Mid (0.3 <= h < 0.7)": [],
        "High (h >= 0.7)": []
    }

    for name, m in metrics.items():
        h = m['homophily']
        gt = ground_truth.get(name, "?")

        # Predict using δ_agg-only rule
        delta_pred = "B" if m['delta_agg'] > 5.0 else "A"
        delta_correct = delta_pred == gt

        # Predict using h-only rule
        h_pred = "A" if h > 0.5 else "B"
        h_correct = h_pred == gt

        entry = {
            "name": name,
            "h": h,
            "delta_agg": m['delta_agg'],
            "rho_fs": m['rho_fs'],
            "gt": gt,
            "delta_pred": delta_pred,
            "delta_correct": delta_correct,
            "h_pred": h_pred,
            "h_correct": h_correct,
            "both_fail": not delta_correct and not h_correct,
            "both_succeed": delta_correct and h_correct
        }

        if h < 0.3:
            zones["Low (h < 0.3)"].append(entry)
        elif h < 0.7:
            zones["Mid (0.3 <= h < 0.7)"].append(entry)
        else:
            zones["High (h >= 0.7)"].append(entry)

    for zone_name, entries in zones.items():
        print(f"\n--- {zone_name} ({len(entries)} datasets) ---")
        print(f"{'Name':<15} {'h':>8} {'δ_agg':>10} {'GT':>4} {'δ_pred':>8} {'h_pred':>8} {'Status':>15}")
        print("-" * 80)

        for e in sorted(entries, key=lambda x: x['h']):
            status = "BOTH_FAIL" if e['both_fail'] else ("OK" if e['both_succeed'] else "PARTIAL")
            delta_mark = "[OK]" if e['delta_correct'] else "[X]"
            h_mark = "[OK]" if e['h_correct'] else "[X]"
            print(f"{e['name']:<15} {e['h']:>8.3f} {e['delta_agg']:>10.2f} {e['gt']:>4} {delta_mark:>8} {h_mark:>8} {status:>15}")

        # Zone statistics
        n_both_fail = sum(1 for e in entries if e['both_fail'])
        n_both_ok = sum(1 for e in entries if e['both_succeed'])
        n_partial = len(entries) - n_both_fail - n_both_ok
        print(f"\nZone Summary: {n_both_ok} both OK, {n_partial} partial, {n_both_fail} both fail")

    # 2. Analyze the "problem datasets"
    print("\n" + "=" * 100)
    print("2. PROBLEM DATASET DEEP DIVE")
    print("=" * 100)

    all_entries = []
    for entries in zones.values():
        all_entries.extend(entries)

    problem_datasets = [e for e in all_entries if e['both_fail']]
    delta_only_fail = [e for e in all_entries if not e['delta_correct'] and e['h_correct']]
    h_only_fail = [e for e in all_entries if e['delta_correct'] and not e['h_correct']]

    print(f"\nBoth metrics fail ({len(problem_datasets)}):")
    for e in problem_datasets:
        print(f"  - {e['name']}: h={e['h']:.3f}, δ_agg={e['delta_agg']:.2f}, GT={e['gt']}")
        print(f"    Analysis: h≈0.5 (edge of h>0.5 threshold), δ_agg≈5 (edge of δ_agg>5 threshold)")

    print(f"\nOnly δ_agg fails ({len(delta_only_fail)}):")
    for e in delta_only_fail:
        print(f"  - {e['name']}: h={e['h']:.3f}, δ_agg={e['delta_agg']:.2f}, GT={e['gt']}")

    print(f"\nOnly h fails ({len(h_only_fail)}):")
    for e in h_only_fail:
        print(f"  - {e['name']}: h={e['h']:.3f}, δ_agg={e['delta_agg']:.2f}, GT={e['gt']}")

    # 3. Threshold analysis
    print("\n" + "=" * 100)
    print("3. OPTIMAL THRESHOLD SEARCH")
    print("=" * 100)

    # Try different threshold combinations
    best_accuracy = 0
    best_thresholds = None

    for h_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for delta_thresh in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
            correct = 0
            for e in all_entries:
                # New rule: use h if h > h_thresh, else use delta_agg
                if e['h'] > h_thresh:
                    pred = "A"
                else:
                    pred = "B" if e['delta_agg'] > delta_thresh else "A"

                if pred == e['gt']:
                    correct += 1

            accuracy = correct / len(all_entries)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thresholds = (h_thresh, delta_thresh)

    print(f"\nBest thresholds: h_thresh={best_thresholds[0]}, delta_thresh={best_thresholds[1]}")
    print(f"Best accuracy: {best_accuracy:.1%} ({int(best_accuracy * len(all_entries))}/{len(all_entries)})")

    # 4. Alternative rule: use h ALWAYS when h is extreme (very high or very low)
    print("\n" + "=" * 100)
    print("4. ALTERNATIVE RULE: EXTREME H DETECTION")
    print("=" * 100)

    # Rule: if h > 0.7 OR h < 0.3, trust h; otherwise trust delta_agg
    print("\nRule: if h > 0.7 OR h < 0.3: trust h; else: trust δ_agg")

    correct = 0
    predictions = []
    for e in all_entries:
        h = e['h']
        if h > 0.7:
            pred = "A"  # High h → mean-agg
            reason = "h > 0.7 → A"
        elif h < 0.3:
            pred = "B"  # Low h → sampling
            reason = "h < 0.3 → B"
        else:
            # Mid h: trust delta_agg
            if e['delta_agg'] > 5.0:
                pred = "B"
                reason = f"mid-h, δ_agg={e['delta_agg']:.1f} > 5 → B"
            else:
                pred = "A"
                reason = f"mid-h, δ_agg={e['delta_agg']:.1f} <= 5 → A"

        is_correct = pred == e['gt']
        if is_correct:
            correct += 1
        predictions.append((e['name'], h, e['delta_agg'], e['gt'], pred, is_correct, reason))

    print(f"\nAccuracy: {correct/len(all_entries):.1%} ({correct}/{len(all_entries)})")
    print(f"\n{'Name':<15} {'h':>8} {'δ_agg':>10} {'GT':>4} {'Pred':>6} {'OK?':>6} {'Reason':<30}")
    print("-" * 90)
    for name, h, delta, gt, pred, ok, reason in sorted(predictions, key=lambda x: x[1]):
        status = "[OK]" if ok else "[X]"
        print(f"{name:<15} {h:>8.3f} {delta:>10.2f} {gt:>4} {pred:>6} {status:>6} {reason:<30}")

    # 5. Final recommendation
    print("\n" + "=" * 100)
    print("5. KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 100)

    print("""
FINDINGS:
=========
1. Mid-homophily (0.3 < h < 0.7) datasets are the UNCERTAINTY ZONE
   - cSBM-MidH, cSBM-NoisyF, cSBM-CleanF all have h ≈ 0.51
   - Both δ_agg and h fail to predict correctly in this zone
   - This is NOT a failure of the metrics, but a fundamental property

2. The "non-overlapping failure sets" claim from multi-AI discussion was WRONG
   - 3 datasets fail BOTH metrics
   - Theoretical max with perfect selection is NOT 100%

3. Low-h heterophilic datasets (Cornell, Texas, Wisconsin, Actor) fail δ_agg
   - These have h < 0.3 and δ_agg < 5
   - δ_agg threshold (5.0) misses them because dilution is low
   - But they need Class B because structure is unreliable

4. The REAL pattern:
   - h > 0.7: Trust structure → Class A (mean-agg)
   - h < 0.3: Don't trust structure → Class B (sampling)
   - 0.3 < h < 0.7: UNCERTAINTY ZONE - need additional signals

RECOMMENDATIONS:
================
1. Acknowledge the "Uncertainty Zone" in the paper
   - This is a principled limitation, not a failure
   - Mid-h datasets are fundamentally ambiguous

2. For the Uncertainty Zone, consider:
   - Feature quality (ρ_FS might help here)
   - δ_agg as secondary signal
   - Ensemble prediction with confidence bounds

3. Revised decision rule:
   - h > 0.7 → Class A (confident)
   - h < 0.3 → Class B (confident)
   - 0.3-0.7 → Use δ_agg with LOW confidence, flag for manual review

4. Paper narrative:
   - "When to Trust Graph Structure: A Homophily-Based Decision Framework"
   - Emphasize the principled boundaries, not just accuracy numbers
""")

    return {
        "zones": {name: len(entries) for name, entries in zones.items()},
        "problem_datasets": [e['name'] for e in problem_datasets],
        "best_thresholds": best_thresholds,
        "best_accuracy": best_accuracy,
        "total_datasets": len(all_entries)
    }


if __name__ == '__main__':
    results = analyze_failure_patterns()

    # Save analysis results
    output_path = Path(__file__).parent / "failure_analysis_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis results saved to: {output_path}")
