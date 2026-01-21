"""
Demonstration of Enhanced Statistical Analysis
===============================================

This script demonstrates the key features of the enhanced statistical
analysis system with example data.

Run: python demo_enhanced_stats.py
"""

import numpy as np
from enhanced_stats import (
    run_complete_analysis,
    format_text_summary,
    format_latex_main_results,
    format_latex_comparison_matrix,
    format_latex_effect_sizes,
    STANDARD_SEEDS
)


def generate_example_data():
    """
    Generate example data simulating FSD-GNN results.

    Scenario: YelpChi dataset (high dilution, δ_agg = 12.57)
    Expected: H2GCN > GraphSAGE > NAA methods
    """
    np.random.seed(42)

    # Simulate 15-seed results with realistic patterns
    data = {
        # Class B methods (good for high dilution)
        'H2GCN': np.random.normal(0.742, 0.028, 15),
        'GraphSAGE': np.random.normal(0.730, 0.030, 15),

        # Class A methods (poor for high dilution)
        'NAA-GAT': np.random.normal(0.679, 0.039, 15),
        'NAA-GCN': np.random.normal(0.672, 0.039, 15),
        'GAT': np.random.normal(0.660, 0.041, 15),
        'GCN': np.random.normal(0.651, 0.040, 15),
    }

    # Clip to valid range [0, 1]
    for method in data:
        data[method] = np.clip(data[method], 0.0, 1.0).tolist()

    return data


def main():
    print("=" * 80)
    print("ENHANCED STATISTICAL ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print()
    print("This demo shows how the enhanced statistical analysis addresses")
    print("reviewer concerns with 15 seeds and rigorous statistical tests.")
    print()

    # Generate example data
    print("Step 1: Generating example data (YelpChi dataset, 15 seeds)...")
    print("-" * 80)
    method_results = generate_example_data()

    print(f"Methods: {list(method_results.keys())}")
    print(f"Seeds per method: {len(method_results['H2GCN'])}")
    print()

    # Show raw data sample
    print("Sample data (first 5 seeds):")
    for method, values in method_results.items():
        print(f"  {method:12s}: {values[:5]}")
    print()

    # Run complete statistical analysis
    print("Step 2: Running comprehensive statistical analysis...")
    print("-" * 80)
    analysis = run_complete_analysis(
        method_results,
        dataset_name='yelpchi',
        metric_name='auc',
        alpha=0.05,
        correction='holm'
    )
    print("[OK] Analysis complete")
    print()

    # Display text summary
    print("Step 3: Statistical Summary")
    print("-" * 80)
    summary = format_text_summary(analysis)
    print(summary)
    print()

    # Generate LaTeX tables
    print("\n" + "=" * 80)
    print("Step 4: Paper-Ready LaTeX Tables")
    print("=" * 80)
    print()

    print("TABLE 1: Main Results")
    print("-" * 80)
    latex_main = format_latex_main_results(analysis)
    print(latex_main)
    print()

    print("\nTABLE 2: Pairwise Comparison Matrix")
    print("-" * 80)
    latex_matrix = format_latex_comparison_matrix(analysis)
    print(latex_matrix)
    print()

    print("\nTABLE 3: Effect Sizes")
    print("-" * 80)
    latex_effects = format_latex_effect_sizes(analysis)
    print(latex_effects)
    print()

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 80)
    print()

    best = analysis.best_method
    best_mean = analysis.method_stats[best].mean
    best_ci = analysis.method_stats[best].ci_lower, analysis.method_stats[best].ci_upper

    print(f"1. Best Method: {best}")
    print(f"   Performance: {best_mean:.4f} (95% CI: [{best_ci[0]:.4f}, {best_ci[1]:.4f}])")
    print()

    print("2. Significant Improvements:")
    sig_count = 0
    for comp in analysis.comparisons:
        if comp.significant_corrected and comp.method1 == best:
            sig_count += 1
            print(f"   • {best} vs {comp.method2}:")
            print(f"     Δ = +{comp.mean_diff:.4f}, p = {comp.p_value_corrected:.4f}, "
                  f"d = {comp.cohens_d:.3f} ({comp.effect_size} effect)")

    if sig_count == 0:
        print("   (No significant differences found after correction)")
    print()

    print("3. Effect Size Interpretation:")
    large_effects = [c for c in analysis.comparisons
                     if abs(c.cohens_d) >= 0.8]
    medium_effects = [c for c in analysis.comparisons
                      if 0.5 <= abs(c.cohens_d) < 0.8]

    print(f"   • Large effects (|d| ≥ 0.8): {len(large_effects)}")
    print(f"   • Medium effects (0.5 ≤ |d| < 0.8): {len(medium_effects)}")
    print()

    print("4. Addressing Reviewer Concerns:")
    print("   [OK] Used 15 seeds (up from 5) -> robust statistical power")
    print("   [OK] Wilcoxon signed-rank test -> non-parametric, robust to outliers")
    print("   [OK] Holm-Bonferroni correction -> controls family-wise error rate")
    print("   [OK] Cohen's d effect sizes -> practical significance assessment")
    print("   [OK] 95% bootstrap CIs -> uncertainty quantification")
    print()

    # Compare with original 5-seed scenario
    print("\n" + "=" * 80)
    print("COMPARISON: 5 Seeds vs 15 Seeds")
    print("=" * 80)
    print()

    # Simulate analysis with only first 5 seeds
    method_results_5seeds = {method: values[:5]
                             for method, values in method_results.items()}

    analysis_5seeds = run_complete_analysis(
        method_results_5seeds,
        dataset_name='yelpchi',
        metric_name='auc',
        alpha=0.05,
        correction='holm'
    )

    # Find H2GCN vs GCN comparison
    comp_15seeds = None
    comp_5seeds = None

    for comp in analysis.comparisons:
        if (comp.method1, comp.method2) == ('H2GCN', 'GCN') or \
           (comp.method2, comp.method1) == ('H2GCN', 'GCN'):
            comp_15seeds = comp
            break

    for comp in analysis_5seeds.comparisons:
        if (comp.method1, comp.method2) == ('H2GCN', 'GCN') or \
           (comp.method2, comp.method1) == ('H2GCN', 'GCN'):
            comp_5seeds = comp
            break

    if comp_15seeds and comp_5seeds:
        print("H2GCN vs GCN Comparison:")
        print()
        print(f"  5 seeds:  p = {comp_5seeds.p_value_corrected:.4f} "
              f"{'(significant)' if comp_5seeds.significant_corrected else '(NOT significant)'}")
        print(f"            d = {comp_5seeds.cohens_d:.3f} ({comp_5seeds.effect_size} effect)")
        print()
        print(f"  15 seeds: p = {comp_15seeds.p_value_corrected:.4f} "
              f"{'(significant)' if comp_15seeds.significant_corrected else '(NOT significant)'}")
        print(f"            d = {comp_15seeds.cohens_d:.3f} ({comp_15seeds.effect_size} effect)")
        print()

        if comp_15seeds.significant_corrected and not comp_5seeds.significant_corrected:
            print("  -> Increasing seeds from 5 to 15 revealed statistical significance!")
        elif comp_15seeds.significant_corrected and comp_5seeds.significant_corrected:
            print("  -> Both significant, but 15 seeds provide more reliable p-value")
        print()

    # Final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR PAPER")
    print("=" * 80)
    print()

    print("1. Update Experimental Setup section:")
    print('   "All experiments use 15 random seeds following TKDE standards,')
    print('    with Wilcoxon tests and Holm-Bonferroni correction."')
    print()

    print("2. Replace result tables with LaTeX output above")
    print()

    print("3. Add to results discussion:")
    print(f'   "H2GCN significantly outperforms mean-aggregation methods')
    print(f'    on YelpChi (p < 0.001, large effects), confirming FSD')
    print(f'    predictions for high-dilution datasets."')
    print()

    print("4. Add Statistical Methods subsection to appendix:")
    print("   • Detailed description of tests")
    print("   • Justification for 15 seeds")
    print("   • Power analysis")
    print("   • Full comparison matrices")
    print()

    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review ENHANCED_STATS_GUIDE.md for detailed usage")
    print("2. Run your actual experiments with 15 seeds")
    print("3. Use enhanced_stats.py to analyze results")
    print("4. Copy LaTeX tables to paper")
    print()


if __name__ == "__main__":
    main()
