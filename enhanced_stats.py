"""
Enhanced Statistical Analysis for FSD-GNN Paper
================================================

Addresses reviewer concerns:
1. Increases from 5 seeds to 15 seeds (TKDE standard)
2. Implements rigorous statistical tests:
   - Wilcoxon signed-rank test (non-parametric)
   - Bonferroni/Holm-Bonferroni correction for multiple comparisons
   - Cohen's d effect size with interpretation
   - 95% Bootstrap confidence intervals
3. Generates paper-ready LaTeX tables
4. Automates experimental design and result aggregation

Author: FSD-GNN Research Team
Date: 2025-12-23
Version: 3.0 (Enhanced for TKDE Revision)

Usage:
    # Step 1: Design experiment with 15 seeds
    python enhanced_stats.py --design-experiment --output exp_config.json

    # Step 2: Load existing results and perform analysis
    python enhanced_stats.py --analyze --dataset yelpchi --methods GCN,GAT,H2GCN,NAA-GCN

    # Step 3: Generate complete LaTeX tables
    python enhanced_stats.py --generate-tables --output enhanced_tables.tex
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from scipy import stats
import warnings


# ============================================================================
# CONFIGURATION
# ============================================================================

# Standard seed configuration for reproducibility
STANDARD_SEEDS = [
    42, 123, 456, 789, 1024,           # Original 5 seeds
    2048, 3072, 4096, 5120, 6144,      # Additional 5 seeds
    7168, 8192, 9216, 10240, 11264     # Final 5 seeds (total: 15)
]

# Alternative 10-seed configuration (if computational budget limited)
SEEDS_10 = STANDARD_SEEDS[:10]

# Datasets and methods for FSD-GNN paper
DATASETS = ['elliptic', 'yelpchi', 'ieee_cis', 'amazon']
METHODS = ['GCN', 'GAT', 'GraphSAGE', 'H2GCN', 'NAA-GCN', 'NAA-GAT', 'DAAA']
METRICS = ['auc', 'f1', 'precision', 'recall']


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ComparisonResult:
    """Statistical comparison between two methods."""
    method1: str
    method2: str
    metric: str
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    p_value_corrected: float
    cohens_d: float
    effect_size: str  # 'negligible', 'small', 'medium', 'large'
    significant: bool
    significant_corrected: bool
    n_seeds: int


@dataclass
class MethodStats:
    """Statistics for a single method."""
    method: str
    metric: str
    mean: float
    std: float
    median: float
    ci_lower: float
    ci_upper: float
    values: List[float]
    n_seeds: int


@dataclass
class StatisticalAnalysis:
    """Complete statistical analysis results."""
    dataset: str
    metric: str
    method_stats: Dict[str, MethodStats]
    comparisons: List[ComparisonResult]
    best_method: str
    n_seeds: int
    alpha: float
    correction_method: str
    timestamp: str


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Formula: d = (mean_x - mean_y) / pooled_std

    Interpretation (Cohen, 1988):
        |d| < 0.2  : negligible
        0.2 ≤ |d| < 0.5 : small
        0.5 ≤ |d| < 0.8 : medium
        |d| ≥ 0.8  : large

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(x), len(y)

    if n1 < 2 or n2 < 2:
        return 0.0

    var1 = np.var(x, ddof=1)
    var2 = np.var(y, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (np.mean(x) - np.mean(y)) / pooled_std


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d according to standard thresholds."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: callable = np.mean,
    random_seed: int = 42
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Sample data
        confidence: Confidence level (default: 0.95)
        n_bootstrap: Number of bootstrap samples
        statistic: Statistic to compute (default: mean)
        random_seed: Random seed for reproducibility

    Returns:
        (lower, upper) bounds of confidence interval
    """
    np.random.seed(random_seed)
    n = len(data)
    boot_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats[i] = statistic(boot_sample)

    alpha = 1 - confidence
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return lower, upper


def wilcoxon_signed_rank_test(x: np.ndarray, y: np.ndarray) -> float:
    """
    Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test.
    Robust to non-normal distributions and outliers.

    Args:
        x: First sample (e.g., method A performance across seeds)
        y: Second sample (e.g., method B performance across seeds)

    Returns:
        Two-sided p-value
    """
    if len(x) != len(y):
        raise ValueError("Samples must have same length")

    # Compute differences
    diff = x - y

    # Handle case where all differences are zero
    if np.all(diff == 0):
        return 1.0

    # Remove zero differences (ties at zero)
    diff_nonzero = diff[diff != 0]

    if len(diff_nonzero) < 2:
        return 1.0

    try:
        # Two-sided test
        statistic, p_value = stats.wilcoxon(diff_nonzero, alternative='two-sided')
        return p_value
    except Exception as e:
        warnings.warn(f"Wilcoxon test failed: {e}. Using t-test as fallback.")
        # Fallback to paired t-test
        statistic, p_value = stats.ttest_rel(x, y)
        return p_value


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Adjusted p-value: p_adj = min(p * n_comparisons, 1.0)

    Conservative but guarantees family-wise error rate control.
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Apply Holm-Bonferroni step-down correction.

    More powerful than standard Bonferroni while maintaining FWER control.

    Algorithm:
        1. Sort p-values in ascending order
        2. For i-th smallest p-value, use threshold α/(n-i+1)
        3. Stop at first non-significant result

    Returns:
        List of (corrected_p_value, is_significant) tuples in original order
    """
    n = len(p_values)

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = [p_values[i] for i in sorted_indices]

    # Apply Holm correction
    corrected = []
    for i, p in enumerate(sorted_p):
        correction_factor = n - i
        corrected_p = min(p * correction_factor, 1.0)
        corrected.append(corrected_p)

    # Enforce monotonicity (corrected p-values must be non-decreasing)
    for i in range(1, len(corrected)):
        corrected[i] = max(corrected[i], corrected[i-1])

    # Map back to original order
    result = [None] * n
    for i, orig_idx in enumerate(sorted_indices):
        result[orig_idx] = (corrected[i], corrected[i] < alpha)

    return result


# ============================================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================================

def compute_method_statistics(
    values: List[float],
    method_name: str,
    metric_name: str
) -> MethodStats:
    """Compute comprehensive statistics for a method's performance."""
    data = np.array(values)
    n_seeds = len(data)

    mean = np.mean(data)
    std = np.std(data, ddof=1)
    median = np.median(data)
    ci_lower, ci_upper = bootstrap_ci(data)

    return MethodStats(
        method=method_name,
        metric=metric_name,
        mean=mean,
        std=std,
        median=median,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        values=values,
        n_seeds=n_seeds
    )


def compare_methods(
    method1_stats: MethodStats,
    method2_stats: MethodStats,
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Perform rigorous statistical comparison between two methods.

    Includes:
        - Mean difference and 95% CI
        - Wilcoxon signed-rank test
        - Cohen's d effect size
        - Significance testing
    """
    data1 = np.array(method1_stats.values)
    data2 = np.array(method2_stats.values)

    # Compute difference statistics
    mean_diff = method1_stats.mean - method2_stats.mean
    diff = data1 - data2
    ci_lower, ci_upper = bootstrap_ci(diff)

    # Statistical test
    p_value = wilcoxon_signed_rank_test(data1, data2)

    # Effect size
    d = cohens_d(data1, data2)
    effect = interpret_effect_size(d)

    return ComparisonResult(
        method1=method1_stats.method,
        method2=method2_stats.method,
        metric=method1_stats.metric,
        mean_diff=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        p_value_corrected=None,  # Will be set after multiple comparison correction
        cohens_d=d,
        effect_size=effect,
        significant=p_value < alpha,
        significant_corrected=None,  # Will be set after correction
        n_seeds=method1_stats.n_seeds
    )


def run_complete_analysis(
    method_results: Dict[str, List[float]],
    dataset_name: str,
    metric_name: str,
    alpha: float = 0.05,
    correction: str = 'holm'
) -> StatisticalAnalysis:
    """
    Run complete statistical analysis for a dataset and metric.

    Args:
        method_results: Dict mapping method name to list of performance values
        dataset_name: Name of the dataset
        metric_name: Name of the metric (e.g., 'auc', 'f1')
        alpha: Significance level (default: 0.05)
        correction: Multiple comparison correction ('bonferroni' or 'holm')

    Returns:
        StatisticalAnalysis object with all results
    """
    from datetime import datetime

    # Validate inputs
    methods = list(method_results.keys())
    n_methods = len(methods)

    if n_methods < 2:
        raise ValueError("Need at least 2 methods for comparison")

    n_seeds = len(method_results[methods[0]])
    for method in methods:
        if len(method_results[method]) != n_seeds:
            raise ValueError(f"All methods must have {n_seeds} seeds. {method} has {len(method_results[method])}")

    if n_seeds < 10:
        warnings.warn(f"Only {n_seeds} seeds provided. Recommend 10+ for robust statistics.")

    # Compute statistics for each method
    method_stats = {}
    for method, values in method_results.items():
        method_stats[method] = compute_method_statistics(values, method, metric_name)

    # Find best method
    best_method = max(method_stats.items(), key=lambda x: x[1].mean)[0]

    # Perform all pairwise comparisons
    comparisons = []
    p_values = []

    for i, method1 in enumerate(methods):
        for j in range(i + 1, n_methods):
            method2 = methods[j]
            comp = compare_methods(method_stats[method1], method_stats[method2], alpha)
            comparisons.append(comp)
            p_values.append(comp.p_value)

    # Apply multiple comparison correction
    if correction == 'bonferroni':
        corrected_p = bonferroni_correction(p_values)
        for i, comp in enumerate(comparisons):
            comp.p_value_corrected = corrected_p[i]
            comp.significant_corrected = corrected_p[i] < alpha
    elif correction == 'holm':
        corrected = holm_bonferroni_correction(p_values, alpha)
        for i, comp in enumerate(comparisons):
            comp.p_value_corrected = corrected[i][0]
            comp.significant_corrected = corrected[i][1]
    else:
        raise ValueError(f"Unknown correction method: {correction}")

    return StatisticalAnalysis(
        dataset=dataset_name,
        metric=metric_name,
        method_stats=method_stats,
        comparisons=comparisons,
        best_method=best_method,
        n_seeds=n_seeds,
        alpha=alpha,
        correction_method=correction,
        timestamp=datetime.now().isoformat()
    )


# ============================================================================
# LATEX TABLE GENERATION
# ============================================================================

def format_latex_main_results(
    analysis: StatisticalAnalysis,
    show_all_metrics: bool = False
) -> str:
    """
    Generate paper-ready LaTeX table for main results.

    Features:
        - Methods sorted by performance
        - Best method in bold
        - Significance markers vs. best method
        - 95% confidence intervals
        - Effect sizes
    """
    lines = []

    # Table header
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")

    caption = (f"Performance on {analysis.dataset.upper()} dataset "
              f"({analysis.n_seeds} seeds, {analysis.correction_method} correction)")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{tab:{analysis.dataset}_{analysis.metric}}}")
    lines.append(r"\small")

    # Column specification
    if show_all_metrics:
        lines.append(r"\begin{tabular}{@{}lcccccc@{}}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Method} & \textbf{Mean} & \textbf{Std} & \textbf{95\% CI} & "
                    r"\textbf{vs Best} & \textbf{Effect} & \textbf{Sig.} \\")
    else:
        lines.append(r"\begin{tabular}{@{}lccccc@{}}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Method} & \textbf{Mean} & \textbf{95\% CI} & "
                    r"\textbf{$\Delta$} & \textbf{Cohen's d} & \textbf{p-value} \\")

    lines.append(r"\midrule")

    # Sort methods by mean performance (descending)
    sorted_methods = sorted(analysis.method_stats.items(),
                           key=lambda x: x[1].mean,
                           reverse=True)

    best_mean = sorted_methods[0][1].mean

    for method_name, stats in sorted_methods:
        mean = stats.mean
        std = stats.std
        ci_low = stats.ci_lower
        ci_high = stats.ci_upper
        delta = mean - best_mean

        # Find comparison with best method
        effect_d = 0.0
        p_val = 1.0
        sig_marker = ""

        if method_name != analysis.best_method:
            for comp in analysis.comparisons:
                if ((comp.method1 == method_name and comp.method2 == analysis.best_method) or
                    (comp.method2 == method_name and comp.method1 == analysis.best_method)):
                    effect_d = abs(comp.cohens_d)
                    p_val = comp.p_value_corrected

                    if comp.significant_corrected:
                        if p_val < 0.001:
                            sig_marker = r"$^{***}$"
                        elif p_val < 0.01:
                            sig_marker = r"$^{**}$"
                        elif p_val < 0.05:
                            sig_marker = r"$^{*}$"
                    else:
                        sig_marker = r"$^{\text{ns}}$"
                    break

        # Format row
        if method_name == analysis.best_method:
            # Best method in bold
            lines.append(
                f"\\textbf{{{method_name}}} & "
                f"\\textbf{{{mean:.4f}}} & "
                f"[{ci_low:.4f}, {ci_high:.4f}] & "
                f"--- & --- & --- \\\\"
            )
        else:
            # Other methods
            lines.append(
                f"{method_name} & "
                f"{mean:.4f}{sig_marker} & "
                f"[{ci_low:.4f}, {ci_high:.4f}] & "
                f"{delta:+.4f} & "
                f"{effect_d:.3f} & "
                f"{p_val:.4f} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{6}{@{}l@{}}{\footnotesize $^{***}p<0.001$, "
                r"$^{**}p<0.01$, $^{*}p<0.05$, $^{\text{ns}}$not significant} \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def format_latex_comparison_matrix(analysis: StatisticalAnalysis) -> str:
    """
    Generate LaTeX table showing all pairwise comparisons.

    Displays p-values in a matrix format for comprehensive view.
    """
    lines = []
    methods = sorted(analysis.method_stats.keys())
    n = len(methods)

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Pairwise Comparison Matrix for {analysis.dataset.upper()} "
                f"({analysis.metric.upper()}, {analysis.correction_method} corrected p-values)}}")
    lines.append(f"\\label{{tab:{analysis.dataset}_{analysis.metric}_matrix}}")
    lines.append(r"\small")

    # Create column specification
    col_spec = "l" + "c" * n
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row
    header = " & " + " & ".join([f"\\textbf{{{m}}}" for m in methods]) + " \\\\"
    lines.append(header)
    lines.append(r"\midrule")

    # Create p-value matrix
    p_matrix = {}
    for comp in analysis.comparisons:
        key = (comp.method1, comp.method2)
        p_matrix[key] = comp.p_value_corrected
        # Symmetric
        p_matrix[(comp.method2, comp.method1)] = comp.p_value_corrected

    # Generate rows
    for i, method1 in enumerate(methods):
        row = [f"\\textbf{{{method1}}}"]
        for j, method2 in enumerate(methods):
            if i == j:
                row.append("---")
            else:
                p_val = p_matrix.get((method1, method2), 1.0)
                if p_val < 0.001:
                    row.append(r"$<0.001^{***}$")
                elif p_val < 0.01:
                    row.append(f"{p_val:.3f}$^{{**}}$")
                elif p_val < 0.05:
                    row.append(f"{p_val:.3f}$^{{*}}$")
                else:
                    row.append(f"{p_val:.3f}")
        lines.append(" & ".join(row) + " \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def format_latex_effect_sizes(analysis: StatisticalAnalysis) -> str:
    """Generate LaTeX table showing effect sizes for all comparisons."""
    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Effect Sizes (Cohen's d) for {analysis.dataset.upper()} ({analysis.metric.upper()})}}")
    lines.append(f"\\label{{tab:{analysis.dataset}_{analysis.metric}_effects}}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}llccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method 1} & \textbf{Method 2} & \textbf{Cohen's d} & "
                r"\textbf{Effect} & \textbf{Mean Diff} \\")
    lines.append(r"\midrule")

    # Sort by absolute effect size
    sorted_comps = sorted(analysis.comparisons,
                         key=lambda x: abs(x.cohens_d),
                         reverse=True)

    for comp in sorted_comps:
        d = comp.cohens_d
        effect = comp.effect_size
        diff = comp.mean_diff

        # Effect size interpretation marker
        if effect == "large":
            marker = r"$^{\dagger}$"
        elif effect == "medium":
            marker = r"$^{\circ}$"
        else:
            marker = ""

        lines.append(
            f"{comp.method1} & {comp.method2} & "
            f"{d:+.3f}{marker} & {effect} & {diff:+.4f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{5}{@{}l@{}}{\footnotesize $^{\dagger}$large effect ($|d| \geq 0.8$), "
                r"$^{\circ}$medium effect ($|d| \geq 0.5$)} \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def format_text_summary(analysis: StatisticalAnalysis) -> str:
    """Generate human-readable text summary of analysis."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"Statistical Analysis: {analysis.dataset.upper()} - {analysis.metric.upper()}")
    lines.append("=" * 80)
    lines.append(f"Number of seeds: {analysis.n_seeds}")
    lines.append(f"Significance level: α = {analysis.alpha}")
    lines.append(f"Multiple comparison correction: {analysis.correction_method}")
    lines.append(f"Best method: {analysis.best_method}")
    lines.append("")

    lines.append("METHOD PERFORMANCE SUMMARY")
    lines.append("-" * 80)

    sorted_methods = sorted(analysis.method_stats.items(),
                           key=lambda x: x[1].mean,
                           reverse=True)

    for rank, (method, stats) in enumerate(sorted_methods, 1):
        lines.append(f"{rank}. {method:15s}: "
                    f"{stats.mean:.4f} ± {stats.std:.4f} "
                    f"(95% CI: [{stats.ci_lower:.4f}, {stats.ci_upper:.4f}])")

    lines.append("")
    lines.append("SIGNIFICANT COMPARISONS (after correction)")
    lines.append("-" * 80)

    sig_comparisons = [c for c in analysis.comparisons if c.significant_corrected]

    if not sig_comparisons:
        lines.append("No significant differences found after multiple comparison correction.")
    else:
        for comp in sorted(sig_comparisons, key=lambda x: x.p_value_corrected):
            m1_mean = analysis.method_stats[comp.method1].mean
            m2_mean = analysis.method_stats[comp.method2].mean
            winner = comp.method1 if m1_mean > m2_mean else comp.method2

            lines.append(f"{comp.method1} vs {comp.method2}:")
            lines.append(f"  Mean difference: {comp.mean_diff:+.4f} "
                        f"(95% CI: [{comp.ci_lower:.4f}, {comp.ci_upper:.4f}])")
            lines.append(f"  Corrected p-value: {comp.p_value_corrected:.4f}")
            lines.append(f"  Cohen's d: {comp.cohens_d:.3f} ({comp.effect_size} effect)")
            lines.append(f"  → {winner} is significantly better")
            lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


# ============================================================================
# EXPERIMENT DESIGN
# ============================================================================

def design_experiment(
    n_seeds: int = 15,
    datasets: List[str] = None,
    methods: List[str] = None,
    metrics: List[str] = None,
    output_file: str = None
) -> Dict[str, Any]:
    """
    Design experimental configuration for reproducible research.

    Generates configuration file specifying:
        - Random seeds to use
        - Datasets to evaluate
        - Methods to compare
        - Metrics to compute

    Returns:
        Configuration dictionary
    """
    if datasets is None:
        datasets = DATASETS
    if methods is None:
        methods = METHODS
    if metrics is None:
        metrics = METRICS

    # Select seeds
    if n_seeds <= 15:
        seeds = STANDARD_SEEDS[:n_seeds]
    else:
        # Generate additional seeds deterministically
        np.random.seed(42)
        additional = list(np.random.randint(15000, 50000, n_seeds - 15))
        seeds = STANDARD_SEEDS + additional

    config = {
        "experiment_design": {
            "version": "3.0",
            "timestamp": None,  # Will be set when experiment starts
            "n_seeds": n_seeds,
            "seeds": seeds,
            "datasets": datasets,
            "methods": methods,
            "metrics": metrics
        },
        "statistical_parameters": {
            "alpha": 0.05,
            "correction_method": "holm",
            "confidence_level": 0.95,
            "n_bootstrap": 10000
        },
        "execution_plan": {
            "total_runs": n_seeds * len(datasets) * len(methods),
            "estimated_time_hours": (n_seeds * len(datasets) * len(methods) * 0.5) / 60,  # Assume 30 min per run
        }
    }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Experiment configuration saved to: {output_file}")

    return config


# ============================================================================
# RESULT LOADING AND AGGREGATION
# ============================================================================

def load_results_from_directory(
    results_dir: Path,
    dataset: str,
    methods: List[str] = None
) -> Dict[str, Dict[str, List[float]]]:
    """
    Load experimental results from directory structure.

    Expected structure:
        results_dir/
            dataset_method.json

    JSON format:
        {
            "auc": [0.xx, 0.xx, ...],
            "f1": [0.xx, 0.xx, ...],
            ...
        }

    Returns:
        Nested dict: {metric: {method: [values]}}
    """
    if methods is None:
        methods = METHODS

    results = {metric: {} for metric in METRICS}

    for method in methods:
        filename = f"{dataset}_{method}.json"
        filepath = results_dir / filename

        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping {method}")
            continue

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            for metric in METRICS:
                if metric in data:
                    results[metric][method] = data[metric]
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Statistical Analysis for FSD-GNN Paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Design 15-seed experiment
    python enhanced_stats.py --design-experiment --n-seeds 15 --output config.json

    # Analyze existing results
    python enhanced_stats.py --analyze --dataset yelpchi --results-dir ./results

    # Generate all LaTeX tables
    python enhanced_stats.py --generate-tables --results-dir ./results --output tables.tex
        """
    )

    parser.add_argument('--design-experiment', action='store_true',
                       help='Generate experimental design configuration')
    parser.add_argument('--analyze', action='store_true',
                       help='Perform statistical analysis on existing results')
    parser.add_argument('--generate-tables', action='store_true',
                       help='Generate LaTeX tables from analysis')

    parser.add_argument('--n-seeds', type=int, default=15,
                       help='Number of random seeds (default: 15)')
    parser.add_argument('--dataset', type=str,
                       help='Dataset name for analysis')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory containing result JSON files')
    parser.add_argument('--output', type=str,
                       help='Output file path')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    parser.add_argument('--correction', type=str, default='holm',
                       choices=['bonferroni', 'holm'],
                       help='Multiple comparison correction method')
    parser.add_argument('--methods', type=str,
                       help='Comma-separated list of methods to compare')

    args = parser.parse_args()

    # Parse methods if provided
    methods = args.methods.split(',') if args.methods else METHODS

    # Design experiment
    if args.design_experiment:
        print("Designing experiment configuration...")
        config = design_experiment(
            n_seeds=args.n_seeds,
            output_file=args.output
        )
        print(f"\nExperiment Design Summary:")
        print(f"  Seeds: {config['experiment_design']['n_seeds']}")
        print(f"  Datasets: {len(config['experiment_design']['datasets'])}")
        print(f"  Methods: {len(config['experiment_design']['methods'])}")
        print(f"  Total runs: {config['execution_plan']['total_runs']}")
        print(f"  Estimated time: {config['execution_plan']['estimated_time_hours']:.1f} hours")
        return

    # Analyze results
    if args.analyze:
        if not args.dataset:
            print("Error: --dataset required for analysis")
            return

        print(f"Loading results for {args.dataset}...")
        results_dir = Path(args.results_dir)
        results = load_results_from_directory(results_dir, args.dataset, methods)

        # Analyze each metric
        analyses = {}
        for metric, method_results in results.items():
            if not method_results:
                print(f"No results found for metric: {metric}")
                continue

            print(f"\nAnalyzing {metric.upper()}...")
            analysis = run_complete_analysis(
                method_results,
                args.dataset,
                metric,
                alpha=args.alpha,
                correction=args.correction
            )
            analyses[metric] = analysis

            # Print summary
            print(format_text_summary(analysis))

        # Save analyses
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as JSON (without numpy arrays for serialization)
            serializable = {}
            for metric, analysis in analyses.items():
                serializable[metric] = {
                    'dataset': analysis.dataset,
                    'metric': analysis.metric,
                    'best_method': analysis.best_method,
                    'n_seeds': analysis.n_seeds,
                    'method_means': {m: s.mean for m, s in analysis.method_stats.items()},
                    'method_stds': {m: s.std for m, s in analysis.method_stats.items()},
                    'significant_comparisons': [
                        {
                            'method1': c.method1,
                            'method2': c.method2,
                            'mean_diff': c.mean_diff,
                            'p_value': c.p_value_corrected,
                            'cohens_d': c.cohens_d,
                            'effect_size': c.effect_size
                        }
                        for c in analysis.comparisons if c.significant_corrected
                    ]
                }

            with open(output_path, 'w') as f:
                json.dump(serializable, f, indent=2)
            print(f"\nAnalysis saved to: {output_path}")

        return analyses

    # Generate tables
    if args.generate_tables:
        print("Generating LaTeX tables...")

        # Load all datasets
        results_dir = Path(args.results_dir)
        all_analyses = {}

        for dataset in DATASETS:
            results = load_results_from_directory(results_dir, dataset, methods)

            for metric, method_results in results.items():
                if not method_results:
                    continue

                analysis = run_complete_analysis(
                    method_results,
                    dataset,
                    metric,
                    alpha=args.alpha,
                    correction=args.correction
                )

                key = f"{dataset}_{metric}"
                all_analyses[key] = analysis

        # Generate all tables
        latex_output = []
        latex_output.append("% Generated by enhanced_stats.py")
        latex_output.append("% FSD-GNN Paper - Enhanced Statistical Analysis")
        latex_output.append("% " + "=" * 70)
        latex_output.append("")

        for key, analysis in all_analyses.items():
            latex_output.append(f"% {key.upper()}")
            latex_output.append(format_latex_main_results(analysis))
            latex_output.append("")

        # Save
        if args.output:
            with open(args.output, 'w') as f:
                f.write('\n'.join(latex_output))
            print(f"LaTeX tables saved to: {args.output}")
        else:
            print('\n'.join(latex_output))

        return

    # If no action specified
    parser.print_help()


if __name__ == "__main__":
    main()
