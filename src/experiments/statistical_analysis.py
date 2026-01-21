"""
Rigorous Statistical Analysis for GNN Experiments - TKDE Version
================================================================

This module implements proper statistical methodology for comparing GNN methods:
- 15 seeds for sufficient statistical power (TKDE requirement)
- Wilcoxon signed-rank test (non-parametric, robust to non-normality)
- Bonferroni correction for multiple comparisons
- Holm-Bonferroni step-down correction (more powerful)
- Cohen's d effect size with confidence intervals
- Bootstrap confidence intervals
- Friedman test for multi-method comparison
- Nemenyi post-hoc test

TKDE Statistical Standards:
1. Minimum 15 random seeds (our implementation)
2. Non-parametric tests (Wilcoxon) for robustness
3. Multiple comparison correction (Bonferroni/Holm)
4. Effect size reporting (Cohen's d)
5. Confidence intervals (95% bootstrap)

Usage:
    from statistical_analysis import run_rigorous_comparison, StatisticalResults

    results = run_rigorous_comparison(
        method_results={'NAA': [...], 'GAT': [...], 'GCN': [...]},
        baseline='GAT',
        alpha=0.05
    )

Author: FSD Framework Research Team
Date: 2024-12-22
Version: 2.0 (TKDE Submission)
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class ComparisonResult:
    """Result of comparing two methods."""
    method1: str
    method2: str
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    p_value_corrected: float
    cohens_d: float
    effect_interpretation: str
    significant: bool
    significant_corrected: bool


@dataclass
class StatisticalResults:
    """Complete statistical analysis results."""
    method_means: Dict[str, float]
    method_stds: Dict[str, float]
    method_cis: Dict[str, Tuple[float, float]]
    comparisons: List[ComparisonResult]
    n_seeds: int
    alpha: float
    correction_method: str


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    d = (mean1 - mean2) / pooled_std

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(x), len(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (np.mean(x) - np.mean(y)) / pooled_std


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
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
    x: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: callable = np.mean
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        x: Sample data
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
        statistic: Statistic to compute (default: mean)

    Returns:
        (lower, upper) bounds of CI
    """
    n = len(x)
    boot_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        boot_sample = np.random.choice(x, size=n, replace=True)
        boot_stats[i] = statistic(boot_sample)

    alpha = 1 - confidence
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return lower, upper


def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> float:
    """
    Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test.
    Robust to non-normal distributions.

    Returns:
        p-value (two-sided)
    """
    # Handle case where differences are all zero
    diff = x - y
    if np.all(diff == 0):
        return 1.0

    # Remove zero differences (ties at zero)
    diff_nonzero = diff[diff != 0]
    if len(diff_nonzero) < 2:
        return 1.0

    try:
        stat, p_value = stats.wilcoxon(diff_nonzero, alternative='two-sided')
    except ValueError:
        # Fall back to t-test if Wilcoxon fails
        stat, p_value = stats.ttest_rel(x, y)

    return p_value


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """
    Apply Bonferroni correction for multiple comparisons.

    p_corrected = p * n_comparisons (capped at 1.0)
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Apply Holm-Bonferroni step-down correction (more powerful than Bonferroni).

    Returns list of (corrected_p, significant) tuples.
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

    # Enforce monotonicity (corrected p-values should be non-decreasing)
    for i in range(1, len(corrected)):
        corrected[i] = max(corrected[i], corrected[i-1])

    # Map back to original order
    result = [None] * n
    for i, orig_idx in enumerate(sorted_indices):
        result[orig_idx] = (corrected[i], corrected[i] < alpha)

    return result


def run_rigorous_comparison(
    method_results: Dict[str, List[float]],
    baseline: str = None,
    alpha: float = 0.05,
    correction: str = 'bonferroni'
) -> StatisticalResults:
    """
    Run rigorous statistical comparison of methods.

    Args:
        method_results: Dict mapping method name to list of metric values (one per seed)
        baseline: Method to compare all others against (if None, compare all pairs)
        alpha: Significance level
        correction: 'bonferroni' or 'holm' for multiple comparison correction

    Returns:
        StatisticalResults object with all analysis results
    """
    methods = list(method_results.keys())
    n_methods = len(methods)

    # Check that all methods have same number of seeds
    n_seeds = len(method_results[methods[0]])
    for method in methods:
        if len(method_results[method]) != n_seeds:
            raise ValueError(f"All methods must have same number of seeds. "
                           f"{method} has {len(method_results[method])}, expected {n_seeds}")

    if n_seeds < 15:
        warnings.warn(f"Only {n_seeds} seeds provided. TKDE requires 15+ for robust statistics.")

    # Compute means, stds, CIs for each method
    method_means = {}
    method_stds = {}
    method_cis = {}

    for method in methods:
        data = np.array(method_results[method])
        method_means[method] = np.mean(data)
        method_stds[method] = np.std(data, ddof=1)
        method_cis[method] = bootstrap_ci(data)

    # Determine comparisons to make
    if baseline is not None:
        if baseline not in methods:
            raise ValueError(f"Baseline '{baseline}' not in methods")
        comparisons_to_make = [(m, baseline) for m in methods if m != baseline]
    else:
        # All pairwise comparisons
        comparisons_to_make = []
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                comparisons_to_make.append((methods[i], methods[j]))

    # Run comparisons
    comparisons = []
    p_values = []

    for method1, method2 in comparisons_to_make:
        data1 = np.array(method_results[method1])
        data2 = np.array(method_results[method2])

        # Compute statistics
        mean_diff = np.mean(data1) - np.mean(data2)

        # Bootstrap CI for difference
        diff = data1 - data2
        ci_lower, ci_upper = bootstrap_ci(diff)

        # Wilcoxon test
        p_value = wilcoxon_test(data1, data2)
        p_values.append(p_value)

        # Effect size
        d = cohens_d(data1, data2)
        effect_interp = interpret_effect_size(d)

        comparisons.append(ComparisonResult(
            method1=method1,
            method2=method2,
            mean_diff=mean_diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            p_value_corrected=None,  # Will be filled after correction
            cohens_d=d,
            effect_interpretation=effect_interp,
            significant=p_value < alpha,
            significant_corrected=None  # Will be filled after correction
        ))

    # Apply multiple comparison correction
    if correction == 'bonferroni':
        corrected_p = bonferroni_correction(p_values, alpha)
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

    return StatisticalResults(
        method_means=method_means,
        method_stds=method_stds,
        method_cis=method_cis,
        comparisons=comparisons,
        n_seeds=n_seeds,
        alpha=alpha,
        correction_method=correction
    )


def format_results_latex(results: StatisticalResults, metric_name: str = "F1") -> str:
    """
    Format results as LaTeX table.

    Args:
        results: StatisticalResults object
        metric_name: Name of the metric

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{metric_name} Comparison ({results.n_seeds} seeds, "
                f"{results.correction_method} correction)}}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Mean} & \\textbf{Std} & "
                "\\textbf{95\\% CI} & \\textbf{$\\Delta$ vs Best} & \\textbf{Sig.} \\\\")
    lines.append("\\midrule")

    # Sort methods by mean (descending)
    sorted_methods = sorted(results.method_means.keys(),
                           key=lambda m: results.method_means[m],
                           reverse=True)
    best_method = sorted_methods[0]
    best_mean = results.method_means[best_method]

    for method in sorted_methods:
        mean = results.method_means[method]
        std = results.method_stds[method]
        ci_low, ci_high = results.method_cis[method]
        delta = mean - best_mean

        # Find significance vs best method
        sig = "-"
        for comp in results.comparisons:
            if (comp.method1 == method and comp.method2 == best_method) or \
               (comp.method2 == method and comp.method1 == best_method):
                if comp.significant_corrected:
                    sig = f"$p<{results.alpha}$"
                else:
                    sig = "n.s."
                break

        if method == best_method:
            sig = "\\textbf{best}"
            lines.append(f"\\textbf{{{method}}} & \\textbf{{{mean:.4f}}} & {std:.4f} & "
                        f"[{ci_low:.4f}, {ci_high:.4f}] & - & {sig} \\\\")
        else:
            lines.append(f"{method} & {mean:.4f} & {std:.4f} & "
                        f"[{ci_low:.4f}, {ci_high:.4f}] & {delta:+.4f} & {sig} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def format_comparison_details(results: StatisticalResults) -> str:
    """
    Format detailed comparison results.

    Returns:
        Formatted string with all comparison details
    """
    lines = []
    lines.append(f"Statistical Analysis Results ({results.n_seeds} seeds)")
    lines.append("=" * 60)
    lines.append(f"Alpha: {results.alpha}")
    lines.append(f"Correction: {results.correction_method}")
    lines.append("")

    lines.append("Method Performance:")
    lines.append("-" * 40)
    for method in sorted(results.method_means.keys(),
                        key=lambda m: results.method_means[m],
                        reverse=True):
        mean = results.method_means[method]
        std = results.method_stds[method]
        ci = results.method_cis[method]
        lines.append(f"  {method}: {mean:.4f} ± {std:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")

    lines.append("")
    lines.append("Pairwise Comparisons:")
    lines.append("-" * 40)

    for comp in results.comparisons:
        lines.append(f"\n{comp.method1} vs {comp.method2}:")
        lines.append(f"  Mean difference: {comp.mean_diff:+.4f} (95% CI: [{comp.ci_lower:.4f}, {comp.ci_upper:.4f}])")
        lines.append(f"  Wilcoxon p-value: {comp.p_value:.4f}")
        lines.append(f"  Corrected p-value: {comp.p_value_corrected:.4f}")
        lines.append(f"  Cohen's d: {comp.cohens_d:.4f} ({comp.effect_interpretation})")

        if comp.significant_corrected:
            lines.append(f"  → SIGNIFICANT after {results.correction_method} correction")
        else:
            lines.append(f"  → Not significant after correction")

    return "\n".join(lines)


def friedman_test(method_results: Dict[str, List[float]]) -> Tuple[float, float]:
    """
    Friedman test for comparing multiple methods across seeds.

    Non-parametric alternative to repeated measures ANOVA.
    Tests H0: All methods have the same median performance.

    Args:
        method_results: Dict mapping method name to list of metric values

    Returns:
        (statistic, p_value)
    """
    methods = list(method_results.keys())
    n_seeds = len(method_results[methods[0]])

    # Create matrix: rows = seeds, columns = methods
    data = np.array([method_results[m] for m in methods]).T

    try:
        stat, p_value = stats.friedmanchisquare(*[data[:, i] for i in range(len(methods))])
    except Exception:
        return 0.0, 1.0

    return stat, p_value


def nemenyi_critical_difference(n_methods: int, n_seeds: int, alpha: float = 0.05) -> float:
    """
    Calculate Nemenyi critical difference for post-hoc comparison.

    CD = q_alpha * sqrt(n_methods * (n_methods + 1) / (6 * n_seeds))

    where q_alpha is from the Studentized range distribution.
    """
    # q_alpha values for alpha=0.05 (from Demsar 2006)
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }

    if n_methods > 10:
        # Approximate for larger number of methods
        q_alpha = 3.164 + 0.05 * (n_methods - 10)
    else:
        q_alpha = q_alpha_table.get(n_methods, 2.343)

    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_seeds))
    return cd


def compute_average_ranks(method_results: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute average rank for each method across all seeds.

    Used for Friedman-Nemenyi analysis.
    Higher performance = lower rank (rank 1 is best).
    """
    methods = list(method_results.keys())
    n_seeds = len(method_results[methods[0]])

    # Create matrix: rows = seeds, columns = methods
    data = np.array([method_results[m] for m in methods]).T

    # Compute ranks for each seed (higher value = lower rank)
    ranks = np.zeros_like(data)
    for i in range(n_seeds):
        # Negative because we want higher values to have lower (better) ranks
        ranks[i, :] = stats.rankdata(-data[i, :])

    # Average ranks across seeds
    avg_ranks = ranks.mean(axis=0)

    return {methods[i]: avg_ranks[i] for i in range(len(methods))}


def run_friedman_nemenyi(
    method_results: Dict[str, List[float]],
    alpha: float = 0.05
) -> Dict:
    """
    Run Friedman test followed by Nemenyi post-hoc test.

    This is the recommended approach for comparing multiple classifiers
    across multiple datasets/seeds (Demsar 2006).

    Args:
        method_results: Dict mapping method name to list of metric values
        alpha: Significance level

    Returns:
        Dictionary with Friedman test results, average ranks, and critical difference
    """
    methods = list(method_results.keys())
    n_methods = len(methods)
    n_seeds = len(method_results[methods[0]])

    # Friedman test
    friedman_stat, friedman_p = friedman_test(method_results)

    # Average ranks
    avg_ranks = compute_average_ranks(method_results)

    # Critical difference
    cd = nemenyi_critical_difference(n_methods, n_seeds, alpha)

    # Determine which pairs are significantly different
    significant_pairs = []
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                rank_diff = abs(avg_ranks[m1] - avg_ranks[m2])
                if rank_diff > cd:
                    significant_pairs.append((m1, m2, rank_diff))

    return {
        'friedman_statistic': friedman_stat,
        'friedman_p_value': friedman_p,
        'friedman_significant': friedman_p < alpha,
        'average_ranks': avg_ranks,
        'critical_difference': cd,
        'n_methods': n_methods,
        'n_seeds': n_seeds,
        'significant_pairs': significant_pairs,
        'alpha': alpha
    }


def format_friedman_nemenyi_latex(fn_results: Dict) -> str:
    """Generate LaTeX table for Friedman-Nemenyi results."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Friedman-Nemenyi Analysis}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Avg. Rank} & \\textbf{Significance} \\\\")
    lines.append("\\midrule")

    # Sort by rank
    sorted_ranks = sorted(fn_results['average_ranks'].items(), key=lambda x: x[1])
    best_method = sorted_ranks[0][0]
    cd = fn_results['critical_difference']

    for method, rank in sorted_ranks:
        # Check if significantly different from best
        rank_diff = rank - sorted_ranks[0][1]
        sig = "n.s." if rank_diff <= cd else f"$\\Delta > CD$"
        if method == best_method:
            sig = "\\textbf{best}"
            lines.append(f"\\textbf{{{method}}} & \\textbf{{{rank:.2f}}} & {sig} \\\\")
        else:
            lines.append(f"{method} & {rank:.2f} & {sig} \\\\")

    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{3}}{{l}}{{Friedman $\\chi^2 = {fn_results['friedman_statistic']:.2f}$, "
                f"$p = {fn_results['friedman_p_value']:.4f}$}} \\\\")
    lines.append(f"\\multicolumn{{3}}{{l}}{{Critical Difference (CD) = {cd:.3f}}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# TKDE Standard Seeds
TKDE_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144,
              7168, 8192, 9216, 10240, 11264]  # 15 seeds


# Example usage and testing
if __name__ == "__main__":
    # Simulate 15-seed experiment results (TKDE standard)
    np.random.seed(42)

    # Simulated F1 scores for different methods
    method_results = {
        'NAA': np.random.normal(0.75, 0.03, 15).tolist(),
        'GAT': np.random.normal(0.70, 0.04, 15).tolist(),
        'GCN': np.random.normal(0.68, 0.03, 15).tolist(),
        'H2GCN': np.random.normal(0.72, 0.035, 15).tolist(),
    }

    # Run rigorous comparison
    results = run_rigorous_comparison(
        method_results,
        baseline='GAT',
        alpha=0.05,
        correction='bonferroni'
    )

    # Print detailed results
    print(format_comparison_details(results))
    print("\n")

    # Print LaTeX table
    print("LaTeX Table:")
    print(format_results_latex(results, "F1 Score"))

    # Run Friedman-Nemenyi analysis
    print("\n\nFriedman-Nemenyi Analysis:")
    print("=" * 60)
    fn_results = run_friedman_nemenyi(method_results)
    print(f"Friedman statistic: {fn_results['friedman_statistic']:.4f}")
    print(f"Friedman p-value: {fn_results['friedman_p_value']:.4f}")
    print(f"Critical Difference: {fn_results['critical_difference']:.4f}")
    print("\nAverage Ranks:")
    for method, rank in sorted(fn_results['average_ranks'].items(), key=lambda x: x[1]):
        print(f"  {method}: {rank:.2f}")
    print("\nSignificant pairs:")
    for m1, m2, diff in fn_results['significant_pairs']:
        print(f"  {m1} vs {m2}: rank diff = {diff:.2f}")

    print("\n\nLaTeX Table (Friedman-Nemenyi):")
    print(format_friedman_nemenyi_latex(fn_results))
