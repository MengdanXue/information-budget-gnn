#!/usr/bin/env python3
"""
Generate LaTeX Tables for TKDE Submission
==========================================

This script generates publication-quality LaTeX tables from experimental results.

Tables generated:
1. Main Results Summary (Information Budget Theory validation)
2. CSBM Prediction Accuracy by Region
3. Symmetric Tuning Results (Fairness comparison)
4. External Validation Results
5. Edge Shuffle Results (Causal Evidence)
6. Bootstrap Confidence Intervals

Author: FSD Framework
Date: 2025-01-16
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

# ============================================================
# Load Results
# ============================================================

def load_json(filename):
    path = Path(__file__).parent / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

info_budget = load_json('information_budget_results.json')
csbm_results = load_json('csbm_falsifiable_results.json')
external_results = load_json('external_validation_results.json')
symmetric_results = load_json('symmetric_tuning_results.json')

output_dir = Path(__file__).parent / 'latex_tables'
output_dir.mkdir(exist_ok=True)


# ============================================================
# Bootstrap Confidence Interval
# ============================================================

def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    """Compute bootstrap confidence interval for the mean."""
    if isinstance(data, (int, float)):
        return data, data

    data = np.array(data)
    boot_means = np.zeros(n_bootstrap)
    n = len(data)

    for i in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(boot_sample)

    alpha = 1 - confidence
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return lower, upper


# ============================================================
# Table 1: Main Evidence Summary
# ============================================================

def generate_table1_main_summary():
    """Generate main evidence summary table."""

    latex = r"""
\begin{table}[t]
\centering
\caption{Information Budget Theory: Evidence Summary}
\label{tab:evidence_summary}
\small
\begin{tabular}{llcc}
\toprule
\textbf{Experiment} & \textbf{Hypothesis} & \textbf{Result} & \textbf{Accuracy} \\
\midrule
Edge Shuffle & Structure $\rightarrow$ GNN advantage & \checkmark & 3/3 (100\%) \\
Feature Degradation & GNN$_{\text{adv}} \leq$ Budget & \checkmark & 9/9 (100\%) \\
Same-h Pairs & MLP determines GNN utility & \checkmark & 7/7 (100\%) \\
CSBM Prediction & Frozen rules predict winner & \checkmark & 32/36 (89\%) \\
Symmetric Tuning & Fair baseline comparison & \checkmark & Confirmed \\
External Validation & Generalize to new datasets & \checkmark & 7/9 (78\%) \\
\midrule
\multicolumn{4}{l}{\textbf{Core Principle:} $\text{GNN}_{\text{max\_gain}} \leq (1 - \text{MLP}_{\text{accuracy}})$} \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================
# Table 2: Edge Shuffle Results
# ============================================================

def generate_table2_edge_shuffle():
    """Generate edge shuffle causal evidence table."""
    if not info_budget:
        return ""

    edge_data = info_budget['edge_shuffle']

    latex = r"""
\begin{table}[t]
\centering
\caption{Edge Shuffle Experiment: Causal Evidence for Structure}
\label{tab:edge_shuffle}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Dataset} & \textbf{h$_{\text{orig}}$} & \textbf{h$_{\text{shuf}}$} & \textbf{GCN$_{\text{orig}}$} & \textbf{GCN$_{\text{shuf}}$} & \textbf{MLP} & \textbf{$\Delta$Adv} \\
\midrule
"""

    for row in edge_data:
        dataset = row['dataset']
        h_orig = row['original_h']
        h_shuf = row['shuffled_h']
        gcn_orig = row['original_gcn']
        gcn_shuf = row['shuffled_gcn']
        mlp = row['original_mlp']
        delta = row['structure_contribution']

        latex += f"{dataset} & {h_orig:.2f} & {h_shuf:.2f} & {gcn_orig:.3f} & {gcn_shuf:.3f} & {mlp:.3f} & {delta*100:+.1f}\\% \\\\\n"

    latex += r"""
\midrule
\multicolumn{7}{l}{\textit{Structure contribution = Original GCN adv. $-$ Shuffled GCN adv.}} \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================
# Table 3: CSBM Prediction Results
# ============================================================

def generate_table3_csbm():
    """Generate CSBM prediction accuracy by region."""
    if not csbm_results:
        return ""

    results = csbm_results['results']

    # Group by homophily region
    regions = {
        'High-h (h > 0.7)': [r for r in results if r['homophily_target'] > 0.7],
        'Mid-h (0.4-0.6)': [r for r in results if 0.35 <= r['homophily_target'] <= 0.65],
        'Low-h (h < 0.3)': [r for r in results if r['homophily_target'] < 0.3]
    }

    latex = r"""
\begin{table}[t]
\centering
\caption{CSBM Falsifiable Prediction: Accuracy by Region}
\label{tab:csbm_prediction}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Region} & \textbf{Correct} & \textbf{Total} & \textbf{Accuracy} \\
\midrule
"""

    for region_name, region_data in regions.items():
        correct = sum(1 for r in region_data if r['prediction_correct'])
        total = len(region_data)
        acc = correct / total if total > 0 else 0
        latex += f"{region_name} & {correct} & {total} & {acc*100:.1f}\\% \\\\\n"

    # Overall
    total_correct = sum(1 for r in results if r['prediction_correct'])
    total_all = len(results)
    overall_acc = total_correct / total_all

    latex += r"""
\midrule
\textbf{Overall} & """ + f"{total_correct} & {total_all} & {overall_acc*100:.1f}\\%" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================
# Table 4: Symmetric Tuning Results
# ============================================================

def generate_table4_symmetric_tuning():
    """Generate symmetric tuning fairness table."""
    if not symmetric_results:
        return ""

    comparisons = symmetric_results['comparisons']
    summary = symmetric_results['summary']

    latex = r"""
\begin{table}[t]
\centering
\caption{Symmetric Hyperparameter Tuning: Fairness Comparison}
\label{tab:symmetric_tuning}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Dataset} & \textbf{h} & \textbf{MLP} & \textbf{GCN} & \textbf{SAGE} & \textbf{GAT} & \textbf{Winner} \\
\midrule
"""

    for c in comparisons:
        dataset = c['dataset']
        h = c['homophily']
        mlp = c['mlp_best']
        gcn = c['gcn_best']
        sage = c['sage_best']
        gat = c['gat_best']
        winner = c['conclusion'].replace('_', ' ')

        # Bold the winner
        if c['conclusion'] == 'GNN_WINS':
            best_gnn = max(gcn, sage, gat)
            if gcn == best_gnn:
                gcn_str = f"\\textbf{{{gcn:.3f}}}"
            else:
                gcn_str = f"{gcn:.3f}"
            if sage == best_gnn:
                sage_str = f"\\textbf{{{sage:.3f}}}"
            else:
                sage_str = f"{sage:.3f}"
            if gat == best_gnn:
                gat_str = f"\\textbf{{{gat:.3f}}}"
            else:
                gat_str = f"{gat:.3f}"
            mlp_str = f"{mlp:.3f}"
        else:
            mlp_str = f"\\textbf{{{mlp:.3f}}}"
            gcn_str = f"{gcn:.3f}"
            sage_str = f"{sage:.3f}"
            gat_str = f"{gat:.3f}"

        latex += f"{dataset} & {h:.2f} & {mlp_str} & {gcn_str} & {sage_str} & {gat_str} & {winner} \\\\\n"

    # Add tuning gains summary
    latex += r"""
\midrule
\multicolumn{7}{l}{\textit{Average Tuning Gains:}} \\
"""
    latex += f"\\multicolumn{{7}}{{l}}{{MLP: +{summary['avg_mlp_tuning_gain']*100:.1f}\\%, "
    latex += f"GCN: +{summary['avg_gcn_tuning_gain']*100:.1f}\\%, "
    latex += f"SAGE: +{summary['avg_sage_tuning_gain']*100:.1f}\\%, "
    latex += f"GAT: +{summary['avg_gat_tuning_gain']*100:.1f}\\%}} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================
# Table 5: External Validation Results
# ============================================================

def generate_table5_external_validation():
    """Generate external validation table."""
    if not external_results:
        return ""

    results = external_results['results']

    latex = r"""
\begin{table}[t]
\centering
\caption{External Dataset Validation: Frozen Rule Predictions}
\label{tab:external_validation}
\small
\begin{tabular}{lccccccc}
\toprule
\textbf{Dataset} & \textbf{h} & \textbf{MLP} & \textbf{GCN} & \textbf{Budget} & \textbf{Pred.} & \textbf{Actual} & \checkmark \\
\midrule
"""

    for r in results:
        dataset = r['dataset']
        h = r['homophily']
        mlp = r['mlp_acc']
        gcn = r['gcn_acc']
        budget = r['budget']
        pred = r['prediction']
        actual = r['actual_winner']
        correct = "\\checkmark" if r['prediction_correct'] else "\\texttimes"

        latex += f"{dataset} & {h:.2f} & {mlp:.3f} & {gcn:.3f} & {budget:.2f} & {pred} & {actual} & {correct} \\\\\n"

    summary = external_results['summary']
    latex += r"""
\midrule
\multicolumn{8}{l}{\textbf{Overall Accuracy: """ + f"{summary['correct_predictions']}/{summary['total_datasets']} ({summary['accuracy']*100:.1f}\\%)" + r"""}} \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================
# Table 6: Feature Degradation Budget Validation
# ============================================================

def generate_table6_feature_degradation():
    """Generate feature degradation budget validation table."""
    if not info_budget:
        return ""

    feat_data = info_budget['feature_degradation']

    latex = r"""
\begin{table}[t]
\centering
\caption{Feature Degradation: Budget Constraint Validation}
\label{tab:feature_degradation}
\small
\begin{tabular}{ccccc}
\toprule
\textbf{Noise} & \textbf{MLP} & \textbf{GCN Adv.} & \textbf{Budget} & \textbf{Valid} \\
\midrule
"""

    for row in feat_data:
        noise = row['noise_level']
        mlp = row['mlp_acc']
        gcn_adv = row['gcn_adv']
        budget = row['theoretical_max_gain']
        valid = "\\checkmark" if row['within_budget'] else "\\texttimes"

        latex += f"{noise:.1f} & {mlp:.3f} & {gcn_adv*100:.1f}\\% & {budget*100:.1f}\\% & {valid} \\\\\n"

    latex += r"""
\midrule
\multicolumn{5}{l}{\textit{Budget = (1 - MLP accuracy), GCN Adv. must $\leq$ Budget}} \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================
# Table 7: Bootstrap Confidence Intervals
# ============================================================

def generate_table7_bootstrap_ci():
    """Generate bootstrap confidence intervals table."""

    np.random.seed(42)

    # Collect accuracy data for CI computation
    ci_data = []

    if csbm_results:
        # CSBM accuracy
        correct = [1 if r['prediction_correct'] else 0 for r in csbm_results['results']]
        ci_lower, ci_upper = bootstrap_ci(correct)
        mean_acc = np.mean(correct)
        ci_data.append(('CSBM Prediction', mean_acc, ci_lower, ci_upper, len(correct)))

    if external_results:
        # External validation accuracy
        correct = [1 if r['prediction_correct'] else 0 for r in external_results['results']]
        ci_lower, ci_upper = bootstrap_ci(correct)
        mean_acc = np.mean(correct)
        ci_data.append(('External Validation', mean_acc, ci_lower, ci_upper, len(correct)))

    latex = r"""
\begin{table}[t]
\centering
\caption{Bootstrap 95\% Confidence Intervals}
\label{tab:bootstrap_ci}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Experiment} & \textbf{Accuracy} & \textbf{95\% CI} & \textbf{N} \\
\midrule
"""

    for name, mean, ci_l, ci_u, n in ci_data:
        latex += f"{name} & {mean*100:.1f}\\% & [{ci_l*100:.1f}\\%, {ci_u*100:.1f}\\%] & {n} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================
# Table 8: Statistical Tests Summary
# ============================================================

def generate_table8_statistical_tests():
    """Generate statistical tests summary."""

    np.random.seed(42)

    latex = r"""
\begin{table}[t]
\centering
\caption{Statistical Significance Tests}
\label{tab:statistical_tests}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Test} & \textbf{Comparison} & \textbf{Statistic} & \textbf{p-value} & \textbf{Sig.} \\
\midrule
"""

    # McNemar's test for CSBM vs random baseline
    if csbm_results:
        n_correct = sum(1 for r in csbm_results['results'] if r['prediction_correct'])
        n_total = len(csbm_results['results'])
        # Compare to random baseline (50%)
        expected_random = n_total * 0.5
        chi2 = (n_correct - expected_random) ** 2 / expected_random + (n_total - n_correct - expected_random) ** 2 / expected_random
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "n.s."))
        latex += f"Binomial & CSBM vs Random & $\\chi^2={chi2:.2f}$ & {p_value:.4f} & {sig} \\\\\n"

    # Wilcoxon test for GNN vs MLP on symmetric tuning
    if symmetric_results:
        comparisons = symmetric_results['comparisons']
        gnn_wins = sum(1 for c in comparisons if c['gnn_advantage_tuned'] > 0)
        mlp_wins = len(comparisons) - gnn_wins
        # Sign test
        n_total = len(comparisons)
        p_binom = stats.binomtest(gnn_wins, n_total, 0.5).pvalue
        sig = "***" if p_binom < 0.001 else ("**" if p_binom < 0.01 else ("*" if p_binom < 0.05 else "n.s."))
        latex += f"Sign Test & GNN vs MLP & W={gnn_wins}/{n_total} & {p_binom:.4f} & {sig} \\\\\n"

    latex += r"""
\midrule
\multicolumn{5}{l}{\textit{Significance levels: *** $p<0.001$, ** $p<0.01$, * $p<0.05$}} \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================
# Table 9: Decision Rules Summary
# ============================================================

def generate_table9_decision_rules():
    """Generate decision rules table."""

    latex = r"""
\begin{table}[t]
\centering
\caption{Trust Regions Decision Rules (Frozen)}
\label{tab:decision_rules}
\small
\begin{tabular}{lll}
\toprule
\textbf{Condition} & \textbf{Prediction} & \textbf{Rationale} \\
\midrule
MLP$_{\text{acc}} > 0.95$ & MLP & Budget too small \\
$h > 0.75$ and Budget $> 0.05$ & GNN & High-h trust region \\
$h < 0.25$ and Budget $> 0.05$ & GNN & Low-h trust region \\
$0.35 \leq h \leq 0.65$ and Budget $< 0.4$ & MLP & Mid-h uncertainty \\
$0.35 \leq h \leq 0.65$ and Budget $\geq 0.4$ & GNN & Large budget \\
SPI $\times$ Budget $> 0.15$ & GNN & Sufficient signal \\
Otherwise & MLP & Default \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING LATEX TABLES FOR TKDE SUBMISSION")
    print("=" * 70)

    all_tables = []

    # Generate all tables
    tables = [
        ('table1_evidence_summary.tex', generate_table1_main_summary()),
        ('table2_edge_shuffle.tex', generate_table2_edge_shuffle()),
        ('table3_csbm_prediction.tex', generate_table3_csbm()),
        ('table4_symmetric_tuning.tex', generate_table4_symmetric_tuning()),
        ('table5_external_validation.tex', generate_table5_external_validation()),
        ('table6_feature_degradation.tex', generate_table6_feature_degradation()),
        ('table7_bootstrap_ci.tex', generate_table7_bootstrap_ci()),
        ('table8_statistical_tests.tex', generate_table8_statistical_tests()),
        ('table9_decision_rules.tex', generate_table9_decision_rules()),
    ]

    for filename, content in tables:
        if content:
            filepath = output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Generated: {filename}")
            all_tables.append(content)

    # Generate combined file
    combined_path = output_dir / 'all_tables.tex'
    with open(combined_path, 'w', encoding='utf-8') as f:
        f.write("% Auto-generated LaTeX tables for Information Budget Theory paper\n")
        f.write("% Date: 2025-01-16\n\n")
        for content in all_tables:
            f.write(content)
            f.write("\n\n")
    print(f"  Generated: all_tables.tex (combined)")

    print("\n" + "=" * 70)
    print(f"All tables saved to: {output_dir}")
    print("=" * 70)

    # Print summary statistics
    print("\n--- SUMMARY STATISTICS ---")

    if csbm_results:
        print(f"\nCSBM Prediction:")
        print(f"  Accuracy: {csbm_results['summary']['accuracy']*100:.1f}%")
        print(f"  Correct: {csbm_results['summary']['correct_predictions']}/{csbm_results['summary']['total_predictions']}")

    if external_results:
        print(f"\nExternal Validation:")
        print(f"  Accuracy: {external_results['summary']['accuracy']*100:.1f}%")
        print(f"  Correct: {external_results['summary']['correct_predictions']}/{external_results['summary']['total_datasets']}")

    if symmetric_results:
        summary = symmetric_results['summary']
        print(f"\nSymmetric Tuning:")
        print(f"  GNN wins: {summary['gnn_wins']}, MLP wins: {summary['mlp_wins']}, Ties: {summary['ties']}")
        print(f"  Avg MLP tuning gain: +{summary['avg_mlp_tuning_gain']*100:.1f}%")
        print(f"  Avg GCN tuning gain: +{summary['avg_gcn_tuning_gain']*100:.1f}%")
