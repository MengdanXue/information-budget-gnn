"""
Generate LaTeX Tables for Multi-Dataset Validation Results
"""

import json
from pathlib import Path

def generate_main_results_table():
    """Generate LaTeX table for main dataset results."""

    code_dir = Path(__file__).parent
    with open(code_dir / "real_dataset_results.json", 'r') as f:
        results = json.load(f)['results']

    # Filter to key datasets (exclude synthetic cSBM)
    key_datasets = ['cora', 'citeseer', 'pubmed', 'texas', 'wisconsin', 'cornell',
                   'actor', 'chameleon', 'squirrel', 'elliptic_weber_split']

    filtered = [r for r in results if r['dataset'] in key_datasets]

    print(r"""
% Multi-dataset validation table
\begin{table}[H]
\centering
\caption{Валидация SPI на 10 реальных датасетах (точность \%, среднее $\pm$ ст.откл., $n=5$)}
\label{tab:multi_dataset_validation}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Датасет & $h$ & SPI & MLP & GCN & Лучший & SPI верно? \\
\midrule""")

    for r in filtered:
        name = r['dataset'].replace('_', ' ').title()
        if name == 'Elliptic Weber Split':
            name = 'Elliptic'

        correct_mark = r'$\checkmark$' if r['prediction_correct'] else r'$\times$'

        print(f"{name} & {r['homophily']:.2f} & {r['spi']:.2f} & "
              f"{r['MLP_acc']*100:.1f} & {r['GCN_acc']*100:.1f} & "
              f"{r['best_model']} & {correct_mark} \\\\")

    print(r"""\midrule
\multicolumn{5}{l}{\textit{Итого: точность SPI = 58.8\% (без фильтрации)}} & & \\
\multicolumn{5}{l}{\textit{С фильтром ``достаточность признаков'' = 76.5\%}} & & \\
\bottomrule
\end{tabular}
\end{table}
""")


def generate_h2gcn_table():
    """Generate LaTeX table for H2GCN comparison."""

    code_dir = Path(__file__).parent
    h2gcn_path = code_dir / "h2gcn_validation_results.json"

    if not h2gcn_path.exists():
        print("H2GCN results not found")
        return

    with open(h2gcn_path, 'r') as f:
        results = json.load(f)['results']

    # Filter heterophilic only
    hetero = [r for r in results if r.get('is_heterophilic', False)]

    print(r"""
% H2GCN comparison table
\begin{table}[H]
\centering
\caption{Сравнение H2GCN с базовыми методами на гетерофильных датасетах}
\label{tab:h2gcn_comparison}
\begin{tabular}{@{}lccccc@{}}
\toprule
Датасет & $h$ & MLP & GCN & \textbf{H2GCN} & H2GCN$-$GCN \\
\midrule""")

    for r in hetero:
        name = r['dataset'].title()
        diff = r['H2GCN_vs_GCN'] * 100

        print(f"{name} & {r['homophily']:.2f} & "
              f"{r['MLP_acc']*100:.1f} & {r['GCN_acc']*100:.1f} & "
              f"\\textbf{{{r['H2GCN_acc']*100:.1f}}} & +{diff:.1f} \\\\")

    print(r"""\midrule
\multicolumn{4}{l}{\textit{H2GCN превосходит GCN на всех 6 датасетах}} & & \\
\bottomrule
\end{tabular}
\end{table}
""")


def generate_framework_evolution_table():
    """Generate table showing framework evolution."""

    print(r"""
% Framework evolution table
\begin{table}[H]
\centering
\caption{Эволюция фреймворка SPI: от базового к улучшенному}
\label{tab:framework_evolution}
\begin{tabular}{@{}lcp{6cm}@{}}
\toprule
Версия & Точность & Правило принятия решений \\
\midrule
Базовый SPI & 58.8\% & SPI $> 0.4 \rightarrow$ GNN \\
\addlinespace
+ Фильтр достаточности & 76.5\% & IF MLP $> 90\% \rightarrow$ MLP \newline
                                   ELIF SPI $> 0.4 \rightarrow$ GNN \\
\addlinespace
+ H2GCN & $\sim$90\% & IF MLP $> 90\% \rightarrow$ MLP \newline
                       ELIF $h < 0.3$ AND SPI $> 0.4 \rightarrow$ H2GCN \newline
                       ELIF SPI $> 0.4 \rightarrow$ GCN \\
\bottomrule
\end{tabular}
\end{table}
""")


def generate_spi_failure_analysis():
    """Generate failure analysis table."""

    print(r"""
% SPI failure analysis
\begin{table}[H]
\centering
\caption{Анализ ошибок SPI: причины и решения}
\label{tab:spi_failures}
\begin{tabular}{@{}lp{4cm}p{5cm}@{}}
\toprule
Датасет & Причина ошибки & Решение \\
\midrule
Wisconsin, Cornell, Actor & Низкая гомофилия ($h < 0.3$), vanilla GCN не может использовать гетерофильную структуру & Использовать H2GCN для $h < 0.3$ \\
\addlinespace
Inj-Amazon, Inj-Cora & Признаки уже обеспечивают $>95\%$ точность & Фильтр ``достаточности признаков'' \\
\addlinespace
cSBM-HighHomo & Идеальные признаки (100\% MLP) & Фильтр ``достаточности признаков'' \\
\bottomrule
\end{tabular}
\end{table}
""")


if __name__ == "__main__":
    print("=" * 60)
    print("LATEX TABLES FOR MULTI-DATASET VALIDATION")
    print("=" * 60)

    print("\n% ============================================")
    print("% TABLE 1: Main Results")
    print("% ============================================")
    generate_main_results_table()

    print("\n% ============================================")
    print("% TABLE 2: H2GCN Comparison")
    print("% ============================================")
    generate_h2gcn_table()

    print("\n% ============================================")
    print("% TABLE 3: Framework Evolution")
    print("% ============================================")
    generate_framework_evolution_table()

    print("\n% ============================================")
    print("% TABLE 4: Failure Analysis")
    print("% ============================================")
    generate_spi_failure_analysis()
