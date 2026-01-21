#!/usr/bin/env python3
"""
Generate Venn Diagram and Visualization for Failure Set Analysis
================================================================

Creates visual representations of the failure patterns discovered in the FSD framework analysis.

Key findings:
- High-h (h > 0.7): 100% accuracy - both metrics work
- Low-h (h < 0.3): h-only works, δ_agg fails on 4 datasets
- Mid-h (0.3-0.7): UNCERTAINTY ZONE - 3 datasets fail both metrics

Author: FSD Framework
Date: 2025-12-23
"""

import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib_venn import venn2
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib or matplotlib-venn not available. Creating text-based visualization.")


def load_failure_data():
    """Load failure analysis results"""
    json_path = Path(__file__).parent / "venn_diagram_data.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def create_ascii_venn():
    """Create ASCII art Venn diagram"""

    data = load_failure_data()
    if data is None:
        print("No venn_diagram_data.json found. Run adaptive_selector.py first.")
        return

    only_delta = data['only_delta_agg_fails']
    only_h = data['only_h_fails']
    both = data['both_fail']
    neither = data['neither_fails']

    venn_art = """
+==============================================================================+
|                    FAILURE SET VENN DIAGRAM (N=19)                           |
+==============================================================================+
|                                                                              |
|                    +-----------------+     +-----------------+               |
|                   /                   \\   /                   \\              |
|                  /    Only delta_agg   \\ /      Only h         \\             |
|                 |      fails (4)       |       fails (0)       |             |
|                 |                      |                       |             |
|                 |  - Cornell          |                       |             |
|                 |  - Texas            |                       |             |
|                 |  - Wisconsin        |                       |             |
|                 |  - Actor            |                       |             |
|                 |                     |                       |             |
|                  \\                   /|\\                     /              |
|                   \\                 / | \\                   /               |
|                    \\---------------+  |  +---------------/                 |
|                                     |  |  |                                  |
|                                    Both fail (3)                             |
|                                    - cSBM-MidH                               |
|                                    - cSBM-NoisyF                             |
|                                    - cSBM-CleanF                             |
|                                                                              |
+------------------------------------------------------------------------------+
|                                                                              |
|  Neither fails (12): Elliptic, Cora, CiteSeer, PubMed, cSBM-HighH,           |
|                      cSBM-LowH, Inj-Cora, Inj-Amazon, Inj-Flickr,            |
|                      Chameleon, Squirrel, Flickr                             |
|                                                                              |
+------------------------------------------------------------------------------+
|  INTERPRETATION:                                                             |
|  - Only delta_agg fails: Low-h datasets (h<0.3) with delta_agg<5             |
|  - Both fail: Mid-h datasets (h~0.5) -> UNCERTAINTY ZONE                     |
|  - Neither fails: High-h (h>0.7) or extreme delta_agg -> confident           |
+==============================================================================+
"""
    print(venn_art)
    return venn_art


def create_homophily_zone_table():
    """Create ASCII table showing homophily zones"""

    table = """
+==============================================================================+
|                    HOMOPHILY ZONE ANALYSIS (N=19)                            |
+====================+=======+================================+================+
| Zone               | Count | Datasets                       | Accuracy       |
+====================+=======+================================+================+
| High (h > 0.7)     |   8   | Elliptic, Cora, CiteSeer,      | 100% (8/8)     |
|                    |       | PubMed, cSBM-HighH, Inj-*      | CONFIDENT      |
+--------------------+-------+--------------------------------+----------------+
| Low (h < 0.3)      |   7   | Cornell, Texas, Wisconsin,     | 100% (7/7)     |
|                    |       | Actor, Chameleon, Squirrel,    | using h-rule   |
|                    |       | cSBM-LowH                      | CONFIDENT      |
+--------------------+-------+--------------------------------+----------------+
| Mid (0.3 <= h<0.7) |   4   | Flickr (OK), cSBM-MidH (FAIL), | 25% (1/4)      |
|                    |       | cSBM-NoisyF (FAIL),            | UNCERTAINTY    |
|                    |       | cSBM-CleanF (FAIL)             | ZONE           |
+--------------------+-------+--------------------------------+----------------+
| TOTAL              |  19   |                                | 84.2% (16/19)  |
+====================+=======+================================+================+

KEY INSIGHT:
- Extreme homophily (h>0.7 or h<0.3) -> 100% prediction accuracy
- Mid-homophily (0.3-0.7) -> Uncertainty zone, fundamentally ambiguous
- The 3 failures are NOT random - they're in the principled "gray area"
"""
    print(table)
    return table


def create_decision_rule_diagram():
    """Create ASCII diagram of the decision rule"""

    diagram = """
+==============================================================================+
|              HOMOPHILY-BASED DECISION FRAMEWORK                              |
+==============================================================================+
|                                                                              |
|                              h (homophily)                                   |
|         <-------------------------------------------------------------->     |
|         0.0                    0.3        0.7                    1.0         |
|          |                      |          |                      |          |
|          |     LOW H ZONE       |  MID H   |    HIGH H ZONE       |          |
|          |    (h < 0.3)         |  ZONE    |    (h > 0.7)         |          |
|          |                      |          |                      |          |
|          |  +---------------+   |  +---+  |  +---------------+   |          |
|          |  | Don't trust   |   |  | ? |  |  | Trust         |   |          |
|          |  | structure     |   |  |   |  |  | structure     |   |          |
|          |  | -> Class B    |   |  |   |  |  | -> Class A    |   |          |
|          |  | (sampling)    |   |  |   |  |  | (mean-agg)    |   |          |
|          |  +---------------+   |  +---+  |  +---------------+   |          |
|          |                      |          |                      |          |
|          |  Accuracy: 100%      |  25%    |  Accuracy: 100%      |          |
|          |  (7/7 datasets)      |  (1/4)  |  (8/8 datasets)      |          |
|                                                                              |
+------------------------------------------------------------------------------+
|  ALGORITHM:                                                                  |
|  1. Compute h (homophily) for the dataset                                    |
|  2. IF h > 0.7:    -> Class A (Mean-aggregation: GCN, GAT, NAA)              |
|     IF h < 0.3:    -> Class B (Sampling: H2GCN, GraphSAGE)                   |
|     ELSE (0.3-0.7):-> UNCERTAIN - use delta_agg as secondary signal          |
|                      OR flag for manual review                               |
+==============================================================================+
"""
    print(diagram)
    return diagram


def create_matplotlib_venn():
    """Create matplotlib Venn diagram if available"""

    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Install with: pip install matplotlib matplotlib-venn")
        return None

    data = load_failure_data()
    if data is None:
        return None

    only_delta = len(data['only_delta_agg_fails'])  # 4
    only_h = len(data['only_h_fails'])              # 0
    both = len(data['both_fail'])                   # 3

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Venn diagram
    ax1 = axes[0]
    v = venn2(subsets=(only_delta, only_h, both),
              set_labels=('δ_agg fails', 'h fails'),
              ax=ax1)

    # Customize colors
    if v.get_patch_by_id('10'):
        v.get_patch_by_id('10').set_color('#ff9999')
        v.get_patch_by_id('10').set_alpha(0.7)
    if v.get_patch_by_id('01'):
        v.get_patch_by_id('01').set_color('#99ff99')
        v.get_patch_by_id('01').set_alpha(0.7)
    if v.get_patch_by_id('11'):
        v.get_patch_by_id('11').set_color('#ffff99')
        v.get_patch_by_id('11').set_alpha(0.7)

    ax1.set_title('Failure Set Overlap (N=19 datasets)\n3 datasets fail BOTH metrics')

    # Add annotations
    annotations = [
        "Only δ_agg fails (4):\nCornell, Texas,\nWisconsin, Actor",
        "Both fail (3):\ncSBM-MidH,\ncSBM-NoisyF,\ncSBM-CleanF",
        "Neither fails (12)"
    ]

    # Homophily zone bar chart
    ax2 = axes[1]
    zones = ['Low\n(h<0.3)', 'Mid\n(0.3-0.7)', 'High\n(h>0.7)']
    correct = [7, 1, 8]
    wrong = [0, 3, 0]

    x = range(len(zones))
    width = 0.35

    bars1 = ax2.bar([i - width/2 for i in x], correct, width, label='Correct', color='#99cc99')
    bars2 = ax2.bar([i + width/2 for i in x], wrong, width, label='Wrong', color='#ff9999')

    ax2.set_ylabel('Number of Datasets')
    ax2.set_title('Prediction Accuracy by Homophily Zone')
    ax2.set_xticks(x)
    ax2.set_xticklabels(zones)
    ax2.legend()

    # Add accuracy labels
    for i, (c, w) in enumerate(zip(correct, wrong)):
        total = c + w
        acc = c / total if total > 0 else 0
        ax2.annotate(f'{acc:.0%}', xy=(i, c + w + 0.3), ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "failure_analysis_venn.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    plt.close()
    return str(output_path)


def main():
    """Generate all visualizations"""

    print("=" * 80)
    print("GENERATING FAILURE ANALYSIS VISUALIZATIONS")
    print("=" * 80)

    # 1. ASCII Venn diagram
    print("\n1. ASCII VENN DIAGRAM")
    print("-" * 40)
    create_ascii_venn()

    # 2. Homophily zone table
    print("\n2. HOMOPHILY ZONE TABLE")
    print("-" * 40)
    create_homophily_zone_table()

    # 3. Decision rule diagram
    print("\n3. DECISION RULE DIAGRAM")
    print("-" * 40)
    create_decision_rule_diagram()

    # 4. Matplotlib Venn (if available)
    print("\n4. MATPLOTLIB VISUALIZATION")
    print("-" * 40)
    result = create_matplotlib_venn()
    if result:
        print(f"Created: {result}")
    else:
        print("Skipped (matplotlib not available)")

    # 5. Save ASCII visualizations to file
    output_path = Path(__file__).parent / "failure_analysis_ascii.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("FAILURE ANALYSIS VISUALIZATIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. VENN DIAGRAM\n")
        f.write(create_ascii_venn() or "")
        f.write("\n\n2. HOMOPHILY ZONE TABLE\n")
        f.write(create_homophily_zone_table() or "")
        f.write("\n\n3. DECISION RULE DIAGRAM\n")
        f.write(create_decision_rule_diagram() or "")

    print(f"\nASCII visualizations saved to: {output_path}")


if __name__ == '__main__':
    main()
