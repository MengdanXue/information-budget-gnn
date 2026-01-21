"""
Complete Experiment Pipeline for Path B Revision

Execute this script step by step to generate all required data for paper revision.

Usage:
    # Step 1: Build IEEE-CIS graph and compute rho_FS (BEFORE seeing results)
    python run_experiments.py --step 1 --data_dir ./ieee_cis_data

    # Step 2: Run 10-seed experiments on all datasets
    python run_experiments.py --step 2

    # Step 3: Statistical analysis
    python run_experiments.py --step 3

    # Step 4: Generate LaTeX tables
    python run_experiments.py --step 4

Prerequisites:
    1. Download IEEE-CIS data from: https://www.kaggle.com/c/ieee-fraud-detection/data
    2. Place train_transaction.csv and train_identity.csv in ./ieee_cis_data/
    3. Install: pip install torch torch_geometric scipy sklearn pandas numpy
"""

import argparse
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List
import numpy as np

# Configuration
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]  # 10 seeds
DATASETS = ['elliptic', 'yelpchi', 'ieee_cis']  # Removed Amazon (label leakage)
METHODS = ['GCN', 'GAT', 'GraphSAGE', 'H2GCN', 'MixHop', 'NAA-GCN', 'NAA-GAT']

OUTPUT_DIR = './experiment_results'


def step1_build_graph_and_predict(data_dir: str):
    """
    Step 1: Build IEEE-CIS graph and compute rho_FS

    IMPORTANT: Record the FSD prediction BEFORE running any experiments.
    This ensures the prediction is truly a priori, not post-hoc.
    """
    print("=" * 60)
    print("STEP 1: Build IEEE-CIS Graph and Compute rho_FS")
    print("=" * 60)
    print("\nIMPORTANT: Record the FSD prediction NOW, before seeing any results!")
    print("This timestamp proves the prediction was made a priori.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Import and run graph builder
    try:
        from ieee_cis_graph_builder import load_ieee_cis_data, build_entity_graph, \
            process_features, compute_rho_fs, create_data_splits

        # Load data
        df = load_ieee_cis_data(data_dir)

        # Build graph
        entity_columns = ['card1', 'card2', 'addr1', 'P_emaildomain', 'DeviceInfo']
        edge_index, edge_stats = build_entity_graph(df, entity_columns)

        # Process features
        features = process_features(df)

        # Compute rho_FS (THE KEY STEP - before any model training)
        rho_fs_results = compute_rho_fs(edge_index, features)

        # Make and record prediction
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'dataset': 'IEEE-CIS',
            'rho_fs': rho_fs_results['rho_fs'],
            'feature_dim': features.shape[1],
            'num_nodes': features.shape[0],
            'num_edges': edge_index.shape[1],
            'fsd_prediction': rho_fs_results['fsd_prediction'],
            'prediction_rule': 'If rho_FS > 0.15 and dim > 50: NAA; If rho_FS < -0.05: H2GCN; Else: No clear winner'
        }

        # Save prediction with timestamp
        prediction_file = os.path.join(OUTPUT_DIR, 'ieee_cis_prediction.json')
        with open(prediction_file, 'w') as f:
            json.dump(prediction_record, f, indent=2)

        print("\n" + "=" * 60)
        print("A PRIORI PREDICTION RECORDED")
        print("=" * 60)
        print(f"Timestamp: {prediction_record['timestamp']}")
        print(f"Dataset: IEEE-CIS")
        print(f"rho_FS: {prediction_record['rho_fs']:.4f}")
        print(f"Feature Dimension: {prediction_record['feature_dim']}")
        print(f"\nFSD PREDICTION: {prediction_record['fsd_prediction']}")
        print(f"\nPrediction saved to: {prediction_file}")
        print("=" * 60)

        # Save processed data for experiments
        labels = df['isFraud'].values
        train_mask, val_mask, test_mask = create_data_splits(df, split_type='temporal')

        data = {
            'edge_index': edge_index,
            'features': features,
            'labels': labels,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'rho_fs_results': rho_fs_results,
        }

        data_file = os.path.join(OUTPUT_DIR, 'ieee_cis_processed.pkl')
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nProcessed data saved to: {data_file}")

        return prediction_record

    except ImportError as e:
        print(f"Error: Could not import graph builder. Make sure ieee_cis_graph_builder.py exists.")
        print(f"Details: {e}")
        return None
    except FileNotFoundError as e:
        print(f"Error: Data files not found in {data_dir}")
        print("Please download IEEE-CIS data from: https://www.kaggle.com/c/ieee-fraud-detection/data")
        print(f"Details: {e}")
        return None


def step2_run_experiments():
    """
    Step 2: Run 10-seed experiments on all datasets

    This generates results for:
    - Elliptic (existing, need to rerun with 10 seeds)
    - YelpChi (existing, need to rerun with 10 seeds)
    - IEEE-CIS (new dataset)
    """
    print("=" * 60)
    print("STEP 2: Run 10-Seed Experiments")
    print("=" * 60)

    print("\nThis step requires running GNN training code.")
    print("Please execute the following for each dataset:\n")

    for dataset in DATASETS:
        print(f"\n--- {dataset.upper()} ---")
        for method in METHODS:
            print(f"python train.py --dataset {dataset} --method {method} --seeds {' '.join(map(str, SEEDS))}")

    print("\n" + "=" * 60)
    print("MANUAL EXECUTION REQUIRED")
    print("=" * 60)
    print("""
After running experiments, save results in the following format:

experiment_results/
├── elliptic_results.json
├── yelpchi_results.json
└── ieee_cis_results.json

Each JSON file should contain:
{
    "GCN": {"auc": [0.xx, ...], "f1": [0.xx, ...]},  # 10 values each
    "GAT": {"auc": [...], "f1": [...]},
    "NAA-GCN": {"auc": [...], "f1": [...]},
    ...
}
""")

    # Create template files
    for dataset in DATASETS:
        template = {method: {"auc": [], "f1": [], "precision": [], "recall": []}
                   for method in METHODS}
        template_file = os.path.join(OUTPUT_DIR, f'{dataset}_results_template.json')
        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"Template created: {template_file}")


def step3_statistical_analysis():
    """
    Step 3: Run rigorous statistical analysis on experiment results
    """
    print("=" * 60)
    print("STEP 3: Statistical Analysis")
    print("=" * 60)

    try:
        from statistical_analysis import run_rigorous_comparison, format_comparison_details
    except ImportError:
        print("Error: Could not import statistical_analysis.py")
        return

    results_summary = {}

    for dataset in DATASETS:
        results_file = os.path.join(OUTPUT_DIR, f'{dataset}_results.json')

        if not os.path.exists(results_file):
            print(f"\nWarning: {results_file} not found. Skipping {dataset}.")
            continue

        print(f"\n--- Analyzing {dataset.upper()} ---")

        with open(results_file, 'r') as f:
            results = json.load(f)

        # Check we have 10 seeds
        for method, metrics in results.items():
            if len(metrics.get('auc', [])) < 10:
                print(f"Warning: {method} has only {len(metrics.get('auc', []))} seeds (need 10)")

        # Run analysis for AUC
        auc_results = {method: metrics['auc'] for method, metrics in results.items()
                      if len(metrics.get('auc', [])) >= 10}

        if auc_results:
            analysis = run_rigorous_comparison(
                auc_results,
                baseline='GCN',
                alpha=0.05,
                correction='bonferroni'
            )

            print(format_comparison_details(analysis))

            results_summary[dataset] = {
                'auc_analysis': analysis,
                'n_seeds': analysis.n_seeds
            }

    # Save analysis results
    summary_file = os.path.join(OUTPUT_DIR, 'statistical_analysis_summary.pkl')
    with open(summary_file, 'wb') as f:
        pickle.dump(results_summary, f)
    print(f"\nAnalysis saved to: {summary_file}")


def step4_generate_latex():
    """
    Step 4: Generate LaTeX tables for paper
    """
    print("=" * 60)
    print("STEP 4: Generate LaTeX Tables")
    print("=" * 60)

    try:
        from statistical_analysis import run_rigorous_comparison, format_results_latex
    except ImportError:
        print("Error: Could not import statistical_analysis.py")
        return

    latex_output = []

    for dataset in DATASETS:
        results_file = os.path.join(OUTPUT_DIR, f'{dataset}_results.json')

        if not os.path.exists(results_file):
            continue

        with open(results_file, 'r') as f:
            results = json.load(f)

        # AUC table
        auc_results = {method: metrics['auc'] for method, metrics in results.items()
                      if len(metrics.get('auc', [])) >= 10}

        if auc_results:
            analysis = run_rigorous_comparison(auc_results, baseline='GCN')
            latex = format_results_latex(analysis, f"AUC on {dataset.upper()}")
            latex_output.append(f"% {dataset.upper()} AUC Table\n{latex}\n")

        # F1 table
        f1_results = {method: metrics['f1'] for method, metrics in results.items()
                     if len(metrics.get('f1', [])) >= 10}

        if f1_results:
            analysis = run_rigorous_comparison(f1_results, baseline='GCN')
            latex = format_results_latex(analysis, f"F1 on {dataset.upper()}")
            latex_output.append(f"% {dataset.upper()} F1 Table\n{latex}\n")

    # Save LaTeX
    latex_file = os.path.join(OUTPUT_DIR, 'latex_tables.tex')
    with open(latex_file, 'w') as f:
        f.write('\n\n'.join(latex_output))

    print(f"LaTeX tables saved to: {latex_file}")
    print("\nCopy these tables to your paper's experiments section.")


def main():
    parser = argparse.ArgumentParser(description='Run experiment pipeline for Path B revision')
    parser.add_argument('--step', type=int, required=True, choices=[1, 2, 3, 4],
                       help='Which step to execute (1-4)')
    parser.add_argument('--data_dir', type=str, default='./ieee_cis_data',
                       help='Directory containing IEEE-CIS data (for step 1)')
    args = parser.parse_args()

    if args.step == 1:
        step1_build_graph_and_predict(args.data_dir)
    elif args.step == 2:
        step2_run_experiments()
    elif args.step == 3:
        step3_statistical_analysis()
    elif args.step == 4:
        step4_generate_latex()


if __name__ == '__main__':
    main()
