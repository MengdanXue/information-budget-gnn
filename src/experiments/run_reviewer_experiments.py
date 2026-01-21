"""
Run All Reviewer-Requested Experiments
Master script to run all experiments required for paper revision
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and report status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            print(f"\n[SUCCESS] {description}")
            return True
        else:
            print(f"\n[FAILED] {description} (return code: {result.returncode})")
            return False
    except Exception as e:
        print(f"\n[ERROR] {description}: {e}")
        return False


def main():
    print("="*60)
    print("REVIEWER-REQUESTED EXPERIMENTS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    experiments = [
        # 1. CSBM Deviation Metrics (fastest, no GPU intensive)
        ("csbm_deviation_metrics.py",
         "CSBM Deviation Metrics (Clustering, Degree Gini, Feature Gaussianity)"),

        # 2. OGB Full Experiments (GPU intensive)
        ("ogb_full_experiments.py",
         "OGB Full Experiments (MLP, GCN, GraphSAGE on ogbn-arxiv, ogbn-products)"),

        # 3. OGB Heterophily Methods (GPU intensive)
        ("ogb_heterophily_methods.py",
         "OGB Heterophily Methods (H2GCN, LINKX, GPR-GNN on ogbn-arxiv)"),
    ]

    results = {}
    for script, desc in experiments:
        success = run_script(script, desc)
        results[script] = success

    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    for script, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {script}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} experiments completed successfully")

    # List output files
    print("\nOutput files generated:")
    output_files = [
        'csbm_deviation_metrics.json',
        'ogb_full_experiments_results.json',
        'ogb_heterophily_methods_results.json',
    ]
    for f in output_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  [EXISTS] {f} ({size:,} bytes)")
        else:
            print(f"  [MISSING] {f}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
