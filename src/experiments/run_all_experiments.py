#!/usr/bin/env python3
"""
Run All Experiments
===================

One-click script to reproduce all experiments for Information Budget Theory paper.

Usage:
    python run_all_experiments.py

Author: FSD Framework
Date: 2025-01-16
"""

import subprocess
import sys
from pathlib import Path
import time


def run_experiment(script_name: str, description: str) -> bool:
    """Run a single experiment and return success status"""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*70}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            capture_output=False,
            timeout=1800  # 30 minutes timeout
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ {description} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"\n✗ {description} failed (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n✗ {description} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"\n✗ {description} error: {e}")
        return False


def main():
    print("="*70)
    print("INFORMATION BUDGET THEORY - FULL EXPERIMENT SUITE")
    print("="*70)
    print("\nThis will run all experiments needed for the paper.")
    print("Estimated total time: ~1 hour\n")

    experiments = [
        ("information_budget_experiment.py", "Core Experiments (Edge Shuffle, Feature Degradation, Same-h Pairs)"),
        ("csbm_falsifiable_experiment.py", "CSBM Falsifiable Prediction Test"),
        ("dual_heterophily_experiment.py", "Dual Heterophily Type Validation"),
        ("external_validation_experiment.py", "External Dataset Validation"),
        ("mlp_tuning_experiment.py", "MLP Systematic Tuning"),
        ("symmetric_tuning_experiment.py", "Symmetric MLP+GNN Tuning"),
    ]

    results = []
    total_start = time.time()

    for script, desc in experiments:
        success = run_experiment(script, desc)
        results.append((desc, success))

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    success_count = 0
    for desc, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "Y" if success else "N"
        print(f"  [{symbol}] {desc}")
        if success:
            success_count += 1

    print(f"\nTotal: {success_count}/{len(results)} experiments passed")
    print(f"Total time: {total_elapsed/60:.1f} minutes")

    # Check for result files
    print("\n" + "-"*70)
    print("Result Files:")
    result_files = [
        "information_budget_results.json",
        "csbm_falsifiable_results.json",
        "dual_heterophily_results.json",
        "external_validation_results.json",
        "mlp_tuning_results.json",
        "symmetric_tuning_results.json",
    ]

    for f in result_files:
        path = Path(__file__).parent / f
        status = "EXISTS" if path.exists() else "MISSING"
        print(f"  [{status}] {f}")

    print("\n" + "="*70)
    if success_count == len(results):
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    else:
        print(f"WARNING: {len(results) - success_count} experiment(s) failed")
    print("="*70)

    return success_count == len(results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
