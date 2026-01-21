#!/bin/bash
# Shell script to run NAA ablation study on all datasets
# Usage: bash run_ablation_study.sh

echo "========================================"
echo "NAA Ablation Study - Full Pipeline"
echo "========================================"
echo ""

# Check Python installation
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Please install Python 3.8+"
    exit 1
fi

echo "Python found:"
python --version
echo ""

# Set default seeds (5 seeds for reasonable runtime)
SEEDS="42 123 456 789 1024"

# Optional: Use 10 seeds for more robust results (slower)
# SEEDS="42 123 456 789 1024 2048 3072 4096 5120 6144"

echo "Using seeds: $SEEDS"
echo ""

# ========================================
# Elliptic Dataset
# ========================================
if [ -f "processed/elliptic_graph.pkl" ]; then
    echo "========================================"
    echo "Running ablation study on Elliptic"
    echo "========================================"
    python ablation_study.py \
        --data_path processed/elliptic_graph.pkl \
        --dataset_name "Elliptic" \
        --seeds $SEEDS \
        --output_dir ablation_results/elliptic

    if [ $? -ne 0 ]; then
        echo "ERROR: Ablation study failed on Elliptic"
        exit 1
    fi
    echo ""
    echo "Elliptic ablation completed successfully!"
    echo ""
else
    echo "WARNING: Elliptic data not found at processed/elliptic_graph.pkl"
    echo "Skipping Elliptic ablation study."
    echo ""
fi

# ========================================
# IEEE-CIS Dataset
# ========================================
if [ -f "processed/ieee_cis_graph.pkl" ]; then
    echo "========================================"
    echo "Running ablation study on IEEE-CIS"
    echo "========================================"
    python ablation_study.py \
        --data_path processed/ieee_cis_graph.pkl \
        --dataset_name "IEEE-CIS" \
        --seeds $SEEDS \
        --output_dir ablation_results/ieee_cis

    if [ $? -ne 0 ]; then
        echo "ERROR: Ablation study failed on IEEE-CIS"
        exit 1
    fi
    echo ""
    echo "IEEE-CIS ablation completed successfully!"
    echo ""
else
    echo "WARNING: IEEE-CIS data not found at processed/ieee_cis_graph.pkl"
    echo "Skipping IEEE-CIS ablation study."
    echo ""
fi

# ========================================
# YelpChi Dataset
# ========================================
if [ -f "processed/yelpchi_graph.pkl" ]; then
    echo "========================================"
    echo "Running ablation study on YelpChi"
    echo "========================================"
    python ablation_study.py \
        --data_path processed/yelpchi_graph.pkl \
        --dataset_name "YelpChi" \
        --seeds $SEEDS \
        --output_dir ablation_results/yelpchi

    if [ $? -ne 0 ]; then
        echo "ERROR: Ablation study failed on YelpChi"
        exit 1
    fi
    echo ""
    echo "YelpChi ablation completed successfully!"
    echo ""
else
    echo "WARNING: YelpChi data not found at processed/yelpchi_graph.pkl"
    echo "Skipping YelpChi ablation study."
    echo ""
fi

# ========================================
# Amazon Dataset
# ========================================
if [ -f "processed/amazon_graph.pkl" ]; then
    echo "========================================"
    echo "Running ablation study on Amazon"
    echo "========================================"
    python ablation_study.py \
        --data_path processed/amazon_graph.pkl \
        --dataset_name "Amazon" \
        --seeds $SEEDS \
        --output_dir ablation_results/amazon

    if [ $? -ne 0 ]; then
        echo "ERROR: Ablation study failed on Amazon"
        exit 1
    fi
    echo ""
    echo "Amazon ablation completed successfully!"
    echo ""
else
    echo "WARNING: Amazon data not found at processed/amazon_graph.pkl"
    echo "Skipping Amazon ablation study."
    echo ""
fi

# ========================================
# Summary
# ========================================
echo ""
echo "========================================"
echo "ABLATION STUDY COMPLETE"
echo "========================================"
echo ""
echo "Results saved to ablation_results/"
echo ""
echo "Generated files for each dataset:"
echo "  - ablation_table.tex          (Component ablation LaTeX table)"
echo "  - lambda_sensitivity_table.tex (Lambda sensitivity LaTeX table)"
echo "  - ablation_results.json        (Complete results in JSON)"
echo ""

# Check if any results exist
if [ -d "ablation_results/" ]; then
    echo "Datasets processed:"
    [ -d "ablation_results/elliptic/" ] && echo "  - Elliptic"
    [ -d "ablation_results/ieee_cis/" ] && echo "  - IEEE-CIS"
    [ -d "ablation_results/yelpchi/" ] && echo "  - YelpChi"
    [ -d "ablation_results/amazon/" ] && echo "  - Amazon"
else
    echo "WARNING: No ablation results found!"
    echo "Please check that dataset files exist in processed/ directory."
fi

echo ""
echo "Next steps:"
echo "1. Review LaTeX tables in ablation_results/{dataset}/*.tex"
echo "2. Copy tables to your paper LaTeX source"
echo "3. Analyze results in ablation_results/{dataset}/ablation_results.json"
echo ""
