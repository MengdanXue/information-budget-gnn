#!/bin/bash
# IEEE-CIS Full Pipeline - From Data to Validation Report
# This script runs the complete prior prediction experiment

set -e  # Exit on error

# Configuration
DATA_DIR="./ieee_cis_data"
PROCESSED_DIR="./processed"
OUTPUT_DIR="./prior_prediction_results"
DEVICE="cuda"  # Change to "cpu" if no GPU

# Seeds for reproducibility
SEEDS="42 123 456 789 1024 2048 3072 4096 5120 6144"

# Methods to evaluate
METHODS="GCN GAT GraphSAGE H2GCN FAGCN GPRGNN NAA-GCN DAAA"

echo "========================================================================"
echo "IEEE-CIS PRIOR PREDICTION EXPERIMENT - FULL PIPELINE"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Build graph from IEEE-CIS data"
echo "  2. Make FSD prediction (Phase 1)"
echo "  3. Run 10-seed experiments (Phase 2) - ~8-16 hours"
echo "  4. Generate validation report (Phase 4)"
echo ""
echo "Configuration:"
echo "  Data directory:   $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Device:           $DEVICE"
echo "  Seeds:            10 (${SEEDS})"
echo "  Methods:          ${METHODS}"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# ============================================================================
# STEP 1: Build Graph from IEEE-CIS Data
# ============================================================================
echo ""
echo "========================================================================"
echo "STEP 1: Building Graph from IEEE-CIS Data"
echo "========================================================================"
echo ""

if [ ! -f "$PROCESSED_DIR/ieee_cis_graph.pkl" ]; then
    echo "Graph not found. Building from CSV files..."

    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: Data directory not found: $DATA_DIR"
        echo ""
        echo "Please download IEEE-CIS data from Kaggle:"
        echo "  https://www.kaggle.com/c/ieee-fraud-detection/data"
        echo ""
        echo "Expected files:"
        echo "  $DATA_DIR/train_transaction.csv"
        echo "  $DATA_DIR/train_identity.csv"
        exit 1
    fi

    python ieee_cis_graph_builder.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$PROCESSED_DIR" \
        --max_edges_per_entity 100

    echo ""
    echo "✅ Graph built successfully"
else
    echo "✅ Graph already exists: $PROCESSED_DIR/ieee_cis_graph.pkl"
fi

# ============================================================================
# STEP 2: Phase 1 - Prior Prediction
# ============================================================================
echo ""
echo "========================================================================"
echo "STEP 2: Phase 1 - Making Prior Prediction"
echo "========================================================================"
echo ""
echo "⚠️  IMPORTANT: This prediction will be timestamped and locked."
echo "    After this step, you cannot modify the prediction."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

python prior_prediction_experiment.py \
    --phase 1 \
    --data_path "$PROCESSED_DIR/ieee_cis_graph.pkl" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "✅ Prediction committed with timestamp"
echo ""
echo "Prediction file: $OUTPUT_DIR/fsd_prediction.json"
echo "Hash file:       $OUTPUT_DIR/prediction_hash.json"
echo ""
echo "You can now:"
echo "  1. Send prediction to collaborators for verification"
echo "  2. Commit to Git for timestamp proof"
echo "  3. Submit to third-party timestamp service"
echo ""
read -p "Press Enter to continue to experiments..."

# ============================================================================
# STEP 3: Phase 2 - Experimental Validation
# ============================================================================
echo ""
echo "========================================================================"
echo "STEP 3: Phase 2 - Running Experiments (This will take 8-16 hours)"
echo "========================================================================"
echo ""
echo "Running ${METHODS}"
echo "With ${SEEDS}"
echo "On device: $DEVICE"
echo ""
echo "Estimated time:"
echo "  - 8 methods × 10 seeds = 80 experiments"
echo "  - ~5-10 min per experiment (GPU) or ~20-30 min (CPU)"
echo "  - Total: 8-16 hours (GPU) or 24-48 hours (CPU)"
echo ""
read -p "Start experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. You can run experiments later with:"
    echo "  python prior_prediction_experiment.py --phase 2 \\"
    echo "    --data_path $PROCESSED_DIR/ieee_cis_graph.pkl \\"
    echo "    --output_dir $OUTPUT_DIR \\"
    echo "    --device $DEVICE"
    exit 0
fi

# Start time
start_time=$(date +%s)

python prior_prediction_experiment.py \
    --phase 2 \
    --data_path "$PROCESSED_DIR/ieee_cis_graph.pkl" \
    --output_dir "$OUTPUT_DIR" \
    --methods $METHODS \
    --seeds $SEEDS \
    --device "$DEVICE"

# End time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))

echo ""
echo "✅ All experiments completed"
echo "   Time elapsed: ${hours}h ${minutes}m"

# ============================================================================
# STEP 4: Phase 4 - Validation Report
# ============================================================================
echo ""
echo "========================================================================"
echo "STEP 4: Phase 4 - Generating Validation Report"
echo "========================================================================"
echo ""

python prior_prediction_experiment.py \
    --phase 4 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "✅ Validation report generated"

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "========================================================================"
echo "EXPERIMENT COMPLETE"
echo "========================================================================"
echo ""
echo "Output files:"
echo "  Prediction:  $OUTPUT_DIR/fsd_prediction.json"
echo "  Results:     $OUTPUT_DIR/experimental_results.json"
echo "  Report:      $OUTPUT_DIR/validation_report.md"
echo ""
echo "Next steps:"
echo "  1. Review the validation report:"
echo "     cat $OUTPUT_DIR/validation_report.md"
echo ""
echo "  2. Check if FSD prediction was correct"
echo ""
echo "  3. Include results in your paper:"
echo "     - Prediction (with timestamp) → Methods section"
echo "     - Validation report → Supplementary material"
echo ""
echo "  4. Respond to reviewers:"
echo "     'Our predictions were timestamped BEFORE experiments."
echo "      See supplementary material S1 for verification.'"
echo ""
echo "========================================================================"
