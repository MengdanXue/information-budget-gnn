#!/bin/bash
#
# Case Study Generation Scripts
# ==============================
#
# Run these commands to generate case study visualizations for the FSD paper.
#

# Set paths
DATA_DIR="D:/Users/11919/Documents/毕业论文/paper/code/data"
OUTPUT_DIR="D:/Users/11919/Documents/毕业论文/paper/figures"

echo "========================================"
echo "FSD Case Study Generation"
echo "========================================"
echo ""

# Step 0: Validate setup
echo "Step 0: Validating setup..."
python test_case_study_setup.py

if [ $? -ne 0 ]; then
    echo "Setup validation failed. Please fix issues before proceeding."
    exit 1
fi

echo ""
echo "========================================"
echo "Setup validated successfully!"
echo "========================================"
echo ""

# Step 1: Generate case studies with training
echo "Step 1: Training models and generating case studies..."
echo "This will take 10-20 minutes with GPU, longer with CPU."
echo ""

python generate_case_study.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_cases 3 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "Case study generation failed."
    exit 1
fi

echo ""
echo "========================================"
echo "Case Study Generation Complete!"
echo "========================================"
echo ""
echo "Generated files in $OUTPUT_DIR:"
echo "  - case_study_elliptic.pdf"
echo "  - case_study_attention_comparison.pdf"
echo "  - case_study_node_*_neighborhood.pdf"
echo ""
echo "Models saved in $OUTPUT_DIR:"
echo "  - naa_gcn_elliptic.pt"
echo "  - gat_elliptic.pt"
echo ""

# Optional: Generate additional cases with different settings
read -p "Generate additional case studies with 5 nodes? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Generating 5-node case study..."
    python generate_case_study.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR/extended" \
        --num_cases 5 \
        --skip_training \
        --device cuda

    echo "Extended case studies saved to $OUTPUT_DIR/extended"
fi

echo ""
echo "All done! Check $OUTPUT_DIR for visualizations."
