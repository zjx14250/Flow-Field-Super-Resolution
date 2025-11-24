#!/bin/bash
# Batch visualization script for all Case1/Case2 configurations
# Usage: bash scripts/batch_visualize_all.sh

set -e  # Exit on error

echo "=== Batch Flow Field SR Visualization ==="
echo "This will generate visualizations for all 6 configurations:"
echo "  - Case1: x2, x4, x8"
echo "  - Case2: x2, x4, x8"
echo ""

OUTPUT_DIR="experiments/visualizations"
DEVICE="cuda:0"

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Using CPU (may be slow)."
    DEVICE="cpu"
fi

echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""

# Loop through all combinations
for case in case1 case2; do
    for scale in 2 4 8; do
        echo "----------------------------------------"
        echo "Processing: ${case} x${scale}"
        echo "----------------------------------------"

        python scripts/visualize_flow_comparison.py \
            --case "$case" \
            --scale "$scale" \
            --output_dir "$OUTPUT_DIR" \
            --device "$DEVICE"

        if [ $? -eq 0 ]; then
            echo "✓ ${case} x${scale} completed successfully"
        else
            echo "✗ ${case} x${scale} failed"
        fi
        echo ""
    done
done

echo "=== All visualizations completed ==="
echo "Results saved to: $OUTPUT_DIR"
