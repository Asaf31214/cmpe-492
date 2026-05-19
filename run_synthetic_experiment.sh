#!/usr/bin/env python3
"""
Generate synthetic failure cases and compare model performance.

Usage:
    ./run_synthetic_experiment.sh
"""

set -e

echo "============================================================"
echo "SYNTHETIC DATA AUGMENTATION EXPERIMENT"
echo "============================================================"

echo -e "\n[1/4] Generating 50 synthetic failure cases..."
python synthetic_data_gen.py --outcome 2 --n 50 --batch-size 5

echo -e "\n[2/4] Baseline: Training on original data only..."
python train_augmented.py --no-synthetic --model all

echo -e "\n[3/4] Augmented: Training on original + synthetic..."
python train_augmented.py --synthetic synthetic_outcome_2.csv --model all

echo -e "\n[4/4] Comparing results..."
echo "Check the output above for:"
echo "  - Failure class recall (class 1)"
echo "  - Overall F1 score"
echo ""
echo "If synthetic data helps, you should see improved failure recall."
