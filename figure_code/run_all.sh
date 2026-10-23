#!/bin/bash
# Regenerate all 9 manuscript figures into ../figures/ (vector PDF, embedded rasters 300 ppi).
# Usage:  bash run_all.sh        (from any directory)
set -e
cd "$(dirname "$0")"
PY="${PY:-python3}"   # needs numpy, matplotlib, torch, torchvision, opencv
for s in fig1_architecture.py fig2_3_4_dataset_models_cm.py fig5_6_convergence_shap.py fig7_8_cnn.py fig9_gradcam.py; do
  echo "### $s"; "$PY" "$s"
done
