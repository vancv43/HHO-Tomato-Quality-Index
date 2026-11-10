#!/usr/bin/env bash
# Runs the analyses in paper order. Raw-image steps run only when the images are present.
set -euo pipefail
PY=${PYTHON:-python}
cd "$(dirname "$0")"
export TOMATO_OUT_DIR="${TOMATO_OUT_DIR:-$PWD/../outputs}"
export TOMATO_DATA_ROOT="${TOMATO_DATA_ROOT:-$PWD/../data_raw/Tomatoes Dataset}"
mkdir -p "$TOMATO_OUT_DIR"

RAW=0
if [ -d "$TOMATO_DATA_ROOT/train" ] && [ -d "$TOMATO_DATA_ROOT/val" ]; then RAW=1; fi

if [ "$RAW" -eq 1 ]; then
  "$PY" extract_all.py
else
  cp ../data_processed/features.npz "$TOMATO_OUT_DIR/"
fi

"$PY" train_baseline.py   # Tables 3-4
"$PY" cv_stats.py         # Tables 5-6
"$PY" ablation.py         # Tables 7-8
"$PY" sensitivity.py      # Table 9
"$PY" convergence.py
"$PY" shap_analysis.py

if [ "$RAW" -eq 1 ]; then
  "$PY" complexity.py     # Table 10
  "$PY" train_cnn.py      # Table 11
  "$PY" cnn_latency.py    # Table 11
  "$PY" gradcam.py
fi
