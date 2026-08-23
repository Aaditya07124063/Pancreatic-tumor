#!/usr/bin/env bash
# Track B: frozen feature extractors + classical/NN classifiers
# (Xception, DenseNet121) x (SVM, Random Forest, AdaBoost, KNN, XGBoost,
#  Bagging, ANN, LSTM, Bi-LSTM), on the SAME deduplicated stratified
# 80/10/10 protocol and the same 5 seeds as Track A.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p outputs_logs

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "Waiting for Track A (PID $WAIT_PID) to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "Track A finished; starting Track B at $(date)"
fi

export TF_CPP_MIN_LOG_LEVEL=2
PYTHON=".venv/bin/python"
LOG="outputs_logs/track_b.log"

echo "=== Track B started $(date) ===" | tee -a "$LOG"
for bb in xception densenet121; do
  echo "--- $bb at $(date) ---" | tee -a "$LOG"
  if $PYTHON -u feature_extraction_pipeline.py \
       --backbone "$bb" --split-mode stratified \
       --seeds 42 7 21 99 123 >> "$LOG" 2>&1; then
    echo "--- $bb OK at $(date) ---" | tee -a "$LOG"
  else
    echo "--- $bb FAILED at $(date) ---" | tee -a "$LOG"
  fi
done
echo "=== Track B finished $(date) ===" | tee -a "$LOG"
