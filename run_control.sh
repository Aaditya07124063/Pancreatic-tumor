#!/usr/bin/env bash
# Provenance-matched control: 4 arms x 5 seeds x 50 epochs (ScratchCNN).
# Chained behind the other jobs so the arms are not competing for the GPU.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p outputs_logs

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "Waiting for PID $WAIT_PID before starting control..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

export EPOCHS=50 PATIENCE=50 SEEDS=42,7,21,99,123 TF_CPP_MIN_LOG_LEVEL=2
LOG="outputs_logs/provenance_control.log"
echo "=== Control started $(date) ===" | tee -a "$LOG"
if .venv/bin/python -u provenance_control.py --epochs 50 >> "$LOG" 2>&1; then
  echo "=== Control finished OK $(date) ===" | tee -a "$LOG"
else
  echo "=== Control FAILED $(date) ===" | tee -a "$LOG"
fi
