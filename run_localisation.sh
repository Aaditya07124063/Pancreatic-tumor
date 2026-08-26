#!/usr/bin/env bash
# Where does the separability live -- inside the anatomy or outside it?
# Chained so it never competes with another job for memory.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p outputs_logs
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "Waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
export EPOCHS=50 PATIENCE=50 SEEDS=42,7,21,99,123 TF_CPP_MIN_LOG_LEVEL=2
LOG="outputs_logs/signal_localisation.log"
echo "=== Localisation started $(date) ===" | tee -a "$LOG"
if .venv/bin/python -u signal_localisation.py --epochs 50 >> "$LOG" 2>&1; then
  echo "=== Localisation OK $(date) ===" | tee -a "$LOG"
else
  echo "=== Localisation FAILED $(date) ===" | tee -a "$LOG"
fi
