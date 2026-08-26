#!/usr/bin/env bash
# Re-run the jobs that crashed under memory pressure at the tail of the
# 19-hour Track A run. Sequential, one process at a time.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p outputs_logs
export EPOCHS=50 PATIENCE=50 SEEDS=42,7,21,99,123 TF_CPP_MIN_LOG_LEVEL=2
PY=".venv/bin/python"
M="outputs_logs/rerun.log"
echo "=== Rerun started $(date) ===" | tee -a "$M"
for s in cnn_scratch_train vit_scratch_train; do
  echo "--- $s at $(date) ---" | tee -a "$M"
  if $PY -u "$s.py" > "outputs_logs/$s.log" 2>&1; then
    echo "--- $s OK at $(date) ---" | tee -a "$M"
  else
    echo "--- $s FAILED at $(date) ---" | tee -a "$M"
  fi
done
echo "--- provenance_control at $(date) ---" | tee -a "$M"
if $PY -u provenance_control.py --epochs 50 > outputs_logs/provenance_control.log 2>&1; then
  echo "--- control OK at $(date) ---" | tee -a "$M"
else
  echo "--- control FAILED at $(date) ---" | tee -a "$M"
fi
echo "=== Rerun finished $(date) ===" | tee -a "$M"
