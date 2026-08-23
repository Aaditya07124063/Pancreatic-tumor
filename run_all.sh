#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p outputs_logs

PYTHON=".venv/bin/python"
SCRIPTS=(
  resnet50_train.py
  inceptionv3_train.py
  mobilevit_train.py
  swin_transformer_train.py
  cnn_scratch_train.py
  vit_scratch_train.py
)

echo "=== Training started at $(date) ===" | tee -a outputs_logs/full_run.log

for script in "${SCRIPTS[@]}"; do
  log="outputs_logs/${script%.py}.log"
  echo "" | tee -a outputs_logs/full_run.log
  echo "--- Starting $script at $(date) ---" | tee -a outputs_logs/full_run.log
  if $PYTHON -u "$script" >> "$log" 2>&1; then
    echo "--- Finished $script OK at $(date) ---" | tee -a outputs_logs/full_run.log
  else
    echo "--- FAILED $script at $(date) ---" | tee -a outputs_logs/full_run.log
  fi
done

$PYTHON summarize_results.py | tee -a outputs_logs/full_run.log
echo "=== Training finished at $(date) ===" | tee -a outputs_logs/full_run.log
