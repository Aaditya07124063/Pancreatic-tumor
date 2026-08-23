#!/usr/bin/env bash
# Full experimental protocol:
#   - deduplicated corpus, stratified 80/10/10, 5 seeds x 50 epochs
#   - PATIENCE=50 so early stopping never fires inside the 50-epoch budget
#     (best-val weights are still restored for evaluation)
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p outputs_logs

export EPOCHS=50
export PATIENCE=50
export SEEDS=42,7,21,99,123
export TF_CPP_MIN_LOG_LEVEL=2

PYTHON=".venv/bin/python"
SCRIPTS=(
  resnet50_train.py
  inceptionv3_train.py
  mobilevit_train.py
  swin_transformer_train.py
  cnn_scratch_train.py
  vit_scratch_train.py
)

MAIN="outputs_logs/full_run.log"
echo "=== Run started $(date) | EPOCHS=$EPOCHS PATIENCE=$PATIENCE SEEDS=$SEEDS ===" | tee -a "$MAIN"

# Integrity audit first -- establishes the shortcut floor the models are read against.
echo "--- leakage_audit.py ---" | tee -a "$MAIN"
$PYTHON -u leakage_audit.py 2>&1 | tee outputs_logs/leakage_audit.log | tail -20 | tee -a "$MAIN"

for script in "${SCRIPTS[@]}"; do
  log="outputs_logs/${script%.py}.log"
  echo "" | tee -a "$MAIN"
  echo "--- Starting $script at $(date) ---" | tee -a "$MAIN"
  if $PYTHON -u "$script" >> "$log" 2>&1; then
    echo "--- Finished $script OK at $(date) ---" | tee -a "$MAIN"
  else
    echo "--- FAILED $script at $(date) (see $log) ---" | tee -a "$MAIN"
  fi
done

$PYTHON summarize_results.py 2>&1 | tee -a "$MAIN"
echo "=== Run finished $(date) ===" | tee -a "$MAIN"
