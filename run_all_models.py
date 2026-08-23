"""
Pancreatic CT Classification — Multi-Model Fine-Tuning Orchestrator
===================================================================
Runs the following 4 model scripts sequentially:
- resnet50_train.py
- inceptionv3_train.py
- mobilevit_train.py
- swin_transformer_train.py

Saves stdout/stderr for each run under './outputs_logs/' and tracks time elapsed.
"""

import os
import sys
import time
import subprocess

MODELS = [
    "resnet50_train.py",
    "inceptionv3_train.py",
    "mobilevit_train.py",
    "swin_transformer_train.py",
    "cnn_scratch_train.py",
    "vit_scratch_train.py",
]

LOG_DIR = "./outputs_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def run_model_training(script_name):
    log_file = os.path.join(LOG_DIR, script_name.replace(".py", ".log"))
    print(f"\n{'-'*65}")
    print(f"  [STARTING] {script_name}")
    print(f"  Log File:  {log_file}")
    print(f"{'-'*65}")

    start_time = time.time()
    
    # Run the script using the virtual environment python interpreter
    with open(log_file, "w") as lf:
        process = subprocess.Popen(
            [".venv/bin/python", "-u", script_name],
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # Monitor the execution and print progress heartbeats
        last_heartbeat = time.time()
        while process.poll() is None:
            time.sleep(2)
            if time.time() - last_heartbeat >= 60:
                elapsed_mins = (time.time() - start_time) / 60
                print(f"    ... {script_name} has been running for {elapsed_mins:.1f} mins ...", flush=True)
                last_heartbeat = time.time()

    elapsed_time = time.time() - start_time
    exit_code = process.returncode

    if exit_code == 0:
        print(f"  [SUCCESS] Completed {script_name} in {elapsed_time/60:.2f} minutes.")
    else:
        print(f"  [FAILED] {script_name} failed with exit code {exit_code} after {elapsed_time/60:.2f} minutes.")
    
    return exit_code, elapsed_time

def main():
    print("=" * 70)
    print("   Pancreatic CT Classification — Multi-Model Training Orchestrator")
    print(f"   Models Scheduled: {', '.join([m.split('_')[0].upper() for m in MODELS])}")
    print("=" * 70)

    start_total = time.time()
    results = []

    for model_script in MODELS:
        if not os.path.exists(model_script):
            print(f"[WARNING] Script '{model_script}' not found. Skipping.")
            results.append((model_script, "NOT FOUND", 0))
            continue

        exit_code, elapsed = run_model_training(model_script)
        status = "SUCCESS" if exit_code == 0 else f"FAILED (code {exit_code})"
        results.append((model_script, status, elapsed))

    total_time = time.time() - start_total
    
    print("\n" + "=" * 70)
    print("              TRAINING ORCHESTRATION SUMMARY")
    print("=" * 70)
    print(f"{'Script Name':<30} {'Status':<15} {'Time Elapsed':<15}")
    print("-" * 70)
    for script, status, elapsed in results:
        print(f"{script:<30} {status:<15} {elapsed/60:>6.2f} mins")
    print("=" * 70)
    print(f"Total time elapsed: {total_time/3600:.2f} hours")
    print("=" * 70)

if __name__ == "__main__":
    main()
