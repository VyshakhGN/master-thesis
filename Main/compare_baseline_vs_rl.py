import subprocess
import re
import numpy as np
import sys
import os

PYTHON_EXEC = sys.executable
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

def extract_run_meta(filepath):
    steps = batch = None
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("RUN_STEPS"):
                steps = int(re.findall(r"\d+", line)[0])
            elif line.startswith("RUN_BATCH"):
                batch = int(re.findall(r"\d+", line)[0])
    return steps, batch

def run_script(filename, pattern="FINAL_HV: ([0-9.]+)", repeat=5):
    values = []
    for i in range(repeat):
        print(f"[{filename} Run {i+1}]")
        try:
            result = subprocess.run([PYTHON_EXEC, filename], capture_output=True, text=True, timeout=5600)
            output = result.stdout + result.stderr
            print("----- Output Start -----")
            print(output.strip())

            print("------ Output End ------")

            matches = re.findall(pattern, output)
            if matches:
                hv = float(matches[-1])
                print(f" → HV = {hv}")
                values.append(hv)
            else:
                print(" → HV not found.")
        except subprocess.TimeoutExpired:
            print(" → Timeout.")
    return values

print("=== Running train_simple_rl.py 15 times ===")
rl_hvs = run_script("train_simple_rl.py", repeat=15)

steps, batch = extract_run_meta("train_simple_rl.py")
rl_output_file = os.path.join(RUNS_DIR, f"steps{steps}_batch{batch}_rl.txt")

with open(rl_output_file, "w") as f:
    for hv in rl_hvs:
        f.write(f"{hv:.4f}\n")
print(f" → Saved RL HVs to {rl_output_file}")

print("\n=== Running main.py 15 times ===")
main_hvs = run_script("main.py", repeat=15)

main_output_file = os.path.join(RUNS_DIR, "main.txt")
with open(main_output_file, "w") as f:
    for hv in main_hvs:
        f.write(f"{hv:.4f}\n")
print(f" → Saved Main HVs to {main_output_file}")

print("\n======== FINAL COMPARISON ========")
def summarize(name, values):
    if values:
        print(f"{name:10}: mean = {np.mean(values):.4f}, std = {np.std(values):.4f}")
    else:
        print(f"{name:10}: No results")

summarize("RL Model", rl_hvs)
summarize("Main.py", main_hvs)
