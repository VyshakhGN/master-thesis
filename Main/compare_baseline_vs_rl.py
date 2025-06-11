import subprocess
import re
import numpy as np
import sys

PYTHON_EXEC = sys.executable

def run_script(filename, pattern="FINAL_HV: ([0-9.]+)", repeat=5):
    values = []
    for i in range(repeat):
        print(f"[{filename} Run {i+1}]")
        try:
            result = subprocess.run([PYTHON_EXEC, filename], capture_output=True, text=True, timeout=900)
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

print("=== Running train_simple_rl.py 5 times ===")
rl_hvs = run_script("train_simple_rl.py", repeat=5)

print("\n=== Running main.py 5 times ===")
main_hvs = run_script("main.py", repeat=5)

print("\n======== FINAL COMPARISON ========")
def summarize(name, values):
    if values:
        print(f"{name:10}: mean = {np.mean(values):.4f}, std = {np.std(values):.4f}")
    else:
        print(f"{name:10}: No results")

summarize("RL Model", rl_hvs)
summarize("Main.py", main_hvs)
