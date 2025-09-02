import random
import matplotlib.pyplot as plt
from operators import MyMutation
from utils import load_smiles_from_file, decode_selfies, get_objectives
from evolution import run_nsga
from pymoo.indicators.hv import HV
import numpy as np

# ------------- run parameters -----------------
FILE = "zinc_subset.txt"

DEBUG = True

if DEBUG:
    K = 100
    POP_SIZE = 100 + K
    NGEN = 50
else:
    K = 30
    POP_SIZE = 100 + K
    NGEN = 100

MUTATION_RATE = 0.1
# ----------------------------------------------

def main():
    pool = load_smiles_from_file(FILE, max_count=1000)

    base_100 = random.sample(pool, 100)
    remainder = [s for s in pool if s not in base_100]
    if len(remainder) < K:
        raise ValueError(f"Not enough remaining molecules to sample K={K}")
    extra_k = random.sample(remainder, K)

    seed_selfies = base_100 + extra_k

    hv, result = run_nsga(seed_selfies, n_gen=NGEN, pop_size=POP_SIZE, return_full=True)
    curve = [HV(ref_point=np.array([1,1,1,1]))(h.pop.get("F")) for h in result.history]
    # print("HV_PER_GEN:", ",".join(f"{v:.6f}" for v in curve))
    print(f"\nHyper-volume reward (episode): {hv:.4f}")
    print(f"FINAL_HV: {hv:.4f}")

    X, F = result.X, result.F
    qed_vals, sa_vals, mpo_vals, tpsa_vals = -F[:, 0], F[:, 1], -F[:, 2], F[:, 3]

    from pymoo.visualization.pcp import PCP

    fig = plt.figure(figsize=(10, 6))
    pcp = PCP(
        title="4D Pareto Front",
        axis_labels=["QED", "SA", "MPO", "TPSA"],
    )
    pcp.add(F)
    # pcp.show()

    print("\nTop 10 Unique Molecules (by QED):")
    unique_smiles = list({decode_selfies(s[0]) for s in X if decode_selfies(s[0]) is not None})
    scored = [(smi, *get_objectives(smi)) for smi in unique_smiles]
    scored.sort(key=lambda x: -x[1])

    for i, (smi, qed, sa, mpo, tpsa) in enumerate(scored[:10], 1):
        print(
            f"{i}. SMILES: {smi} | QED: {qed:.3f} | SA: {sa:.3f} | "
            f"MPO: {mpo:.3f} | TPSA: {tpsa:.2f}"
        )

    print(f"\nTotal unique molecules in Pareto front: {len(unique_smiles)}")

if __name__ == "__main__":
    main()