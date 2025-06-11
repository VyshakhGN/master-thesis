
import random
import matplotlib.pyplot as plt
import numpy as np

from operators import MyMutation
from utils import load_smiles_from_file, decode_selfies, get_objectives
from evolution import run_nsga

random.seed(38)
np.random.seed(38)

FILE = "zinc_subset.txt"

DEBUG = True

if DEBUG:
    POP_SIZE = 150
    NGEN = 20
else:
    POP_SIZE = 200
    NGEN = 100

MUTATION_RATE = 0.1

def build_seed_population(file_path: str, pop_size: int) -> list[str]:
    full_pool = load_smiles_from_file(file_path, max_count=1000)
    if len(full_pool) < pop_size:
        raise ValueError(f"Not enough molecules in pool ({len(full_pool)}) to sample {pop_size}")
    return random.sample(full_pool, pop_size)


def main():
    from utils import load_smiles_from_file
    pool = load_smiles_from_file(FILE, max_count=1000)
    seed_selfies = pool[:120]

    hv, result = run_nsga(
        seed_selfies,
        n_gen=NGEN,
        pop_size=POP_SIZE,
        mutation_rate=MUTATION_RATE,
        return_full=True,
        random_seed=42
    )
    print(f"\nHyper-volume reward (episode): {hv:.4f}")
    print(f"FINAL_HV: {hv:.4f}")

    X, F = result.X, result.F
    qed_vals, sa_vals, mpo_vals, inv_sa_vals = -F[:, 0], F[:, 1], -F[:, 2], -F[:, 3]

    from pymoo.visualization.pcp import PCP

    fig = plt.figure(figsize=(10, 6))
    pcp = PCP(
        title="5D Pareto Front",
        axis_labels=["QED", "SA", "MPO", "1-SA", "RTB"],
    )
    pcp.add(F)
    pcp.show()

    print("\nTop 10 Unique Molecules (by QED):")
    unique_smiles = list(
        {
            decode_selfies(s[0]) for s in X if decode_selfies(s[0]) is not None
        }
    )
    scored = [(smi, *get_objectives(smi)) for smi in unique_smiles]
    scored.sort(key=lambda x: -x[1])  # sort descend by QED

    for i, (smi, qed, sa, mpo, inv_sa, rtb) in enumerate(scored[:10], 1):
        print(
            f"{i}. SMILES: {smi} | QED: {qed:.3f} | SA: {sa:.3f} | "
            f"MPO: {mpo:.3f} | RTB: {rtb:.2f}"
        )

    print(f"\nTotal unique molecules in Pareto front: {len(unique_smiles)}")


if __name__ == "__main__":
    main()