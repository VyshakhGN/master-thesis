# main.py
import random
import matplotlib.pyplot as plt
import numpy as np

from operators import MyMutation
from utils import load_smiles_from_file, decode_selfies, get_objectives
from evolution import run_nsga  # << new helper with duplicate eliminator

# ------------- run parameters -----------------
FILE = "zinc_subset.txt"

DEBUG = True

if DEBUG:
    POP_SIZE = 150
    NGEN = 20
else:
    POP_SIZE = 200
    NGEN = 100

MUTATION_RATE = 0.1
# ----------------------------------------------


def build_seed_population(file_path: str, pop_size: int) -> list[str]:
    """Load/expand SELFIES until we have exactly `pop_size` unique entries."""
    selfies = load_smiles_from_file(file_path, max_count=pop_size)

    if len(selfies) < pop_size:
        print(f"Expanding {len(selfies)} valid SELFIES to {pop_size} via mutation …")
        mutator = MyMutation(mutation_rate=0.9)
        pool = selfies[:]
        while len(pool) < pop_size:
            seed_selfie = random.choice(selfies)
            mutated = mutator._do(None, np.array([[seed_selfie]], dtype=object))[0, 0]
            if decode_selfies(mutated):  # keep only chemically valid
                pool.append(mutated)
        selfies = pool
    else:
        selfies = selfies[:pop_size]

    return selfies


def main():
    seed_selfies = build_seed_population(FILE, POP_SIZE)

    # ---------- evolutionary episode ----------
    hv, result = run_nsga(
        seed_selfies,
        n_gen=NGEN,
        pop_size=POP_SIZE,
        mutation_rate=MUTATION_RATE,
        return_full=True,
    )
    print(f"\nHyper-volume reward (episode): {hv:.4f}")

    # ---------- extract objectives ----------
    X, F = result.X, result.F
    qed_vals, sa_vals, mpo_vals, inv_sa_vals = -F[:, 0], F[:, 1], -F[:, 2], -F[:, 3]

    # ---------- Parallel-coordinate plot ----------
    from pymoo.visualization.pcp import PCP

    fig = plt.figure(figsize=(10, 6))
    pcp = PCP(
        title="5D Pareto Front",
        axis_labels=["QED", "SA", "MPO", "1-SA", "RTB"],
    )
    pcp.add(F)
    pcp.show()

    # ---------- top-10 by QED ----------
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