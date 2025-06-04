import random
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.visualization.pcp import PCP
from rdkit import Chem
from utils import load_smiles_from_file, decode_selfies, get_objectives
from problem import MolecularOptimization
from operators import MySampling, MyCrossover, MyMutation
from pymoo.core.duplicate import DuplicateElimination

class SelfiesDuplicateEliminator(DuplicateElimination):

    def _canonical_smiles(self, selfie):
        smi = decode_selfies(selfie)
        if smi is None:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)   # deterministic

    def is_equal(self, a, b):
        smi_a = self._canonical_smiles(a[0])
        smi_b = self._canonical_smiles(b[0])
        # If either fails to decode, fall back to old behaviour
        if smi_a is None or smi_b is None:
            return a[0] == b[0]
        return smi_a == smi_b

def main():
    FILE = "zinc_subset.txt"
    NGEN = 100
    POP_SIZE = 200
    CROSSOVER_PROB = 0.9
    MUTATION_RATE = 0.1
    seed = random.randint(0, 10000)

    selfies = load_smiles_from_file(FILE, max_count=POP_SIZE)
    if len(selfies) < POP_SIZE:
        print(f"Expanding {len(selfies)} valid SELFIES to {POP_SIZE} using mutation...")
        mutator = MyMutation(mutation_rate=0.9)
        pool = selfies[:]
        while len(pool) < POP_SIZE:
            seed_selfie = random.choice(selfies)
            mutated = mutator._do(None, np.array([[seed_selfie]], dtype=object))[0, 0]
            if decode_selfies(mutated):
                pool.append(mutated)
        selfies = pool
    else:
        selfies = selfies[:POP_SIZE]

    problem = MolecularOptimization(selfies)
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=MySampling(selfies),
        crossover=MyCrossover(),
        mutation=MyMutation(mutation_rate=MUTATION_RATE),
        eliminate_duplicates=SelfiesDuplicateEliminator()
    )
    termination = get_termination("n_gen", NGEN)

    result = minimize(problem, algorithm, termination, seed=seed, verbose=True)

    # --- Extract and Plot Objectives ---
    X, F = result.X, result.F
    qed_vals, sa_vals, mpo_vals, inv_sa_vals = -F[:, 0], F[:, 1], -F[:, 2], -F[:, 3]

    # Parallel Coordinate Plot
    fig = plt.figure(figsize=(10, 6))
    pcp = PCP(
        title="5D Pareto Front",
        axis_labels=["QED", "SA", "MPO", "1 - SA", "RTB"]
    )
    pcp.add(F)
    pcp.show()

    # --- Top Molecules by QED ---
    print("\nTop 10 Unique Molecules (by QED):")
    unique_smiles = list(set([decode_selfies(s[0]) for s in X if decode_selfies(s[0])]))
    scored = [(smi, *get_objectives(smi)) for smi in unique_smiles]
    scored.sort(key=lambda x: -x[1])  # Sort by QED descending

    for i, (smi, qed, sa, mpo, inv_sa, rtb) in enumerate(scored[:10], 1):
        print(f"{i}. SMILES: {smi} | QED: {qed:.3f} | SA: {sa:.3f} | MPO: {mpo:.3f} | RTB: {rtb:.2f}")

    print(f"\nTotal unique molecules in Pareto front: {len(unique_smiles)}")


if __name__ == "__main__":
    main()