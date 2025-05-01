import numpy as np
import random
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from utils import (
    load_smiles_from_file, decode_selfies,
    crossover_selfies, mutate_selfies
)
from problem import MolecularOptimization

def evolve_population(selfies_list, generations, pop_size, crossover_rate, mutation_rate):
    population = selfies_list[:]
    for _ in range(generations):
        new_gen = []
        while len(new_gen) < pop_size:
            p1, p2 = np.random.choice(population, 2, replace=False)
            c1, c2 = crossover_selfies(p1, p2) if np.random.rand() < crossover_rate else (p1, p2)
            c1, c2 = mutate_selfies(c1, mutation_rate), mutate_selfies(c2, mutation_rate)
            for child in [c1, c2]:
                smiles = decode_selfies(child)
                if smiles:
                    new_gen.append(child)
                if len(new_gen) >= pop_size:
                    break
        population = new_gen
    return [decode_selfies(s) for s in population if decode_selfies(s)]

def main():
    FILE = "zinc_subset.txt"
    NGEN_EA = 50
    NGEN_NS = 100
    CROSSOVER = 0.9
    MUTATION = 0.1
    DEFAULT_POP_SIZE = 200
    seed = random.randint(0, 10000)

    selfies = load_smiles_from_file(FILE)
    if len(selfies) < DEFAULT_POP_SIZE:
        print(f"Expanding {len(selfies)} valid SELFIES to {DEFAULT_POP_SIZE} using mutation...")
        pool = selfies[:]
        while len(pool) < DEFAULT_POP_SIZE:
            seed_selfie = random.choice(selfies)
            mutated = mutate_selfies(seed_selfie, mutation_rate=0.9)
            if decode_selfies(mutated):
                pool.append(mutated)
        selfies = pool
    else:
        selfies = selfies[:DEFAULT_POP_SIZE]

    POP_SIZE = DEFAULT_POP_SIZE

    evolved_smiles = evolve_population(selfies, NGEN_EA, POP_SIZE, CROSSOVER, MUTATION)

    if len(evolved_smiles) < 10:
        print("Too few molecules.")
        return

    problem = MolecularOptimization(evolved_smiles)
    algorithm = NSGA2(pop_size=len(evolved_smiles))
    termination = get_termination("n_gen", NGEN_NS)

    result = minimize(problem, algorithm, termination, seed=seed, verbose=True)
    X, F = result.X, result.F
    qed_vals, sa_vals, sim_vals = -F[:, 0], F[:, 1], -F[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(qed_vals, sa_vals, sim_vals)
    ax.set_xlabel("QED")
    ax.set_ylabel("SA")
    ax.set_zlabel("Similarity")
    plt.title("Pareto Front")
    plt.show()

    print("\nTop 10 Molecules (by QED):")
    seen = set()
    count = 0
    for i in qed_vals.argsort()[::-1]:
        smi = evolved_smiles[int(X[i][0])]
        if smi not in seen:
            count += 1
            print(
                f"{count}. SMILES: {smi} | QED: {qed_vals[i]:.3f} | SA: {sa_vals[i]:.3f} | Similarity: {sim_vals[i]:.3f}")
            seen.add(smi)
        if count >= 10:
            break

    print(f"\nTotal unique molecules in Pareto front: {len(set(evolved_smiles))}")


if __name__ == "__main__":
    main()
