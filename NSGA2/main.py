import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from utils import generate_random_selfies, decode_selfies, mutate_selfies, crossover_selfies
from problem import MolecularOptimization
import numpy as np
import random
from rdkit import Chem


def evolve_population_with_crossover(selfies_list, generations, pop_size, crossover_rate, mutation_rate):
    current_population = selfies_list.copy()

    for gen in range(generations):
        next_gen = []
        while len(next_gen) < pop_size:
            if len(current_population) < 2:
                current_population += generate_random_selfies(2)

            p1, p2 = np.random.choice(current_population, 2, replace=False)

            # Crossover
            if np.random.rand() < crossover_rate:
                child1, child2 = crossover_selfies(p1, p2)
            else:
                child1, child2 = p1, p2

            # # Mutation
            # child1 = mutate_selfies(child1, mutation_rate)
            # child2 = mutate_selfies(child2, mutation_rate)

            for child in [child1, child2]:
                smiles = decode_selfies(child)
                if smiles and len(smiles) > 5 and Chem.MolFromSmiles(smiles).GetNumAtoms() >= 5:
                    next_gen.append(child)
                if len(next_gen) >= pop_size:
                    break

        current_population = next_gen

    final_smiles = [decode_selfies(s) for s in current_population if decode_selfies(s)]
    return final_smiles


def main():
    # === Literature-recommended Parameters ===
    POP_SIZE = 500
    NGEN_EA = 200
    NGEN_NS = 200
    CROSSOVER_RATE = 0.9
    MUTATION_RATE = 0.1

    seed = random.randint(1, 100)
    selfies_list = generate_random_selfies(POP_SIZE)

    evolved_smiles = evolve_population_with_crossover(
        selfies_list,
        generations=NGEN_EA,
        pop_size=POP_SIZE,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE
    )

    if len(evolved_smiles) < 10:
        print("❌ Too few valid molecules generated. Try rerunning or adjusting parameters.")
        return

    problem = MolecularOptimization(evolved_smiles)
    algorithm = NSGA2(pop_size=len(evolved_smiles))
    termination = get_termination("n_gen", NGEN_NS)
    result = minimize(problem, algorithm, termination, seed=seed, verbose=True)

    X, F = result.X, result.F
    qed_vals = -F[:, 0]
    sa_vals = F[:, 1]
    sim_vals = -F[:, 2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(qed_vals, sa_vals, sim_vals)
    ax.set_xlabel("QED")
    ax.set_ylabel("SA")
    ax.set_zlabel("Similarity")
    ax.set_title("3D Pareto Front")
    plt.show()

    top_indices = qed_vals.argsort()[::-1]
    seen = set()
    count = 0
    print("\nTop 10 Molecules (by QED):")
    for i in top_indices:
        idx = int(X[i][0])
        smiles = evolved_smiles[idx]
        if smiles not in seen:
            print(f"{count+1}. SMILES: {smiles} | QED: {qed_vals[i]:.3f} | SA: {sa_vals[i]:.3f} | Similarity: {sim_vals[i]:.3f}")
            seen.add(smiles)
            count += 1
        if count >= 10:
            break


if __name__ == "__main__":
    main()
