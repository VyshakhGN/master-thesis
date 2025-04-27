import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from utils import generate_random_smiles
from problem import MolecularOptimization
from utils import mutate_selfies
import numpy as np
from utils import crossover_selfies, decode_selfies

from utils import decode_selfies

def evolve_population_with_crossover(selfies_list, generations=10, pop_size=100, crossover_rate=0.5, mutation_rate=0.1):
    current_population = selfies_list.copy()

    for gen in range(generations):
        new_population = []

        for _ in range(pop_size // 2):
            if len(current_population) < 2:
                break

            p1, p2 = np.random.choice(current_population, 2, replace=False)

            if decode_selfies(p1) is None or decode_selfies(p2) is None:
                child1, child2 = p1, p2
            else:
                if np.random.rand() < crossover_rate:
                    child1, child2 = crossover_selfies(p1, p2)
                else:
                    child1, child2 = p1, p2

            # Tiny mutation
            if np.random.rand() < mutation_rate:
                child1 = mutate_selfies(child1)
            if np.random.rand() < mutation_rate:
                child2 = mutate_selfies(child2)

            new_population.extend([child1, child2])

        # Decode new_population back to SMILES
        decoded_smiles = []
        for selfie in new_population:
            smiles = decode_selfies(selfie)
            if smiles:
                decoded_smiles.append(smiles)

        # Backup: if too few valid molecules, add some random new ones
        if len(decoded_smiles) < pop_size:
            print(f"Population too small after generation {gen}, adding random molecules...")
            decoded_smiles += generate_random_smiles(pop_size - len(decoded_smiles))

        # Final population clipping
        current_population = decoded_smiles[:pop_size]

    return current_population


def main():
    from random import randint

    # Step 1: Generate random molecules
    selfies_list = generate_random_smiles(2000)

    # Step 2: Evolve with crossover
    evolved_smiles_list = evolve_population_with_crossover(selfies_list, generations=10, pop_size=100, crossover_rate=0.5)


    # Step 2.5: SAFETY check
    if len(evolved_smiles_list) == 0:
        print("No valid molecules generated after crossover. Exiting safely...")
        exit(1)

    # Step 3: Set population size
    pop_size = min(len(evolved_smiles_list), 100)

    # Step 4: Setup problem and run
    problem = MolecularOptimization(evolved_smiles_list)
    algorithm = NSGA2(pop_size=pop_size)
    termination = get_termination("n_gen", 50)
    seed = randint(1, 100)

    result = minimize(problem, algorithm, termination, seed=seed, verbose=True)

    X = result.X
    F = result.F

    qed_vals = -F[:, 0]
    sa_vals = F[:, 1]
    similarity_vals = -F[:, 2]

    # 3D Pareto Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(qed_vals, sa_vals, similarity_vals)
    ax.set_xlabel("QED")
    ax.set_ylabel("SA Score")
    ax.set_zlabel("Tanimoto Similarity")
    ax.set_title("3D Pareto Front: QED vs SA vs Similarity")
    plt.show()

    # Print top 10 by QED
    print("\nTop 10 Molecules (by QED):")
    seen = set()
    top_indices = qed_vals.argsort()[::-1]
    count = 0
    for i in top_indices:
        mol_idx = int(X[i][0])
        smiles = evolved_smiles_list[mol_idx]
        if smiles not in seen:
            print(f"{count+1}. SMILES: {smiles} | QED: {qed_vals[i]:.3f} | SA: {sa_vals[i]:.3f} | Similarity: {similarity_vals[i]:.3f}")
            seen.add(smiles)
            count += 1
        if count >= 10:
            break

if __name__ == "__main__":
    main()

