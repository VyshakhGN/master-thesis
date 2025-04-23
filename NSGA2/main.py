import matplotlib.pyplot as plt
import numpy as np
from utils import generate_random_smiles, get_objectives
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from random import randint

class MolecularOptimization(Problem):
    def __init__(self, smiles_list):
        super().__init__(n_var=1, n_obj=2, n_constr=0,
                         xl=np.array([0.0]), xu=np.array([len(smiles_list) - 1]))
        self.smiles_list = smiles_list

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for row in X:
            idx = int(row[0])
            qed, sa = get_objectives(self.smiles_list[idx])
            F.append([-qed, sa])
        out["F"] = np.array(F)

def main():

    smiles_list = generate_random_smiles(200)
    problem = MolecularOptimization(smiles_list)
    algorithm = NSGA2(pop_size=100, eliminate_duplicates=True)
    termination = get_termination("n_gen", 100)
    seed = randint(1, 100)
    result = minimize(problem, algorithm, termination, seed=seed, verbose=True)


    F = result.F
    X = result.X

    qed_vals = -F[:, 0]
    sa_vals = F[:, 1]


    plt.figure(figsize=(8, 6))
    plt.scatter(qed_vals, sa_vals)
    plt.xlabel("QED")
    plt.ylabel("SA Score")
    plt.title("Pareto Front (QED vs SA)")
    plt.grid(True)
    plt.show()


    print("\nTop 10 Molecules (by QED):")
    seen = set()
    top_indices = qed_vals.argsort()[::-1]
    count = 0
    for i in top_indices:
        mol_idx = int(X[i][0])
        smiles = smiles_list[mol_idx]
        if smiles not in seen:
            print(f"{count+1}. SMILES: {smiles} | QED: {qed_vals[i]:.3f} | SA: {sa_vals[i]:.3f}")
            seen.add(smiles)
            count += 1
        if count >= 10:
            break

if __name__ == "__main__":
    main()
