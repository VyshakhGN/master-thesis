from pymoo.core.problem import Problem
import numpy as np
from utils import get_objectives

class MolecularOptimization(Problem):
    def __init__(self, smiles_list):
        super().__init__(
            n_var=1,
            n_obj=3,  # Now 3 objectives
            n_constr=0,
            xl=np.array([0.0]),
            xu=np.array([max(0, len(smiles_list) - 1)])
        )
        self.smiles_list = smiles_list

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for row in X:
            idx = int(row[0])
            qed, sa, similarity = get_objectives(self.smiles_list[idx])
            F.append([-qed, sa, -similarity])  # Minimize SA, maximize QED and similarity
        out["F"] = np.array(F)
