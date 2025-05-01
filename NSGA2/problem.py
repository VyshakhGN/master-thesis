from pymoo.core.problem import Problem
import numpy as np
from utils import get_objectives

class MolecularOptimization(Problem):
    def __init__(self, smiles_list):
        super().__init__(
            n_var=1,
            n_obj=3,
            n_constr=0,
            xl=np.array([0.0]),
            xu=np.array([len(smiles_list) - 1])
        )
        self.smiles_list = smiles_list

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for row in X:
            idx = int(row[0])
            qed, sa, sim = get_objectives(self.smiles_list[idx])
            F.append([-qed, sa, -sim])  # maximize QED/similarity, minimize SA
        out["F"] = np.array(F)
