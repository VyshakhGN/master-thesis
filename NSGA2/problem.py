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
        out["F"] = np.array([
            [-q, s, -t]
            for [q, s, t] in (get_objectives(self.smiles_list[int(x[0])]) for x in X)
        ])
