from pymoo.core.problem import ElementwiseProblem
from utils import get_objectives

class MolecularOptimization(ElementwiseProblem):
    def __init__(self, initial_selfies):
        super().__init__(n_var=1, n_obj=3, n_constr=0, xl=0, xu=len(initial_selfies) - 1, type_var=object)
        self.initial_selfies = initial_selfies

    def _evaluate(self, x, out, *args, **kwargs):
        selfie = x[0]
        qed, sa, sim = get_objectives(selfie)
        out["F"] = [-qed, sa, -sim]  # Minimize SA, maximize QED and Similarity