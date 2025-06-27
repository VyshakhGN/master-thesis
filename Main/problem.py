from pymoo.core.problem import ElementwiseProblem
from utils import get_objectives

class MolecularOptimization(ElementwiseProblem):
    def __init__(self, initial_selfies):
        super().__init__(n_var=1, n_obj=4, n_constr=0, xl=0, xu=len(initial_selfies) - 1, type_var=object)
        self.initial_selfies = initial_selfies

    def _evaluate(self, x, out, *args, **kwargs):
        selfie = x[0]
        objectives = get_objectives(selfie)
        # Minimize SA,TPSA, and maximize QED,MPO.
        out["F"] = [-objectives[0], objectives[1], -objectives[2], objectives[3]]
