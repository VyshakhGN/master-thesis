
import numpy as np
from rdkit import Chem

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.duplicate import DuplicateElimination
from pymoo.indicators.hv import HV

from problem import MolecularOptimization
from operators import MySampling, MyCrossover, MyMutation
from utils import decode_selfies


class SelfiesDuplicateEliminator(DuplicateElimination):

    @staticmethod
    def _canonical_smiles(selfie: str) -> str | None:
        smi = decode_selfies(selfie)
        if smi is None:
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def is_equal(self, a, b) -> bool:
        smi_a = self._canonical_smiles(a[0])
        smi_b = self._canonical_smiles(b[0])
        if smi_a is None or smi_b is None:
            return a[0] == b[0]
        return smi_a == smi_b


def run_nsga(
    seed_selfies: list[str],
    n_gen: int = 100,
    pop_size: int = 200,
    mutation_rate: float = 0.1,
    crossover_prob: float = 0.9,
    return_full: bool = False,
):

    problem = MolecularOptimization(seed_selfies)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MySampling(seed_selfies),
        crossover=MyCrossover(),
        mutation=MyMutation(mutation_rate=mutation_rate),
        eliminate_duplicates=SelfiesDuplicateEliminator(),
    )
    termination = get_termination("n_gen", n_gen)

    result = minimize(
        problem,
        algorithm,
        termination,
        verbose=False,
        save_history=True
    )

    ref_point = np.array([1.0, 1.0, 1.0, 1.0])
    hv_reward = HV(ref_point=ref_point)(result.F)

    return (hv_reward, result) if return_full else hv_reward