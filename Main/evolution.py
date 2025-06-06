# evolution.py
"""
Black-box wrapper around the existing NSGA-II pipeline.
`run_nsga()` takes a list of SELFIES, runs the GA for N generations
and returns a scalar hyper-volume reward (and optionally the full
pymoo result object).
"""

import random
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


# ------------------------------------------------------------------
#  Duplicate eliminator based on canonical SMILES equality
# ------------------------------------------------------------------
class SelfiesDuplicateEliminator(DuplicateElimination):
    """Treat two individuals as identical if they decode to the same
    canonical SMILES (ignoring SELFIES string differences)."""

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
        # fall back to raw SELFIES comparison if decoding fails
        if smi_a is None or smi_b is None:
            return a[0] == b[0]
        return smi_a == smi_b


# ------------------------------------------------------------------
#  One-shot evolutionary run returning a scalar reward
# ------------------------------------------------------------------
def run_nsga(
    seed_selfies: list[str],
    n_gen: int = 100,
    pop_size: int = 200,
    mutation_rate: float = 0.1,
    crossover_prob: float = 0.9,
    random_seed: int | None = None,
    return_full: bool = False,
):
    """
    Evolve `pop_size` molecules for `n_gen` generations.

    Parameters
    ----------
    seed_selfies : list[str]
        Initial population (length must equal `pop_size`).
    n_gen : int
        Number of generations.
    pop_size : int
        Population size (should match len(seed_selfies)).
    mutation_rate : float
        Per-token insertion probability in MyMutation.
    crossover_prob : float
        Not currently used (the custom crossover is deterministic).
    random_seed : int | None
        Reproducibility control.
    return_full : bool
        If True, also return the full `pymoo.optimize.MinimizationResult`.

    Returns
    -------
    float  (or tuple[float, result] if `return_full`)
        Hyper-volume reward; higher == better.
    """
    if random_seed is None:
        random_seed = random.randint(0, 10_000)

    # ---------- configure GA ----------
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
        seed=random_seed,
        verbose=False,   # keep console clean (RL loop will be chatty enough)
    )

    # ---------- scalar reward ----------
    ref_point = np.array([1.0, 1.0, 1.0, 1.0, 1.0])   # dominated by all ideal points
    hv_reward = HV(ref_point=ref_point)(result.F)

    return (hv_reward, result) if return_full else hv_reward
