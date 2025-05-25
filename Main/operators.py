import numpy as np
import selfies as sf
from utils import decode_selfies
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation


class MySampling(Sampling):
    def __init__(self, initial_selfies):
        super().__init__()
        self.initial_selfies = initial_selfies

    def _do(self, problem, n_samples, **kwargs):
        X = np.array([[s] for s in self.initial_selfies], dtype=object)
        return X


class MyCrossover(Crossover):
    def __init__(self):
        # 2 parents -> 2 offspring
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        n_matings = X.shape[1]
        offsprings = np.full((self.n_offsprings, n_matings, 1), None, dtype=object)

        for k in range(n_matings):
            parent1 = X[0, k, 0]
            parent2 = X[1, k, 0]

            t1 = list(sf.split_selfies(parent1))
            t2 = list(sf.split_selfies(parent2))

            if len(t1) < 2 or len(t2) < 2:
                offsprings[0, k, 0] = parent1
                offsprings[1, k, 0] = parent2
                continue

            pt1 = np.random.randint(1, len(t1))
            pt2 = np.random.randint(1, len(t2))

            c1 = ''.join(t1[:pt1] + t2[pt2:])
            c2 = ''.join(t2[:pt2] + t1[pt1:])

            offsprings[0, k, 0] = c1
            offsprings[1, k, 0] = c2

        return offsprings


class MyMutation(Mutation):
    def __init__(self, mutation_rate=0.1):
        super().__init__()
        self.mutation_rate = mutation_rate
        self.alphabet = list(sf.get_semantic_robust_alphabet())

    def _do(self, problem, X, **kwargs):
        mutated = []

        for i in range(X.shape[0]):
            selfie = X[i, 0]
            mutated_selfie = self._safe_insert_token_mutation(selfie)
            mutated.append([mutated_selfie])

        return np.array(mutated, dtype=object)

    def _safe_insert_token_mutation(self, selfie, max_attempts=10):
        tokens = list(sf.split_selfies(selfie))
        attempts = 0
        while attempts < max_attempts:
            if np.random.rand() < self.mutation_rate:
                insert_token = np.random.choice(self.alphabet)
                insert_pos = np.random.randint(0, len(tokens) + 1)
                mutated_tokens = tokens[:insert_pos] + [insert_token] + tokens[insert_pos:]
                mutated_selfie = ''.join(mutated_tokens)
                if decode_selfies(mutated_selfie):
                    return mutated_selfie
            attempts += 1
        return selfie