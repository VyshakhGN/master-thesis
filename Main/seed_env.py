import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
from evolution import run_nsga
from utils import load_smiles_from_file, get_objectives


POOL_FILE = "zinc_subset.txt"
POOL_SIZE = 300

def build_filtered_pool():
    selfies = load_smiles_from_file(POOL_FILE, max_count=POOL_SIZE)
    props = [get_objectives(s)[0:2] for s in selfies]  # QED, SA
    return selfies, np.array(props, dtype=np.float32)

class SeedEnvClean(gym.Env):
    metadata = {"render_modes": []}
    FIXED_FILE = "fixed_seeds.pkl"

    def __init__(self, pool, props, picks=200, n_gen=30):
        super().__init__()
        self.pool = pool
        self.props = props
        self.N = len(pool)
        self.PICKS = picks
        self.n_gen = n_gen


        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.N + 3,), dtype=np.float32)

        self.reset()

    def _get_obs(self):
        return np.concatenate([
            self.available.astype(np.float32),
            np.array([
                self.sum_qed / self.PICKS,
                self.sum_sa / self.PICKS,
                len(self.selected) / self.PICKS
            ], dtype=np.float32)
        ])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.available = np.ones(self.N, dtype=np.int8)
        self.selected = []
        self.sum_qed = 0.0
        self.sum_sa = 0.0

        self.selected = []
        self.sum_qed = 0.0
        self.sum_sa = 0.0
        # All available — nothing is pre-filled now

        return self._get_obs(), {}

    def step(self, action):
        if self.available[action] == 0:
            raise ValueError(f"Invalid action taken: {action}")

        self.available[action] = 0
        self.selected.append(action)
        qed, sa = self.props[action]
        self.sum_qed += float(qed)
        self.sum_sa  += float(sa)

        done = len(self.selected) == self.PICKS
        reward = 0.0

        if done:
            selfies = [self.pool[i] for i in self.selected]
            reward = run_nsga(selfies, n_gen=self.n_gen, pop_size=self.PICKS)
            print(f"[env] Done. HV reward = {reward:.4f}")

        return self._get_obs(), reward, done, False, {}

def load_env(picks=200, n_gen=30):
    selfies, props = build_filtered_pool()
    return SeedEnvClean(selfies, props, picks=picks, n_gen=n_gen)
