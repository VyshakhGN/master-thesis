import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils import decode_selfies
from utils import get_objectives
from evolution import run_nsga

class SimpleSeedEnv(gym.Env):

    def __init__(self, pool, props, K=20, n_gen=20, base_seeds=100, rng=None):
        super().__init__()
        self.pool = pool
        self.props = props
        self.K = K
        self.N = len(pool)
        self.n_gen = n_gen
        self.base_seeds = base_seeds
        self.rng = np.random.default_rng(rng)

        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.N, 4), dtype=np.float32)
        self.last_hv = 0.0

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.available = np.ones(self.N, dtype=np.int8)
        self.fixed_idx = self.rng.choice(self.N, size=self.base_seeds, replace=False).tolist()
        for i in self.fixed_idx:
            self.available[i] = 0
        self.selected = []
        self.last_hv = 0.0
        return self._obs(), {}

    def _obs(self):
        return np.where(self.available[:, None] == 1, self.props, 0.0).astype(np.float32)

    def step(self, action: int):
        if self.available[action] == 0:
            return self._obs(), -1.0, True, False, {}

        self.available[action] = 0
        self.selected.append(action)
        done = len(self.selected) >= self.K

        reward = 0.0
        if done:
            indices = self.fixed_idx + self.selected
            selfies = [self.pool[i] for i in indices]
            hv = run_nsga(selfies, n_gen=self.n_gen, pop_size=len(indices))
            self.last_hv = hv
            reward = hv
            print(f"[env] Done. HV = {hv:.4f}, Total Reward = {reward:.4f}")

        return self._obs(), reward, done, False, {}
