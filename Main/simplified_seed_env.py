# simplified_seed_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
from evolution import run_nsga

class SimpleSeedEnv(gym.Env):
    """
    RL agent selects K molecules from pool (excluding fixed set).
    Final reward is the NSGA-II hypervolume using fixed + selected.
    """
    metadata = {"render.modes": []}

    def __init__(self, pool, fixed_idx, K=20, n_gen=20):
        super().__init__()
        self.pool = pool
        self.fixed_idx = fixed_idx
        self.K = K
        self.N = len(pool)
        self.n_gen = n_gen

        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.available = np.ones(self.N, dtype=np.int8)
        for i in self.fixed_idx:
            self.available[i] = 0
        self.selected = []
        return self._obs(), {}

    def _obs(self):
        return self.available.astype(np.float32)

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
            reward = run_nsga(selfies, n_gen=self.n_gen, pop_size=len(indices),random_seed=42)
            print(f"[env] Done. HV reward = {reward:.4f}")

        return self._obs(), reward, done, False, {}
