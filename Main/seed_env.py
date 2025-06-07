# seed_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from evolution import run_nsga


# ---------- helper for MaskablePPO ---------- #
def action_mask(env):
    """
    Returns a Boolean mask where True means the action (index)
    is still available, False means it was already picked.
    sb3-contrib’s ActionMasker will use this before each step.
    """
    return env.available.astype(bool)


class SeedEnv(gym.Env):
    """
    Agent picks `PICKS` molecules. Observation = mask + avg(QED, SA) + progress.
    Reward = hyper-volume from run_nsga() after final pick.
    """
    metadata = {"render.modes": []}

    def __init__(self, pool, prop_table, picks=200, n_gen=100):
        super().__init__()
        self.pool = pool
        self.prop = prop_table          # shape (N, 2)  [QED, SA]
        self.N = len(pool)
        self.PICKS = picks
        self.n_gen = n_gen

        self.action_space = spaces.Discrete(self.N)
        obs_len = self.N + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )

        self.available = None
        self.selected = None
        self.sum_qed = None
        self.sum_sa = None

    def _obs(self):
        return np.concatenate(
            [
                self.available.astype(np.float32),
                np.array(
                    [
                        self.sum_qed / self.PICKS,
                        self.sum_sa / self.PICKS,
                        len(self.selected) / self.PICKS,
                    ],
                    dtype=np.float32,
                ),
            ]
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.available = np.ones(self.N, dtype=np.int8)
        self.selected = []
        self.sum_qed = 0.0
        self.sum_sa = 0.0
        return self._obs(), {}

    def step(self, action: int):
        if self.available[action] == 0:
            # should not happen with masking, but keep a safeguard
            return self._obs(), -1.0, True, False, {}

        # record valid pick
        self.available[action] = 0
        self.selected.append(action)
        self.sum_qed += float(self.prop[action, 0])
        self.sum_sa += float(self.prop[action, 1])

        done = len(self.selected) == self.PICKS
        reward = 0.0

        if done:
            selfies = [self.pool[i] for i in self.selected]
            reward = run_nsga(
                selfies,
                n_gen=self.n_gen,
                pop_size=self.PICKS,
            )

        return self._obs(), reward, done, False, {}
