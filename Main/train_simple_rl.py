import pickle, numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.wrappers import ActionMasker
from simplified_seed_env import SimpleSeedEnv

DEBUG = True
RUN_STEPS = 5000
RUN_BATCH = 128

if DEBUG:
    POOL_FILE = "pool_with_props.pkl"
    K = 100
    NGEN = 50
    TOTAL_STEPS = 7000
else:
    K = 30
    NGEN = 50
    TOTAL_STEPS = 20000

def evaluate_once(model, env):
    obs, _ = env.reset()
    done = False
    while not done:
        valid = np.flatnonzero(env.env.available)
        action, _ = model.predict(obs, deterministic=True)
        if env.env.available[action] == 0:
            action = int(np.random.choice(valid))
        obs, reward, done, _, _ = env.step(action)
    return float(env.env.last_hv)

class HVPerUpdateCallback(BaseCallback):
    def __init__(self, eval_env, eval_every_updates=1, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_every_updates = eval_every_updates
        self.update_idx = 0
        self.hv_history = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        self.update_idx += 1
        if self.update_idx % self.eval_every_updates == 0:
            hv = evaluate_once(self.model, self.eval_env)
            self.hv_history.append(hv)
            if self.verbose:
                print(f"[RL] Update {self.update_idx}: HV = {hv:.4f}")
        return True

def load_env():
    pool, props = pickle.load(open(POOL_FILE, "rb"))
    return SimpleSeedEnv(pool, props, K=K, n_gen=NGEN, base_seeds=100)

def mask_fn(env):
    return env.available.astype(bool)[None, :]

def main():
    env = load_env()
    env = ActionMasker(env, mask_fn)

    eval_env = load_env()
    eval_env = ActionMasker(eval_env, mask_fn)

    model = MaskablePPO("MlpPolicy", env, verbose=1, n_steps=700, batch_size=700)
    model.learn(total_timesteps=TOTAL_STEPS)

    cb = HVPerUpdateCallback(eval_env, eval_every_updates=1, verbose=1)
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)

    print("\n=== RL Convergence (HV per PPO update) ===")
    print("HV_PER_UPDATE:", ",".join(f"{v:.6f}" for v in cb.hv_history))

    hv_list = []
    for _ in range(3):
        obs, _ = env.reset()
        done = False
        while not done:
            valid = np.flatnonzero(env.env.available)
            action, _ = model.predict(obs, deterministic=True)
            if env.env.available[action] == 0:
                action = int(np.random.choice(valid))
            obs, reward, done, _, _ = env.step(action)
        hv_list.append(env.env.last_hv)

    print("\n=== Evaluation ===")
    print("Pure Hypervolumes:", hv_list)
    print(f"FINAL_HV: {np.mean(hv_list):.4f}")

if __name__ == "__main__":
    main()