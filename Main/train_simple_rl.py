# train_simple_rl.py
import pickle, numpy as np, os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from simplified_seed_env import SimpleSeedEnv

DEBUG = True
if DEBUG:
    POOL_FILE = "pool_with_props.pkl"
    K = 20
    NGEN = 20
    TOTAL_STEPS = 5000
else:
    K = 30
    NGEN = 50
    TOTAL_STEPS = 20000

RUNS_DIR = "runs"

def load_env():
    pool, props = pickle.load(open(POOL_FILE, "rb"))
    fixed_100 = pool[:100]
    fixed_idx = list(range(100))
    return SimpleSeedEnv(pool, fixed_idx, props, K=K, n_gen=NGEN)

def mask_fn(env):
    return env.available.astype(bool)[None, :]

def main():
    env = load_env()
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=256,
                        tensorboard_log=os.path.join(RUNS_DIR, "tb"))
    model.learn(total_timesteps=TOTAL_STEPS)

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
    os.makedirs(RUNS_DIR, exist_ok=True)
    main()
