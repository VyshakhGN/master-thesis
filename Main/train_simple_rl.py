import pickle, numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from simplified_seed_env import SimpleSeedEnv

DEBUG = True
RUN_STEPS = 5000
RUN_BATCH = 128

if DEBUG:
    POOL_FILE = "pool_with_props.pkl"
    K = 50
    NGEN = 75
    TOTAL_STEPS = 15000
else:
    K = 30
    NGEN = 50
    TOTAL_STEPS = 20000

def load_env():
    pool, props = pickle.load(open(POOL_FILE, "rb"))
    return SimpleSeedEnv(pool, props, K=K, n_gen=NGEN, base_seeds=100)

def mask_fn(env):
    return env.available.astype(bool)[None, :]

def main():
    env = load_env()
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO("MlpPolicy", env, verbose=1, n_steps=512, batch_size=512)
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
    main()
