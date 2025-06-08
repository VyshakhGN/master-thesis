import os
import pickle
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from seed_env import load_env

DEBUG = True  # Set to False for full run

if DEBUG:
    PICKS = 150
    N_GEN = 20
    TOTAL_STEPS = 7500
    SAVE_EVERY = 2500
else:
    PICKS = 200
    N_GEN = 30
    TOTAL_STEPS = 20000
    SAVE_EVERY = 10000

RUNS_DIR = "runs"



def env_mask(env):
    return env.available.astype(bool)[None, :]

def main():
    base_env = load_env(picks=PICKS, n_gen=N_GEN)
    env = ActionMasker(base_env, env_mask)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=256,
        tensorboard_log=os.path.join(RUNS_DIR, "tb")
    )

    model.learn(total_timesteps=TOTAL_STEPS)
    model.save(os.path.join(RUNS_DIR, "ppo_seedenv"))

    # Evaluate
    rewards = []
    for _ in range(2):
        obs, _ = env.reset()
        done = False
        while not done:
            valid = np.flatnonzero(env.env.available)
            if len(valid) == 0:
                break
            action, _ = model.predict(obs, deterministic=True)
            if env.env.available[action] == 0:
                action = int(np.random.choice(valid))
            obs, reward, done, _, _ = env.step(action)
        rewards.append(float(reward))

    print("\n=== Evaluation ===")
    print("Rewards:", rewards)
    print("Mean HV:", np.mean(rewards))

if __name__ == "__main__":
    os.makedirs(RUNS_DIR, exist_ok=True)
    main()
