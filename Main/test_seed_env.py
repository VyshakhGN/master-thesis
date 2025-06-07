import pickle, random
from seed_env import SeedEnv

pool, props = pickle.load(open("pool_with_props.pkl", "rb"))
env = SeedEnv(pool, props)

obs, _ = env.reset()
done = False
while not done:
    # choose a random *available* index
    valid = [i for i, m in enumerate(obs[:len(pool)]) if m]
    action = random.choice(valid)
    obs, reward, done, _, _ = env.step(action)

print("Hyper-volume reward:", reward)
print("Observation length :", obs.size)   # should be N + 3
