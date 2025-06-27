import pickle
import random
from utils import load_smiles_from_file

POOL_FILE   = "zinc_subset.txt"
POOL_SIZE   = 5000
FIXED_COUNT = 100
OUT_FILE    = "fixed_seeds.pkl"

pool = load_smiles_from_file(POOL_FILE, max_count=POOL_SIZE)
fixed_100 = random.sample(pool, FIXED_COUNT)

with open(OUT_FILE, "wb") as f:
    pickle.dump((fixed_100, pool), f)

print(f"Saved {FIXED_COUNT} fixed seeds → {OUT_FILE}")
