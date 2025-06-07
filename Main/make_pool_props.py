# make_pool_props.py
import pickle, numpy as np
from utils import load_smiles_from_file, get_objectives

POOL_FILE = "zinc_subset.txt"   # your larger ZINC slice
N_POOL    = 5000                    # how many to keep

def main():
    smiles = load_smiles_from_file(POOL_FILE, max_count=N_POOL)

    props = []
    for smi in smiles:
        qed, sa, *_ = get_objectives(smi)   # we only need QED & SA
        props.append([qed, sa])

    props = np.asarray(props, dtype=np.float32)           # shape (N, 2)

    with open("pool_with_props.pkl", "wb") as f:
        pickle.dump((smiles, props), f)

    print(f"Saved {len(smiles)} molecules → pool_with_props.pkl")

if __name__ == "__main__":
    main()
