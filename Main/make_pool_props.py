import pickle
import numpy as np
from utils import load_smiles_from_file, get_objectives, decode_selfies

POOL_FILE = "zinc_subset.txt"
N_POOL = 1000

def main():
    selfies_list = load_smiles_from_file(POOL_FILE, max_count=N_POOL)
    props = []
    filtered_selfies = []

    for selfie in selfies_list:
        decoded = decode_selfies(selfie)
        if not decoded:
            print(f"[skip] Could not decode SELFIES: {selfie}")
            continue
        try:
            result = get_objectives(decoded)
            if result == [0.0, 1.0, 0.0, 0.0]:
                print(f"[fallback] {decoded} → {result}")
            else:
                print(f"[valid]    {decoded} → {result}")
            qed, sa, mpo, rtb = result
            props.append([qed, sa, mpo, rtb])
            filtered_selfies.append(selfie)
        except Exception as e:
            print(f"[skip] {decoded} failed with: {e}")
            continue

    props = np.asarray(props, dtype=np.float32)

    with open("pool_with_props.pkl", "wb") as f:
        pickle.dump((filtered_selfies, props), f)

    print(f"Saved {len(filtered_selfies)} valid molecules → pool_with_props.pkl")

if __name__ == "__main__":
    main()
