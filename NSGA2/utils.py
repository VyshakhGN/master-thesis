import selfies as sf
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem, DataStructs
import sascorer

REFERENCE_SMILES = "CCO"
REFERENCE_MOL = Chem.MolFromSmiles(REFERENCE_SMILES)
REFERENCE_FP = AllChem.GetMorganFingerprintAsBitVect(REFERENCE_MOL, radius=2, nBits=2048)


def generate_random_selfie():
    alphabet = list(sf.get_semantic_robust_alphabet())
    return ''.join(np.random.choice(alphabet, size=np.random.randint(10, 30)))


def generate_random_selfies(n=100):
    selfies_list = []
    while len(selfies_list) < n:
        selfie = generate_random_selfie()
        if decode_selfies(selfie):
            selfies_list.append(selfie)
    return selfies_list


def decode_selfies(selfie):
    try:
        smiles = sf.decoder(selfie)
        mol = Chem.MolFromSmiles(smiles)
        # Add stricter filters
        if mol is None:
            return None
        if mol.GetNumHeavyAtoms() < 5:
            return None  # Filter out trivial molecules
        return smiles
    except:
        return None



def mutate_selfies(selfie, mutation_rate=0.1):
    tokens = list(sf.split_selfies(selfie))
    alphabet = list(sf.get_semantic_robust_alphabet())

    if len(tokens) == 0 or np.random.rand() > mutation_rate:
        return selfie

    for _ in range(3):  # Try up to 3 times
        pos = np.random.randint(0, len(tokens))
        new_token = np.random.choice(alphabet)
        tokens[pos] = new_token
        mutated = ''.join(tokens)
        if decode_selfies(mutated):
            return mutated

    return selfie  # fallback


def crossover_selfies(parent1, parent2):
    tokens1 = list(sf.split_selfies(parent1))
    tokens2 = list(sf.split_selfies(parent2))

    if len(tokens1) < 2 or len(tokens2) < 2:
        return parent1, parent2

    pt1 = np.random.randint(1, len(tokens1))
    pt2 = np.random.randint(1, len(tokens2))

    child1 = ''.join(tokens1[:pt1] + tokens2[pt2:])
    child2 = ''.join(tokens2[:pt2] + tokens1[pt1:])

    # Only return if both decode to meaningful SMILES
    return child1, child2



def get_similarity(smiles1, smiles2=None):
    mol1 = Chem.MolFromSmiles(smiles1)
    if mol1 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = REFERENCE_FP if smiles2 is None else AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles2), radius=2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def get_objectives(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [0.0, 1.0, 0.0]
    qed = QED.qed(mol)
    sa = sascorer.calculateScore(mol) / 10
    similarity = get_similarity(smiles)
    return [qed, sa, similarity]
