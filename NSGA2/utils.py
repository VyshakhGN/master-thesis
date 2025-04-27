import selfies as sf
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem, DataStructs
import sascorer

# Define a reference molecule
REFERENCE_SMILES = "CCO"  # Ethanol (temporary - can change later)
REFERENCE_MOL = Chem.MolFromSmiles(REFERENCE_SMILES)
REFERENCE_FP = AllChem.GetMorganFingerprintAsBitVect(REFERENCE_MOL, radius=2, nBits=2048)

def generate_random_smiles(n=100):
    alphabet = list(sf.get_semantic_robust_alphabet())
    population = []

    while len(population) < n:
        rand_selfie = ''.join(np.random.choice(alphabet, size=np.random.randint(10, 30)))

        try:
            smiles = sf.decoder(rand_selfie)
            mol = Chem.MolFromSmiles(smiles)

            if mol and mol.GetNumHeavyAtoms() >= 5:
                population.append(smiles)

        except Exception:
            continue

    return population

def mutate_selfies(selfie):
    alphabet = list(sf.get_semantic_robust_alphabet())

    try:
        tokens = list(sf.split_selfies(selfie))
    except Exception:
        # If splitting fails, return original selfie without mutation
        return selfie

    if len(tokens) == 0:
        return selfie

    i = np.random.randint(len(tokens))
    tokens[i] = np.random.choice(alphabet)

    return ''.join(tokens)




def get_similarity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    similarity = DataStructs.TanimotoSimilarity(fp, REFERENCE_FP)
    return similarity

def get_objectives(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [0.0, 1.0, 0.0]  # Worst scores
    qed = QED.qed(mol)
    sa = sascorer.calculateScore(mol) / 10  # Normalize SA
    similarity = get_similarity(smiles)
    return [qed, sa, similarity]

import selfies as sf

def crossover_selfies(parent1, parent2):
    tokens1 = list(sf.split_selfies(parent1))
    tokens2 = list(sf.split_selfies(parent2))

    # SAFETY: Require at least 2 tokens for crossover
    if len(tokens1) < 2 or len(tokens2) < 2:
        return parent1, parent2

    pt1 = np.random.randint(1, len(tokens1))
    pt2 = np.random.randint(1, len(tokens2))

    child1_tokens = tokens1[:pt1] + tokens2[pt2:]
    child2_tokens = tokens2[:pt2] + tokens1[pt1:]

    child1_selfies = ''.join(child1_tokens)
    child2_selfies = ''.join(child2_tokens)

    child1 = sf.encoder(child1_selfies)
    child2 = sf.encoder(child2_selfies)

    return child1, child2


def decode_selfies(selfie):
    try:
        smiles = sf.decoder(selfie)
        mol = Chem.MolFromSmiles(smiles)
        return smiles if mol else None
    except:
        return None
