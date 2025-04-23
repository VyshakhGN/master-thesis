import selfies as sf
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
import sascorer

def generate_random_smiles(n=100):
    alphabet = list(sf.get_semantic_robust_alphabet())
    population = []
    while len(population) < n:
        rand_selfie = ''.join(np.random.choice(alphabet, size=np.random.randint(10, 30)))
        smiles = sf.decoder(rand_selfie)
        if Chem.MolFromSmiles(smiles):
            population.append(smiles)
    return population

def get_objectives(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [0.0, 1.0]
    qed = QED.qed(mol)
    sa = sascorer.calculateScore(mol) / 10
    return [qed, sa]
