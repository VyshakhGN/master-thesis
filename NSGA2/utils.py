import selfies as sf
from rdkit import Chem
from rdkit.Chem import QED, AllChem, DataStructs
import sascorer

REFERENCE_SMILES = "CCO"
REFERENCE_MOL = Chem.MolFromSmiles(REFERENCE_SMILES)
REFERENCE_FP = AllChem.GetMorganFingerprintAsBitVect(REFERENCE_MOL, 2, 2048)

def decode_selfies(selfie):
    try:
        smiles = sf.decoder(selfie)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() < 5:
            return None
        return smiles
    except:
        return None

def encode_smiles(smiles):
    try:
        selfie = sf.encoder(smiles)
        if decode_selfies(selfie):  # Round-trip check
            return selfie
    except:
        return None

def load_smiles_from_file(path):
    with open(path, "r") as f:
        smiles = [line.strip() for line in f if line.strip()]
    valid = [encode_smiles(smi) for smi in smiles]
    return [s for s in valid if s]

def get_similarity(smiles1, smiles2=None):
    mol1 = Chem.MolFromSmiles(smiles1)
    if not mol1:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    fp2 = REFERENCE_FP if smiles2 is None else AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles2), 2, 2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def get_objectives(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [0.0, 1.0, 0.0]
    qed = QED.qed(mol)
    sa = sascorer.calculateScore(mol) / 10
    sim = get_similarity(smiles)
    return [qed, sa, sim]