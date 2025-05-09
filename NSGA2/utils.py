import selfies as sf
from rdkit import Chem
from rdkit.Chem import QED, AllChem, DataStructs, Descriptors, Lipinski
import sascorer

REFERENCE_SMILES = "CCO"  # Replace this with your target/reference molecule if needed
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


def get_similarity(smiles1, smiles2=None):
    mol1 = Chem.MolFromSmiles(smiles1)
    if not mol1:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    fp2 = REFERENCE_FP if smiles2 is None else AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smiles2), 2, 2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def get_objectives(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [0.0, 1.0, 0.0]
    qed = QED.qed(mol)
    sa = sascorer.calculateScore(mol) / 10  # Normalize to [0,1]
    sim = get_similarity(smiles)
    return [qed, sa, sim]


# Drug-likeness and synthetic filters
def passes_drug_filters(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()

    return (
        150 <= mw <= 500 and
        logp <= 5 and
        h_donors <= 5 and
        h_acceptors <= 10 and
        rot_bonds <= 10 and
        heavy_atoms >= 5
    )


def load_smiles_from_file(path, max_count=200):
    with open(path, "r") as f:
        smiles_lines = [line.strip().split()[0] for line in f if line.strip()]

    filtered = []
    for smi in smiles_lines:
        mol = Chem.MolFromSmiles(smi)
        if mol and passes_drug_filters(mol):
            encoded = encode_smiles(smi)
            if encoded:
                filtered.append(encoded)
            if len(filtered) >= max_count:
                break

    return filtered
