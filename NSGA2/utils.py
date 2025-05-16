import selfies as sf
from rdkit import Chem
from rdkit.Chem import QED, AllChem, DataStructs
import sascorer
from rdkit.Chem import Descriptors, Lipinski
from guacamol.benchmark_suites import goal_directed_benchmark_suite
from guacamol.utils.chemistry import canonicalize

# Load the GuacaMol Cobimetinib MPO task
benchmark = goal_directed_benchmark_suite(version_name="v1")
task = [b for b in benchmark if b.name == "Cobimetinib MPO"][0]

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

def get_objectives(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [0.0, 1.0, 0.0, 0.0, 1.0]

    qed = QED.qed(mol)
    sa = sascorer.calculateScore(mol) / 10
    mpo_score = task.objective.score(canonicalize(smiles))
    inv_sa = 1 - sa
    rtb = Lipinski.NumRotatableBonds(mol) / 10  # Normalize to [0, 1]

    return [qed, sa, mpo_score, inv_sa, rtb]

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
