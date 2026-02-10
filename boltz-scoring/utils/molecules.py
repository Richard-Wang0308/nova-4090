import math
import numpy as np
import pandas as pd
import time
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
import bittensor as bt
from dotenv import load_dotenv

load_dotenv(override=True)


def is_boltz_safe_smiles(smiles: str) -> tuple[bool, str | None]:
    """
    Replicates Boltz atom-name generation and enforces <= 4 characters.
    Returns (ok, reason). ok=False means the SMILES should be rejected for Boltz.
    """
    try:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return False, "RDKit failed to parse SMILES"
        mol = AllChem.AddHs(mol)
        canonical_order = AllChem.CanonicalRankAtoms(mol)
        for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
            atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
            if len(atom_name) > 4:
                return False, f"Atom name would exceed 4 chars: {atom_name}"
        return True, None
    except Exception as e:
        return False, f"Boltz safety check failed: {e}"


def get_heavy_atom_count(smiles: str) -> int:
    """
    Calculate the number of heavy atoms in a molecule from its SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        bt.logging.warning(f"Could not parse SMILES string: {smiles}, returning 0")
        return 0
    return mol.GetNumHeavyAtoms()


def compute_maccs_entropy(smiles_list: list[str]) -> float:
    """
    Computes fingerprint entropy from MACCS keys for a list of SMILES.

    Parameters:
        smiles_list (list of str): Molecules in SMILES format.

    Returns:
        avg_entropy (float): Average entropy per bit.
    """
    n_bits = 167  # RDKit uses 167 bits (index 0 is always 0)
    bit_counts = np.zeros(n_bits)
    valid_mols = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.array(fp)
            bit_counts += arr
            valid_mols += 1

    if valid_mols == 0:
        raise ValueError("No valid molecules found.")

    probs = bit_counts / valid_mols
    entropy_per_bit = np.array([
        -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0
        for p in probs
    ])

    avg_entropy = np.mean(entropy_per_bit)

    return avg_entropy
