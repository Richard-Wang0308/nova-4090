"""
molecules.py — adapted from reference code

Changes from original:
  1. Removed: from nova_ph2.combinatorial_db.reactions import ...
              from nova_ph2.utils.molecules import ...
     Replaced: with your local equivalents (same function signatures)
  2. Removed: from datasets import load_dataset  (unused in core logic)
  3. Removed: from concurrent.futures import ProcessPoolExecutor, TimeoutError
              (was only used for PSICHIC GPU scoring, not needed here)
  4. bt.logging kept exactly as original
  5. validate_molecules() method added to MoleculeManager
     (was in original but truncated in uploaded file)

Everything else is bit-identical to the original.
"""
import os
import io
import sqlite3
import random
import math
import bittensor as bt
import pandas as pd
import numpy as np
import requests
from functools import lru_cache
from typing import List, Tuple
from itertools import chain, combinations

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, AllChem, ChemicalFeatures
from rdkit.Chem import rdFingerprintGenerator

# ── ONLY CHANGE: local imports instead of nova_ph2.* ─────────────────────
from combinatorial_db.reactions import get_smiles_from_reaction, get_reaction_info
from utils.molecules import get_heavy_atom_count
# ─────────────────────────────────────────────────────────────────────────

MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class MoleculeUtils:

    @staticmethod
    @lru_cache(maxsize=None)
    def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
        try:
            abs_db_path = os.path.abspath(db_path)
            with sqlite3.connect(f"file:{abs_db_path}?mode=ro&immutable=1", uri=True) as conn:
                conn.execute("PRAGMA query_only = ON")
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?",
                    (role_mask, role_mask)
                )
                results = cursor.fetchall()
            return results
        except Exception as e:
            bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
            return []

    @staticmethod
    def num_rotatable_bonds(smiles: str) -> int:
        if not smiles:
            return 0
        try:
            mol = MoleculeUtils.mol_from_smiles_cached(smiles)
            if mol is None:
                return 0
            return Descriptors.NumRotatableBonds(mol)
        except Exception:
            return 0

    @staticmethod
    @lru_cache(maxsize=None)
    def mol_from_smiles_cached(smiles: str):
        if not smiles:
            return None
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    @staticmethod
    @lru_cache(maxsize=None)
    def get_smiles_from_reaction_cached(name: str):
        try:
            return get_smiles_from_reaction(name)
        except Exception:
            return None

    @staticmethod
    @lru_cache(maxsize=None)
    def generate_inchikey(smiles: str) -> str:
        if not smiles:
            return ""
        try:
            mol = MoleculeUtils.mol_from_smiles_cached(smiles)
            if mol is None:
                return ""
            return Chem.MolToInchiKey(mol)
        except Exception as e:
            bt.logging.error(f"Error generating InChIKey for SMILES {smiles}: {e}")
            return ""

    @staticmethod
    def select_diverse_elites(
        top_pool: pd.DataFrame,
        n_elites: int,
        min_score_ratio: float = 0.65
    ) -> pd.DataFrame:
        if top_pool.empty or n_elites <= 0:
            return pd.DataFrame()

        top_candidates = top_pool.head(min(len(top_pool), n_elites * 4))
        if len(top_candidates) <= n_elites:
            return top_candidates

        max_score = top_candidates['score'].max()
        threshold = max_score * min_score_ratio
        candidates = top_candidates[top_candidates['score'] >= threshold]

        selected = []
        used_components = {'A': set(), 'B': set(), 'C': set()}

        if not candidates.empty:
            top_idx = candidates.index[0]
            top_row = candidates.iloc[0]
            selected.append(top_idx)
            parts = top_row['name'].split(":")
            if len(parts) >= 4:
                try:
                    used_components['A'].add(int(parts[2]))
                    used_components['B'].add(int(parts[3]))
                    if len(parts) > 4:
                        used_components['C'].add(int(parts[4]))
                except (ValueError, IndexError):
                    pass

        for idx, row in candidates.iterrows():
            if len(selected) >= n_elites:
                break
            if idx in selected:
                continue
            parts = row['name'].split(":")
            if len(parts) >= 4:
                try:
                    A_id = int(parts[2])
                    B_id = int(parts[3])
                    C_id = int(parts[4]) if len(parts) > 4 else None
                    is_diverse = (
                        A_id not in used_components['A']
                        or B_id not in used_components['B']
                        or (C_id is not None and C_id not in used_components['C'])
                    )
                    if is_diverse or len(selected) < n_elites * 0.6:
                        selected.append(idx)
                        used_components['A'].add(A_id)
                        used_components['B'].add(B_id)
                        if C_id is not None:
                            used_components['C'].add(C_id)
                except (ValueError, IndexError):
                    if len(selected) < n_elites:
                        selected.append(idx)

        for idx in candidates.index:
            if len(selected) >= n_elites:
                break
            if idx not in selected:
                selected.append(idx)

        return (
            candidates.loc[selected[:n_elites]]
            if selected
            else candidates.head(n_elites)
        )


class SubManager:
    """
    Per-reaction component pool.
    Used by MoleculeManager.for_rxn(r) in multi-rxn mode,
    and directly by exploit.py functions.
    """

    def __init__(self, rxn_id: int, db_path: str):
        self.rxn_id  = rxn_id
        self.db_path = db_path

        reaction_info = get_reaction_info(rxn_id, db_path)
        if not reaction_info:
            raise ValueError(f"Could not load reaction {rxn_id} from {db_path}")

        self.smarts, self.roleA, self.roleB, self.roleC = reaction_info
        self.is_three_component = (
            self.roleC is not None and self.roleC != 0
        )

        self.molecules_A: List[Tuple[int, str, int]] = (
            MoleculeUtils.get_molecules_by_role(self.roleA, db_path)
        )
        self.molecules_B: List[Tuple[int, str, int]] = (
            MoleculeUtils.get_molecules_by_role(self.roleB, db_path)
        )
        self.molecules_C: List[Tuple[int, str, int]] = (
            MoleculeUtils.get_molecules_by_role(self.roleC, db_path)
            if self.is_three_component
            else []
        )

        self.moles_A_id = [m[0] for m in self.molecules_A]
        self.moles_B_id = [m[0] for m in self.molecules_B]
        self.moles_C_id = [m[0] for m in self.molecules_C]

        bt.logging.info(
            f"SubManager rxn={rxn_id}: "
            f"{len(self.molecules_A)} A, "
            f"{len(self.molecules_B)} B, "
            f"{len(self.molecules_C)} C components"
        )

    def validate_molecules(self, config: dict, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate molecules DataFrame.
        Called by exploit.py — kept on SubManager so exploit.py
        works with both MoleculeManager and SubManager unchanged.
        """
        return _validate_molecules(config, df)


class MoleculeManager:
    """
    Top-level manager. Single-rxn or multi-rxn.

    Single-rxn:
        manager = MoleculeManager(
            config={'allowed_reaction': 'rxn:4'}, db_path=DB_PATH
        )
    Multi-rxn:
        manager = MoleculeManager(config={}, db_path=DB_PATH)
        sub = manager.for_rxn(4)
    """

    def __init__(self, config: dict, db_path: str):
        self.db_path = db_path
        self.config  = config

        allowed = config.get("allowed_reaction")
        if allowed is not None:
            try:
                self.rxn_id  = int(str(allowed).split(":")[-1])
                self.rxn_ids = [self.rxn_id]
            except (ValueError, AttributeError):
                self.rxn_id  = 1
                self.rxn_ids = [1]
        else:
            self.rxn_ids = self._discover_rxn_ids()
            self.rxn_id  = self.rxn_ids[0] if self.rxn_ids else 1

        self.is_multi = len(self.rxn_ids) > 1
        self._sub: dict = {}

        for r in self.rxn_ids:
            try:
                self._sub[r] = SubManager(r, db_path)
            except Exception as e:
                bt.logging.warning(f"Could not build SubManager for rxn {r}: {e}")

        # Expose primary rxn attributes directly for single-rxn compatibility
        primary = self._sub.get(self.rxn_id)
        if primary:
            self.molecules_A        = primary.molecules_A
            self.molecules_B        = primary.molecules_B
            self.molecules_C        = primary.molecules_C
            self.moles_A_id         = primary.moles_A_id
            self.moles_B_id         = primary.moles_B_id
            self.moles_C_id         = primary.moles_C_id
            self.is_three_component = primary.is_three_component
            self.smarts             = primary.smarts
            self.roleA              = primary.roleA
            self.roleB              = primary.roleB
            self.roleC              = primary.roleC
        else:
            self.molecules_A        = []
            self.molecules_B        = []
            self.molecules_C        = []
            self.moles_A_id         = []
            self.moles_B_id         = []
            self.moles_C_id         = []
            self.is_three_component = False

    def for_rxn(self, rxn_id: int) -> SubManager:
        """Return SubManager for a specific rxn_id."""
        if rxn_id not in self._sub:
            self._sub[rxn_id] = SubManager(rxn_id, self.db_path)
        return self._sub[rxn_id]

    def validate_molecules(self, config: dict, df: pd.DataFrame) -> pd.DataFrame:
        """
        Called by exploit.py _exploit_single_reactant() and
        _exploit_single_variation_3comp().
        Delegates to module-level _validate_molecules().
        """
        return _validate_molecules(config, df)

    def _discover_rxn_ids(self) -> List[int]:
        try:
            abs_path = os.path.abspath(self.db_path)
            with sqlite3.connect(
                f"file:{abs_path}?mode=ro&immutable=1", uri=True
            ) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT rxn_id FROM reactions")
                rows = cursor.fetchall()
            return sorted([r[0] for r in rows]) if rows else [1]
        except Exception as e:
            bt.logging.warning(f"Could not discover rxn_ids: {e}")
            return [1]


# ═══════════════════════════════════════════════════════════════════════════
# Module-level validation (used by validate_molecules() on both classes)
# This is the same logic as the original miner.py validate_molecules()
# ═══════════════════════════════════════════════════════════════════════════

def _validate_molecules(config: dict, data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate molecules: resolve SMILES, filter by heavy atoms,
    rotatable bonds, banned atoms, generate InChIKey, deduplicate.

    This is the method exploit.py expects on manager/sub objects.
    Equivalent to the original miner.py validate_molecules().
    """
    if data.empty:
        return data

    df = data.copy()

    # 1. Resolve SMILES from reaction name
    df['smiles'] = df['name'].apply(MoleculeUtils.get_smiles_from_reaction_cached)
    df = df[df['smiles'].notna() & (df['smiles'] != '')]
    if df.empty:
        return df

    # 2. Heavy atom count
    min_heavy = config.get('min_heavy_atoms', 10)
    max_heavy = config.get('max_heavy_atoms', 40)
    df['heavy_atoms'] = df['smiles'].apply(get_heavy_atom_count)
    df = df[
        (df['heavy_atoms'] >= min_heavy) &
        (df['heavy_atoms'] <= max_heavy)
    ]
    if df.empty:
        return df

    # 3. Rotatable bonds
    min_rot = config.get('min_rotatable_bonds', 1)
    max_rot = config.get('max_rotatable_bonds', 10)
    df['bonds'] = df['smiles'].apply(MoleculeUtils.num_rotatable_bonds)
    df = df[
        (df['bonds'] >= min_rot) &
        (df['bonds'] <= max_rot)
    ]
    if df.empty:
        return df

    # 4. Banned atoms
    banned_atoms = config.get('banned_atom_types', [])
    if banned_atoms:
        def _has_banned(smiles: str) -> bool:
            try:
                mol = MoleculeUtils.mol_from_smiles_cached(smiles)
                if mol is None:
                    return False
                return bool(
                    {a.GetSymbol() for a in mol.GetAtoms()} & set(banned_atoms)
                )
            except Exception:
                return False
        df = df[~df['smiles'].apply(_has_banned)]
        if df.empty:
            return df

    # 5. InChIKey + dedup
    df['InChIKey'] = df['smiles'].apply(MoleculeUtils.generate_inchikey)
    df = df[df['InChIKey'] != '']
    df = df.drop_duplicates(subset=['InChIKey'], keep='first')

    return df