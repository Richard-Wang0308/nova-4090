"""
molecules.py — Blueprint molecule layer adapted for small-molecule competition.

Changes vs Blueprint:
  - Local combinatorial_db / utils imports (no nova_ph2)
  - Single fixed reaction via config["allowed_reaction"]
  - Validity: HA 10–40, rotatable bonds, banned atoms (incl. S)
"""
import os
import sqlite3
import math
import bittensor as bt
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import List, Tuple
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys

from combinatorial_db.reactions import get_smiles_from_reaction, get_reaction_info
from utils.molecules import get_heavy_atom_count


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
                    (role_mask, role_mask),
                )
                return cursor.fetchall()
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
    def select_diverse_elites(top_pool: pd.DataFrame, n_elites: int, min_score_ratio: float = 0.65) -> pd.DataFrame:
        if top_pool.empty or n_elites <= 0:
            return pd.DataFrame()

        top_candidates = top_pool.head(min(len(top_pool), n_elites * 4))
        if len(top_candidates) <= n_elites:
            return top_candidates

        max_score = top_candidates["score"].max()
        threshold = max_score * min_score_ratio
        candidates = top_candidates[top_candidates["score"] >= threshold]

        selected = []
        used_components = {"A": set(), "B": set(), "C": set()}

        if not candidates.empty:
            top_idx = candidates.index[0]
            top_row = candidates.iloc[0]
            selected.append(top_idx)
            parts = top_row["name"].split(":")
            if len(parts) >= 4:
                try:
                    used_components["A"].add(int(parts[2]))
                    used_components["B"].add(int(parts[3]))
                    if len(parts) > 4:
                        used_components["C"].add(int(parts[4]))
                except (ValueError, IndexError):
                    pass

        for idx, row in candidates.iterrows():
            if len(selected) >= n_elites:
                break
            if idx in selected:
                continue

            parts = row["name"].split(":")
            if len(parts) >= 4:
                try:
                    A_id = int(parts[2])
                    B_id = int(parts[3])
                    C_id = int(parts[4]) if len(parts) > 4 else None

                    is_diverse = (
                        A_id not in used_components["A"]
                        or B_id not in used_components["B"]
                        or (C_id is not None and C_id not in used_components["C"])
                    )

                    if is_diverse or len(selected) < n_elites * 0.6:
                        selected.append(idx)
                        used_components["A"].add(A_id)
                        used_components["B"].add(B_id)
                        if C_id is not None:
                            used_components["C"].add(C_id)
                except (ValueError, IndexError):
                    if len(selected) < n_elites:
                        selected.append(idx)

        for idx, row in candidates.iterrows():
            if len(selected) >= n_elites:
                break
            if idx not in selected:
                selected.append(idx)

        return candidates.loc[selected[:n_elites]] if selected else candidates.head(n_elites)

    @staticmethod
    @lru_cache(maxsize=None)
    def maccs_fp_from_smiles_cached(smiles: str):
        if not smiles:
            return None
        try:
            mol = MoleculeUtils.mol_from_smiles_cached(smiles)
            if mol is None:
                return None
            return MACCSkeys.GenMACCSKeys(mol)
        except Exception:
            return None

    @staticmethod
    def parse_components(name: str) -> tuple:
        parts = name.split(":")
        if len(parts) < 4:
            return None, None, None
        A = int(parts[2])
        B = int(parts[3])
        C = int(parts[4]) if len(parts) > 4 else None
        return A, B, C

    @staticmethod
    def _heavy_atoms_dict_from_bitcounts(bitcounts: pd.DataFrame) -> dict:
        if bitcounts is None or bitcounts.empty or "heavy_atoms" not in bitcounts.columns:
            return {}
        return dict(zip(bitcounts["mol_id"], bitcounts["heavy_atoms"]))


class SubManager:
    def __init__(self, rxn_id: int, db_path: str):
        self.rxn_id = rxn_id
        self.db_path = db_path

        reaction_info = get_reaction_info(self.rxn_id, db_path)
        _, self.roleA, self.roleB, self.roleC = reaction_info
        self.is_three_component = self.roleC is not None and self.roleC != 0

        self.molecules_A = MoleculeUtils.get_molecules_by_role(self.roleA, db_path)
        self.molecules_B = MoleculeUtils.get_molecules_by_role(self.roleB, db_path)
        self.molecules_C = (
            MoleculeUtils.get_molecules_by_role(self.roleC, db_path)
            if self.is_three_component
            else []
        )

        self.moles_A_id = [mol[0] for mol in self.molecules_A]
        self.moles_B_id = [mol[0] for mol in self.molecules_B]
        self.moles_C_id = [mol[0] for mol in self.molecules_C] if self.is_three_component else None

        self.role_A_bitcounts = pd.DataFrame(
            self.molecules_A, columns=["mol_id", "smiles", "_"]
        )[["mol_id", "smiles"]]
        self.role_A_bitcounts["heavy_atoms"] = self.role_A_bitcounts["smiles"].apply(
            get_heavy_atom_count
        )

        self.role_B_bitcounts = pd.DataFrame(
            self.molecules_B, columns=["mol_id", "smiles", "_"]
        )[["mol_id", "smiles"]]
        self.role_B_bitcounts["heavy_atoms"] = self.role_B_bitcounts["smiles"].apply(
            get_heavy_atom_count
        )

        if self.is_three_component:
            self.role_C_bitcounts = pd.DataFrame(
                self.molecules_C, columns=["mol_id", "smiles", "_"]
            )[["mol_id", "smiles"]]
            self.role_C_bitcounts["heavy_atoms"] = self.role_C_bitcounts["smiles"].apply(
                get_heavy_atom_count
            )
        else:
            self.role_C_bitcounts = None

        self.dict_A = MoleculeUtils._heavy_atoms_dict_from_bitcounts(self.role_A_bitcounts)
        self.dict_B = MoleculeUtils._heavy_atoms_dict_from_bitcounts(self.role_B_bitcounts)
        self.dict_C = (
            MoleculeUtils._heavy_atoms_dict_from_bitcounts(self.role_C_bitcounts)
            if self.role_C_bitcounts is not None
            else {}
        )

    def validate_molecules(self, config: dict, data: pd.DataFrame, time_elapsed: int = 0) -> pd.DataFrame:
        return MoleculeManager._validate_single(self, config, data, time_elapsed)


class MoleculeManager:
    """Single fixed-reaction manager for small-molecule mining."""

    def __init__(self, config: dict, db_path: str):
        self.db_path = db_path
        allowed = config.get("allowed_reaction")
        if not allowed:
            raise ValueError("MoleculeManager requires config['allowed_reaction'] e.g. 'rxn:1'")

        self.rxn_ids = [int(str(allowed).split(":")[-1])]
        self.subs: dict = {}
        for r in self.rxn_ids:
            self.subs[r] = SubManager(r, db_path)

        primary = self.subs[self.rxn_ids[0]]
        self.rxn_id = primary.rxn_id
        self.is_three_component = primary.is_three_component
        self.molecules_A = primary.molecules_A
        self.molecules_B = primary.molecules_B
        self.molecules_C = primary.molecules_C
        self.moles_A_id = primary.moles_A_id
        self.moles_B_id = primary.moles_B_id
        self.moles_C_id = primary.moles_C_id
        self.role_A_bitcounts = primary.role_A_bitcounts
        self.role_B_bitcounts = primary.role_B_bitcounts
        self.role_C_bitcounts = primary.role_C_bitcounts
        self.dict_A = primary.dict_A
        self.dict_B = primary.dict_B
        self.dict_C = primary.dict_C
        self.roleA = primary.roleA
        self.roleB = primary.roleB
        self.roleC = primary.roleC
        self.is_multi = False

    def for_rxn(self, rxn_id: int) -> SubManager:
        sub = self.subs.get(rxn_id)
        if sub is None:
            raise KeyError(f"rxn {rxn_id} not loaded; have {self.rxn_ids}")
        return sub

    @staticmethod
    def parse_rxn_from_name(name: str):
        try:
            return int(name.split(":")[1])
        except (IndexError, ValueError):
            return None

    def validate_molecules(self, config: dict, data: pd.DataFrame, time_elapsed: int = 0) -> pd.DataFrame:
        if data.empty:
            return data
        return self._validate_single(self.subs[self.rxn_ids[0]], config, data, time_elapsed)

    @staticmethod
    def _validate_single(sub: SubManager, config: dict, data: pd.DataFrame, time_elapsed: int = 0) -> pd.DataFrame:
        if data.empty:
            return data

        data = data.copy()
        data["smiles"] = data["name"].map(MoleculeUtils.get_smiles_from_reaction_cached)
        data = data[data["smiles"].notna()]
        if data.empty:
            return data

        min_heavy = config.get("min_heavy_atoms", 10)
        max_heavy = config.get("max_heavy_atoms", 40)
        data["heavy_atoms"] = data["smiles"].map(get_heavy_atom_count)
        data["bonds"] = data["smiles"].map(MoleculeUtils.num_rotatable_bonds)

        mask = (
            (data["heavy_atoms"] >= min_heavy)
            & (data["heavy_atoms"] <= max_heavy)
            & (data["bonds"] >= config.get("min_rotatable_bonds", 1))
            & (data["bonds"] <= config.get("max_rotatable_bonds", 10))
        )
        data = data[mask].reset_index(drop=True)
        if data.empty:
            return data

        banned = config.get("banned_atom_types") or []
        if banned:
            banned_set = set(banned)

            def _has_banned(smiles: str) -> bool:
                try:
                    mol = MoleculeUtils.mol_from_smiles_cached(smiles)
                    if mol is None:
                        return True
                    return bool({a.GetSymbol() for a in mol.GetAtoms()} & banned_set)
                except Exception:
                    return True

            data = data[~data["smiles"].map(_has_banned)].reset_index(drop=True)

        return data
