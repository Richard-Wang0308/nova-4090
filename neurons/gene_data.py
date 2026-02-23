import os
import sys
import random
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib
import time
import sqlite3
import pandas as pd
import numpy as np
import math
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
from dotenv import load_dotenv
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from rdkit.Chem import rdFingerprintGenerator
from collections import defaultdict
from functools import lru_cache

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
HARDCODED_RXN_ID = 2
STARTING_EPOCH = 21074
REACTION_TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'mols.csv')
SCORE_RESULTS_DB = os.path.join(BASE_DIR, "score_results.sqlite")

from config.config_loader import load_config
from utils import (
    get_smiles,
    get_heavy_atom_count,
    compute_maccs_entropy,
    molecule_unique_for_protein_hf,
    find_chemically_identical,
    is_reaction_allowed,
    contains_atom_type
)
from utils.molecules import molecule_unique_for_protein_hf
from molecules_base import generate_inchikey, SynthonLibrary, generate_molecules_from_synthon_library
from combinatorial_db.reactions import get_smiles_from_reaction, get_reaction_info

BOLTZ_AVAILABLE = False
BoltzWrapper = None

# Create global Morgan fingerprint generator
MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ============================================================================
# CACHING FUNCTIONS (from molecules.py)
# ============================================================================

@lru_cache(maxsize=1000_000)
def _get_smiles_from_reaction_cached(name: str):
    """Cache SMILES retrieval to avoid repeated database queries."""
    try:
        return get_smiles_from_reaction(name)
    except Exception:
        return None

@lru_cache(maxsize=1000_000)
def _mol_from_smiles_cached(smiles: str):
    """Cache molecule parsing to avoid repeated SMILES parsing."""
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

@lru_cache(maxsize=1000_000)
def _maccs_fp_from_smiles_cached(smiles: str):
    """Cache MACCS fingerprints for SMILES strings for fast Tanimoto similarity."""
    if not smiles:
        return None
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return None
        return MACCSkeys.GenMACCSKeys(mol)
    except Exception:
        return None

@lru_cache(maxsize=1000_000)
def _inchikey_from_name_cached(name: str) -> str:
    """Cache InChIKey generation from molecule name to avoid repeated computation."""
    try:
        s = _get_smiles_from_reaction_cached(name)
        if not s:
            return ""
        return generate_inchikey(s)
    except Exception:
        return ""

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
        print(f"Error getting molecules by role {role_mask}: {e}")
        return []


# ============================================================================
# COMPONENT WEIGHTING & DIVERSITY FUNCTIONS
# ============================================================================

def build_component_weights(top_pool: pd.DataFrame, rxn_id: int) -> Dict[str, Dict[int, float]]:
    """
    Build component weights based on scores of molecules containing them.
    Uses exponential weighting for top molecules to emphasize best components.
    """
    weights = {'A': defaultdict(float), 'B': defaultdict(float), 'C': defaultdict(float)}
    counts = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}
    
    if top_pool.empty:
        return weights
    
    max_score = top_pool['score'].max() if not top_pool.empty else 1.0
    
    for idx, row in top_pool.iterrows():
        name = row['name']
        score = row['score']
        
        # Exponential weighting: top molecules contribute more
        rank = idx + 1
        rank_weight = 2.5 * math.exp(-rank / 18.0)
        weighted_score = max(0, score) * rank_weight
        
        parts = name.split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                weights['A'][A_id] += weighted_score
                weights['B'][B_id] += weighted_score
                counts['A'][A_id] += 1
                counts['B'][B_id] += 1
                
                if len(parts) > 4:
                    C_id = int(parts[4])
                    weights['C'][C_id] += weighted_score
                    counts['C'][C_id] += 1
            except (ValueError, IndexError):
                continue
    
    # Normalize by count and add smoothing
    for role in ['A', 'B', 'C']:
        for comp_id in weights[role]:
            if counts[role][comp_id] > 0:
                avg_weight = weights[role][comp_id] / counts[role][comp_id]
                weights[role][comp_id] = avg_weight + 0.15
    
    return weights


def select_diverse_elites(top_pool: pd.DataFrame, n_elites: int, min_score_ratio: float = 0.65) -> pd.DataFrame:
    """
    Select diverse elite molecules: top by score, but ensure diversity in component space.
    """
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
    
    # First, add top scorer
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
    
    # Add diverse molecules
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
                
                is_diverse = (A_id not in used_components['A'] or 
                             B_id not in used_components['B'] or
                             (C_id is not None and C_id not in used_components['C']))
                
                if is_diverse or len(selected) < n_elites * 0.6:
                    selected.append(idx)
                    used_components['A'].add(A_id)
                    used_components['B'].add(B_id)
                    if C_id is not None:
                        used_components['C'].add(C_id)
            except (ValueError, IndexError):
                if len(selected) < n_elites:
                    selected.append(idx)
    
    # Fill remaining slots
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx not in selected:
            selected.append(idx)
    
    return candidates.loc[selected[:n_elites]] if selected else candidates.head(n_elites)


def compute_tanimoto_similarity_to_pool(
    candidate_smiles: pd.Series,
    pool_smiles: pd.Series,
) -> pd.Series:
    """
    Compute, for each candidate SMILES, the maximum MACCS Tanimoto similarity
    to any molecule in the reference pool.
    """
    if candidate_smiles.empty or pool_smiles.empty:
        return pd.Series(0.0, index=candidate_smiles.index, dtype=float)

    pool_fps = []
    for smi in pool_smiles.dropna().unique():
        fp = _maccs_fp_from_smiles_cached(smi)
        if fp is not None:
            pool_fps.append(fp)

    if not pool_fps:
        return pd.Series(0.0, index=candidate_smiles.index, dtype=float)

    similarities = {}
    for idx, smi in candidate_smiles.items():
        fp_cand = _maccs_fp_from_smiles_cached(smi)
        if fp_cand is None:
            similarities[idx] = 0.0
            continue
        max_sim = 0.0
        for fp_ref in pool_fps:
            try:
                sim = DataStructs.TanimotoSimilarity(fp_cand, fp_ref)
            except Exception:
                sim = 0.0
            if sim > max_sim:
                max_sim = sim
        similarities[idx] = float(max_sim)

    return pd.Series(similarities, dtype=float)


# ============================================================================
# ENHANCED SYNTHON LIBRARY (from molecules.py)
# ============================================================================

class EnhancedSynthonLibrary:
    """Enhanced synthon library with similarity search using Morgan fingerprints."""
    
    def __init__(self, db_path: str, rxn_id: int):
        self.db_path = db_path
        self.rxn_id = rxn_id
        self.reaction_info = get_reaction_info(rxn_id, db_path)
        
        if not self.reaction_info:
            raise ValueError(f"Could not load reaction {rxn_id}")
        
        self.smarts, self.roleA, self.roleB, self.roleC = self.reaction_info
        self.is_three_component = self.roleC is not None and self.roleC != 0
        
        # Load all components
        self.molecules_A = get_molecules_by_role(self.roleA, db_path)
        self.molecules_B = get_molecules_by_role(self.roleB, db_path)
        self.molecules_C = get_molecules_by_role(self.roleC, db_path) if self.is_three_component else []
        
        # Build fingerprint indices
        self.fps_A = self._build_fingerprint_index(self.molecules_A)
        self.fps_B = self._build_fingerprint_index(self.molecules_B)
        self.fps_C = self._build_fingerprint_index(self.molecules_C) if self.is_three_component else {}
        
        print(f"EnhancedSynthonLibrary initialized: {len(self.fps_A)} A components, "
                       f"{len(self.fps_B)} B components" + 
                       (f", {len(self.fps_C)} C components" if self.is_three_component else ""))
    
    def _build_fingerprint_index(self, molecules: List[Tuple[int, str, int]]) -> Dict[int, object]:
        """Build fingerprint index for fast similarity search."""
        fps = {}
        for mol_id, smiles, _ in molecules:
            mol = _mol_from_smiles_cached(smiles)
            if mol:
                fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                fps[mol_id] = fp
        return fps
    
    def find_similar_components(
        self, 
        target_smiles: str, 
        role: str = 'A',
        top_k: int = 80,
        min_similarity: float = 0.5
    ) -> List[Tuple[int, float]]:
        """Find components similar to target molecule."""
        target_mol = _mol_from_smiles_cached(target_smiles)
        if not target_mol:
            return []
        
        target_fp = MORGAN_FP_GENERATOR.GetFingerprint(target_mol)
        
        if role == 'A':
            fps_dict = self.fps_A
        elif role == 'B':
            fps_dict = self.fps_B
        elif role == 'C' and self.is_three_component:
            fps_dict = self.fps_C
        else:
            return []
        
        similarities = []
        for mol_id, fp in fps_dict.items():
            try:
                sim = DataStructs.TanimotoSimilarity(target_fp, fp)
                if sim >= min_similarity:
                    similarities.append((mol_id, sim))
            except Exception:
                continue
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_similar_to_molecule_name(
        self,
        molecule_name: str,
        vary_component: str = 'both',
        top_k_per_component: int = 10,
        min_similarity: float = 0.6
    ) -> Dict[str, List[int]]:
        """Given a high-scoring molecule name, find similar components."""
        parts = molecule_name.split(":")
        if len(parts) < 4:
            return {}
        
        try:
            if len(parts) == 4:
                _, rxn, A_id, B_id = parts
                A_id, B_id = int(A_id), int(B_id)
                C_id = None
            else:
                _, rxn, A_id, B_id, C_id = parts
                A_id, B_id, C_id = int(A_id), int(B_id), int(C_id)
        except (ValueError, IndexError):
            return {}
        
        result = {}
        
        if vary_component in ['A', 'both', 'all']:
            A_smiles = self._get_component_smiles(A_id, 'A')
            if A_smiles:
                similar_As = self.find_similar_components(
                    A_smiles, 'A', top_k_per_component, min_similarity
                )
                result['A'] = [mol_id for mol_id, _ in similar_As if mol_id != A_id]
        
        if vary_component in ['B', 'both', 'all']:
            B_smiles = self._get_component_smiles(B_id, 'B')
            if B_smiles:
                similar_Bs = self.find_similar_components(
                    B_smiles, 'B', top_k_per_component, min_similarity
                )
                result['B'] = [mol_id for mol_id, _ in similar_Bs if mol_id != B_id]
        
        if self.is_three_component and C_id and vary_component in ['C', 'all']:
            C_smiles = self._get_component_smiles(C_id, 'C')
            if C_smiles:
                similar_Cs = self.find_similar_components(
                    C_smiles, 'C', top_k_per_component, min_similarity
                )
                result['C'] = [mol_id for mol_id, _ in similar_Cs if mol_id != C_id]
        
        return result
    
    def _get_component_smiles(self, mol_id: int, role: str) -> str:
        """Get SMILES for a component by ID and role."""
        if role == 'A':
            molecules = self.molecules_A
        elif role == 'B':
            molecules = self.molecules_B
        elif role == 'C':
            molecules = self.molecules_C
        else:
            return None
        
        for mid, smiles, _ in molecules:
            if mid == mol_id:
                return smiles
        return None
    
    def generate_similar_molecules(
        self,
        base_molecule_names: List[str],
        n_per_base: int = 5,
        min_similarity: float = 0.6
    ) -> List[str]:
        """Generate new molecule names by finding similar components to base molecules."""
        new_molecules = []
        
        is_single_molecule = len(base_molecule_names) == 1
        if is_single_molecule:
            effective_n_per_base = n_per_base * 3 if n_per_base < 80 else n_per_base
        else:
            effective_n_per_base = n_per_base
        
        for base_name in base_molecule_names:
            parts = base_name.split(":")
            if len(parts) < 4:
                continue
            
            try:
                if len(parts) == 4:
                    _, rxn, A_id, B_id = parts
                    A_id, B_id = int(A_id), int(B_id)
                    
                    similar_comps = self.find_similar_to_molecule_name(
                        base_name, 'both', effective_n_per_base, min_similarity
                    )
                    
                    for new_A in similar_comps.get('A', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{new_A}:{B_id}")
                    
                    for new_B in similar_comps.get('B', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{new_B}")
                
                else:  # 3-component
                    _, rxn, A_id, B_id, C_id = parts
                    A_id, B_id, C_id = int(A_id), int(B_id), int(C_id)
                    
                    similar_comps = self.find_similar_to_molecule_name(
                        base_name, 'all', effective_n_per_base, min_similarity
                    )
                    
                    for new_A in similar_comps.get('A', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{new_A}:{B_id}:{C_id}")
                    
                    for new_B in similar_comps.get('B', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{new_B}:{C_id}")
                    
                    for new_C in similar_comps.get('C', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{B_id}:{new_C}")
            
            except (ValueError, IndexError) as e:
                print(f"Could not parse molecule name {base_name}: {e}")
                continue
        
        return list(dict.fromkeys(new_molecules))


def generate_molecules_from_enhanced_synthon_library(
    synthon_lib: EnhancedSynthonLibrary,
    top_molecules: pd.DataFrame,
    n_samples: int,
    min_similarity: float = 0.6,
    n_per_base: int = 10
) -> pd.DataFrame:
    """Generate new molecules using enhanced synthon similarity search."""
    if top_molecules.empty:
        return pd.DataFrame(columns=["name"])
    
    if len(top_molecules) == 1:
        seed_names = top_molecules["name"].tolist()
        effective_n_per_base = n_per_base * 4 if n_per_base < 80 else n_per_base
    else:
        n_seeds = min(30, len(top_molecules))
        seed_names = top_molecules.head(n_seeds)["name"].tolist()
        effective_n_per_base = n_per_base
    
    new_names = synthon_lib.generate_similar_molecules(
        seed_names,
        n_per_base=effective_n_per_base,
        min_similarity=min_similarity
    )
    
    if not new_names:
        return pd.DataFrame(columns=["name"])
    
    if len(new_names) > n_samples * 3.0:
        new_names = random.sample(new_names, int(n_samples * 2.0))
    
    return pd.DataFrame({"name": new_names})


# ============================================================================
# GENETIC ALGORITHM FUNCTIONS
# ============================================================================

def _parse_components(name: str) -> tuple:
    """Parse molecule name into components."""
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None
    A = int(parts[2])
    B = int(parts[3])
    C = int(parts[4]) if len(parts) > 4 else None
    return A, B, C


def _ids_from_pool(pool):
    """Extract IDs from molecule pool."""
    return [x[0] for x in pool]


def generate_offspring_from_elites(
    rxn_id: int, 
    n: int,
    is_three_component: bool,
    pool_A_ids: list,
    pool_B_ids: list,
    pool_C_ids: list,
    mutation_prob: float = 0.1, 
    seed: int | None = None,
    avoid_names: set[str] = None,
    avoid_inchikeys: set[str] = None,
    max_tries: int = 10,
    elite_As: set[int] = None,
    elite_Bs: set[int] = None,
    elite_Cs: set[int] = None
) -> list[str]:
    """Generate offspring from elite molecules with mutation."""
    rng = random.Random(seed) if seed is not None else random
    
    elite_As_list = list(elite_As) if elite_As else []
    elite_Bs_list = list(elite_Bs) if elite_Bs else []
    elite_Cs_list = list(elite_Cs) if elite_Cs else []

    out = []
    local_names = set()
    check_inchikeys = avoid_inchikeys is not None and len(avoid_inchikeys) > 0
    
    for _ in range(n):
        cand = None
        name = None
        for _try in range(max_tries):
            use_mutA = (not elite_As) or (rng.random() < mutation_prob)
            use_mutB = (not elite_Bs) or (rng.random() < mutation_prob)
            use_mutC = (not elite_Cs) or (rng.random() < mutation_prob)

            A = rng.choice(pool_A_ids) if use_mutA else rng.choice(elite_As_list)
            B = rng.choice(pool_B_ids) if use_mutB else rng.choice(elite_Bs_list)
            if is_three_component:
                C = rng.choice(pool_C_ids) if use_mutC else rng.choice(elite_Cs_list)
                name = f"rxn:{rxn_id}:{A}:{B}:{C}"
            else:
                name = f"rxn:{rxn_id}:{A}:{B}"

            if avoid_names and name in avoid_names:
                continue
            if name in local_names:
                continue

            if check_inchikeys:
                try:
                    key = _inchikey_from_name_cached(name)
                    if key and key in avoid_inchikeys:
                        continue
                except Exception:
                    pass

            cand = name
            break

        if cand is None:
            if name is None:
                A = rng.choice(pool_A_ids)
                B = rng.choice(pool_B_ids)
                if is_three_component:
                    C = rng.choice(pool_C_ids) if pool_C_ids else 0
                    name = f"rxn:{rxn_id}:{A}:{B}:{C}"
                else:
                    name = f"rxn:{rxn_id}:{A}:{B}"
            cand = name
        out.append(cand)
        local_names.add(cand)
        if avoid_names is not None:
            avoid_names.add(cand)
    return out


def generate_molecules_from_pools(
    rxn_id: int, 
    n: int, 
    molecules_A: List[Tuple], 
    molecules_B: List[Tuple], 
    molecules_C: List[Tuple], 
    is_three_component: bool, 
    seed: int = None,
    component_weights: dict = None
) -> List[str]:
    """Generate molecules from component pools with optional weighting."""
    rng = random.Random(seed) if seed is not None else random
    
    A_ids = [a[0] for a in molecules_A]
    B_ids = [b[0] for b in molecules_B]
    C_ids = [c[0] for c in molecules_C] if is_three_component else None
    
    if component_weights:
        weights_A = [component_weights.get('A', {}).get(aid, 1.0) for aid in A_ids]
        weights_B = [component_weights.get('B', {}).get(bid, 1.0) for bid in B_ids]
        weights_C = [component_weights.get('C', {}).get(cid, 1.0) for cid in C_ids] if is_three_component else None
        
        if weights_A:
            sum_w = sum(weights_A)
            weights_A = [w / sum_w if sum_w > 0 else 1.0/len(weights_A) for w in weights_A]
        if weights_B:
            sum_w = sum(weights_B)
            weights_B = [w / sum_w if sum_w > 0 else 1.0/len(weights_B) for w in weights_B]
        if weights_C:
            sum_w = sum(weights_C)
            weights_C = [w / sum_w if sum_w > 0 else 1.0/len(weights_C) for w in weights_C]
        
        picks_A = rng.choices(A_ids, weights=weights_A, k=n) if weights_A else rng.choices(A_ids, k=n)
        picks_B = rng.choices(B_ids, weights=weights_B, k=n) if weights_B else rng.choices(B_ids, k=n)
        if is_three_component:
            picks_C = rng.choices(C_ids, weights=weights_C, k=n) if weights_C else rng.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    else:
        picks_A = rng.choices(A_ids, k=n)
        picks_B = rng.choices(B_ids, k=n)
        if is_three_component:
            picks_C = rng.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    
    names = list(dict.fromkeys(names))
    return names


def validate_molecules_batch(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Validate molecules by checking heavy atom count and rotatable bonds."""
    if data.empty:
        return data
    
    data = data.copy()
    data['smiles'] = data["name"].apply(_get_smiles_from_reaction_cached)
    
    data = data[data['smiles'].notna()]
    if data.empty:
        return data
    
    data['heavy_atoms'] = data["smiles"].apply(get_heavy_atom_count)
    data['bonds'] = data["smiles"].apply(lambda s: Descriptors.NumRotatableBonds(_mol_from_smiles_cached(s)) if _mol_from_smiles_cached(s) else 0)
    
    mask = (
        (data['heavy_atoms'] >= config['min_heavy_atoms']) &
        (data['bonds'] >= config['min_rotatable_bonds']) &
        (data['bonds'] <= config['max_rotatable_bonds'])
    )
    data = data[mask]
    
    if not data.empty:
        data['InChIKey'] = data["smiles"].apply(generate_inchikey)
    
    return data


def generate_valid_random_molecules_batch(
    rxn_id: int,
    n_samples: int,
    db_path: str,
    subnet_config: dict,
    batch_size: int = 200,
    seed: int = None,
    elite_names: list[str] | None = None,
    elite_frac: float = 0.5,
    mutation_prob: float = 0.1,
    avoid_inchikeys: set[str] | None = None,
    component_weights: dict | None = None,
) -> pd.DataFrame:
    """Generate valid random molecules using genetic algorithm."""
    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        print(f"Could not get reaction info for rxn_id {rxn_id}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    
    smarts, roleA, roleB, roleC = reaction_info
    is_three_component = roleC is not None and roleC != 0
    
    molecules_A = get_molecules_by_role(roleA, db_path)
    molecules_B = get_molecules_by_role(roleB, db_path)
    molecules_C = get_molecules_by_role(roleC, db_path) if is_three_component else []

    if not molecules_A or not molecules_B or (is_three_component and not molecules_C):
        print(f"No molecules found for roles A={roleA}, B={roleB}, C={roleC}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    elite_As, elite_Bs, elite_Cs = set(), set(), set()
    if elite_names:
        for name in elite_names:
            A, B, C = _parse_components(name)
            if A is not None: 
                elite_As.add(A)
            if B is not None: 
                elite_Bs.add(B)
            if C is not None and is_three_component: 
                elite_Cs.add(C)

    pool_A_ids = _ids_from_pool(molecules_A)
    pool_B_ids = _ids_from_pool(molecules_B)
    pool_C_ids = _ids_from_pool(molecules_C) if is_three_component else []
    valid_dfs = []
    seen_keys = set()
    total_valid = 0
    
    while total_valid < n_samples:
        needed = n_samples - total_valid
        batch_size_actual = min(max(batch_size, 300), needed * 2)
        
        emitted_names = set()
        if elite_names:
            n_elite = max(0, min(batch_size_actual, int(batch_size_actual * elite_frac)))
            n_rand = batch_size_actual - n_elite

            elite_batch = generate_offspring_from_elites(
                rxn_id=rxn_id,
                n=n_elite,
                pool_A_ids=pool_A_ids,
                pool_B_ids=pool_B_ids,
                pool_C_ids=pool_C_ids,
                is_three_component=is_three_component,
                mutation_prob=mutation_prob,
                seed=seed,
                avoid_names=emitted_names,
                avoid_inchikeys=avoid_inchikeys,
                max_tries=10,
                elite_As=elite_As,
                elite_Bs=elite_Bs,
                elite_Cs=elite_Cs,
            )
            emitted_names.update(elite_batch)

            rand_batch = generate_molecules_from_pools(
                rxn_id, n_rand, molecules_A, molecules_B, molecules_C, is_three_component, seed, component_weights
            )
            rand_batch = [n for n in rand_batch if n and (n not in emitted_names)]
            batch_molecules = elite_batch + rand_batch

        else:
            batch_molecules = generate_molecules_from_pools(
                rxn_id, batch_size_actual, molecules_A, molecules_B, molecules_C, is_three_component, seed, component_weights
            )

        
        if not batch_molecules:
            continue
            
        batch_df = pd.DataFrame({"name": batch_molecules})
        batch_df = batch_df[batch_df["name"].notna()]
        if batch_df.empty:
            continue
            
        batch_df = validate_molecules_batch(batch_df, subnet_config)
        
        if batch_df.empty:
            continue

        batch_df = batch_df.drop_duplicates(subset=["InChIKey"], keep="first")
        
        mask = ~batch_df["InChIKey"].isin(seen_keys)
        if avoid_inchikeys:
            mask = mask & ~batch_df["InChIKey"].isin(avoid_inchikeys)
        batch_df = batch_df[mask]
        
        if batch_df.empty:
            continue
        
        seen_keys.update(batch_df["InChIKey"].values)
        valid_dfs.append(batch_df[["name", "smiles", "InChIKey"]].copy())
        total_valid += len(batch_df)
        
        if total_valid >= n_samples:
            break
        
    if not valid_dfs:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    
    result_df = pd.concat(valid_dfs, ignore_index=True)
    return result_df.head(n_samples).copy()


# ============================================================================
# EXISTING VALIDATION FUNCTIONS (keep as-is)
# ============================================================================

def validate_molecule_smiles(molecule_name: str, smiles: str) -> Tuple[bool, str]:
    """Validate SMILES string with RDKit."""
    if not smiles:
        return False, "No SMILES provided"
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, f"RDKit cannot parse SMILES: {smiles}"
        return True, ""
    except Exception as e:
        return False, f"RDKit parsing error: {str(e)}"


def validate_molecule_heavy_atoms(
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate heavy atom count."""
    try:
        heavy_atom_count = get_heavy_atom_count(smiles)
        min_atoms = config.get('min_heavy_atoms', 10)
        
        if heavy_atom_count < min_atoms:
            return False, f"Insufficient heavy atoms: {heavy_atom_count} < {min_atoms}"
        return True, ""
    except Exception as e:
        return False, f"Heavy atom count error: {str(e)}"


def validate_molecule_banned_atoms(
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate molecule doesn't contain banned atom types."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Cannot parse SMILES for banned atom check"
        
        banned_atoms = config.get('banned_atom_types', [])
        if not banned_atoms:
            return True, ""
        
        if contains_atom_type(mol, banned_atoms):
            return False, f"Contains banned atom types: {banned_atoms}"
        return True, ""
    except Exception as e:
        return False, f"Banned atom check error: {str(e)}"


def validate_molecule_rotatable_bonds(
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate rotatable bonds are within acceptable range."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Cannot parse SMILES for rotatable bonds check"
        
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        min_bonds = config.get('min_rotatable_bonds', 1)
        max_bonds = config.get('max_rotatable_bonds', 10)
        
        if num_rotatable_bonds < min_bonds or num_rotatable_bonds > max_bonds:
            return False, f"Rotatable bonds out of range: {num_rotatable_bonds} (expected {min_bonds}-{max_bonds})"
        return True, ""
    except Exception as e:
        return False, f"Rotatable bonds check error: {str(e)}"


async def validate_molecule_huggingface_unique(
    state: Dict[str, Any],
    molecule_name: str,
    smiles: str
) -> Tuple[bool, str]:
    """Validate molecule is unique (NOT in HuggingFace dataset)."""
    if not state.get('current_challenge_targets'):
        return False, "No target proteins available"
    
    primary_target = state['current_challenge_targets'][0]
    
    try:
        is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
        
        if not is_unique_hf:
            return False, f"Molecule already in HuggingFace dataset for {primary_target}"
        return True, ""
    except Exception as e:
        return False, f"HuggingFace uniqueness check error: {str(e)}"


async def validate_molecule_complete(
    state: Dict[str, Any],
    molecule_name: str,
    smiles: str,
    config: Dict[str, Any] = None
) -> Tuple[bool, List[str]]:
    """Perform complete validation on a molecule."""
    if config is None:
        config = state.get('config', {})
    
    errors = []
    
    is_valid, error_msg = validate_molecule_smiles(molecule_name, smiles)
    if not is_valid:
        errors.append(f"[SMILES] {error_msg}")
        return False, errors
    
    is_valid, error_msg = validate_molecule_heavy_atoms(molecule_name, smiles, config)
    if not is_valid:
        errors.append(f"[HEAVY_ATOMS] {error_msg}")
    
    is_valid, error_msg = validate_molecule_banned_atoms(molecule_name, smiles, config)
    if not is_valid:
        errors.append(f"[BANNED_ATOMS] {error_msg}")
    
    is_valid, error_msg = validate_molecule_rotatable_bonds(molecule_name, smiles, config)
    if not is_valid:
        errors.append(f"[ROTATABLE_BONDS] {error_msg}")
    
    is_valid, error_msg = await validate_molecule_huggingface_unique(state, molecule_name, smiles)
    if not is_valid:
        errors.append(f"[HF_UNIQUE] {error_msg}")
    
    return len(errors) == 0, errors


# ============================================================================
# DATABASE FUNCTIONS (keep as-is)
# ============================================================================

def init_score_results_db(db_path: str = None) -> None:
    """
    Initialize/create the score_results.sqlite database.
    
    Creates a table with molecule_name and score fields.
    """
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scored_molecules (
                molecule_name TEXT PRIMARY KEY,
                score REAL NOT NULL,
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                available BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Create index on score for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)
        """)
        
        conn.commit()
        conn.close()
        bt.logging.debug(f"Initialized score_results database at {db_path}")
    except Exception as e:
        bt.logging.error(f"Error initializing score_results database: {e}")

def get_score_from_db(molecule_name: str, db_path: str = None) -> Optional[float]:
    """Get score for a molecule from the database."""
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT score FROM scored_molecules WHERE molecule_name = ?", (molecule_name,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return float(result[0])
        return None
    except Exception as e:
        print(f"Error getting score from DB for {molecule_name}: {e}")
        return None


def write_scores_to_db(molecules: List[Dict[str, Any]], db_path: str = None) -> None:
    """Write scored molecules to the database."""
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    
    if not molecules:
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        to_insert = []
        for mol in molecules:
            molecule_name = mol.get('name')
            score = mol.get('boltz_score')
            
            if molecule_name and score is not None:
                to_insert.append((molecule_name, float(score), True))
        
        if to_insert:
            cursor.executemany(
                "INSERT INTO scored_molecules (molecule_name, score, available) VALUES (?, ?, ?)",
                to_insert
            )
            conn.commit()
            print(f"✅ Wrote {len(to_insert)} scored molecules to database")
        
        conn.close()
    except Exception as e:
        print(f"Error writing scores to database: {e}")

def batch_get_scores_from_db(molecule_names: List[str], db_path: str = None) -> Dict[str, float]:
    """Get scores for multiple molecules from the database in batch."""
    if db_path is None:
        db_path = SCORE_RESULTS_DB
    
    if not molecule_names:
        return {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(molecule_names))
        cursor.execute(
            f"SELECT molecule_name, score FROM scored_molecules WHERE molecule_name IN ({placeholders})",
            molecule_names
        )
        results = cursor.fetchall()
        conn.close()
        
        return {name: float(score) for name, score in results}
    except Exception as e:
        print(f"Error batch getting scores from DB: {e}")
        return {}


def load_molecules_from_db_with_validation(
    db_path: str,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from SQLite database with validation from config.yaml."""
    if config is None:
        config = {}
    
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    try:
        print(f"Loading molecules from database {db_path} for rxn_id={rxn_id}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT molecule_name, score FROM scored_molecules")
        db_results = cursor.fetchall()
        conn.close()
        
        if not db_results:
            print("No molecules found in database")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        result_rows = []
        successful_count = 0
        failed_count = 0
        banned_atom_count = 0
        heavy_atom_count = 0
        wrong_rxn_id_count = 0
        
        for molecule_name, score in db_results:
            try:
                if not molecule_name.startswith(f"rxn:{rxn_id}:"):
                    wrong_rxn_id_count += 1
                    continue
                
                smiles = get_smiles_from_reaction(molecule_name)
                
                if not smiles:
                    failed_count += 1
                    continue
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed_count += 1
                    continue
                
                banned_atoms = config.get('banned_atom_types', [])
                if banned_atoms and contains_atom_type(mol, banned_atoms):
                    banned_atom_count += 1
                    continue
                
                min_heavy_atoms = config.get('min_heavy_atoms', 10)
                heavy_atom_count_val = get_heavy_atom_count(smiles)
                if heavy_atom_count_val < min_heavy_atoms:
                    heavy_atom_count += 1
                    continue
                
                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    failed_count += 1
                    continue
                
                result_rows.append({
                    'name': molecule_name,
                    'smiles': smiles,
                    'InChIKey': inchikey,
                    'score': float(score) if score is not None else None,
                })
                successful_count += 1
                
            except Exception as e:
                failed_count += 1
                continue
        
        result_df = pd.DataFrame(result_rows)
        if not result_df.empty:
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')
            
            if 'score' in result_df.columns:
                result_df = result_df.sort_values(by='score', ascending=False, na_position='last')
                print(
                    f"✅ Loaded {len(result_df)} molecules from database "
                    f"(successful: {successful_count}, failed: {failed_count}, "
                    f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count}, "
                    f"wrong rxn_id: {wrong_rxn_id_count})"
                )
                if len(result_df) > 0:
                    scores = result_df['score'].dropna()
                    if len(scores) > 0:
                        print(
                            f"   Score range: {scores.min():.6f} to {scores.max():.6f} "
                            f"(top 3: {scores.head(3).tolist()})"
                        )
        else:
            print(
                f"No valid molecules loaded from database "
                f"(successful: {successful_count}, failed: {failed_count}, "
                f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count}, "
                f"wrong rxn_id: {wrong_rxn_id_count})"
            )
        
        return result_df
        
    except Exception as e:
        print(f"Error loading molecules from database: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_from_csv_with_validation(
    csv_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from CSV file with validation from config.yaml."""
    if config is None:
        config = {}
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    try:
        print(
            f"Loading molecules from {csv_path} for targets {target_proteins}, "
            f"epoch >= {starting_epoch}, rxn_id={rxn_id}"
        )
        df = pd.read_csv(csv_path)
        
        if 'epoch' in df.columns:
            df = df[df['epoch'] >= starting_epoch]
        else:
            print("CSV file does not have 'epoch' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        if 'molecule_name' in df.columns:
            df = df[df['molecule_name'].str.startswith(f"rxn:{rxn_id}:", na=False)]
        else:
            print("CSV file does not have 'molecule_name' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        if df.empty:
            print("No matching molecules found in CSV")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        result_rows = []
        successful_count = 0
        failed_count = 0
        banned_atom_count = 0
        heavy_atom_count = 0
        
        for _, row in df.iterrows():
            molecule_name = row['molecule_name']
            
            try:
                smiles = get_smiles_from_reaction(molecule_name)
                
                if not smiles:
                    failed_count += 1
                    continue
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed_count += 1
                    continue
                
                banned_atoms = config.get('banned_atom_types', [])
                if banned_atoms and contains_atom_type(mol, banned_atoms):
                    banned_atom_count += 1
                    continue
                
                min_heavy_atoms = config.get('min_heavy_atoms', 10)
                heavy_atom_count_val = get_heavy_atom_count(smiles)
                if heavy_atom_count_val < min_heavy_atoms:
                    heavy_atom_count += 1
                    continue
                
                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    failed_count += 1
                    continue
                
                final_score = row.get('final_score', None)
                if pd.isna(final_score):
                    final_score = None
                else:
                    final_score = float(final_score)
                
                result_rows.append({
                    'name': molecule_name,
                    'smiles': smiles,
                    'InChIKey': inchikey,
                    'score': final_score,
                })
                successful_count += 1
                
            except Exception as e:
                failed_count += 1
                continue
        
        result_df = pd.DataFrame(result_rows)
        if not result_df.empty:
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')
            
            if 'score' in result_df.columns:
                result_df = result_df.sort_values(by='score', ascending=False, na_position='last')
                print(
                    f"✅ Loaded {len(result_df)} molecules from CSV "
                    f"(successful: {successful_count}, failed: {failed_count}, "
                    f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count})"
                )
                if len(result_df) > 0:
                    scores = result_df['score'].dropna()
                    if len(scores) > 0:
                        print(
                            f"   Score range: {scores.min():.6f} to {scores.max():.6f} "
                            f"(top 3: {scores.head(3).tolist()})"
                        )
        else:
            print(
                f"No valid molecules loaded from CSV "
                f"(successful: {successful_count}, failed: {failed_count}, "
                f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count})"
            )
        
        return result_df
        
    except Exception as e:
        print(f"Error loading molecules from CSV: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_combined(
    csv_path: str,
    db_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from both CSV and database, merge them, and deduplicate."""
    if config is None:
        config = {}
    
    print(f"🔄 Loading molecules from CSV and database...")
    
    csv_df = load_molecules_from_csv_with_validation(
        csv_path, target_proteins, starting_epoch, rxn_id, config
    )
    
    db_df = load_molecules_from_db_with_validation(
        db_path, rxn_id, config
    )
    
    if csv_df.empty and db_df.empty:
        print("No molecules loaded from either CSV or database")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    if csv_df.empty:
        print("No molecules from CSV, using database only")
        return db_df
    
    if db_df.empty:
        print("No molecules from database, using CSV only")
        return csv_df
    
    csv_df['source'] = 'csv'
    db_df['source'] = 'database'
    
    combined_df = pd.concat([csv_df, db_df], ignore_index=True)
    
    combined_df = combined_df.sort_values(
        by=['score', 'source'],
        ascending=[False, True],
        na_position='last'
    )
    
    combined_df = combined_df.drop_duplicates(subset=['InChIKey'], keep='first')
    combined_df = combined_df.drop(columns=['source'])
    combined_df = combined_df.sort_values(by='score', ascending=False, na_position='last')
    
    csv_count = len(csv_df)
    db_count = len(db_df)
    combined_count = len(combined_df)
    duplicates_removed = csv_count + db_count - combined_count
    
    print(
        f"✅ Combined loading complete: "
        f"{csv_count} from CSV, {db_count} from database, "
        f"{combined_count} unique molecules after deduplication "
        f"({duplicates_removed} duplicates removed)"
    )
    
    if combined_count > 0:
        scores = combined_df['score'].dropna()
        if len(scores) > 0:
            print(
                f"   Combined score range: {scores.min():.6f} to {scores.max():.6f} "
                f"(top 3: {scores.head(3).tolist()})"
            )
    
    return combined_df


async def load_submissions_from_csv(
    state: Dict[str, Any],
    csv_path: str,
    start_epoch: int,
    rxn_id: int,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """Load submissions from both CSV and database, merge them, and return top 200."""
    try:
        target_proteins = state.get('current_challenge_targets', [])
        molecules_df = load_molecules_combined(
            csv_path,
            SCORE_RESULTS_DB,
            target_proteins,
            start_epoch,
            rxn_id,
            config
        )
        
        if molecules_df.empty:
            print("⚠️  No valid molecules loaded from CSV or database")
            return pd.DataFrame()
        
        top_200 = molecules_df.head(200)
        
        print(f"✅ Selected top 200 molecules from combined CSV and database")
        
        if len(top_200) > 0 and 'score' in top_200.columns:
            scores = top_200['score'].dropna()
            if len(scores) > 0:
                print(
                    f"   Top 200 score range: {scores.min():.6f} to {scores.max():.6f} "
                    f"(top 3: {scores.head(3).tolist()})"
                )
        
        return top_200
        
    except Exception as e:
        print(f"Error loading molecules from CSV and database: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()


# ============================================================================
# SCORING FUNCTIONS (keep as-is)
# ============================================================================

async def score_molecules_with_boltz_batched(
    state: Dict[str, Any],
    molecules: List[Dict[str, Any]],
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """Score molecules using BoltzWrapper in batches."""
    if state.get('boltz_wrapper') is None:
        print("BoltzWrapper not available, skipping scoring")
        return molecules
    
    if not molecules:
        return molecules
    
    print(f"🔬 Processing {len(molecules)} molecules for scoring in batches of {batch_size}...")
    
    init_score_results_db()
    
    all_scored_molecules = []
    total_batches = (len(molecules) + batch_size - 1) // batch_size
    
    # Process molecules in batches
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(molecules))
        batch = molecules[start_idx:end_idx]
        
        print(
            f"📦 Batch {batch_idx + 1}/{total_batches}: "
            f"Scoring {len(batch)} molecules"
        )
        
        # Score this batch
        molecules_to_score = []
        molecules_with_db_scores = []
        molecules_in_hf = []
        
        target_proteins = state.get('current_challenge_targets', [])
        primary_target = target_proteins[0] if target_proteins else None
        
        molecule_names = [mol['name'] for mol in batch]
        db_scores = batch_get_scores_from_db(molecule_names)
        
        print(f"   Found {len(db_scores)} molecules already in database")
        
        # Separate molecules by source
        for mol in batch:
            molecule_name = mol['name']
            smiles = mol.get('smiles')
            
            if molecule_name in db_scores:
                mol['boltz_score'] = db_scores[molecule_name]
                mol['boltz_score_source'] = 'database'
                molecules_with_db_scores.append(mol)
                print(f"   ✓ {molecule_name}: score from DB = {db_scores[molecule_name]:.6f}")
                continue
            
            if primary_target and smiles:
                try:
                    is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
                    if not is_unique_hf:
                        print(f"   ⏭️  {molecule_name}: already in HuggingFace, skipping")
                        molecules_in_hf.append(mol)
                        continue
                except Exception as e:
                    print(f"   Error checking HuggingFace for {molecule_name}: {e}")
            
            molecules_to_score.append(mol)
        
        print(
            f"   Breakdown: {len(molecules_with_db_scores)} from DB, "
            f"{len(molecules_in_hf)} in HuggingFace (skipped), "
            f"{len(molecules_to_score)} need scoring"
        )
        
        newly_scored_molecules = []
        if molecules_to_score:
            print(f"   Scoring {len(molecules_to_score)} new molecules with Boltz...")
            
            boltz = state['boltz_wrapper']
            config = state['config']
            target_proteins = state.get('current_challenge_targets', [])
            antitarget_proteins = state.get('current_challenge_antitargets', [])
            
            if not target_proteins:
                print("No target proteins available for scoring")
                all_scored_molecules.extend(molecules_with_db_scores)
                continue
            
            primary_target = target_proteins[0]
            
            try:
                output_dir = os.path.join(boltz.output_dir, 'boltz_results_inputs')
                if os.path.exists(output_dir):
                    try:
                        lightning_logs_dir = os.path.join(output_dir, 'lightning_logs')
                        if os.path.exists(lightning_logs_dir):
                            import shutil
                            shutil.rmtree(lightning_logs_dir, ignore_errors=True)
                            print(f"Cleaned up old lightning_logs directory")
                    except Exception as cleanup_err:
                        print(f"Could not clean up old logs: {cleanup_err}")
                
                processed_dir = os.path.join(output_dir, 'processed')
                structures_dir = os.path.join(processed_dir, 'structures')
                records_dir = os.path.join(processed_dir, 'records')
                msa_dir = os.path.join(processed_dir, 'msa')
                predictions_dir = os.path.join(output_dir, 'predictions')
                
                os.makedirs(structures_dir, exist_ok=True)
                os.makedirs(records_dir, exist_ok=True)
                os.makedirs(msa_dir, exist_ok=True)
                os.makedirs(predictions_dir, exist_ok=True)
                
                valid_molecules_by_uid = {
                    0: {
                        'smiles': [mol['smiles'] for mol in molecules_to_score],
                        'names': [mol['name'] for mol in molecules_to_score]
                    }
                }
                
                score_dict = {
                    0: {
                        "target_scores": [[]],
                        "antitarget_scores": [[]],
                        "entropy": None,
                        "entropy_boltz": None,
                        "block_submitted": None,
                        "push_time": ""
                    }
                }
                
                num_molecules_to_score = len(molecules_to_score)
                subnet_config = {
                    'weekly_target': primary_target,
                    'num_antitargets': len(antitarget_proteins),
                    'binding_pocket': config.get('binding_pocket'),
                    'max_distance': config.get('max_distance'),
                    'force': config.get('force', False),
                    'num_molecules_boltz': num_molecules_to_score,
                    'boltz_metric': config.get('boltz_metric', ['affinity_probability_binary', 'affinity_pred_value']),
                    'combination_strategy': config.get('combination_strategy', 'heavy_atom_normalization'),
                    'sample_selection': config.get('sample_selection', 'first'),
                }
                
                final_block_hash = "0x" + "0" * 64
                
                print(f"   Running Boltz scoring for {len(molecules_to_score)} molecules...")
                start_time = time.time()
                
                def run_scoring():
                    boltz.score_molecules_target(
                        valid_molecules_by_uid,
                        score_dict,
                        subnet_config,
                        final_block_hash
                    )
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, run_scoring)
                
                elapsed = time.time() - start_time
                print(f"   ✅ Boltz scoring completed in {elapsed:.2f} seconds")
                
                uid = 0
                smiles_to_score = {}
                if uid in boltz.per_molecule_metric:
                    smiles_to_score = boltz.per_molecule_metric[uid].copy()
                    print(f"   ✅ Loaded {len(smiles_to_score)} unique SMILES scores")
                
                target_scores_list = None
                target_scores = score_dict[uid].get('target_scores', [[]])
                if target_scores and len(target_scores[0]) > 0:
                    target_scores_list = target_scores[0] if isinstance(target_scores[0], list) else [target_scores[0]]
                
                avg_score = None
                if not smiles_to_score and not target_scores_list:
                    avg_score = score_dict[uid].get('boltz_score')
                
                for mol_idx, mol in enumerate(molecules_to_score):
                    smiles = mol['smiles']
                    score = None
                    
                    if smiles in smiles_to_score:
                        score = smiles_to_score[smiles]
                    elif target_scores_list and mol_idx < len(target_scores_list):
                        score = target_scores_list[mol_idx]
                    elif target_scores_list:
                        try:
                            valid_idx = valid_molecules_by_uid[uid]['smiles'].index(smiles)
                            if valid_idx < len(target_scores_list):
                                score = target_scores_list[valid_idx]
                        except (ValueError, IndexError):
                            pass
                    
                    if score is None and avg_score is not None:
                        score = avg_score
                    
                    mol['boltz_score'] = score
                    if score is not None:
                        newly_scored_molecules.append(mol)
                
                if newly_scored_molecules:
                    print(f"Newly scored molecules: {newly_scored_molecules}")
                    for mol in newly_scored_molecules:
                        print(f"  - {mol['name']}: {mol['boltz_score']}")
                    write_scores_to_db(newly_scored_molecules)
            
            except Exception as e:
                print(f"❌ Error scoring batch with Boltz: {e}")
                import traceback
                print(traceback.format_exc())
        
        # Combine results from this batch
        batch_results = molecules_with_db_scores + newly_scored_molecules
        
        for mol in molecules_in_hf:
            mol['boltz_score'] = None
            mol['boltz_score_source'] = 'huggingface_skipped'
            batch_results.append(mol)
        
        all_scored_molecules.extend(batch_results)
        
        print(
            f"   ✅ Batch {batch_idx + 1} complete: "
            f"{len(molecules_with_db_scores)} from DB, "
            f"{len(newly_scored_molecules)} newly scored, "
            f"{len(molecules_in_hf)} skipped"
        )
    
    # Sort all results by score
    scored_molecules = sorted(
        all_scored_molecules,
        key=lambda m: m.get('boltz_score') if m.get('boltz_score') is not None else float('-inf'),
        reverse=True
    )
    
    print(f"✅ Batch scoring complete: {len(scored_molecules)} total molecules scored")
    
    return scored_molecules


def _import_boltz_wrapper():
    """Import BoltzWrapper following DataGenerator pattern."""
    global BOLTZ_AVAILABLE, BoltzWrapper
    
    try:
        BOLTZ_SCORING_DIR = os.path.join(BASE_DIR, "boltz-scoring")
        BOLTZ_SRC_DIR = os.path.join(BOLTZ_SCORING_DIR, "boltz", "src")
        
        if not os.path.exists(BOLTZ_SCORING_DIR):
            print(f"⚠️  Boltz-scoring directory not found at {BOLTZ_SCORING_DIR}")
            return False
        
        if BOLTZ_SCORING_DIR not in sys.path:
            sys.path.append(BOLTZ_SCORING_DIR)
        
        if BOLTZ_SRC_DIR not in sys.path:
            sys.path.insert(0, BOLTZ_SRC_DIR)
        
        boltz_utils_path = os.path.join(BOLTZ_SCORING_DIR, 'utils')
        if os.path.exists(boltz_utils_path) and boltz_utils_path not in sys.path:
            sys.path.insert(0, boltz_utils_path)
        
        from boltz.wrapper import BoltzWrapper as BW
        BoltzWrapper = BW
        BOLTZ_AVAILABLE = True
        print(f"✅ BoltzWrapper imported successfully")
        return True
        
    except ImportError as e:
        print(f"⚠️  Failed to import BoltzWrapper: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Error setting up BoltzWrapper: {e}")
        return False


# ============================================================================
# STARTUP & GENERATION FUNCTIONS
# ============================================================================

async def startup_phase(state: Dict[str, Any]) -> None:
    """
    Startup phase:
    1. Initialize score_results database
    2. Import and initialize BoltzWrapper
    3. Load molecules from CSV and database with validation
    4. Initialize enhanced synthon library
    5. Prepare top_pool
    """
    print("🚀 Starting STARTUP phase: Initialize DB, Boltz, & Load CSV/DB...")
    
    try:
        # Initialize score_results database
        print("💾 Initializing score_results database...")
        init_score_results_db()
        print(f"✅ Score results database initialized")
        
        # Log validation config
        config = state['config']
        print(
            f"✅ Loaded validation config from config.yaml:"
            f"\n   - min_heavy_atoms: {config.get('min_heavy_atoms', 10)}"
            f"\n   - min_rotatable_bonds: {config.get('min_rotatable_bonds', 1)}"
            f"\n   - max_rotatable_bonds: {config.get('max_rotatable_bonds', 10)}"
            f"\n   - banned_atom_types: {config.get('banned_atom_types', [])}"
        )
        
        # Import BoltzWrapper
        print("🔬 Importing BoltzWrapper...")
        boltz_imported = _import_boltz_wrapper()
        
        # Initialize BoltzWrapper
        if boltz_imported and BoltzWrapper is not None:
            print("🔬 Initializing BoltzWrapper...")
            try:
                state['boltz_wrapper'] = BoltzWrapper()
                print("✅ BoltzWrapper initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize BoltzWrapper: {e}")
                import traceback
                print(traceback.format_exc())
                state['boltz_wrapper'] = None
        else:
            print("⚠️  BoltzWrapper not available, scoring will be skipped")
            state['boltz_wrapper'] = None
        
        # Load submissions from both CSV and database
        print("📂 Loading submissions from CSV and database...")
        top_200_df = await load_submissions_from_csv(
            state,
            REACTION_TRAIN_CSV,
            STARTING_EPOCH,
            HARDCODED_RXN_ID,
            config
        )
        
        # Store top_200_df in state
        state['top_200_df'] = top_200_df
        
        if top_200_df.empty:
            print("⚠️  No molecules loaded from CSV or database")
            return
        
        print(f"✅ Loaded {len(top_200_df)} top molecules from CSV and database")
        
        # Initialize enhanced synthon library
        print(f"🔬 Initializing EnhancedSynthonLibrary for rxn_id={HARDCODED_RXN_ID}...")
        try:
            synthon_lib_start = time.time()
            synthon_lib = EnhancedSynthonLibrary(DB_PATH, HARDCODED_RXN_ID)
            state['synthon_lib'] = synthon_lib
            state['synthon_lib_ready'] = True
            elapsed = time.time() - synthon_lib_start
            print(f"✅ EnhancedSynthonLibrary initialized in {elapsed:.2f}s")
        except Exception as e:
            print(f"❌ Failed to initialize EnhancedSynthonLibrary: {e}")
            import traceback
            print(traceback.format_exc())
            state['synthon_lib'] = None
            state['synthon_lib_ready'] = False
        
        # Update state with top molecules
        state['top_pool'] = top_200_df.copy()
        state['seen_inchikeys'].update(top_200_df['InChIKey'].tolist())
        
        print(
            f"✅ STARTUP COMPLETE:"
            f"\n   Total molecules in pool: {len(state['top_pool'])}"
            f"\n   Top 200 molecules: {len(state['top_200_df'])}"
            f"\n   BoltzWrapper: {'✅ Ready' if state.get('boltz_wrapper') else '❌ Not available'}"
            f"\n   SynthonLibrary: {'✅ Ready' if state.get('synthon_lib_ready') else '❌ Not available'}"
        )
        
        state['startup_complete'] = True
    
    except Exception as e:
        print(f"Error in startup phase: {e}")
        import traceback
        print(traceback.format_exc())


async def generate_unique_molecules_adaptive(
    state: Dict[str, Any],
    top_200_df: pd.DataFrame,
    desired_count: int = 100
) -> List[Dict[str, Any]]:
    """
    Generate unique molecules using ADAPTIVE strategy from miner.py:
    - Synthon search with multi-range strategy (tight/medium/broad)
    - Genetic algorithm with elite selection and mutation
    - Component weighting based on top molecules
    """
    if top_200_df.empty:
        print("Top 200 DataFrame is empty")
        return []
    
    config = state['config']
    rxn_id = state['rxn_id']
    synthon_lib = state.get('synthon_lib')
    synthon_lib_ready = state.get('synthon_lib_ready', False)
    
    # Get adaptive parameters from state
    iteration = state.get('iteration', 1)
    prev_avg_score = state.get('prev_avg_score')
    current_avg_score = top_200_df['score'].mean() if not top_200_df.empty else None
    score_improvement_rate = state.get('score_improvement_rate', 0.0)
    no_improvement_counter = state.get('no_improvement_counter', 0)
    
    # Calculate score improvement
    if current_avg_score is not None and prev_avg_score is not None:
        score_improvement_rate = (current_avg_score - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
    
    # Update state
    state['prev_avg_score'] = current_avg_score
    state['score_improvement_rate'] = score_improvement_rate
    
    print(
        f"🧬 Iteration {iteration}: Generating {desired_count} molecules "
        f"(improvement: {score_improvement_rate:.4f}, no_improve: {no_improvement_counter})"
    )
    
    new_molecules = []
    generated_molecules = state.get('generated_molecules', set())
    generated_inchikeys = state.get('generated_inchikeys', set())
    
    # Build component weights and elite pool
    component_weights = build_component_weights(top_200_df, rxn_id) if not top_200_df.empty else None
    elite_df = select_diverse_elites(top_200_df, min(150, len(top_200_df))) if not top_200_df.empty else pd.DataFrame()
    elite_names = elite_df["name"].tolist() if not elite_df.empty else None
    
    # Adaptive parameters
    mutation_prob = state.get('mutation_prob', 0.5)
    elite_frac = state.get('elite_frac', 0.4)
    
    # STRATEGY SELECTION based on improvement and iteration
    if synthon_lib_ready and iteration > 1 and not top_200_df.empty:
        current_max_score = top_200_df['score'].max()
        has_high_score = current_max_score > 0.01
        has_very_high_score = current_max_score > 0.015
        
        # MULTI-RANGE SYNTHON STRATEGY (from miner.py)
        if score_improvement_rate <= 0.01:
            print(f"   Using MULTI-RANGE synthon strategy (low improvement)")
            
            if has_very_high_score:
                # Ultra-aggressive: TOP 1 + tight + medium + broad
                n_synthon_top1 = int(desired_count * 0.21)
                synthon_top1_df = generate_molecules_from_enhanced_synthon_library(
                    synthon_lib, top_200_df.head(1), n_synthon_top1,
                    min_similarity=0.85, n_per_base=50
                )
                
                n_synthon_tight = int(desired_count * 0.07)
                synthon_tight_df = generate_molecules_from_enhanced_synthon_library(
                    synthon_lib, top_200_df.head(5), n_synthon_tight,
                    min_similarity=0.80, n_per_base=30
                )
                
                n_synthon_medium = int(desired_count * 0.21)
                seed_medium = top_200_df.iloc[10:40] if len(top_200_df) > 40 else top_200_df.iloc[5:]
                synthon_medium_df = generate_molecules_from_enhanced_synthon_library(
                    synthon_lib, seed_medium, n_synthon_medium,
                    min_similarity=0.55, n_per_base=15
                )
                
                n_synthon_broad = int(desired_count * 0.21)
                synthon_broad_df = generate_molecules_from_enhanced_synthon_library(
                    synthon_lib, top_200_df.head(50), n_synthon_broad,
                    min_similarity=0.40, n_per_base=20
                )
                
                synthon_df = pd.concat([synthon_top1_df, synthon_tight_df, synthon_medium_df, synthon_broad_df], ignore_index=True)
            else:
                # Standard multi-range: tight + medium + broad
                n_synthon_tight = int(desired_count * 0.28)
                synthon_tight_df = generate_molecules_from_enhanced_synthon_library(
                    synthon_lib, top_200_df.head(5), n_synthon_tight,
                    min_similarity=0.80, n_per_base=30
                )
                
                n_synthon_medium = int(desired_count * 0.21)
                seed_medium = top_200_df.iloc[10:40] if len(top_200_df) > 40 else top_200_df.iloc[5:]
                synthon_medium_df = generate_molecules_from_enhanced_synthon_library(
                    synthon_lib, seed_medium, n_synthon_medium,
                    min_similarity=0.55, n_per_base=15
                )
                
                n_synthon_broad = int(desired_count * 0.21)
                synthon_broad_df = generate_molecules_from_enhanced_synthon_library(
                    synthon_lib, top_200_df.head(50), n_synthon_broad,
                    min_similarity=0.40, n_per_base=20
                )
                
                synthon_df = pd.concat([synthon_tight_df, synthon_medium_df, synthon_broad_df], ignore_index=True)
            
            synthon_df = synthon_df.drop_duplicates(subset=["name"], keep="first")
            
            if not synthon_df.empty:
                synthon_df = validate_molecules_batch(synthon_df, config)
                print(f"   ✅ {len(synthon_df)} multi-range synthon candidates validated")
            
            # Fill remaining with GA
            n_traditional = desired_count - len(synthon_df)
            if n_traditional > 0:
                traditional_df = generate_valid_random_molecules_batch(
                    rxn_id, n_traditional, DB_PATH, config,
                    batch_size=400, elite_names=elite_names,
                    elite_frac=elite_frac, mutation_prob=mutation_prob,
                    avoid_inchikeys=state['seen_inchikeys'],
                    component_weights=component_weights
                )
            else:
                traditional_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
            
            data = pd.concat([synthon_df, traditional_df], ignore_index=True)
            data = data.drop_duplicates(subset=["name"], keep="first")
            print(f"   Combined: {len(data)} total ({len(synthon_df)} synthon + {len(traditional_df)} GA)")
        
        elif score_improvement_rate > 0.05:
            # High improvement: tight exploration
            print(f"   Using TIGHT synthon strategy (high improvement)")
            n_synthon = int(desired_count * 0.75)
            synthon_df = generate_molecules_from_enhanced_synthon_library(
                synthon_lib, top_200_df.head(20), n_synthon,
                min_similarity=0.75, n_per_base=15
            )
            
            if not synthon_df.empty:
                synthon_df = validate_molecules_batch(synthon_df, config)
            
            n_traditional = desired_count - len(synthon_df)
            if n_traditional > 0:
                traditional_df = generate_valid_random_molecules_batch(
                    rxn_id, n_traditional, DB_PATH, config,
                    batch_size=300, elite_names=elite_names,
                    elite_frac=elite_frac, mutation_prob=mutation_prob,
                    avoid_inchikeys=state['seen_inchikeys'],
                    component_weights=component_weights
                )
            else:
                traditional_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
            
            data = pd.concat([synthon_df, traditional_df], ignore_index=True)
            data = data.drop_duplicates(subset=["name"], keep="first")
            print(f"   Combined: {len(data)} total ({len(synthon_df)} synthon + {len(traditional_df)} GA)")
        
        else:
            # Medium improvement: balanced exploration
            print(f"   Using BALANCED synthon strategy (medium improvement)")
            n_synthon = int(desired_count * 0.70)
            synthon_df = generate_molecules_from_enhanced_synthon_library(
                synthon_lib, top_200_df.head(30), n_synthon,
                min_similarity=0.65, n_per_base=20
            )
            
            if not synthon_df.empty:
                synthon_df = validate_molecules_batch(synthon_df, config)
            
            n_traditional = desired_count - len(synthon_df)
            if n_traditional > 0:
                traditional_df = generate_valid_random_molecules_batch(
                    rxn_id, n_traditional, DB_PATH, config,
                    batch_size=300, elite_names=elite_names,
                    elite_frac=elite_frac, mutation_prob=mutation_prob,
                    avoid_inchikeys=state['seen_inchikeys'],
                    component_weights=component_weights
                )
            else:
                traditional_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
            
            data = pd.concat([synthon_df, traditional_df], ignore_index=True)
            data = data.drop_duplicates(subset=["name"], keep="first")
            print(f"   Combined: {len(data)} total ({len(synthon_df)} synthon + {len(traditional_df)} GA)")
    
    else:
        # Standard genetic algorithm
        print(f"   Using standard GENETIC ALGORITHM")
        data = generate_valid_random_molecules_batch(
            rxn_id, desired_count, DB_PATH, config,
            batch_size=400, elite_names=elite_names,
            elite_frac=elite_frac, mutation_prob=mutation_prob,
            avoid_inchikeys=state['seen_inchikeys'],
            component_weights=component_weights
        )
    
    # Convert to molecule list and validate
    validation_stats = {
        'total_generated': 0,
        'passed_validation': 0,
        'failed_validation': 0,
    }
    
    for _, row in data.iterrows():
        molecule_name = row['name']
        smiles = row.get('smiles')
        inchikey = row.get('InChIKey')
        
        validation_stats['total_generated'] += 1
        
        # Check if already generated
        if molecule_name in generated_molecules:
            continue
        
        if inchikey and inchikey in generated_inchikeys:
            continue
        
        # Validate molecule (already validated in batch, but double-check)
        is_valid, errors = await validate_molecule_complete(state, molecule_name, smiles, config)
        
        if not is_valid:
            validation_stats['failed_validation'] += 1
            continue
        
        # Add to unique molecules
        new_molecules.append({
            'name': molecule_name,
            'smiles': smiles,
            'InChIKey': inchikey,
            'type': 'adaptive'
        })
        
        generated_molecules.add(molecule_name)
        if inchikey:
            generated_inchikeys.add(inchikey)
        validation_stats['passed_validation'] += 1
        
        if len(new_molecules) >= desired_count:
            break
    
    # Update state
    state['generated_molecules'] = generated_molecules
    state['generated_inchikeys'] = generated_inchikeys
    
    # Update adaptive parameters based on duplication rate
    if validation_stats['total_generated'] > 0:
        dup_ratio = 1.0 - (validation_stats['passed_validation'] / validation_stats['total_generated'])
        
        if dup_ratio > 0.7:
            mutation_prob = min(0.9, mutation_prob * 1.5)
            elite_frac = max(0.15, elite_frac * 0.7)
            print(f"   ⚠️  High duplication ({dup_ratio:.2%}), adjusting: mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
        elif dup_ratio > 0.5:
            mutation_prob = min(0.7, mutation_prob * 1.3)
            elite_frac = max(0.2, elite_frac * 0.8)
        elif dup_ratio < 0.15 and iteration > 10:
            mutation_prob = max(0.05, mutation_prob * 0.95)
            elite_frac = min(0.85, elite_frac * 1.05)
    
    state['mutation_prob'] = mutation_prob
    state['elite_frac'] = elite_frac
    
    # Update no_improvement_counter
    if score_improvement_rate == 0.0:
        state['no_improvement_counter'] = no_improvement_counter + 1
    else:
        state['no_improvement_counter'] = 0
    
    print(
        f"✅ Generated {len(new_molecules)} valid molecules"
        f"\n   Stats: {validation_stats['total_generated']} generated, "
        f"{validation_stats['passed_validation']} passed, "
        f"{validation_stats['failed_validation']} failed"
    )
    
    return new_molecules


async def run_continuous_genetic_loop(state: Dict[str, Any]) -> None:
    """
    Continuous genetic algorithm loop with ADAPTIVE strategy:
    1. Generate molecules using adaptive strategy (synthon + GA)
    2. Score them in batches
    3. Update top pool
    4. Repeat infinitely
    """
    print("🚀 Starting CONTINUOUS genetic algorithm loop with ADAPTIVE strategy...")
    
    desired_unique_count = 100
    batch_size = 10
    round_number = 0
    
    # Initialize adaptive parameters
    state['iteration'] = 0
    state['prev_avg_score'] = None
    state['score_improvement_rate'] = 0.0
    state['no_improvement_counter'] = 0
    state['mutation_prob'] = 0.5
    state['elite_frac'] = 0.4
    
    while not state['shutdown_event'].is_set():
        try:
            round_number += 1
            state['iteration'] = round_number
            print(f"\n{'='*70}")
            print(f"🔄 Generation Round {round_number}")
            print(f"{'='*70}")
            
            # Use existing top_200_df from state
            top_200_df = state.get('top_200_df', pd.DataFrame())
            
            if top_200_df.empty:
                print("No top 200 molecules found, waiting...")
                await asyncio.sleep(10)
                continue
            
            # Generate unique molecules with ADAPTIVE strategy
            print(f"🧬 Generating {desired_unique_count} molecules with ADAPTIVE strategy...")
            unique_molecules = await generate_unique_molecules_adaptive(
                state, top_200_df, desired_unique_count
            )
            
            if not unique_molecules:
                print("Failed to generate unique molecules, retrying in 10 seconds...")
                await asyncio.sleep(10)
                continue
            
            print(f"✅ Generated {len(unique_molecules)} valid unique molecules")
            
            # Score molecules in batches
            print(f"🔬 Round {round_number}: Scoring {len(unique_molecules)} molecules in batches of {batch_size}...")
            
            scored_molecules = await score_molecules_with_boltz_batched(
                state, unique_molecules, batch_size=batch_size
            )
            
            # Filter molecules with valid scores
            molecules_with_scores = [m for m in scored_molecules if m.get('boltz_score') is not None]
            
            if molecules_with_scores:
                # Sort by score
                molecules_with_scores.sort(key=lambda m: m.get('boltz_score', float('-inf')), reverse=True)
                
                # Get best molecule
                best_molecule = molecules_with_scores[0]
                best_score = best_molecule.get('boltz_score')
                
                print(
                    f"\n{'='*70}"
                    f"\n🏆 Round {round_number} Best Molecule:"
                    f"\n   Name: {best_molecule['name']}"
                    f"\n   Score: {best_score:.6f}"
                    f"\n   SMILES: {best_molecule.get('smiles', 'N/A')}"
                    f"\n{'='*70}\n"
                )
                
                # Update state with best molecule
                state['best_molecule'] = best_molecule
                state['best_score'] = best_score
                
                # Update top_200_df with new scored molecules
                new_rows = []
                for mol in molecules_with_scores:
                    new_rows.append({
                        'name': mol['name'],
                        'smiles': mol['smiles'],
                        'InChIKey': mol['InChIKey'],
                        'score': mol['boltz_score']
                    })
                
                if new_rows:
                    new_df = pd.DataFrame(new_rows)
                    combined_df = pd.concat([state['top_200_df'], new_df], ignore_index=True)
                    
                    # Deduplicate by InChIKey, keeping highest score
                    combined_df = combined_df.sort_values(by='score', ascending=False, na_position='last')
                    combined_df = combined_df.drop_duplicates(subset=['InChIKey'], keep='first')
                    
                    # Keep top 200
                    state['top_200_df'] = combined_df.head(200)
                    
                    print(f"✅ Updated top_200_df: {len(state['top_200_df'])} molecules")
            else:
                print("⚠️  No molecules with valid scores in this round")
            
            print(f"✅ Round {round_number} complete. Starting next round...")
            
            # Small delay before next round
            await asyncio.sleep(1)
        
        except Exception as e:
            print(f"Error in continuous GA loop: {e}")
            import traceback
            print(traceback.format_exc())
            await asyncio.sleep(10)


async def run_generator() -> None:
    """Main generation loop."""
    print("🚀 Starting molecule generator...")
    
    # Load config from config.yaml
    config_dict = load_config()
    
    # Get target protein from config
    target_protein = config_dict.get('weekly_target', 'KRAS')
    
    print(f"Configuration loaded from config.yaml:")
    print(f"  Target protein: {target_protein}")
    print(f"  Reaction ID: {HARDCODED_RXN_ID}")
    print(f"  Database: {DB_PATH}")
    print(f"  Score DB: {SCORE_RESULTS_DB}")
    
    state: Dict[str, Any] = {
        'config': config_dict,
        'startup_complete': False,
        'shutdown_event': asyncio.Event(),
        'current_challenge_targets': [target_protein],
        'current_challenge_antitargets': [],
        'rxn_id': HARDCODED_RXN_ID,
        'top_pool': pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"]),
        'seen_inchikeys': set(),
        'generated_molecules': set(),
        'generated_inchikeys': set(),
        'boltz_wrapper': None,
        'top_200_df': pd.DataFrame(),
        'best_molecule': None,
        'best_score': float('-inf'),
        'synthon_lib': None,
        'synthon_lib_ready': False,
    }
    
    print("🚀 Entering main generator loop...")
    
    # Run startup phase
    await startup_phase(state)
    
    # Run continuous genetic loop with adaptive strategy
    await run_continuous_genetic_loop(state)


def main():
    """Main entry point."""
    try:
        asyncio.run(run_generator())
    except KeyboardInterrupt:
        print("\n🛑 Generator interrupted by user")
    except Exception as e:
        print(f"Fatal error in generator: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
    