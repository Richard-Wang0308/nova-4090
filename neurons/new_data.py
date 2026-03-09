"""
mini_data.py - DPEX_DJA + Boltz-2 Integration
==============================================

Main competition loop with full DPEX_DJA metaheuristic.
Replaces blind crossover with intelligent hybrid search.

KEY FIXES:
- Mark molecules as generated AFTER successful scoring (not before)
- Proper filtering order in CandidateQualityFilter
- Correct AdaptiveBudgetController initialization
"""

import os
import sys
import random
import logging
import asyncio
import time
import sqlite3
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
HARDCODED_RXN_ID = 5
STARTING_EPOCH = 21213
REACTION_TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'mols.csv')
SCORE_RESULTS_DB = os.path.join(BASE_DIR, "score_results.sqlite")

from config.config_loader import load_config
from utils import (
    get_smiles,
    get_heavy_atom_count,
    molecule_unique_for_protein_hf,
    contains_atom_type
)
from molecules_base import generate_inchikey
from combinatorial_db.reactions import get_smiles_from_reaction

# Import DPEX_DJA components
from dpex_dja_boltz_core import (
    DPEXDJABoltzState,
    CandidateQualityFilter,
    AdaptiveBudgetController,
    DJAGenerator,
    TabuGenerator,
    ExploitModeGenerator,
    initialize_dpex_config,
    get_dpex_config,
)

BOLTZ_AVAILABLE = False
BoltzWrapper = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ════════════════════════════════════════════════════════════════

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
        min_atoms = 21
        max_atoms = 26

        if heavy_atom_count < min_atoms:
            return False, f"Insufficient heavy atoms: {heavy_atom_count} < {min_atoms}"
        if heavy_atom_count > max_atoms:
            return False, f"Too many heavy atoms: {heavy_atom_count} > {max_atoms}"
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
    
    # 1. SMILES validity
    is_valid, error_msg = validate_molecule_smiles(molecule_name, smiles)
    if not is_valid:
        errors.append(f"[SMILES] {error_msg}")
        return False, errors
    
    # 2. Heavy atom count
    is_valid, error_msg = validate_molecule_heavy_atoms(molecule_name, smiles, config)
    if not is_valid:
        errors.append(f"[HEAVY_ATOMS] {error_msg}")
    
    # 3. Banned atoms
    is_valid, error_msg = validate_molecule_banned_atoms(molecule_name, smiles, config)
    if not is_valid:
        errors.append(f"[BANNED_ATOMS] {error_msg}")
    
    # 4. Rotatable bonds
    is_valid, error_msg = validate_molecule_rotatable_bonds(molecule_name, smiles, config)
    if not is_valid:
        errors.append(f"[ROTATABLE_BONDS] {error_msg}")
    
    # 5. HuggingFace uniqueness
    is_valid, error_msg = await validate_molecule_huggingface_unique(state, molecule_name, smiles)
    if not is_valid:
        errors.append(f"[HF_UNIQUE] {error_msg}")
    
    return len(errors) == 0, errors


# ════════════════════════════════════════════════════════════════
# DATABASE FUNCTIONS
# ════════════════════════════════════════════════════════════════

def init_score_results_db(db_path: str = None) -> None:
    """Initialize/create the score_results.sqlite database."""
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
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Score results database initialized at {db_path}")
    except Exception as e:
        logger.error(f"❌ Error initializing score_results database: {e}")


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
        logger.debug(f"Error getting score from DB for {molecule_name}: {e}")
        return None


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
        logger.debug(f"Error batch getting scores from DB: {e}")
        return {}


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
            logger.info(f"✅ Wrote {len(to_insert)} scored molecules to database")
        
        conn.close()
    except Exception as e:
        logger.error(f"❌ Error writing scores to database: {e}")


# ════════════════════════════════════════════════════════════════
# MOLECULE LOADING FUNCTIONS
# ════════════════════════════════════════════════════════════════

def load_molecules_from_db_with_validation(
    db_path: str,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from SQLite database with validation."""
    if config is None:
        config = {}
    
    if not os.path.exists(db_path):
        logger.warning(f"Database file not found at {db_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    try:
        logger.info(f"Loading molecules from database {db_path} for rxn_id={rxn_id}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT molecule_name, score FROM scored_molecules")
        db_results = cursor.fetchall()
        conn.close()
        
        if not db_results:
            logger.info("No molecules found in database")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        result_rows = []
        successful_count = 0
        failed_count = 0
        
        for molecule_name, score in db_results:
            try:
                if not molecule_name.startswith(f"rxn:{rxn_id}:"):
                    continue
                
                smiles = get_smiles_from_reaction(molecule_name)
                if not smiles:
                    failed_count += 1
                    continue
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed_count += 1
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
                logger.debug(f"Could not process {molecule_name}: {e}")
                failed_count += 1
                continue
        
        result_df = pd.DataFrame(result_rows)
        if not result_df.empty:
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')
            result_df = result_df.sort_values(by='score', ascending=False, na_position='last')
            logger.info(f"✅ Loaded {len(result_df)} molecules (successful: {successful_count}, failed: {failed_count})")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error loading molecules from database: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_from_csv_with_validation(
    csv_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from CSV file with validation."""
    if config is None:
        config = {}
    
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found at {csv_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    try:
        logger.info(f"Loading molecules from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if 'epoch' in df.columns:
            df = df[df['epoch'] >= starting_epoch]
        
        if 'molecule_name' in df.columns:
            df = df[df['molecule_name'].str.startswith(f"rxn:{rxn_id}:", na=False)]
        
        if df.empty:
            logger.info("No matching molecules found in CSV")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        result_rows = []
        successful_count = 0
        failed_count = 0
        
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
                logger.debug(f"Could not process {molecule_name}: {e}")
                failed_count += 1
                continue
        
        result_df = pd.DataFrame(result_rows)
        if not result_df.empty:
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')
            result_df = result_df.sort_values(by='score', ascending=False, na_position='last')
            logger.info(f"✅ Loaded {len(result_df)} molecules from CSV (successful: {successful_count}, failed: {failed_count})")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error loading molecules from CSV: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])


def load_molecules_combined(
    csv_path: str,
    db_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from both CSV and database, merge and deduplicate."""
    if config is None:
        config = {}
    
    logger.info(f"🔄 Loading molecules from CSV and database...")
    
    csv_df = load_molecules_from_csv_with_validation(
        csv_path, target_proteins, starting_epoch, rxn_id, config
    )
    
    db_df = load_molecules_from_db_with_validation(db_path, rxn_id, config)
    
    if csv_df.empty and db_df.empty:
        logger.warning("No molecules loaded from either CSV or database")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    if csv_df.empty:
        logger.info("No molecules from CSV, using database only")
        return db_df
    
    if db_df.empty:
        logger.info("No molecules from database, using CSV only")
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
    
    logger.info(
        f"✅ Combined loading complete: "
        f"{csv_count} from CSV, {db_count} from database, "
        f"{combined_count} unique molecules after deduplication "
        f"({duplicates_removed} duplicates removed)"
    )
    
    return combined_df


# ════════════════════════════════════════════════════════════════
# SCORING FUNCTION
# ════════════════════════════════════════════════════════════════

async def score_molecules_with_boltz_batched(
    state: Dict[str, Any],
    molecules: List[Dict[str, Any]],
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """Score molecules using BoltzWrapper in batches."""
    if state.get('boltz_wrapper') is None:
        logger.warning("BoltzWrapper not available, skipping scoring")
        return molecules
    
    if not molecules:
        return molecules
    
    logger.info(f"🔬 Processing {len(molecules)} molecules for scoring in batches of {batch_size}...")
    
    init_score_results_db()
    
    all_scored_molecules = []
    total_batches = (len(molecules) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(molecules))
        batch = molecules[start_idx:end_idx]
        
        logger.info(
            f"📦 Batch {batch_idx + 1}/{total_batches}: "
            f"Scoring {len(batch)} molecules"
        )
        
        molecules_to_score = []
        molecules_with_db_scores = []
        molecules_in_hf = []
        
        target_proteins = state.get('current_challenge_targets', [])
        primary_target = target_proteins[0] if target_proteins else None
        
        molecule_names = [mol['name'] for mol in batch]
        db_scores = batch_get_scores_from_db(molecule_names)
        
        logger.info(f"   Found {len(db_scores)} molecules already in database")
        
        for mol in batch:
            molecule_name = mol['name']
            smiles = mol.get('smiles')
            
            if molecule_name in db_scores:
                mol['boltz_score'] = db_scores[molecule_name]
                mol['boltz_score_source'] = 'database'
                molecules_with_db_scores.append(mol)
                logger.debug(f"   ✓ {molecule_name}: score from DB = {db_scores[molecule_name]:.6f}")
                continue
            
            if primary_target and smiles:
                try:
                    is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
                    if not is_unique_hf:
                        logger.debug(f"   ⏭️  {molecule_name}: already in HuggingFace, skipping")
                        molecules_in_hf.append(mol)
                        continue
                except Exception as e:
                    logger.debug(f"   Error checking HuggingFace for {molecule_name}: {e}")
            
            molecules_to_score.append(mol)
        
        logger.info(
            f"   Breakdown: {len(molecules_with_db_scores)} from DB, "
            f"{len(molecules_in_hf)} in HuggingFace (skipped), "
            f"{len(molecules_to_score)} need scoring"
        )
        
        newly_scored_molecules = []
        if molecules_to_score:
            logger.info(f"   Scoring {len(molecules_to_score)} new molecules with Boltz...")
            
            boltz = state['boltz_wrapper']
            config = state['config']
            target_proteins = state.get('current_challenge_targets', [])
            antitarget_proteins = state.get('current_challenge_antitargets', [])
            
            if not target_proteins:
                logger.warning("No target proteins available for scoring")
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
                            logger.debug(f"Cleaned up old lightning_logs directory")
                    except Exception as cleanup_err:
                        logger.debug(f"Could not clean up old logs: {cleanup_err}")
                
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
                    'binding_pocket': getattr(config, 'binding_pocket', None),
                    'max_distance': getattr(config, 'max_distance', None),
                    'force': getattr(config, 'force', False),
                    'num_molecules_boltz': num_molecules_to_score,
                    'boltz_metric': getattr(config, 'boltz_metric', ['affinity_probability_binary', 'affinity_pred_value']),
                    'combination_strategy': getattr(config, 'combination_strategy', 'heavy_atom_normalization'),
                    'sample_selection': getattr(config, 'sample_selection', 'first'),
                }
                
                final_block_hash = "0x" + "0" * 64
                
                logger.info(f"   Running Boltz scoring for {len(molecules_to_score)} molecules...")
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
                logger.info(f"   ✅ Boltz scoring completed in {elapsed:.2f} seconds")
                
                uid = 0
                smiles_to_score = {}
                if uid in boltz.per_molecule_metric:
                    smiles_to_score = boltz.per_molecule_metric[uid].copy()
                    logger.info(f"   ✅ Loaded {len(smiles_to_score)} unique SMILES scores")
                
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
                    for mol in newly_scored_molecules:
                        logger.debug(f"Molecule {mol['name']} scored {mol['boltz_score']}")
                    write_scores_to_db(newly_scored_molecules)
            
            except Exception as e:
                logger.error(f"❌ Error scoring batch with Boltz: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Combine results from this batch
        batch_results = molecules_with_db_scores + newly_scored_molecules
        
        for mol in molecules_in_hf:
            mol['boltz_score'] = None
            mol['boltz_score_source'] = 'huggingface_skipped'
            batch_results.append(mol)
        
        all_scored_molecules.extend(batch_results)
        
        logger.info(
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
    
    logger.info(f"✅ Batch scoring complete: {len(scored_molecules)} total molecules scored")
    
    return scored_molecules


# ════════════════════════════════════════════════════════════════
# BOLTZ WRAPPER IMPORT
# ════════════════════════════════════════════════════════════════

def _import_boltz_wrapper():
    """Import BoltzWrapper following DataGenerator pattern."""
    global BOLTZ_AVAILABLE, BoltzWrapper
    
    try:
        BOLTZ_SCORING_DIR = os.path.join(BASE_DIR, "boltz-scoring")
        BOLTZ_SRC_DIR = os.path.join(BOLTZ_SCORING_DIR, "boltz", "src")
        
        if not os.path.exists(BOLTZ_SCORING_DIR):
            logger.warning(f"⚠️  Boltz-scoring directory not found at {BOLTZ_SCORING_DIR}")
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
        logger.info(f"✅ BoltzWrapper imported successfully")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️  Failed to import BoltzWrapper: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Error setting up BoltzWrapper: {e}")
        return False


# ════════════════════════════════════════════════════════════════
# MAIN DPEX_DJA GENERATION AND SCORING LOOP
# ════════════════════════════════════════════════════════════════

async def run_generation_and_scoring_loop_dpex(state: Dict[str, Any]) -> None:
    """
    Main loop with full DPEX_DJA integration.
    
    Key features:
    1. Adaptive budget controller
    2. DJA global exploration
    3. Tabu local refinement
    4. Quality filtering before scoring
    5. Stagnation detection + exploit mode
    
    FIXED: Molecules marked as generated AFTER successful scoring
    """
    
    logger.info("🚀 Starting DPEX_DJA + Boltz-2 loop...")
    logger.info("Press Ctrl+C to stop")
    
    # Initialize DPEX state
    dpex_state = DPEXDJABoltzState()
    dpex_config = state.get('dpex_config')
    
    # Initialize budget controller with config
    budget_controller = AdaptiveBudgetController(dpex_config=dpex_config)
    
    # Initialize generators
    dja_gen = DJAGenerator(state.get('molecule_manager'))
    tabu_gen = TabuGenerator(state.get('molecule_manager'))
    exploit_gen = ExploitModeGenerator(state.get('molecule_manager'))
    
    round_number = 0
    wall_clock_start = time.time()
    wall_clock_limit = 3600  # 1 hour for demo
    
    try:
        while time.time() - wall_clock_start < wall_clock_limit:
            round_number += 1
            iteration_start = time.time()
            wall_clock_remaining = wall_clock_limit - (iteration_start - wall_clock_start)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔄 Round {round_number} [DPEX_DJA + Boltz-2]")
            logger.info(f"{'='*70}")
            
            # ── Step 1: Reload molecules ──────────────────────────────────
            logger.info("📂 Reloading molecules from CSV and database...")
            config = state['config']
            molecules_df = load_molecules_combined(
                REACTION_TRAIN_CSV,
                SCORE_RESULTS_DB,
                state['current_challenge_targets'],
                STARTING_EPOCH,
                HARDCODED_RXN_ID,
                config
            )
            
            if molecules_df.empty:
                logger.warning("No molecules loaded, waiting...")
                await asyncio.sleep(10)
                continue
            
            # Use top 700 as elite pool
            elite_df = molecules_df.head(700)
            dpex_state.top_pool = molecules_df.copy()
            state['top_pool'] = molecules_df.copy()
            
            logger.info(f"✅ Loaded {len(molecules_df)} molecules (elite pool: {len(elite_df)})")
            
            # ── Step 2: Compute adaptive budget ────────────────────────────
            improvement_rate = None
            if len(dpex_state.score_history) >= 2:
                recent_scores = list(dpex_state.score_history)[-2:]
                improvement_rate = (recent_scores[-1] - recent_scores[0]) / max(abs(recent_scores[0]), 1e-6)
            
            num_candidates, budget_info = budget_controller.get_candidate_budget(
                iteration=round_number,
                improvement_rate=improvement_rate,
                wall_clock_remaining=wall_clock_remaining
            )
            
            logger.info(
                f"💰 Budget allocation:"
                f"\n   Candidates to generate: {num_candidates}"
                f"\n   Reasoning: {budget_info['reasoning']}"
            )
            
            # ── Step 3: Compute DJA/Tabu split ────────────────────────────
            num_dja, num_tabu, split_info = budget_controller.get_dja_tabu_split(
                iteration=round_number,
                improvement_rate=improvement_rate
            )
            
            logger.info(
                f"🧬 Generation split:"
                f"\n   DJA (exploration): {num_dja} candidates"
                f"\n   Tabu (refinement): {num_tabu} candidates"
                f"\n   Reasoning: {split_info['reasoning']}"
            )
            
            # ── Step 4: Initialize populations if needed ───────────────────
            if not dpex_state.pop_A:
                logger.info("🔄 Initializing Population A (exploration) from elite molecules...")
                dpex_state.pop_A = elite_df.head(50).to_dict('records')
                for mol in dpex_state.pop_A:
                    mol['score'] = mol.get('score', 0.0)
            
            if not dpex_state.pop_B:
                logger.info("🔄 Initializing Population B (refinement) from elite molecules...")
                dpex_state.pop_B = elite_df.head(50).to_dict('records')
                for mol in dpex_state.pop_B:
                    mol['score'] = mol.get('score', 0.0)
            
            # ── Step 5: Generate DJA candidates ────────────────────────────
            logger.info(f"🧬 [DJA] Generating {num_dja} candidates via global exploration...")
            dja_candidates = dja_gen.generate_candidates(dpex_state.pop_A, num_dja, dpex_state)
            logger.info(f"✅ [DJA] Generated {len(dja_candidates)} valid candidates")
            
            # ── Step 6: Generate Tabu candidates ───────────────────────────
            logger.info(f"🧬 [Tabu] Generating {num_tabu} candidates via local refinement...")
            tabu_candidates = tabu_gen.generate_candidates(dpex_state.pop_B, num_tabu, dpex_state)
            logger.info(f"✅ [Tabu] Generated {len(tabu_candidates)} valid candidates")
            
            # ── Step 7: Generate Exploit candidates (if stagnating) ────────
            exploit_candidates = []
            if dpex_state.use_exploit_mode and dpex_state.best_molecule:
                logger.info(f"🎯 [Exploit] Activating deep search mode...")
                exploit_candidates = exploit_gen.generate_candidates(
                    dpex_state.best_molecule,
                    int(num_candidates * 0.2),  # 20% of budget
                    dpex_state
                )
                logger.info(f"✅ [Exploit] Generated {len(exploit_candidates)} candidates via deep search")
            
            # ── Step 8: Combine all candidates ─────────────────────────────
            all_raw_candidates = dja_candidates + tabu_candidates + exploit_candidates
            logger.info(f"📊 Total raw candidates: {len(all_raw_candidates)}")
            
            if not all_raw_candidates:
                logger.warning("No candidates generated, retrying...")
                await asyncio.sleep(5)
                continue
            
            # ── Step 9: Filter candidates by quality (BEFORE scoring) ─────
            logger.info("🔍 Filtering candidates by quality...")
            quality_filter = CandidateQualityFilter(elite_df)
            filtered_candidates, filter_stats = quality_filter.filter_candidates(
                all_raw_candidates,
                dpex_state,
                max_candidates=num_candidates
            )
            
            logger.info(
                f"✅ Filtering complete:"
                f"\n   Input: {filter_stats['input_count']}"
                f"\n   Output: {filter_stats['output_count']}"
                f"\n   Removed (duplicates): {filter_stats['removed_duplicates']}"
                f"\n   Removed (low diversity): {filter_stats['removed_low_diversity']}"
                f"\n   Removed (low similarity): {filter_stats['removed_low_similarity']}"
                f"\n   Removed (tabu): {filter_stats['removed_tabu']}"
            )
            
            if not filtered_candidates:
                logger.warning("No candidates passed filtering, retrying...")
                await asyncio.sleep(5)
                continue
            
            # ── Step 10: Score filtered candidates ──────────────────────────
            logger.info(f"🔬 Scoring {len(filtered_candidates)} filtered candidates with Boltz-2...")
            
            batch_size = 10
            total_batches = (len(filtered_candidates) + batch_size - 1) // batch_size
            all_scored = []
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(filtered_candidates))
                batch = filtered_candidates[start_idx:end_idx]
                
                logger.info(
                    f"   📦 Batch {batch_idx + 1}/{total_batches}: "
                    f"Scoring {len(batch)} molecules"
                )
                
                scored_batch = await score_molecules_with_boltz_batched(
                    state, batch, batch_size=len(batch)
                )
                
                # ── FIXED: Mark as generated AFTER successful scoring ──
                for mol in scored_batch:
                    if mol.get('boltz_score') is not None:
                        dpex_state.generated_molecules.add(mol['name'])
                        all_scored.append(mol)
            
            logger.info(f"✅ Scored {len(all_scored)} molecules")
            
            # ── Step 11: Update best and track improvement ──────────────────
            best_score_this_round = float('-inf')
            best_mol_this_round = None
            
            for mol in all_scored:
                score = mol.get('boltz_score')
                if score is not None and score > best_score_this_round:
                    best_score_this_round = score
                    best_mol_this_round = mol
            
            if best_mol_this_round:
                improved = dpex_state.update_best(best_mol_this_round, best_score_this_round)
                if improved:
                    logger.info(
                        f"🏆 NEW BEST: {best_mol_this_round['name']} "
                        f"(score: {best_score_this_round:.6f})"
                    )
            
            # ── Step 12: Detect stagnation and trigger exploit mode ────────
            is_stagnating = dpex_state.detect_stagnation()
            
            if is_stagnating:
                logger.warning(
                    f"⚠️  STAGNATION DETECTED (no improvement for {dpex_state.no_improvement_counter} iterations)"
                )
                dpex_state.use_exploit_mode = True
                logger.info("🎯 Activating EXPLOIT MODE for deep local search")
            else:
                dpex_state.use_exploit_mode = False
            
            # ── Step 13: Update populations ────────────────────────────────
            logger.info("🔄 Updating populations...")
            
            # Inject best-of-A into B
            if dpex_state.pop_A:
                best_A = max(dpex_state.pop_A, key=lambda x: x.get('score', float('-inf')))
                dpex_state.pop_B.append(best_A)
            
            # Refresh populations from top_pool
            top_50 = molecules_df.head(50).to_dict('records')
            for mol in top_50:
                mol['score'] = mol.get('score', 0.0)
            
            dpex_state.pop_A = top_50.copy()
            dpex_state.pop_B = top_50.copy()
            
            # ── Step 14: Summary ────────────────────────────────────────────
            iteration_time = time.time() - iteration_start
            logger.info(
                f"\n✅ Round {round_number} complete:"
                f"\n   Time: {iteration_time:.1f}s"
                f"\n   Candidates generated: {len(all_raw_candidates)}"
                f"\n   Candidates filtered: {len(filtered_candidates)}"
                f"\n   Candidates scored: {len(all_scored)}"
                f"\n   Best this round: {best_score_this_round:.6f if best_mol_this_round else 'N/A'}"
                f"\n   Global best: {dpex_state.best_score:.6f}"
                f"\n   Stagnation counter: {dpex_state.no_improvement_counter}"
                f"\n   Exploit mode: {'ACTIVE' if dpex_state.use_exploit_mode else 'inactive'}"
            )
            
            await asyncio.sleep(2)
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping...")
    
    finally:
        # Log final summary
        budget_controller.log_budget_summary()
        logger.info(
            f"\n🏁 Final Result:"
            f"\n   Best molecule: {dpex_state.best_molecule['name'] if dpex_state.best_molecule else 'None'}"
            f"\n   Best score: {dpex_state.best_score:.6f}"
            f"\n   Iterations: {round_number}"
            f"\n   Total candidates generated: {len(dpex_state.generated_molecules)}"
        )


# ════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════

async def main():
    """Main entry point with DPEX config integration."""
    logger.info("🚀 Starting mini_data.py - DPEX_DJA + Boltz-2 Integration")
    
    # ════════════════════════════════════════════════════════════════
    # STEP 1: Initialize DPEX Configuration from YAML
    # ════════════════════════════════════════════════════════════════
    
    logger.info("📋 Loading DPEX configuration from dpex_pool_config.yaml...")
    try:
        # Try to load config from default location
        dpex_config_path = os.path.join(
            os.path.dirname(__file__),
            'dpex_pool_config.yaml'
        )
        
        if not os.path.exists(dpex_config_path):
            logger.warning(f"Config file not found at {dpex_config_path}")
            logger.info("Checking alternative locations...")
            
            # Try common alternative locations
            alt_paths = [
                os.path.join(BASE_DIR, 'dpex_pool_config.yaml'),
                os.path.join(BASE_DIR, 'config', 'dpex_pool_config.yaml'),
                'dpex_pool_config.yaml',
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    dpex_config_path = alt_path
                    logger.info(f"Found config at {alt_path}")
                    break
            else:
                logger.warning("Config file not found in any location, using defaults")
        
        dpex_config = initialize_dpex_config(dpex_config_path)
        logger.info("✅ DPEX configuration loaded successfully")
        
        # Log pool configuration
        logger.info(
            f"📊 Pool Configuration:"
            f"\n   Population A size: {dpex_config.pool.pop_A_size}"
            f"\n   Population B size: {dpex_config.pool.pop_B_size}"
            f"\n   Elite pool size: {dpex_config.pool.elite_pool_size}"
            f"\n   Tabu A maxlen: {dpex_config.tabu.tabu_A_maxlen}"
            f"\n   Tabu B maxlen: {dpex_config.tabu.tabu_B_maxlen}"
            f"\n   Seed budget: {dpex_config.candidates.seed_budget}"
            f"\n   Early budget: {dpex_config.candidates.early_budget}"
            f"\n   Normal budget: {dpex_config.candidates.normal_budget}"
            f"\n   Stagnation budget: {dpex_config.candidates.stagnation_budget}"
        )
    
    except ImportError as e:
        logger.error(f"❌ Failed to import DPEX config loader: {e}")
        logger.warning("Using default DPEX configuration")
        dpex_config = None
    
    except Exception as e:
        logger.error(f"❌ Failed to load DPEX config: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("Using default DPEX configuration")
        dpex_config = None
    
    # ════════════════════════════════════════════════════════════════
    # STEP 2: Load Main Configuration
    # ════════════════════════════════════════════════════════════════
    
    logger.info("📋 Loading main configuration...")
    try:
        config = load_config()
        logger.info("✅ Main config loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load main config: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # ════════════════════════════════════════════════════════════════
    # STEP 3: Initialize State Dictionary
    # ════════════════════════════════════════════════════════════════
    
    logger.info("🔧 Initializing state dictionary...")
    
    state: Dict[str, Any] = {
        'config': config,
        'dpex_config': dpex_config,  # NEW: DPEX configuration
        'startup_complete': False,
        'current_challenge_targets': [],
        'current_challenge_antitargets': [],
        'rxn_id': HARDCODED_RXN_ID,
        'top_pool': pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"]),
        'seen_inchikeys': set(),
        'generated_molecules': set(),
        'boltz_wrapper': None,
        'molecule_manager': None,  # Will be initialized from database
    }
    
    logger.info("✅ State dictionary initialized")
    
    # ════════════════════════════════════════════════════════════════
    # STEP 4: Extract Target Proteins from Configuration
    # ════════════════════════════════════════════════════════════════
    
    logger.info("🎯 Extracting target proteins from configuration...")
    
    try:
        # Try different config formats
        if hasattr(config, 'weekly_target'):
            state['current_challenge_targets'] = [config.weekly_target]
            logger.info(f"   Found weekly_target (attribute): {config.weekly_target}")
        
        elif isinstance(config, dict) and 'weekly_target' in config:
            state['current_challenge_targets'] = [config['weekly_target']]
            logger.info(f"   Found weekly_target (dict): {config['weekly_target']}")
        
        else:
            logger.warning("No weekly_target found in config, using default")
            state['current_challenge_targets'] = ['P31652']
        
        # Try to get antitargets
        if hasattr(config, 'antitargets'):
            state['current_challenge_antitargets'] = config.antitargets
            logger.info(f"   Found antitargets: {len(config.antitargets)} proteins")
        
        elif isinstance(config, dict) and 'antitargets' in config:
            state['current_challenge_antitargets'] = config['antitargets']
            logger.info(f"   Found antitargets: {len(config['antitargets'])} proteins")
        
        else:
            logger.info("   No antitargets found in config")
            state['current_challenge_antitargets'] = []
    
    except Exception as e:
        logger.warning(f"Error extracting target proteins: {e}")
        logger.info("Using default target protein")
        state['current_challenge_targets'] = ['P31652']
        state['current_challenge_antitargets'] = []
    
    logger.info(f"✅ Target protein: {state['current_challenge_targets'][0]}")
    if state['current_challenge_antitargets']:
        logger.info(f"✅ Antitargets: {state['current_challenge_antitargets']}")
    
    # ════════════════════════════════════════════════════════════════
    # STEP 5: Initialize Score Results Database
    # ════════════════════════════════════════════════════════════════
    
    logger.info("💾 Initializing score_results database...")
    try:
        init_score_results_db()
        logger.info("✅ Score results database initialized")
    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        logger.warning("Continuing without database...")
    
    # ════════════════════════════════════════════════════════════════
    # STEP 6: Import BoltzWrapper
    # ════════════════════════════════════════════════════════════════
    
    logger.info("🔬 Importing BoltzWrapper...")
    try:
        boltz_imported = _import_boltz_wrapper()
        
        if boltz_imported:
            logger.info("✅ BoltzWrapper imported successfully")
        else:
            logger.warning("⚠️  BoltzWrapper import failed")
    
    except Exception as e:
        logger.error(f"❌ Error importing BoltzWrapper: {e}")
        boltz_imported = False
    
    # ════════════════════════════════════════════════════════════════
    # STEP 7: Initialize BoltzWrapper Instance
    # ════════════════════════════════════════════════════════════════
    
    if boltz_imported and BoltzWrapper is not None:
        logger.info("🔬 Initializing BoltzWrapper instance...")
        try:
            state['boltz_wrapper'] = BoltzWrapper()
            logger.info("✅ BoltzWrapper initialized successfully")
            
            # Log BoltzWrapper info
            logger.info(
                f"   Output directory: {state['boltz_wrapper'].output_dir}"
            )
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize BoltzWrapper: {e}")
            import traceback
            logger.error(traceback.format_exc())
            state['boltz_wrapper'] = None
            logger.warning("Continuing without BoltzWrapper (scoring disabled)")
    
    else:
        logger.warning("⚠️  BoltzWrapper not available, scoring will be skipped")
        state['boltz_wrapper'] = None
    
    # ════════════════════════════════════════════════════════════════
    # STEP 8: Log Startup Summary
    # ════════════════════════════════════════════════════════════════
    
    logger.info("\n" + "="*70)
    logger.info("🚀 STARTUP SUMMARY")
    logger.info("="*70)
    logger.info(
        f"Configuration Status:"
        f"\n   ✅ Main config: loaded"
        f"\n   {'✅' if dpex_config else '⚠️ '} DPEX config: {'loaded' if dpex_config else 'using defaults'}"
        f"\n   ✅ Score database: initialized"
        f"\n   {'✅' if state['boltz_wrapper'] else '⚠️ '} BoltzWrapper: {'ready' if state['boltz_wrapper'] else 'disabled'}"
        f"\n"
        f"Target Configuration:"
        f"\n   Target protein: {state['current_challenge_targets'][0]}"
        f"\n   Antitargets: {len(state['current_challenge_antitargets'])}"
        f"\n   Reaction ID: {state['rxn_id']}"
        f"\n"
        f"DPEX Configuration:"
        f"\n   Pop A size: {dpex_config.pool.pop_A_size if dpex_config else 'default'}"
        f"\n   Pop B size: {dpex_config.pool.pop_B_size if dpex_config else 'default'}"
        f"\n   Elite pool: {dpex_config.pool.elite_pool_size if dpex_config else 'default'}"
    )
    logger.info("="*70 + "\n")
    
    # ════════════════════════════════════════════════════════════════
    # STEP 9: Run Main Generation and Scoring Loop
    # ════════════════════════════════════════════════════════════════
    
    logger.info("🎬 Starting main generation and scoring loop...\n")
    
    try:
        await run_generation_and_scoring_loop_dpex(state)
    
    except KeyboardInterrupt:
        logger.info("\n" + "="*70)
        logger.info("🛑 Program stopped by user (Ctrl+C)")
        logger.info("="*70)
    
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("❌ Fatal error occurred")
        logger.error("="*70)
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # ── Cleanup ──────────────────────────────────────────────
        logger.info("\n" + "="*70)
        logger.info("🧹 Cleanup and Final Summary")
        logger.info("="*70)
        
        # Log final statistics
        if state.get('top_pool') is not None and not state['top_pool'].empty:
            logger.info(
                f"Final Statistics:"
                f"\n   Total molecules in pool: {len(state['top_pool'])}"
                f"\n   Best score: {state['top_pool']['score'].max() if 'score' in state['top_pool'].columns else 'N/A'}"
            )
        
        logger.info("✅ Program completed")
        logger.info("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())