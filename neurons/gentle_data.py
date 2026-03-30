import os
import sys
import random
import logging
import asyncio
import time
import sqlite3
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
HARDCODED_RXN_ID = 5
# STARTING_EPOCH = 21492
STARTING_EPOCH = 21771
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
from enhanced_generation_dpex import (
    EnhancedMoleculeGenerator,
    SynthonLibraryEnhanced,
    MoleculeCandidate
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
        min_atoms = 14
        max_atoms = 30

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


def load_components_from_db(role: str, rxn_id: int) -> List[Tuple[int, str, int]]:
    """
    Load component molecules from database.
    
    Args:
        role: 'A', 'B', or 'C'
        rxn_id: Reaction ID
    
    Returns:
        List of (mol_id, smiles, role_mask) tuples
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Map role to role_mask
        role_mask = {'A': 1, 'B': 2, 'C': 4}.get(role, 0)
        
        cursor.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?",
            (role_mask, role_mask)
        )
        results = cursor.fetchall()
        conn.close()
        
        logger.info(f"✅ Loaded {len(results)} {role} components from database")
        return results
    except Exception as e:
        logger.error(f"❌ Error loading {role} components: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def extract_component_ids(molecules: List[Tuple[int, str, int]]) -> List[int]:
    """Extract component IDs from molecule tuples."""
    return [mol[0] for mol in molecules]


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
        logger.info(f"✅ Initialized score_results database at {db_path}")
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
                "INSERT OR REPLACE INTO scored_molecules (molecule_name, score, available) VALUES (?, ?, ?)",
                to_insert
            )
            conn.commit()
            logger.info(f"✅ Wrote {len(to_insert)} scored molecules to database")
        
        conn.close()
    except Exception as e:
        logger.error(f"❌ Error writing scores to database: {e}")


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


def load_molecules_from_db_with_validation(
    db_path: str,
    rxn_id: int,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Load molecules from SQLite database with validation from config.yaml."""
    if config is None:
        config = {}
    
    if not os.path.exists(db_path):
        logger.warning(f"Database file not found at {db_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    try:
        logger.info(
            f"Loading molecules from database {db_path} for rxn_id={rxn_id}"
        )
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query all scored molecules
        cursor.execute("SELECT molecule_name, score FROM scored_molecules")
        db_results = cursor.fetchall()
        conn.close()
        
        if not db_results:
            logger.info("No molecules found in database")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        result_rows = []
        successful_count = 0
        failed_count = 0
        banned_atom_count = 0
        heavy_atom_count = 0
        wrong_rxn_id_count = 0
        
        for molecule_name, score in db_results:
            try:
                # Filter by rxn_id
                if not molecule_name.startswith(f"rxn:{rxn_id}:"):
                    wrong_rxn_id_count += 1
                    continue
                
                smiles = get_smiles_from_reaction(molecule_name)
                
                if not smiles:
                    logger.debug(f"No SMILES found for {molecule_name}")
                    failed_count += 1
                    continue
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.debug(f"Cannot parse SMILES for {molecule_name}")
                    failed_count += 1
                    continue
                
                # Check banned atoms
                banned_atoms = config.get('banned_atom_types', [])
                if banned_atoms and contains_atom_type(mol, banned_atoms):
                    logger.debug(f"Molecule {molecule_name} contains banned atoms {banned_atoms}, skipping")
                    banned_atom_count += 1
                    continue
                
                # Check heavy atom count
                min_heavy_atoms = 14
                max_heavy_atoms = 30
                heavy_atom_count_val = get_heavy_atom_count(smiles)
                if heavy_atom_count_val < min_heavy_atoms:
                    logger.debug(f"Molecule {molecule_name} has insufficient heavy atoms ({heavy_atom_count_val} < {min_heavy_atoms}), skipping")
                    heavy_atom_count += 1
                    continue
                if heavy_atom_count_val > max_heavy_atoms:
                    logger.debug(f"Molecule {molecule_name} has too many heavy atoms ({heavy_atom_count_val} > {max_heavy_atoms}), skipping")
                    heavy_atom_count += 1
                    continue
                
                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    logger.debug(f"Could not generate InChIKey for {molecule_name}")
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
            
            if 'score' in result_df.columns:
                result_df = result_df.sort_values(by='score', ascending=False, na_position='last')
                logger.info(
                    f"✅ Loaded {len(result_df)} molecules from database "
                    f"(successful: {successful_count}, failed: {failed_count}, "
                    f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count}, "
                    f"wrong rxn_id: {wrong_rxn_id_count})"
                )
                if len(result_df) > 0:
                    scores = result_df['score'].dropna()
                    if len(scores) > 0:
                        logger.info(
                            f"   Score range: {scores.min():.6f} to {scores.max():.6f} "
                            f"(top 3: {scores.head(3).tolist()})"
                        )
        else:
            logger.warning(
                f"No valid molecules loaded from database "
                f"(successful: {successful_count}, failed: {failed_count}, "
                f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count}, "
                f"wrong rxn_id: {wrong_rxn_id_count})"
            )
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error loading molecules from database: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
        logger.warning(f"CSV file not found at {csv_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    try:
        logger.info(
            f"Loading molecules from {csv_path} for targets {target_proteins}, "
            f"epoch >= {starting_epoch}, rxn_id={rxn_id}"
        )
        df = pd.read_csv(csv_path)
        
        if 'epoch' in df.columns:
            df = df[df['epoch'] >= starting_epoch]
        else:
            logger.warning("CSV file does not have 'epoch' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        if 'molecule_name' in df.columns:
            df = df[df['molecule_name'].str.startswith(f"rxn:{rxn_id}:", na=False)]
        else:
            logger.warning("CSV file does not have 'molecule_name' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
        
        if df.empty:
            logger.info("No matching molecules found in CSV")
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
                    logger.debug(f"No SMILES found for {molecule_name}")
                    failed_count += 1
                    continue
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.debug(f"Cannot parse SMILES for {molecule_name}")
                    failed_count += 1
                    continue
                
                # Check banned atoms
                banned_atoms = config.get('banned_atom_types', [])
                if banned_atoms and contains_atom_type(mol, banned_atoms):
                    logger.debug(f"Molecule {molecule_name} contains banned atoms {banned_atoms}, skipping")
                    banned_atom_count += 1
                    continue
                
                # Check heavy atom count
                # min_heavy_atoms = config.get('min_heavy_atoms', 10)
                min_heavy_atoms = 14    
                max_heavy_atoms = 30

                heavy_atom_count_val = get_heavy_atom_count(smiles)
                if heavy_atom_count_val < min_heavy_atoms:
                    logger.debug(f"Molecule {molecule_name} has insufficient heavy atoms ({heavy_atom_count_val} < {min_heavy_atoms}), skipping")
                    heavy_atom_count += 1
                    continue

                if heavy_atom_count_val > max_heavy_atoms:
                    logger.debug(f"Molecule {molecule_name} has too many heavy atoms ({heavy_atom_count_val} > {max_heavy_atoms}), skipping")
                    heavy_atom_count += 1
                    continue
                
                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    logger.debug(f"Could not generate InChIKey for {molecule_name}")
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
            
            if 'score' in result_df.columns:
                result_df = result_df.sort_values(by='score', ascending=False, na_position='last')
                logger.info(
                    f"✅ Loaded {len(result_df)} molecules from CSV "
                    f"(successful: {successful_count}, failed: {failed_count}, "
                    f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count})"
                )
                if len(result_df) > 0:
                    scores = result_df['score'].dropna()
                    if len(scores) > 0:
                        logger.info(
                            f"   Score range: {scores.min():.6f} to {scores.max():.6f} "
                            f"(top 3: {scores.head(3).tolist()})"
                        )
        else:
            logger.warning(
                f"No valid molecules loaded from CSV "
                f"(successful: {successful_count}, failed: {failed_count}, "
                f"banned atoms: {banned_atom_count}, insufficient heavy atoms: {heavy_atom_count})"
            )
        
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
    """
    Load molecules from both CSV and database, merge them, and deduplicate.
    When duplicates exist (by InChIKey), prefer the one with the higher score.
    """
    if config is None:
        config = {}
    
    logger.info(f"🔄 Loading molecules from CSV and database...")
    
    # Load from CSV
    csv_df = load_molecules_from_csv_with_validation(
        csv_path, target_proteins, starting_epoch, rxn_id, config
    )
    
    # Load from database
    db_df = load_molecules_from_db_with_validation(
        db_path, rxn_id, config
    )
    
    # Combine both dataframes
    if csv_df.empty and db_df.empty:
        logger.warning("No molecules loaded from either CSV or database")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    
    if csv_df.empty:
        logger.info("No molecules from CSV, using database only")
        return db_df
    
    if db_df.empty:
        logger.info("No molecules from database, using CSV only")
        return csv_df
    
    # Merge dataframes
    csv_df['source'] = 'csv'
    db_df['source'] = 'database'
    
    combined_df = csv_df
    
    # Deduplicate by InChIKey, keeping the one with the highest score
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
    
    if combined_count > 0:
        scores = combined_df['score'].dropna()
        if len(scores) > 0:
            logger.info(
                f"   Combined score range: {scores.min():.6f} to {scores.max():.6f} "
                f"(top 3: {scores.head(3).tolist()})"
            )
    
    return combined_df


async def score_molecules_with_boltz_batched(
    state: Dict[str, Any],
    molecules: List[Dict[str, Any]],
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Score molecules using BoltzWrapper in batches.
    
    Args:
        state: State dictionary
        molecules: List of molecules to score
        batch_size: Number of molecules per batch (default 10)
        
    Returns:
        List of scored molecules
    """
    if state.get('boltz_wrapper') is None:
        logger.warning("BoltzWrapper not available, skipping scoring")
        return molecules
    
    if not molecules:
        return molecules
    
    logger.info(f"🔬 Processing {len(molecules)} molecules for scoring in batches of {batch_size}...")
    
    init_score_results_db()
    
    all_scored_molecules = []
    total_batches = (len(molecules) + batch_size - 1) // batch_size
    
    # Process molecules in batches
    for batch_idx in range(total_batches):
        # Get batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(molecules))
        batch = molecules[start_idx:end_idx]
        
        logger.info(
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
        
        logger.info(f"   Found {len(db_scores)} molecules already in database")
        
        # Separate molecules by source
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


async def generate_unique_molecules_enhanced(
    state: Dict[str, Any],
    top_molecules_df: pd.DataFrame,
    desired_count: int = 1000,
    strategy: str = "hybrid"
) -> List[Dict[str, Any]]:
    """
    Generate unique molecules using enhanced DPEX-DJA strategy.
    
    Args:
        state: State dictionary with synthon library and component pool
        top_molecules_df: DataFrame of top molecules to use as seed
        desired_count: Number of molecules to generate
        strategy: "hybrid" (default), "dja", "tabu", "exploit", "crossover"
    
    Returns:
        List of generated molecules with SMILES and metadata
    """
    
    if top_molecules_df.empty:
        logger.warning("❌ Top molecules DataFrame is empty")
        return []
    
    # Initialize enhanced generator if not already done
    if 'enhanced_generator' not in state or state['enhanced_generator'] is None:
        logger.info("🧬 Initializing EnhancedMoleculeGenerator...")
        
        try:
            # Load component pool from database
            logger.info("📂 Loading component pools from database...")
            component_pool_A = load_components_from_db('A', HARDCODED_RXN_ID)
            component_pool_B = load_components_from_db('B', HARDCODED_RXN_ID)
            component_pool_C = load_components_from_db('C', HARDCODED_RXN_ID)
            
            if not component_pool_A or not component_pool_B:
                logger.error("❌ Failed to load component pools from database")
                logger.error(f"   A components: {len(component_pool_A)}, B components: {len(component_pool_B)}, C components: {len(component_pool_C)}")
                return []
            
            logger.info(f"✅ Loaded component pools: A={len(component_pool_A)}, B={len(component_pool_B)}, C={len(component_pool_C)}")
            
            # Initialize synthon library
            logger.info("🧪 Initializing SynthonLibrary...")
            synthon_lib = SynthonLibraryEnhanced(
                molecules_A=component_pool_A,
                molecules_B=component_pool_B,
                molecules_C=component_pool_C if component_pool_C else None
            )
            
            # Extract component IDs
            component_ids_A = extract_component_ids(component_pool_A)
            component_ids_B = extract_component_ids(component_pool_B)
            component_ids_C = extract_component_ids(component_pool_C) if component_pool_C else []
            
            logger.info(f"✅ Extracted component IDs: A={len(component_ids_A)}, B={len(component_ids_B)}, C={len(component_ids_C)}")
            
            # Initialize generator
            logger.info("🔧 Initializing EnhancedMoleculeGenerator...")
            generator = EnhancedMoleculeGenerator(
                rxn_id=HARDCODED_RXN_ID,
                synthon_library=synthon_lib,
                component_ids_A=component_ids_A,
                component_ids_B=component_ids_B,
                component_ids_C=component_ids_C
            )
            
            if generator is None:
                logger.error("❌ EnhancedMoleculeGenerator initialization returned None")
                return []
            
            state['enhanced_generator'] = generator
            state['synthon_library'] = synthon_lib
            state['component_ids_A'] = component_ids_A
            state['component_ids_B'] = component_ids_B
            state['component_ids_C'] = component_ids_C
            
            logger.info(f"✅ EnhancedMoleculeGenerator initialized successfully")
        
        except Exception as e:
            logger.error(f"❌ Error initializing EnhancedMoleculeGenerator: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    generator = state['enhanced_generator']
    
    if generator is None:
        logger.error("❌ Generator is still None after initialization")
        return []
    
    # Convert DataFrame to list of dicts for generation
    top_molecules = top_molecules_df.head(1000).to_dict('records')
    
    logger.info(
        f"🧬 Generating {desired_count} molecules using '{strategy}' strategy "
        f"from {len(top_molecules)} seed molecules..."
    )
    
    unique_molecules = []
    generated_names = state.get('generated_molecules', set())
    generated_inchikeys = state.get('generated_inchikeys', set())
    
    attempts = 0
    max_attempts = 1000
    batch_size = 50
    
    validation_stats = {
        'total_generated': 0,
        'passed_validation': 0,
        'failed_smiles': 0,
        'failed_heavy_atoms': 0,
        'failed_banned_atoms': 0,
        'failed_rotatable_bonds': 0,
        'failed_hf_unique': 0,
    }
    
    while len(unique_molecules) < desired_count and attempts < max_attempts:
        attempts += 1
        
        # Generate batch
        try:
            candidates = generator.generate_batch(
                top_molecules,
                strategy=strategy,
                batch_size=batch_size
            )
        except Exception as e:
            logger.error(f"❌ Error generating batch: {e}")
            import traceback
            logger.error(traceback.format_exc())
            break
        
        for candidate in candidates:
            if len(unique_molecules) >= desired_count:
                break
            
            # Skip if already generated
            if candidate.name in generated_names:
                continue
            
            # Get SMILES from reaction
            try:
                smiles = get_smiles_from_reaction(candidate.name)
                if not smiles:
                    validation_stats['failed_smiles'] += 1
                    continue
            except Exception as e:
                logger.debug(f"Error getting SMILES for {candidate.name}: {e}")
                validation_stats['failed_smiles'] += 1
                continue
            
            # Validate molecule
            is_valid, errors = await validate_molecule_complete(
                state, candidate.name, smiles, state['config']
            )
            
            if not is_valid:
                for error in errors:
                    if "[HEAVY_ATOMS]" in error:
                        validation_stats['failed_heavy_atoms'] += 1
                    elif "[BANNED_ATOMS]" in error:
                        validation_stats['failed_banned_atoms'] += 1
                    elif "[ROTATABLE_BONDS]" in error:
                        validation_stats['failed_rotatable_bonds'] += 1
                    elif "[HF_UNIQUE]" in error:
                        validation_stats['failed_hf_unique'] += 1
                continue
            
            # Generate InChIKey
            try:
                inchikey = generate_inchikey(smiles)
                if inchikey in generated_inchikeys:
                    continue
            except Exception as e:
                logger.debug(f"Error generating InChIKey: {e}")
                continue
            
            # Add to results
            unique_molecules.append({
                'name': candidate.name,
                'smiles': smiles,
                'InChIKey': inchikey,
                'type': candidate.generation_method
            })
            
            generated_names.add(candidate.name)
            generated_inchikeys.add(inchikey)
            validation_stats['passed_validation'] += 1
            validation_stats['total_generated'] += 1
        
        # Update stagnation tracking
        if top_molecules and 'score' in top_molecules[0]:
            generator.update_stagnation(top_molecules[0]['score'])
        
        await asyncio.sleep(0.01)
    
    state['generated_molecules'] = generated_names
    state['generated_inchikeys'] = generated_inchikeys
    
    # Log statistics
    gen_stats = generator.get_statistics()
    logger.info(
        f"✅ Generated {len(unique_molecules)} valid molecules "
        f"(attempts: {attempts}, strategy: {strategy})"
        f"\n   Validation: {validation_stats['passed_validation']}/{validation_stats['total_generated']}"
        f"\n   Generator stats:"
        f"\n   - Total generated: {gen_stats['total_generated']}"
        f"\n   - Pop A size: {gen_stats['pop_a_size']}"
        f"\n   - Pop B size: {gen_stats['pop_b_size']}"
        f"\n   - Stagnation counter: {gen_stats['stagnation_counter']}"
        f"\n   - Best score: {gen_stats['last_best_score']:.6f}"
    )
    
    return unique_molecules


async def run_generation_and_scoring_loop_enhanced(state: Dict[str, Any]) -> None:
    """
    Enhanced generation and scoring loop with DPEX-DJA strategy rotation.
    """
    logger.info("🚀 Starting enhanced generation and scoring loop...")
    logger.info("Press Ctrl+C to stop")
    
    desired_unique_count = 250
    batch_size = 10
    round_number = 0
    strategy_rotation = ["hybrid", "dja", "tabu", "exploit"]
    strategy_idx = 0
    
    try:
        while True:
            round_number += 1
            current_strategy = strategy_rotation[strategy_idx % len(strategy_rotation)]
            strategy_idx += 1
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔄 Round {round_number} (Strategy: {current_strategy})")
            logger.info(f"{'='*70}")
            
            # Reload molecules
            molecules_df = load_molecules_combined(
                REACTION_TRAIN_CSV,
                SCORE_RESULTS_DB,
                state['current_challenge_targets'],
                STARTING_EPOCH,
                HARDCODED_RXN_ID,
                state['config']
            )
            
            if molecules_df.empty:
                logger.warning("⚠️  No molecules loaded, waiting...")
                await asyncio.sleep(10)
                continue
            
            # top_molecules_df = molecules_df[30:500]
            top_molecules_df = molecules_df.head(100)
            state['top_pool'] = molecules_df.copy()
            state['seen_inchikeys'].update(molecules_df['InChIKey'].tolist())
            
            logger.info(f"✅ Loaded {len(molecules_df)} molecules (top 700 for generation)")
            
            # Generate with enhanced strategy
            unique_molecules = await generate_unique_molecules_enhanced(
                state,
                top_molecules_df,
                desired_count=desired_unique_count,
                strategy=current_strategy
            )
            
            if not unique_molecules:
                logger.warning("⚠️  Failed to generate molecules, waiting...")
                await asyncio.sleep(10)
                continue
            
            logger.info(f"✅ Generated {len(unique_molecules)} valid molecules")
            
            # Score molecules
            logger.info(f"🔬 Scoring {len(unique_molecules)} molecules...")
            scored_molecules = await score_molecules_with_boltz_batched(
                state, unique_molecules, batch_size=batch_size
            )
            
            # Update generator with scores
            if 'enhanced_generator' in state and state['enhanced_generator'] is not None:
                scored_dict = {
                    m['name']: m.get('boltz_score')
                    for m in scored_molecules
                    if m.get('boltz_score') is not None
                }
                
                # Convert to MoleculeCandidate objects
                candidates = [
                    MoleculeCandidate(
                        name=m['name'],
                        smiles=m.get('smiles', ''),
                        score=m.get('boltz_score'),
                        generation_method=m.get('type', 'unknown')
                    )
                    for m in scored_molecules
                    if m.get('boltz_score') is not None
                ]
                
                if candidates:
                    state['enhanced_generator'].dja_manager.update_populations(
                        candidates, scored_dict
                    )
                    logger.info(
                        f"✅ Updated populations with {len(candidates)} scored molecules"
                    )
            
            # Find best molecule in this round
            best_in_round = None
            best_score_in_round = float('-inf')
            for mol in scored_molecules:
                score = mol.get('boltz_score')
                if score is not None and score > best_score_in_round:
                    best_score_in_round = score
                    best_in_round = mol
            
            logger.info(
                f"\n✅ Round {round_number} complete:"
                f"\n   - Strategy: {current_strategy}"
                f"\n   - Generated: {len(unique_molecules)} molecules"
                f"\n   - Scored: {len(scored_molecules)} molecules"
                f"\n   - Best in round: {best_in_round['name'] if best_in_round else 'None'} "
                f"(score: {best_score_in_round:.6f})"
            )
            
            # Wait before next round
            await asyncio.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping enhanced generation loop...")
        raise


async def main():
    """Main entry point."""
    logger.info("🚀 Starting mini_data.py - DPEX-DJA Enhanced Generation and Scoring Tool")
    
    # Load config
    try:
        config = load_config()
        logger.info("✅ Config loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return
    
    # Initialize state
    state: Dict[str, Any] = {
        'config': config,
        'startup_complete': False,
        'current_challenge_targets': [],
        'current_challenge_antitargets': [],
        'rxn_id': HARDCODED_RXN_ID,
        'top_pool': pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"]),
        'seen_inchikeys': set(),
        'generated_molecules': set(),
        'generated_inchikeys': set(),
        'boltz_wrapper': None,
        'enhanced_generator': None,
        'synthon_library': None,
        'component_ids_A': [],
        'component_ids_B': [],
        'component_ids_C': [],
    }
    
    # Get target proteins from config
    if hasattr(config, 'weekly_target'):
        state['current_challenge_targets'] = [config.weekly_target]
    elif isinstance(config, dict) and 'weekly_target' in config:
        state['current_challenge_targets'] = [config['weekly_target']]
    else:
        logger.warning("No weekly_target found in config, using default")
        state['current_challenge_targets'] = ['P31652']
    
    logger.info(f"🎯 Target protein: {state['current_challenge_targets'][0]}")
    
    # Initialize score_results database
    logger.info("💾 Initializing score_results database...")
    init_score_results_db()
    logger.info(f"✅ Score results database initialized")
    
    # Import BoltzWrapper
    logger.info("🔬 Importing BoltzWrapper...")
    boltz_imported = _import_boltz_wrapper()
    
    # Initialize BoltzWrapper
    if boltz_imported and BoltzWrapper is not None:
        logger.info("🔬 Initializing BoltzWrapper...")
        try:
            state['boltz_wrapper'] = BoltzWrapper()
            logger.info("✅ BoltzWrapper initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize BoltzWrapper: {e}")
            import traceback
            logger.error(traceback.format_exc())
            state['boltz_wrapper'] = None
    else:
        logger.warning("⚠️  BoltzWrapper not available, scoring will be skipped")
        state['boltz_wrapper'] = None
    
    # Run the enhanced main loop
    try:
        await run_generation_and_scoring_loop_enhanced(state)
    except KeyboardInterrupt:
        logger.info("✅ Program stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())