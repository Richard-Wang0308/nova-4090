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
STARTING_EPOCH = 22050
REACTION_TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'rxn5.csv')
SCORE_RESULTS_DB = os.path.join(BASE_DIR, "score_results_5.sqlite")

from config.config_loader import load_config
from utils import (
    get_smiles,
    get_heavy_atom_count,
    molecule_unique_for_protein_hf,
    contains_atom_type
)
from molecules_base import generate_inchikey
from combinatorial_db.reactions import get_smiles_from_reaction

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
        # min_atoms = config.get('min_heavy_atoms', 10)
        min_atoms = 25
        max_atoms = 35

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


class GeneticAlgorithmOperator:
    """Performs genetic algorithm operations on molecules (CROSSOVER ONLY)."""
    
    def __init__(self, rxn_id: int, db_path: str):
        """Initialize GA operator."""
        self.rxn_id = rxn_id
        self.db_path = db_path
        self.generated_molecule_names: Set[str] = set()
    
    def crossover_molecules(self, mol_name_1: str, mol_name_2: str) -> Optional[str]:
        """Crossover two molecules by swapping random components."""
        try:
            parts1 = mol_name_1.split(':')
            parts2 = mol_name_2.split(':')
            
            # logger.debug(f"Crossover: {mol_name_1} x {mol_name_2}")
            
            if (parts1[0] != 'rxn' or parts2[0] != 'rxn'):
                logger.debug(f"Invalid format: must start with 'rxn'")
                return None
            
            if len(parts1) != len(parts2):
                logger.debug(f"Invalid format: different number of components")
                return None
            
            if len(parts1) not in [4, 5]:
                logger.debug(f"Invalid format: expected 4 or 5 parts, got {len(parts1)}")
                return None
            
            try:
                rxn_id_1 = int(parts1[1])
                rxn_id_2 = int(parts2[1])
                if rxn_id_1 != self.rxn_id or rxn_id_2 != self.rxn_id:
                    logger.debug(f"Wrong rxn_ids: {rxn_id_1}, {rxn_id_2}")
                    return None
            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing rxn_ids: {e}")
                return None
            
            num_components = len(parts1) - 2
            component_indices = list(range(2, 2 + num_components))
            swap_idx = random.choice(component_indices)
            
            offspring_parts = parts1.copy()
            offspring_parts[swap_idx] = parts2[swap_idx]
            offspring_name = ':'.join(offspring_parts)
            
            if offspring_name in self.generated_molecule_names:
                # logger.debug(f"⚠️  Offspring {offspring_name} already generated in this batch")
                return None
            
            try:
                offspring_smiles = get_smiles_from_reaction(offspring_name)
                if offspring_smiles:
                    mol = Chem.MolFromSmiles(offspring_smiles)
                    if mol is not None:
                        self.generated_molecule_names.add(offspring_name)
                        # logger.info(f"✅ Crossover successful: {mol_name_1} × {mol_name_2} → {offspring_name}")
                        return offspring_name
                    else:
                        logger.debug(f"Invalid SMILES from RDKit: {offspring_smiles}")
                else:
                    logger.debug(f"No SMILES generated for offspring")
            except Exception as e:
                logger.debug(f"Error validating crossover: {e}")
            
            return None
        
        except Exception as e:
            logger.debug(f"Error in crossover_molecules: {e}")
            return None
    
    def apply_genetic_operations(
        self,
        top_molecules: List[str],
        num_crossovers: int = 5
    ) -> List[Dict[str, Any]]:
        """Apply genetic operations (CROSSOVER ONLY) to top molecules."""
        new_molecules = []
        self.generated_molecule_names.clear()
        
        # logger.info(f"🧬 Applying CROSSOVER-ONLY genetic operations to top {len(top_molecules)} molecules...")
        
        crossovers_created = 0
        
        for i in range(num_crossovers):
            parent1 = random.choice(top_molecules)
            parent2 = random.choice(top_molecules)
            
            # logger.info(f"   Attempting crossover {i+1}/{num_crossovers}: {parent1} x {parent2}")
            
            if parent1 != parent2:
                offspring = self.crossover_molecules(parent1, parent2)
                
                if offspring:
                    try:
                        smiles = get_smiles_from_reaction(offspring)
                        inchikey = generate_inchikey(smiles)
                        
                        if smiles and inchikey:
                            new_molecules.append({
                                'name': offspring,
                                'smiles': smiles,
                                'InChIKey': inchikey,
                                'type': 'crossover'
                            })
                            crossovers_created += 1
                            # logger.info(f"   ✅ Crossover #{crossovers_created}: {offspring}")
                    
                    except Exception as e:
                        logger.debug(f"Error processing offspring: {e}")
        
        # logger.info(f"   Crossovers: {crossovers_created}/{num_crossovers} successful")
        return new_molecules


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
        print(f"Initialized score_results database at {db_path}")
    except Exception as e:
        print(f"Error initializing score_results database: {e}")


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
                # min_heavy_atoms = config.get('min_heavy_atoms', 10)
                min_heavy_atoms = 25
                max_heavy_atoms = 35
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
                min_heavy_atoms = 25
                max_heavy_atoms = 35
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
    # Add source column for tracking
    csv_df['source'] = 'csv'
    db_df['source'] = 'database'
    
    # Combine dataframes
    # combined_df = pd.concat([csv_df, db_df], ignore_index=True)
    combined_df = csv_df
    
    # Deduplicate by InChIKey, keeping the one with the highest score
    # If scores are equal, prefer database (more recent scoring)
    combined_df = combined_df.sort_values(
        by=['score', 'source'],
        ascending=[False, True],  # Higher score first, then 'database' before 'csv'
        na_position='last'
    )
    
    # Keep first occurrence (highest score, or database if scores equal)
    combined_df = combined_df.drop_duplicates(subset=['InChIKey'], keep='first')
    
    # Remove source column
    combined_df = combined_df.drop(columns=['source'])
    
    # Sort by score
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


async def generate_unique_molecules_from_top200(
    state: Dict[str, Any],
    top_200_df: pd.DataFrame,
    desired_count: int = 100
) -> List[Dict[str, Any]]:
    """Generate unique molecules using genetic algorithm from top 200 molecules."""
    if top_200_df.empty:
        logger.warning("Top 200 DataFrame is empty")
        return []
    
    ga_operator = GeneticAlgorithmOperator(HARDCODED_RXN_ID, DB_PATH)
    all_names = top_200_df['name'].tolist()
    
    pool_sizes = [30, 50, 100, 150, 200]
    current_pool_size_idx = 0
    current_pool_size = min(pool_sizes[current_pool_size_idx], len(all_names))
    
    logger.info(f"🧬 Generating {desired_count} unique molecules with validation (starting with top {current_pool_size})...")
    
    unique_molecules = []
    attempts = 0
    max_attempts = 500
    last_successful_attempt = 0
    
    generated_molecules = state.get('generated_molecules', set())
    generated_inchikeys = state.get('generated_inchikeys', set())
    
    validation_stats = {
        'total_generated': 0,
        'passed_validation': 0,
        'failed_smiles': 0,
        'failed_heavy_atoms': 0,
        'failed_banned_atoms': 0,
        'failed_rotatable_bonds': 0,
        'failed_hf_unique': 0,
        'failed_other': 0,
    }
    
    while len(unique_molecules) < desired_count and attempts < max_attempts:
        attempts += 1
        
        if attempts - last_successful_attempt >= 100 and current_pool_size_idx < len(pool_sizes) - 1:
            current_pool_size_idx += 1
            new_pool_size = min(pool_sizes[current_pool_size_idx], len(all_names))
            if new_pool_size > current_pool_size:
                current_pool_size = new_pool_size
                logger.info(f"📈 Increasing pool size to top {current_pool_size}")
                last_successful_attempt = attempts
        
        current_pool_names = all_names[:current_pool_size]
        new_molecules = ga_operator.apply_genetic_operations(current_pool_names, num_crossovers=10)
        
        for mol in new_molecules:
            if len(unique_molecules) >= desired_count:
                break
            
            molecule_name = mol['name']
            smiles = mol.get('smiles')
            
            validation_stats['total_generated'] += 1
            
            if molecule_name in [m['name'] for m in unique_molecules]:
                continue
            
            if molecule_name in generated_molecules:
                logger.debug(f"   ⏭️  Molecule {molecule_name} already generated")
                continue
            
            inchikey = None
            try:
                inchikey = generate_inchikey(smiles) if smiles else None
                if inchikey and inchikey in generated_inchikeys:
                    logger.debug(f"   ⏭️  Molecule {molecule_name} (InChIKey: {inchikey}) already generated")
                    continue
            except Exception as e:
                logger.debug(f"   Could not generate InChIKey for {molecule_name}: {e}")
            
            # Validate with config.yaml settings
            is_valid, errors = await validate_molecule_complete(state, molecule_name, smiles, state['config'])
            
            if not is_valid:
                for error in errors:
                    logger.debug(f"   ❌ {molecule_name}: {error}")
                    if "[SMILES]" in error:
                        validation_stats['failed_smiles'] += 1
                    elif "[HEAVY_ATOMS]" in error:
                        validation_stats['failed_heavy_atoms'] += 1
                    elif "[BANNED_ATOMS]" in error:
                        validation_stats['failed_banned_atoms'] += 1
                    elif "[ROTATABLE_BONDS]" in error:
                        validation_stats['failed_rotatable_bonds'] += 1
                    elif "[HF_UNIQUE]" in error:
                        validation_stats['failed_hf_unique'] += 1
                    else:
                        validation_stats['failed_other'] += 1
                continue
            
            unique_molecules.append(mol)
            generated_molecules.add(molecule_name)
            if inchikey:
                generated_inchikeys.add(inchikey)
            last_successful_attempt = attempts
            validation_stats['passed_validation'] += 1
            
            # logger.info(
            #     f"   ✅ Added valid molecule {molecule_name} "
            #     f"({len(unique_molecules)}/{desired_count})"
            # )
        
        if len(unique_molecules) >= desired_count:
            break
        
        await asyncio.sleep(0.1)
    
    state['generated_molecules'] = generated_molecules
    state['generated_inchikeys'] = generated_inchikeys
    
    logger.info(
        f"✅ Generated {len(unique_molecules)} valid molecules (attempts: {attempts})"
        f"\n   Validation stats:"
        f"\n   - Total generated: {validation_stats['total_generated']}"
        f"\n   - Passed validation: {validation_stats['passed_validation']}"
        f"\n   - Failed SMILES: {validation_stats['failed_smiles']}"
        f"\n   - Failed heavy atoms: {validation_stats['failed_heavy_atoms']}"
        f"\n   - Failed banned atoms: {validation_stats['failed_banned_atoms']}"
        f"\n   - Failed rotatable bonds: {validation_stats['failed_rotatable_bonds']}"
        f"\n   - Failed HF uniqueness: {validation_stats['failed_hf_unique']}"
        f"\n   - Failed other: {validation_stats['failed_other']}"
    )
    
    return unique_molecules


async def run_generation_and_scoring_loop(state: Dict[str, Any]) -> None:
    """
    Main loop that continuously generates and scores molecules until interrupted.
    """
    logger.info("🚀 Starting generation and scoring loop...")
    logger.info("Press Ctrl+C to stop")
    
    desired_unique_count = 1000
    batch_size = 10
    round_number = 0
    
    try:
        while True:
            round_number += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"🔄 Round {round_number}")
            logger.info(f"{'='*70}")
            
            # Reload molecules from CSV and database
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
                logger.warning("⚠️  No valid molecules loaded from CSV or database, waiting...")
                await asyncio.sleep(10)
                continue
            
            # Get top 200 molecules (already sorted by score)
            top_200_df = molecules_df.head(150)
            # top_200_df = molecules_df[50:500]
            # top_200_df = molecules_df[10:110]
            
            # Update state with new molecules
            state['top_pool'] = molecules_df.copy()
            state['seen_inchikeys'].update(molecules_df['InChIKey'].tolist())
            state['top_200_df'] = top_200_df
            
            logger.info(f"✅ Reloaded {len(molecules_df)} molecules from CSV and database (top 200: {len(top_200_df)})")
            
            # Generate unique molecules
            logger.info(f"🧬 Generating {desired_unique_count} unique molecules with validation...")
            unique_molecules = await generate_unique_molecules_from_top200(
                state, top_200_df, desired_unique_count
            )
            
            if not unique_molecules:
                logger.warning("Failed to generate unique molecules, waiting before retry...")
                await asyncio.sleep(10)
                continue
            
            logger.info(f"✅ Generated {len(unique_molecules)} valid unique molecules")
            
            # Score this batch of molecules in batches of 10
            total_batches = (len(unique_molecules) + batch_size - 1) // batch_size
            logger.info(f"🔬 Round {round_number}: Scoring {len(unique_molecules)} molecules in {total_batches} batches of {batch_size}...")
            
            all_scored_molecules = []
            best_molecule_so_far = None
            best_score_so_far = float('-inf')
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(unique_molecules))
                batch = unique_molecules[start_idx:end_idx]
                
                logger.info(
                    f"   📦 Round {round_number}, Batch {batch_idx + 1}/{total_batches}: "
                    f"Scoring {len(batch)} molecules"
                )
                
                scored_batch = await score_molecules_with_boltz_batched(
                    state, batch, batch_size=len(batch)
                )
                
                # Filter molecules with valid scores
                batch_with_scores = [m for m in scored_batch if m.get('boltz_score') is not None]
                all_scored_molecules.extend(batch_with_scores)
                
                # Update best molecule so far
                for mol in batch_with_scores:
                    score = mol.get('boltz_score')
                    if score is not None and score > best_score_so_far:
                        best_score_so_far = score
                        best_molecule_so_far = mol
                        source = mol.get('boltz_score_source', 'unknown')
                        logger.info(
                            f"   🏆 New best in round {round_number}, batch {batch_idx + 1}: "
                            f"{mol['name']} (score: {score:.6f}, source: {source})"
                        )
            
            # Summary for this round
            best_score_str = f"{best_score_so_far:.6f}" if best_molecule_so_far else 'N/A'
            logger.info(
                f"\n✅ Round {round_number} complete:"
                f"\n   - Generated: {len(unique_molecules)} molecules"
                f"\n   - Scored: {len(all_scored_molecules)} molecules"
                f"\n   - Best molecule: {best_molecule_so_far['name'] if best_molecule_so_far else 'None'}"
                f"\n   - Best score: {best_score_str}"
            )
            
            # Wait a bit before next round
            await asyncio.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping generation and scoring loop...")
        raise


async def main():
    """Main entry point."""
    logger.info("🚀 Starting mini_data.py - Generation and Scoring Tool")
    
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
        'top_200_df': pd.DataFrame(),
    }
    
    # Get target proteins from config
    if hasattr(config, 'weekly_target'):
        state['current_challenge_targets'] = [config.weekly_target]
    elif isinstance(config, dict) and 'weekly_target' in config:
        state['current_challenge_targets'] = [config['weekly_target']]
    else:
        logger.warning("No weekly_target found in config, using default")
        state['current_challenge_targets'] = ['P31652']  # Default target
    
    logger.info(f"Target protein: {state['current_challenge_targets'][0]}")
    
    # Initialize score_results database
    logger.info("💾 Initializing score_results database...")
    init_score_results_db()
    logger.info(f"✅ Score results database initialized")
    
    # Log validation config
    logger.info(
        f"✅ Loaded validation config:"
        f"\n   - min_heavy_atoms: {config.get('min_heavy_atoms', 10) if isinstance(config, dict) else getattr(config, 'min_heavy_atoms', 10)}"
        f"\n   - min_rotatable_bonds: {config.get('min_rotatable_bonds', 1) if isinstance(config, dict) else getattr(config, 'min_rotatable_bonds', 1)}"
        f"\n   - max_rotatable_bonds: {config.get('max_rotatable_bonds', 10) if isinstance(config, dict) else getattr(config, 'max_rotatable_bonds', 10)}"
        f"\n   - banned_atom_types: {config.get('banned_atom_types', []) if isinstance(config, dict) else getattr(config, 'banned_atom_types', [])}"
    )
    
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
    
    # Run the main loop
    try:
        await run_generation_and_scoring_loop(state)
    except KeyboardInterrupt:
        logger.info("✅ Program stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
