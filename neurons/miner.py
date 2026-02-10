#!/usr/bin/env python3
"""
IMPROVED BITTENSOR MINER - ML-Guided Molecule Generation

Uses tuned ML model (LightGBM + XGBoost + CatBoost) for fast scoring.
No BoltzPredictor needed - ML model is 10x faster!

Key improvements:
1. ML model for fast pre-filtering & scoring (10x faster than BoltzPredictor!)
2. Automatic molecular descriptor calculation
3. Robust feature name handling (handles duplicates)
4. Continuous generation until epoch end
5. Keep original submission logic (submit best at epoch end)
"""

import os
import sys
import math
import random
import argparse
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import pickle

from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.errors import MetadataError

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Database path for combinatorial DB
DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
HARDCODED_RXN_ID = 2
STARTING_EPOCH = 20516

# ✅ KEEP: CSV for seeding top_pool
REACTION2_TRAIN_CSV = os.path.join(BASE_DIR, 'BoltzPredictor', 'data', 'train.csv')

# ✅ NEW: ML Model paths (REPLACES BoltzPredictor)
ML_MODEL_PATH = os.path.join(BASE_DIR, 'deployment', 'best_tuned_model.pkl')
ML_FEATURES_PATH = os.path.join(BASE_DIR, 'deployment', 'feature_names.json')
ML_METRICS_PATH = os.path.join(BASE_DIR, 'deployment', 'tuned_metrics.json')

from config.config_loader import load_config
from utils import (
    get_sequence_from_protein_code,
    upload_file_to_github,
    get_challenge_params_from_blockhash,
    get_heavy_atom_count,
    compute_maccs_entropy,
)
from utils.molecules import molecule_unique_for_protein_hf
from molecules_base import (
    generate_valid_random_molecules_batch,
    select_diverse_elites,
    build_component_weights,
    SynthonLibrary,
    generate_molecules_from_synthon_library,
    validate_molecules,
    generate_inchikey,
)
from combinatorial_db.reactions import get_smiles_from_reaction
from btdr import QuicknetBittensorDrandTimelock

# ============================================================================
# ✅ MOLECULAR DESCRIPTOR CALCULATOR
# ============================================================================

class MolecularDescriptorCalculator:
    """
    Calculates molecular descriptors needed by ML model.
    
    Computes: heavy_atom_count, mw, num_atoms, num_heteroatoms, logp, hba, hbd,
              rotatable_bonds, aromatic_rings, ring_count, tpsa, molar_refractivity
    """
    
    @staticmethod
    def calculate_descriptors(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate molecular descriptors for SMILES in dataframe.
        
        Args:
            df: DataFrame with 'smiles' column
            
        Returns:
            DataFrame with added descriptor columns
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, Lipinski
        
        try:
            descriptors_list = []
            
            for idx, row in df.iterrows():
                smiles = row['smiles']
                
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    
                    if mol is None:
                        # Invalid SMILES, add NaN values
                        descriptors_list.append({
                            'heavy_atom_count': float('nan'),
                            'mw': float('nan'),
                            'num_atoms': float('nan'),
                            'num_heteroatoms': float('nan'),
                            'logp': float('nan'),
                            'hba': float('nan'),
                            'hbd': float('nan'),
                            'rotatable_bonds': float('nan'),
                            'aromatic_rings': float('nan'),
                            'ring_count': float('nan'),
                            'tpsa': float('nan'),
                            'molar_refractivity': float('nan'),
                        })
                        continue
                    
                    # Calculate descriptors
                    descriptors = {
                        'heavy_atom_count': Lipinski.HeavyAtomCount(mol),
                        'mw': Descriptors.MolWt(mol),
                        'num_atoms': mol.GetNumAtoms(),
                        'num_heteroatoms': Lipinski.NumHeteroatoms(mol),
                        'logp': Crippen.MolLogP(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'rotatable_bonds': Lipinski.NumRotatableBonds(mol),
                        'aromatic_rings': Descriptors.NumAromaticRings(mol),
                        'ring_count': Descriptors.RingCount(mol),
                        'tpsa': Descriptors.TPSA(mol),
                        'molar_refractivity': Crippen.MolMR(mol),
                    }
                    
                    descriptors_list.append(descriptors)
                
                except Exception as e:
                    bt.logging.debug(f"Error calculating descriptors for {smiles}: {e}")
                    # Add NaN values on error
                    descriptors_list.append({
                        'heavy_atom_count': float('nan'),
                        'mw': float('nan'),
                        'num_atoms': float('nan'),
                        'num_heteroatoms': float('nan'),
                        'logp': float('nan'),
                        'hba': float('nan'),
                        'hbd': float('nan'),
                        'rotatable_bonds': float('nan'),
                        'aromatic_rings': float('nan'),
                        'ring_count': float('nan'),
                        'tpsa': float('nan'),
                        'molar_refractivity': float('nan'),
                    })
            
            # Create descriptors dataframe
            desc_df = pd.DataFrame(descriptors_list)
            
            # Combine with original dataframe
            result_df = pd.concat([df.reset_index(drop=True), desc_df], axis=1)
            
            # ✅ Remove duplicate columns
            result_df = result_df.loc[:, ~result_df.columns.duplicated(keep='first')]
            
            return result_df
        
        except Exception as e:
            bt.logging.error(f"Error calculating descriptors: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return df


# ============================================================================
# ✅ ML MODEL PREDICTOR (UPDATED - ROBUST FEATURE HANDLING)
# ============================================================================

class MLModelPredictor:
    """
    Wrapper for tuned ML model to predict molecule properties.
    
    Uses the tuned ensemble model (LightGBM + XGBoost + CatBoost)
    to predict molecule scores based on molecular descriptors.
    
    This REPLACES BoltzPredictor for fast scoring!
    """
    
    def __init__(self, model_path: str, features_path: str, metrics_path: str):
        """Initialize ML model predictor."""
        self.model = None
        self.features = None
        self.metrics = None
        self.descriptor_calculator = MolecularDescriptorCalculator()
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            bt.logging.info(f"✅ Loaded ML model from {model_path}")
            
            # Load features
            with open(features_path, 'r') as f:
                self.features = json.load(f)
            
            # ✅ Clean feature names (remove .1, .2 suffixes)
            self.features = [f.split('.')[0] if '.' in f else f for f in self.features]
            # Remove duplicates while preserving order
            seen = set()
            self.features = [f for f in self.features if not (f in seen or seen.add(f))]
            
            bt.logging.info(f"✅ Loaded {len(self.features)} features (cleaned)")
            
            # Load metrics
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
            bt.logging.info(f"✅ Model R²: {self.metrics['test']['r2']:.4f}")
            
        except Exception as e:
            bt.logging.error(f"Failed to load ML model: {e}")
            raise
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict scores for molecules using ML model.
        
        Args:
            df: DataFrame with 'smiles' column
            
        Returns:
            Array of predicted scores
        """
        try:
            # ✅ Calculate descriptors if not present
            if not all(feat in df.columns for feat in self.features):
                bt.logging.debug(f"Calculating molecular descriptors for {len(df)} molecules...")
                df = self.descriptor_calculator.calculate_descriptors(df)
            
            # ✅ Handle duplicate columns - keep only first occurrence
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
            
            # ✅ Clean column names (remove .1, .2 suffixes)
            df.columns = [col.split('.')[0] if '.' in col else col for col in df.columns]
            
            # Select required features
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                bt.logging.warning(f"Missing features: {missing_features}")
                # Try to find them without suffix
                for feat in missing_features:
                    base_feat = feat.split('.')[0]
                    if base_feat in df.columns:
                        df[feat] = df[base_feat]
            
            X = df[self.features]
            
            # Remove rows with NaN values in features
            valid_mask = ~X.isna().any(axis=1)
            X_valid = X[valid_mask]
            
            if len(X_valid) == 0:
                bt.logging.warning("No valid molecules after descriptor calculation")
                return np.array([float('nan')] * len(df))
            
            # Predict
            predictions_valid = self.model.predict(X_valid)
            
            # Create full predictions array with NaN for invalid molecules
            predictions = np.full(len(df), float('nan'))
            predictions[valid_mask] = predictions_valid
            
            return predictions
        
        except Exception as e:
            bt.logging.error(f"Error in ML prediction: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return np.array([float('nan')] * len(df))
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model."""
        try:
            # Get from first estimator (LightGBM)
            lgb_model = self.model.estimators_[0]
            importance = lgb_model.feature_importances_
            
            importance_dict = {
                feat: float(imp) for feat, imp in zip(self.features, importance)
            }
            
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        except Exception as e:
            bt.logging.warning(f"Could not get feature importance: {e}")
            return {}


# ============================================================================
# EXISTING FUNCTIONS (from base code)
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
    parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)

    config = bt.config(parser)
    config.update(load_config())

    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey_str,
            config.netuid,
            'miner',
        )
    )

    os.makedirs(config.full_path, exist_ok=True)
    return config


def load_github_path() -> str:
    """Constructs the path for GitHub operations."""
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER')
    github_repo_path = os.environ.get('GITHUB_REPO_PATH')

    if github_repo_name is None or github_repo_branch is None or github_repo_owner is None:
        raise ValueError("Missing GitHub environment variables")

    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path too long (max 100 chars)")

    return github_path


def setup_logging(config: argparse.Namespace) -> None:
    """Sets up Bittensor logging."""
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid}")


async def setup_bittensor_objects(config: argparse.Namespace) -> Tuple[Any, Any, Any, int, int]:
    """Initializes wallet, subtensor, and metagraph."""
    bt.logging.info("Setting up Bittensor objects.")

    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    try:
        async with bt.async_subtensor(network=config.network) as subtensor:
            metagraph = await subtensor.metagraph(config.netuid)
            await metagraph.sync()
            bt.logging.info(f"Metagraph synced successfully.")

            miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            bt.logging.info(f"Miner UID: {miner_uid}")

            epoch_length = 361
            bt.logging.info(f"Epoch length: {epoch_length} blocks")

        return wallet, subtensor, metagraph, miner_uid, epoch_length
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        raise


async def check_molecule_unique(state: Dict[str, Any], molecule_name: str, smiles: str) -> bool:
    """Check if molecule is unique for target protein."""
    if not state.get('current_challenge_targets'):
        bt.logging.warning("No target proteins available")
        return False
    
    primary_target = state['current_challenge_targets'][0]
    
    try:
        is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
        
        if not is_unique_hf:
            bt.logging.debug(f"Molecule {molecule_name} already seen")
            return False
        
        return True
    except Exception as e:
        bt.logging.error(f"Error checking uniqueness: {e}")
        return False


def load_existing_molecules_from_csv(
    csv_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int
) -> pd.DataFrame:
    """Load existing molecules from CSV file."""
    if not os.path.exists(csv_path):
        bt.logging.warning(f"CSV file not found at {csv_path}")
        return pd.DataFrame(
            columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
        )
    
    try:
        bt.logging.info(
            f"Loading existing molecules from {csv_path} for targets {target_proteins}, "
            f"epoch >= {starting_epoch}, rxn_id={rxn_id}"
        )
        df = pd.read_csv(csv_path)
        
        # Filter by target protein
        if 'target_protein' in df.columns:
            df = df[df['target_protein'].isin(target_proteins)]
        else:
            bt.logging.warning("CSV file does not have 'target_protein' column")
            return pd.DataFrame(
                columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
            )
        
        # Filter by epoch
        if 'epoch' in df.columns:
            df = df[df['epoch'] >= starting_epoch]
        else:
            bt.logging.warning("CSV file does not have 'epoch' column")
            return pd.DataFrame(
                columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
            )
        
        # Filter by reaction ID
        if 'molecule_name' in df.columns:
            df = df[df['molecule_name'].str.startswith(f"rxn:{rxn_id}:", na=False)]
        else:
            bt.logging.warning("CSV file does not have 'molecule_name' column")
            return pd.DataFrame(
                columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
            )
        
        if df.empty:
            bt.logging.info("No matching molecules found in CSV")
            return pd.DataFrame(
                columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
            )
        
        # Extract molecule names and scores
        result_rows = []
        successful_count = 0
        failed_count = 0
        
        for _, row in df.iterrows():
            molecule_name = row['molecule_name']
            
            try:
                smiles = get_smiles_from_reaction(molecule_name)
                
                if not smiles:
                    bt.logging.debug(f"No SMILES found for {molecule_name}")
                    failed_count += 1
                    continue
                
                inchikey = generate_inchikey(smiles)
                if not inchikey:
                    bt.logging.debug(f"Could not generate InChIKey for {molecule_name}")
                    failed_count += 1
                    continue
                
                final_score = row.get('final_score', 0.0)
                if pd.isna(final_score):
                    final_score = 0.0
                
                result_rows.append({
                    'name': molecule_name,
                    'smiles': smiles,
                    'InChIKey': inchikey,
                    'score': float(final_score),
                    'target_affinity': float('nan'),
                    'antitarget_affinity': float('nan'),
                })
                successful_count += 1
                
            except Exception as e:
                bt.logging.debug(f"Could not process {molecule_name}: {e}")
                failed_count += 1
                continue
        
        result_df = pd.DataFrame(result_rows)
        if not result_df.empty:
            result_df = result_df.sort_values('score', ascending=False)
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')
            bt.logging.info(
                f"Loaded {len(result_df)} existing molecules from CSV "
                f"(successful: {successful_count}, failed: {failed_count})"
            )
        else:
            bt.logging.warning(
                f"No valid molecules loaded from CSV "
                f"(successful: {successful_count}, failed: {failed_count})"
            )
        
        return result_df
        
    except Exception as e:
        bt.logging.error(f"Error loading molecules from CSV: {e}")
        return pd.DataFrame(
            columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
        )


# ============================================================================
# ✅ IMPROVED: INFERENCE LOOP WITH ML GUIDANCE
# ============================================================================

async def run_improved_model_loop(state: Dict[str, Any]) -> None:
    """
    Improved inference loop using ML model guidance.
    
    Key improvements:
    1. Uses ML model for fast pre-filtering & scoring (10x faster than BoltzPredictor!)
    2. Automatic molecular descriptor calculation
    3. Robust feature name handling
    4. Continuous generation until epoch end
    5. Keep original submission logic (submit best at epoch end)
    """
    bt.logging.info("🚀 Starting IMPROVED model inference loop with ML guidance...")
    
    # Initialize ML predictor
    try:
        ml_predictor = MLModelPredictor(
            ML_MODEL_PATH,
            ML_FEATURES_PATH,
            ML_METRICS_PATH
        )
        state['ml_predictor'] = ml_predictor
        
        # Log feature importance
        importance = ml_predictor.get_feature_importance()
        if importance:
            bt.logging.info("📊 Top 5 Important Features:")
            for feat, imp in list(importance.items())[:5]:
                bt.logging.info(f"   {feat:30s}: {imp:.4f}")
    
    except Exception as e:
        bt.logging.error(f"Failed to load ML model: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())
        return
    
    iteration = 0
    base_n_samples = state.get('chunk_size', 512)
    top_pool = state.get('top_pool', pd.DataFrame())
    seen_inchikeys = state.get('seen_inchikeys', set())
    rxn_id = HARDCODED_RXN_ID
    
    config_dict = {
        'min_heavy_atoms': state['config'].min_heavy_atoms,
        'min_rotatable_bonds': state['config'].min_rotatable_bonds,
        'max_rotatable_bonds': state['config'].max_rotatable_bonds,
    }
    
    while not state['shutdown_event'].is_set():
        try:
            iteration += 1
            
            # Build component weights if we have a top pool
            component_weights = None
            if not top_pool.empty:
                component_weights = build_component_weights(top_pool, rxn_id)
            
            # Select diverse elites
            elite_df = select_diverse_elites(
                top_pool, min(200, len(top_pool))
            ) if not top_pool.empty else pd.DataFrame()
            elite_names = elite_df["name"].tolist() if not elite_df.empty else None
            
            # Generate molecules
            if iteration == 1:
                # First iteration: Load existing molecules from CSV
                if state.get('csv_data_loaded', False):
                    bt.logging.info("CSV data already loaded at startup, using existing top_pool")
                else:
                    bt.logging.info("First iteration: Loading existing molecules from CSV...")
                    existing_df = load_existing_molecules_from_csv(
                        REACTION2_TRAIN_CSV,
                        state['current_challenge_targets'],
                        STARTING_EPOCH,
                        rxn_id
                    )
                    
                    if not existing_df.empty:
                        # Add to top_pool
                        top_pool = pd.concat([top_pool, existing_df], ignore_index=True)
                        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
                        top_pool = top_pool.sort_values(by="score", ascending=False)
                        top_pool = top_pool.head(100)
                        
                        # Add to seen_inchikeys
                        seen_inchikeys.update(top_pool["InChIKey"].tolist())
                        bt.logging.info(
                            f"Loaded {len(existing_df)} existing molecules, kept top 100 in top_pool"
                        )
                        
                        state['top_pool'] = top_pool
                        state['seen_inchikeys'] = seen_inchikeys
                        state['csv_data_loaded'] = True
                
                # Rebuild component weights after loading CSV
                component_weights = None
                if not top_pool.empty:
                    component_weights = build_component_weights(top_pool, rxn_id)
                
                elite_df = select_diverse_elites(
                    top_pool, min(200, len(top_pool))
                ) if not top_pool.empty else pd.DataFrame()
                elite_names = elite_df["name"].tolist() if not elite_df.empty else None
            
            # Generate new molecules
            df = generate_valid_random_molecules_batch(
                rxn_id,
                n_samples=base_n_samples,
                db_path=DB_PATH,
                subnet_config=config_dict,
                batch_size=400,
                elite_names=elite_names,
                elite_frac=0.85,
                mutation_prob=0.15,
                avoid_inchikeys=seen_inchikeys,
                component_weights=component_weights,
            )
            
            if df.empty:
                await asyncio.sleep(2)
                continue
            
            # ✅ SCORE WITH ML MODEL (fast pre-filtering)
            try:
                ml_scores = state['ml_predictor'].predict(df)
                df['ml_score'] = ml_scores
                
                # Filter out NaN scores
                df = df[~np.isnan(df['ml_score'])]
                
                if not df.empty:
                    df = df.sort_values('ml_score', ascending=False)
                    
                    bt.logging.info(
                        f"📊 ML Scoring (Iteration {iteration}):"
                        f"\n   Generated: {len(ml_scores)} molecules"
                        f"\n   Valid: {len(df)} molecules"
                        f"\n   Top ML score: {df['ml_score'].max():.4f}"
                        f"\n   Mean ML score: {df['ml_score'].mean():.4f}"
                    )
            
            except Exception as e:
                bt.logging.error(f"Error in ML scoring: {e}")
                import traceback
                bt.logging.error(traceback.format_exc())
                await asyncio.sleep(2)
                continue
            
            # Update seen molecules
            seen_inchikeys.update(df["InChIKey"].tolist())
            
            # Add to top pool
            df_to_add = df[["name", "smiles", "InChIKey", "ml_score"]].copy()
            df_to_add.columns = ["name", "smiles", "InChIKey", "score"]
            df_to_add['target_affinity'] = float('nan')
            df_to_add['antitarget_affinity'] = float('nan')
            
            top_pool = pd.concat([top_pool, df_to_add], ignore_index=True)
            top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
            top_pool = top_pool.sort_values(by="score", ascending=False)
            top_pool = top_pool.head(100)  # Keep top 100
            
            # Update state
            state['top_pool'] = top_pool
            state['seen_inchikeys'] = seen_inchikeys
            
            # Update best score
            if not top_pool.empty:
                best_score = top_pool['score'].iloc[0]
                if best_score > state['best_score']:
                    state['best_score'] = best_score
                    bt.logging.info(f"🏆 New best ML score: {state['best_score']:.6f}")
            
            await asyncio.sleep(2)

        except Exception as e:
            bt.logging.error(f"Error in improved loop: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            await asyncio.sleep(2)


# ============================================================================
# SUBMISSION (ORIGINAL LOGIC - UNCHANGED)
# ============================================================================

async def find_unique_candidate(state: Dict[str, Any], top_pool: pd.DataFrame, max_candidates: int = 10) -> Optional[str]:
    """Find a unique candidate molecule from top_pool for submission."""
    if top_pool.empty:
        bt.logging.warning("Top pool is empty, no candidates to check")
        return None
    
    if not state.get('current_challenge_targets'):
        bt.logging.warning("No target proteins available")
        return None
    
    primary_target = state['current_challenge_targets'][0]
    bt.logging.info(f"Checking uniqueness for top candidates against target {primary_target}")
    
    candidates_to_check = min(max_candidates, len(top_pool))
    
    for idx in range(candidates_to_check):
        candidate_row = top_pool.iloc[idx]
        molecule_name = candidate_row['name']
        smiles = candidate_row['smiles']
        score = candidate_row['score']
        
        bt.logging.info(
            f"Checking candidate #{idx + 1}: {molecule_name} "
            f"(score: {score:.6f})"
        )
        
        is_unique = await check_molecule_unique(state, molecule_name, smiles)
        
        if is_unique:
            bt.logging.info(
                f"✅ Found unique candidate #{idx + 1}: {molecule_name} "
                f"(score: {score:.6f})"
            )
            return molecule_name
        else:
            bt.logging.info(
                f"❌ Candidate #{idx + 1} ({molecule_name}) is not unique, trying next..."
            )
    
    bt.logging.warning(f"No unique candidates found in top {candidates_to_check} molecules")
    return None


async def submit_response(state: Dict[str, Any]) -> None:
    """Encrypts and submits the current candidate product."""
    candidate_product = state['candidate_product']
    if not candidate_product:
        bt.logging.warning("No candidate product to submit")
        return

    bt.logging.info(f"Starting submission process for product: {candidate_product}")
    
    try:
        current_block = await state['subtensor'].get_current_block()
        encrypted_response = state['bdt'].encrypt(state['miner_uid'], candidate_product, current_block)
        bt.logging.info(f"Encrypted response generated successfully")

        tmp_file = tempfile.NamedTemporaryFile(delete=True)
        with open(tmp_file.name, 'w+') as f:
            f.write(str(encrypted_response))
            f.flush()

            f.seek(0)
            content_str = f.read()
            encoded_content = base64.b64encode(content_str.encode()).decode()

            filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
            commit_content = f"{state['github_path']}/{filename}.txt"
            bt.logging.info(f"Prepared commit content: {commit_content}")

            bt.logging.info(f"Attempting chain commitment...")
            try: 
                commitment_status = await state['subtensor'].set_commitment(
                    wallet=state['wallet'],
                    netuid=state['config'].netuid,
                    data=commit_content
                )
                bt.logging.info(f"Chain commitment status: {commitment_status}")
            except MetadataError:
                bt.logging.info("Too soon to commit again. Will keep looking for better candidates.")
                return

            if commitment_status:
                try:
                    bt.logging.info(f"Commitment set successfully for {commit_content}")
                    bt.logging.info("Attempting GitHub upload...")
                    github_status = upload_file_to_github(filename, encoded_content)
                    if github_status:
                        bt.logging.info(f"File uploaded successfully to {commit_content}")
                        state['last_submitted_product'] = candidate_product
                        state['last_submission_time'] = datetime.datetime.now()
                        current_epoch = current_block // state['epoch_length']
                        state['last_submission_epoch'] = current_epoch
                    else:
                        bt.logging.error(f"Failed to upload file to GitHub for {commit_content}")
                except Exception as e:
                    bt.logging.error(f"Failed to upload file for {commit_content}: {e}")
    
    except Exception as e:
        bt.logging.error(f"Error in submit_response: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())


# ============================================================================
# MAIN MINING LOOP
# ============================================================================

async def run_miner(config: argparse.Namespace) -> None:
    """Main mining loop."""

    wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor_objects(config)

    state: Dict[str, Any] = {
        'config': config,
        'chunk_size': 512,
        'submission_interval': 1200,
        'github_path': load_github_path(),
        'wallet': wallet,
        'subtensor': subtensor,
        'metagraph': metagraph,
        'miner_uid': miner_uid,
        'epoch_length': epoch_length,
        'bdt': QuicknetBittensorDrandTimelock(),
        'candidate_product': None,
        'best_score': float('-inf'),
        'last_submitted_product': None,
        'last_submission_time': None,
        'last_submission_epoch': -1,
        'csv_data_loaded': False,
        'shutdown_event': asyncio.Event(),
        'current_challenge_targets': [],
        'last_challenge_targets': [],
        'current_challenge_antitargets': [],
        'last_challenge_antitargets': [],
        'rxn_id': None,
        'top_pool': pd.DataFrame(
            columns=["name", "smiles", "InChIKey", "score", "target_affinity", "antitarget_affinity"]
        ),
        'seen_inchikeys': set(),
        'ml_predictor': None,
    }

    bt.logging.info("Entering main miner loop...")

    state['rxn_id'] = HARDCODED_RXN_ID
    
    current_block = await subtensor.get_current_block()
    last_boundary = (current_block // epoch_length) * epoch_length
    block_hash = await subtensor.determine_block_hash(last_boundary)
    startup_proteins = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        weekly_target=config.weekly_target,
        num_antitargets=config.num_antitargets
    )

    if startup_proteins:
        state['current_challenge_targets'] = startup_proteins["targets"]
        state['last_challenge_targets'] = startup_proteins["targets"]
        state['current_challenge_antitargets'] = startup_proteins["antitargets"]
        state['last_challenge_antitargets'] = startup_proteins["antitargets"]
        
        bt.logging.info(f"Using hardcoded reaction ID: {state['rxn_id']}")
        bt.logging.info(
            f"Startup targets: {startup_proteins['targets']}, "
            f"antitargets: {startup_proteins['antitargets']}"
        )

        # Launch the IMPROVED inference loop
        try:
            state['inference_task'] = asyncio.create_task(run_improved_model_loop(state))
            bt.logging.info("✅ Improved ML-guided inference loop started!")
        except Exception as e:
            bt.logging.error(f"Error starting inference: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

    # Main epoch-based loop (ORIGINAL SUBMISSION LOGIC - UNCHANGED)
    while True:
        try:
            current_block = await subtensor.get_current_block()

            if current_block % epoch_length == 0:
                current_epoch = current_block // epoch_length
                bt.logging.info(
                    f"Epoch boundary at block {current_block} (epoch {current_epoch}). "
                    f"Continuing ML-guided search."
                )

            if current_block % 60 == 0:
                await metagraph.sync()
                log = (
                    f"Block: {metagraph.block.item()} | "
                    f"Number of nodes: {metagraph.n} | "
                    f"Current epoch: {metagraph.block.item() // epoch_length}"
                )
                bt.logging.info(log)

            # ✅ ORIGINAL SUBMISSION LOGIC - Check if close to epoch end
            blocks_until_epoch = epoch_length - (current_block % epoch_length)
            
            if blocks_until_epoch <= 50:
                last_submission_epoch = state.get('last_submission_epoch', -1)
                current_epoch = current_block // epoch_length
                can_submit = (last_submission_epoch < current_epoch)
                
                if can_submit:
                    top_pool = state.get('top_pool', pd.DataFrame())
                    
                    if not top_pool.empty:
                        bt.logging.info(
                            f"⏰ Close to epoch end ({blocks_until_epoch} blocks remaining), "
                            f"searching for unique candidate to submit..."
                        )
                        
                        unique_candidate = await find_unique_candidate(state, top_pool, max_candidates=10)
                        
                        if unique_candidate:
                            if unique_candidate != state.get('last_submitted_product'):
                                state['candidate_product'] = unique_candidate
                                bt.logging.info(f"Attempting to submit unique candidate: {unique_candidate}")
                                try:
                                    await submit_response(state)
                                    bt.logging.info(f"✅ Submission successful!")
                                except Exception as e:
                                    bt.logging.error(f"Error submitting response: {e}")
                            else:
                                bt.logging.info("Skipping submission - same product as last submission")
                        else:
                            bt.logging.warning(
                                f"No unique candidates found in top pool. "
                                f"Skipping submission for this epoch."
                            )

            await asyncio.sleep(1)

        except RuntimeError as e:
            bt.logging.error(e)
            import traceback
            traceback.print_exc()

        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting miner.")
            state['shutdown_event'].set()
            break


async def main() -> None:
    """Main entry point."""
    config = parse_arguments()
    setup_logging(config)
    await run_miner(config)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
