#!/usr/bin/env python3
"""
SIMPLIFIED BITTENSOR MINER - Genetic Algorithm Based Molecule Generation

Simple workflow:
1. Load molecules from CSV at startup
2. Apply genetic operations (CROSSOVER ONLY)
3. Check if molecule is unique (NOT on HuggingFace)
4. Submit if unique
5. Do not submit again in the same epoch
"""

import os
import sys
import random
import argparse
import asyncio
import datetime
import tempfile
import traceback
import base64
import hashlib
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path

from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.errors import MetadataError

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Database path for combinatorial DB
DB_PATH = os.path.join(BASE_DIR, "combinatorial_db", "molecules.sqlite")
HARDCODED_RXN_ID = 2
STARTING_EPOCH = 20656

# ✅ CSV for loading initial molecules
REACTION_TRAIN_CSV = os.path.join(BASE_DIR, 'BoltzPredictor', 'data', 'mols_2.csv')

from config.config_loader import load_config
from utils import (
    upload_file_to_github,
    get_challenge_params_from_blockhash,
)
from utils.molecules import molecule_unique_for_protein_hf
from molecules_base import (
    generate_inchikey,
)
from combinatorial_db.reactions import get_smiles_from_reaction
from btdr import QuicknetBittensorDrandTimelock


# ============================================================================
# ✅ GENETIC ALGORITHM OPERATIONS (CROSSOVER ONLY)
# ============================================================================

class GeneticAlgorithmOperator:
    """Performs genetic algorithm operations on molecules (CROSSOVER ONLY)."""
    
    def __init__(self, rxn_id: int, db_path: str):
        """Initialize GA operator."""
        self.rxn_id = rxn_id
        self.db_path = db_path
        self.generated_molecule_names: Set[str] = set()  # Track generated molecule names
    
    def crossover_molecules(self, mol_name_1: str, mol_name_2: str) -> Optional[str]:
        """
        Crossover two molecules by swapping random components.
        
        Format: rxn:1:comp1:comp2
        We can swap comp1 or comp2 between two molecules
        
        Args:
            mol_name_1: First parent molecule name
            mol_name_2: Second parent molecule name
            
        Returns:
            New molecule name from crossover, or None if crossover failed
        """
        try:
            from rdkit import Chem
            
            # Parse molecule names
            parts1 = mol_name_1.split(':')
            parts2 = mol_name_2.split(':')
            
            bt.logging.debug(f"Crossover: {mol_name_1} x {mol_name_2}")
            bt.logging.debug(f"  Parts1: {parts1}, Parts2: {parts2}")
            
            # Expected format: rxn:1:comp1:comp2 (4 parts each)
            if (len(parts1) != 4 or len(parts2) != 4 or 
                parts1[0] != 'rxn' or parts2[0] != 'rxn'):
                bt.logging.debug(f"Invalid format for crossover")
                return None
            
            try:
                rxn_id_1 = int(parts1[1])
                rxn_id_2 = int(parts2[1])
                if rxn_id_1 != self.rxn_id or rxn_id_2 != self.rxn_id:
                    bt.logging.debug(f"Wrong rxn_ids: {rxn_id_1}, {rxn_id_2}")
                    return None
            except (ValueError, IndexError) as e:
                bt.logging.debug(f"Error parsing rxn_ids: {e}")
                return None
            
            # Randomly select which component to swap: either comp1 (index 2) or comp2 (index 3)
            swap_idx = random.choice([2, 3])
            bt.logging.debug(f"Swapping component at index {swap_idx}")
            
            # Create offspring by swapping one component
            offspring_parts = parts1.copy()
            offspring_parts[swap_idx] = parts2[swap_idx]
            offspring_name = ':'.join(offspring_parts)
            
            bt.logging.debug(f"Offspring: {offspring_name}")
            
            # ✅ CHECK IF WE ALREADY GENERATED THIS MOLECULE
            if offspring_name in self.generated_molecule_names:
                bt.logging.debug(f"⚠️  Offspring {offspring_name} already generated in this batch")
                return None
            
            # Validate offspring
            try:
                offspring_smiles = get_smiles_from_reaction(offspring_name)
                if offspring_smiles:
                    mol = Chem.MolFromSmiles(offspring_smiles)
                    if mol is not None:
                        # ✅ TRACK THIS MOLECULE NAME
                        self.generated_molecule_names.add(offspring_name)
                        bt.logging.info(f"✅ Crossover successful: {mol_name_1} × {mol_name_2} → {offspring_name}")
                        return offspring_name
                    else:
                        bt.logging.debug(f"Invalid SMILES from RDKit: {offspring_smiles}")
                else:
                    bt.logging.debug(f"No SMILES generated for offspring")
            except Exception as e:
                bt.logging.debug(f"Error validating crossover: {e}")
            
            return None
        
        except Exception as e:
            bt.logging.debug(f"Error in crossover_molecules: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            return None
    
    def apply_genetic_operations(
        self,
        top_molecules: List[str],
        num_crossovers: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Apply genetic operations (CROSSOVER ONLY) to top molecules.
        
        Args:
            top_molecules: List of top molecule names
            num_crossovers: Number of crossovers to attempt
            
        Returns:
            List of new molecules with their SMILES and names (in order generated)
        """
        new_molecules = []
        
        # ✅ RESET TRACKING FOR THIS BATCH
        self.generated_molecule_names.clear()
        
        bt.logging.info(f"🧬 Applying CROSSOVER-ONLY genetic operations to top {len(top_molecules)} molecules...")
        bt.logging.info(f"   Sample molecules: {top_molecules[:3]}")
        
        # Apply crossovers only
        crossover_attempts = 0
        crossovers_created = 0
        
        for i in range(num_crossovers):
            parent1 = random.choice(top_molecules)
            parent2 = random.choice(top_molecules)
            crossover_attempts += 1
            
            bt.logging.info(f"   Attempting crossover {i+1}/{num_crossovers}: {parent1} x {parent2}")
            
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
                            bt.logging.info(f"   ✅ Crossover #{crossovers_created}: {parent1} × {parent2} → {offspring}")
                    
                    except Exception as e:
                        bt.logging.debug(f"Error processing offspring: {e}")
                else:
                    bt.logging.debug(f"   ❌ Crossover {i+1} failed (duplicate or invalid)")
            else:
                bt.logging.debug(f"   ⏭️  Crossover {i+1}: parents are identical, skipping")
        
        bt.logging.info(f"   Crossovers: {crossovers_created}/{num_crossovers} successful")
        bt.logging.info(f"🧬 Generated {len(new_molecules)} new molecules from crossover operations")
        
        # ✅ VERIFY ORDER IS PRESERVED
        bt.logging.info(f"   Generated molecules in order: {[m['name'] for m in new_molecules]}")
        
        return new_molecules


# ============================================================================
# HELPER FUNCTIONS
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
    """
    Check if molecule is unique for target protein (NOT in HuggingFace dataset).
    
    Returns:
        True if molecule is NOT in HuggingFace (i.e., it's unique/new)
        False if molecule IS in HuggingFace (i.e., it's already known)
    """
    if not state.get('current_challenge_targets'):
        bt.logging.warning("No target proteins available")
        return False
    
    primary_target = state['current_challenge_targets'][0]
    
    try:
        is_unique_hf = molecule_unique_for_protein_hf(primary_target, smiles)
        
        if not is_unique_hf:
            bt.logging.debug(f"❌ Molecule {molecule_name} already in HuggingFace dataset")
            return False
        
        bt.logging.info(f"✅ Molecule {molecule_name} is NOT in HuggingFace (unique!)")
        return True
    except Exception as e:
        bt.logging.error(f"Error checking uniqueness: {e}")
        return False


def load_molecules_from_csv(
    csv_path: str,
    target_proteins: List[str],
    starting_epoch: int,
    rxn_id: int
) -> pd.DataFrame:
    """Load molecules from CSV file."""
    if not os.path.exists(csv_path):
        bt.logging.warning(f"CSV file not found at {csv_path}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    
    try:
        bt.logging.info(
            f"Loading molecules from {csv_path} for targets {target_proteins}, "
            f"epoch >= {starting_epoch}, rxn_id={rxn_id}"
        )
        df = pd.read_csv(csv_path)
        
        # Filter by target protein
        if 'target_protein' in df.columns:
            df = df[df['target_protein'].isin(target_proteins)]
        else:
            bt.logging.warning("CSV file does not have 'target_protein' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
        
        # Filter by epoch
        if 'epoch' in df.columns:
            df = df[df['epoch'] >= starting_epoch]
        else:
            bt.logging.warning("CSV file does not have 'epoch' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
        
        # Filter by reaction ID
        if 'molecule_name' in df.columns:
            df = df[df['molecule_name'].str.startswith(f"rxn:{rxn_id}:", na=False)]
        else:
            bt.logging.warning("CSV file does not have 'molecule_name' column")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
        
        if df.empty:
            bt.logging.info("No matching molecules found in CSV")
            return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
        
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
                
                result_rows.append({
                    'name': molecule_name,
                    'smiles': smiles,
                    'InChIKey': inchikey,
                })
                successful_count += 1
                
            except Exception as e:
                bt.logging.debug(f"Could not process {molecule_name}: {e}")
                failed_count += 1
                continue
        
        result_df = pd.DataFrame(result_rows)
        if not result_df.empty:
            result_df = result_df.drop_duplicates(subset=['InChIKey'], keep='first')
            bt.logging.info(
                f"✅ Loaded {len(result_df)} molecules from CSV "
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
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])


# ============================================================================
# ✅ ADAPTIVE GENETIC OPERATIONS WITH SUBMISSION LOGIC (CROSSOVER ONLY)
# ============================================================================

async def run_adaptive_genetic_loop(state: Dict[str, Any]) -> None:
    """
    Adaptive genetic algorithm loop (CROSSOVER ONLY):
    1. Generate molecules from top 5 using crossover
    2. Check each molecule - if NOT on HuggingFace, submit it
    3. If all are on HuggingFace, apply GA on top 10 molecules
    4. Do not submit again in same epoch after successful submission
    """
    bt.logging.info("🚀 Starting ADAPTIVE genetic algorithm loop (CROSSOVER ONLY)...")
    
    ga_operator = GeneticAlgorithmOperator(HARDCODED_RXN_ID, DB_PATH)
    top_pool = state.get('top_pool', pd.DataFrame())
    
    if top_pool.empty:
        bt.logging.warning("Top pool is empty, cannot start GA loop")
        return
    
    generation = 0
    
    while not state['shutdown_event'].is_set():
        try:
            generation += 1
            
            # Get current epoch
            current_block = await state['subtensor'].get_current_block()
            current_epoch = current_block // state['epoch_length']
            last_submission_epoch = state.get('last_submission_epoch', -1)
            
            # Check if we already submitted in this epoch
            if last_submission_epoch == current_epoch:
                bt.logging.info(
                    f"⏭️  Already submitted in epoch {current_epoch}, waiting for next epoch..."
                )
                await asyncio.sleep(10)
                continue
            
            bt.logging.info(f"\n{'='*70}")
            bt.logging.info(f"🧬 Generation {generation} - Epoch {current_epoch}")
            bt.logging.info(f"{'='*70}")
            
            # Get top molecules for GA
            top_pool = state.get('top_pool', pd.DataFrame())
            
            if top_pool.empty:
                bt.logging.warning("Top pool is empty")
                await asyncio.sleep(10)
                continue
            
            # Start with top 5
            top_n = 5
            top_molecules_df = top_pool.head(top_n)
            top_molecules_names = top_molecules_df['name'].tolist()
            
            bt.logging.info(f"📊 Using top {top_n} molecules for crossover operations")
            bt.logging.info(f"   Molecules: {top_molecules_names}")
            
            # Apply genetic operations (CROSSOVER ONLY)
            new_molecules = ga_operator.apply_genetic_operations(
                top_molecules_names,
                num_crossovers=5  # 5 crossover attempts
            )
            
            if not new_molecules:
                bt.logging.warning("No new molecules generated from crossover")
                await asyncio.sleep(10)
                continue
            
            bt.logging.info(
                f"✅ Generated {len(new_molecules)} new molecules"
            )
            
            # ✅ CHECK EACH MOLECULE - IF NOT ON HUGGINGFACE, SUBMIT IT
            all_on_huggingface = True
            
            for idx, molecule in enumerate(new_molecules):
                molecule_name = molecule['name']
                smiles = molecule['smiles']
                
                bt.logging.info(
                    f"\n🔍 Checking molecule #{idx+1}/{len(new_molecules)}: {molecule_name}"
                )
                
                # Check if unique (NOT on HuggingFace)
                is_unique = await check_molecule_unique(state, molecule_name, smiles)
                
                if is_unique:
                    # ✅ Found a unique molecule - SUBMIT IT
                    bt.logging.info(
                        f"✅ Found unique molecule! Attempting submission..."
                    )
                    
                    state['candidate_product'] = molecule_name
                    
                    try:
                        await submit_response(state)
                        bt.logging.info(f"✅ Submission successful for {molecule_name}!")
                        all_on_huggingface = False
                        break  # Exit loop after successful submission
                    
                    except Exception as e:
                        bt.logging.error(f"❌ Error submitting response: {e}")
                        import traceback
                        bt.logging.error(traceback.format_exc())
                
                else:
                    # Molecule is already on HuggingFace
                    bt.logging.info(
                        f"❌ Molecule {molecule_name} is already on HuggingFace, trying next..."
                    )
            
            # ✅ IF ALL GENERATED MOLECULES ARE ON HUGGINGFACE, APPLY GA ON TOP 10
            if all_on_huggingface:
                bt.logging.warning(
                    f"⚠️  All {len(new_molecules)} generated molecules are already on HuggingFace!"
                )
                bt.logging.info(
                    f"🧬 Applying crossover on top 10 molecules instead..."
                )
                
                top_n = 10
                top_molecules_df = top_pool.head(top_n)
                top_molecules_names = top_molecules_df['name'].tolist()
                
                bt.logging.info(f"📊 Using top {top_n} molecules for crossover operations")
                
                # Apply genetic operations on top 10
                new_molecules_top10 = ga_operator.apply_genetic_operations(
                    top_molecules_names,
                    num_crossovers=8  # 8 crossover attempts with larger pool
                )
                
                if new_molecules_top10:
                    bt.logging.info(
                        f"✅ Generated {len(new_molecules_top10)} new molecules from top 10"
                    )
                    
                    # Check molecules from top 10 GA
                    for idx, molecule in enumerate(new_molecules_top10):
                        molecule_name = molecule['name']
                        smiles = molecule['smiles']
                        
                        bt.logging.info(
                            f"\n🔍 Checking top-10 molecule #{idx+1}/{len(new_molecules_top10)}: {molecule_name}"
                        )
                        
                        # if idx == 0:
                        #     bt.logging.info("Trying to skip submission for first molecule")
                        #     continue

                        is_unique = await check_molecule_unique(state, molecule_name, smiles)
                        
                        if is_unique:
                            bt.logging.info(
                                f"✅ Found unique molecule from top-10 crossover! Attempting submission..."
                            )
                            
                            state['candidate_product'] = molecule_name
                            
                            try:
                                await submit_response(state)
                                bt.logging.info(f"✅ Submission successful for {molecule_name}!")
                                break
                            
                            except Exception as e:
                                bt.logging.error(f"❌ Error submitting response: {e}")
            
            await asyncio.sleep(10)
        
        except Exception as e:
            bt.logging.error(f"Error in adaptive GA loop: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            await asyncio.sleep(10)


# ============================================================================
# ✅ STARTUP: LOAD CSV
# ============================================================================

async def startup_phase(state: Dict[str, Any]) -> None:
    """
    Startup phase:
    1. Load molecules from CSV
    2. Prepare top_pool
    """
    bt.logging.info("🚀 Starting STARTUP phase: Load CSV...")
    
    try:
        # Load molecules from CSV
        bt.logging.info("📂 Loading molecules from CSV...")
        
        molecules_df = load_molecules_from_csv(
            REACTION_TRAIN_CSV,
            state['current_challenge_targets'],
            STARTING_EPOCH,
            HARDCODED_RXN_ID
        )
        
        if molecules_df.empty:
            bt.logging.warning("No molecules loaded from CSV")
            return
        
        # Update state
        state['top_pool'] = molecules_df.copy()
        state['seen_inchikeys'].update(molecules_df['InChIKey'].tolist())
        
        bt.logging.info(
            f"✅ STARTUP COMPLETE:"
            f"\n   Total molecules in pool: {len(state['top_pool'])}"
            f"\n   Sample molecules: {state['top_pool']['name'].head(3).tolist()}"
        )
        
        state['startup_complete'] = True
    
    except Exception as e:
        bt.logging.error(f"Error in startup phase: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())


# ============================================================================
# SUBMISSION LOGIC
# ============================================================================

async def submit_response(state: Dict[str, Any]) -> None:
    """Encrypts and submits the current candidate product."""
    candidate_product = state['candidate_product']
    if not candidate_product:
        bt.logging.warning("No candidate product to submit")
        return

    bt.logging.info(f"📤 Starting submission process for product: {candidate_product}")
    
    try:
        current_block = await state['subtensor'].get_current_block()
        encrypted_response = state['bdt'].encrypt(state['miner_uid'], candidate_product, current_block)
        bt.logging.info(f"🔐 Encrypted response generated successfully")

        tmp_file = tempfile.NamedTemporaryFile(delete=True)
        with open(tmp_file.name, 'w+') as f:
            f.write(str(encrypted_response))
            f.flush()

            f.seek(0)
            content_str = f.read()
            encoded_content = base64.b64encode(content_str.encode()).decode()

            filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
            commit_content = f"{state['github_path']}/{filename}.txt"
            bt.logging.info(f"📝 Prepared commit content: {commit_content}")

            bt.logging.info(f"⛓️  Attempting chain commitment...")
            try: 
                commitment_status = await state['subtensor'].set_commitment(
                    wallet=state['wallet'],
                    netuid=state['config'].netuid,
                    data=commit_content
                )
                bt.logging.info(f"⛓️  Chain commitment status: {commitment_status}")
            except MetadataError:
                bt.logging.info("⏳ Too soon to commit again. Will try next epoch.")
                return

            if commitment_status:
                try:
                    bt.logging.info(f"✅ Commitment set successfully for {commit_content}")
                    bt.logging.info("📤 Attempting GitHub upload...")
                    github_status = upload_file_to_github(filename, encoded_content)
                    if github_status:
                        bt.logging.info(f"✅ File uploaded successfully to {commit_content}")
                        state['last_submitted_product'] = candidate_product
                        state['last_submission_time'] = datetime.datetime.now()
                        current_epoch = current_block // state['epoch_length']
                        state['last_submission_epoch'] = current_epoch
                        bt.logging.info(f"✅ Submission recorded for epoch {current_epoch}")
                    else:
                        bt.logging.error(f"❌ Failed to upload file to GitHub for {commit_content}")
                except Exception as e:
                    bt.logging.error(f"❌ Failed to upload file for {commit_content}: {e}")
    
    except Exception as e:
        bt.logging.error(f"❌ Error in submit_response: {e}")
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
        'github_path': load_github_path(),
        'wallet': wallet,
        'subtensor': subtensor,
        'metagraph': metagraph,
        'miner_uid': miner_uid,
        'epoch_length': epoch_length,
        'bdt': QuicknetBittensorDrandTimelock(),
        'candidate_product': None,
        'last_submitted_product': None,
        'last_submission_time': None,
        'last_submission_epoch': -1,
        'startup_complete': False,
        'shutdown_event': asyncio.Event(),
        'current_challenge_targets': [],
        'last_challenge_targets': [],
        'current_challenge_antitargets': [],
        'last_challenge_antitargets': [],
        'rxn_id': None,
        'top_pool': pd.DataFrame(columns=["name", "smiles", "InChIKey"]),
        'seen_inchikeys': set(),
    }

    bt.logging.info("🚀 Entering main miner loop...")

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

        # Run startup phase
        try:
            await startup_phase(state)
            bt.logging.info("✅ Startup phase completed!")
        except Exception as e:
            bt.logging.error(f"Error in startup: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

        # Launch adaptive GA loop
        try:
            state['ga_task'] = asyncio.create_task(run_adaptive_genetic_loop(state))
            bt.logging.info("✅ Adaptive GA loop started!")
        except Exception as e:
            bt.logging.error(f"Error starting GA loop: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

    # Main monitoring loop
    while True:
        try:
            current_block = await subtensor.get_current_block()

            if current_block % epoch_length == 0:
                current_epoch = current_block // epoch_length
                bt.logging.info(
                    f"⏰ Epoch boundary at block {current_block} (epoch {current_epoch})"
                )

            if current_block % 60 == 0:
                await metagraph.sync()
                log = (
                    f"Block: {metagraph.block.item()} | "
                    f"Number of nodes: {metagraph.n} | "
                    f"Current epoch: {metagraph.block.item() // epoch_length}"
                )
                bt.logging.info(log)

            await asyncio.sleep(1)

        except RuntimeError as e:
            bt.logging.error(e)
            import traceback
            traceback.print_exc()

        except KeyboardInterrupt:
            bt.logging.success("⛔ Keyboard interrupt detected. Exiting miner.")
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
