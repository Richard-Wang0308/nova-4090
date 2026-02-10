#!/usr/bin/env python3
"""
Boltz Scoring Script

This script reads target and antitarget from config.yaml, gets molecules,
runs the Boltz model for scoring, and writes results to result.json.

Uses the exact same logic as nova/neurons/validator/validator.py
"""

import os
import sys
import argparse
import yaml
import json
import math
import time
import bittensor as bt

# Add project root to path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from boltz.wrapper import BoltzWrapper

def setup_logging():
    """Setup bittensor logging - matches nova/neurons/validator/setup.py"""
    # Create a minimal parser and config object for logging
    parser = argparse.ArgumentParser('Boltz Scoring')
    config = bt.config(parser)
    bt.logging(config=config, logging_dir=BASE_DIR, record_log=False)
    bt.logging.set_debug(True)

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from config.yaml file.
    
    Expected structure:
    - protein_selection:
        - weekly_target: "Q01959"
        - antitarget: "P12345"
    - protein_constraints:
        - binding_pocket: null
        - max_distance: null
        - force: false
    - molecule_validation:
        - num_molecules_boltz: 1
        - boltz_metric: ["affinity_probability_binary", "affinity_pred_value"]
        - combination_strategy: "heavy_atom_normalization"
        - sample_selection: "first"
    - molecule:
        - name: "molecule_1"
        - smiles: "CCO"
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def prepare_molecules_data(config: dict) -> tuple[dict, dict]:
    """
    Prepare valid_molecules_by_uid and score_dict from config.
    
    Returns:
        valid_molecules_by_uid: dict mapping UID to {'smiles': [...], 'names': [...]}
        score_dict: dict mapping UID to score data structure
    """
    molecule = config.get('molecule', {})
    
    if not molecule:
        raise ValueError("No molecule found in config.yaml")
    
    smiles = molecule.get('smiles')
    name = molecule.get('name', 'molecule_1')
    
    if not smiles:
        raise ValueError("Molecule SMILES not found in config.yaml")
    
    # For simplicity, we'll use UID 0
    # In the validator, UIDs come from the network, but here we just score one molecule
    uid = 0
    
    valid_molecules_by_uid = {
        uid: {
            'smiles': [smiles],
            'names': [name]
        }
    }
    
    # Initialize score_dict structure matching validator
    score_dict = {
        uid: {
            "target_scores": [[]],
            "antitarget_scores": [[]],
            "entropy": None,
            "entropy_boltz": None,
            "block_submitted": None,
            "push_time": ""
        }
    }
    
    return valid_molecules_by_uid, score_dict

def main():
    """Main execution function"""
    setup_logging()
    
    bt.logging.info("Starting Boltz scoring...")
    
    # Load configuration
    try:
        config = load_config()
        bt.logging.info("Configuration loaded successfully")
    except Exception as e:
        bt.logging.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Extract subnet config (matching validator structure)
    target = config['protein_selection']['weekly_target']
    antitarget = config['protein_selection'].get('antitarget')
    
    subnet_config = {
        'weekly_target': target,
        'num_antitargets': 1 if antitarget else 0,
        'binding_pocket': config.get('protein_constraints', {}).get('binding_pocket'),
        'max_distance': config.get('protein_constraints', {}).get('max_distance'),
        'force': config.get('protein_constraints', {}).get('force', False),
        'num_molecules_boltz': config['molecule_validation']['num_molecules_boltz'],
        'boltz_metric': config['molecule_validation']['boltz_metric'],
        'combination_strategy': config['molecule_validation']['combination_strategy'],
        'sample_selection': config['molecule_validation'].get('sample_selection', 'first'),
    }
    
    # Prepare molecules data
    try:
        valid_molecules_by_uid, score_dict = prepare_molecules_data(config)
        molecule_name = valid_molecules_by_uid[0]['names'][0]
        molecule_smiles = valid_molecules_by_uid[0]['smiles'][0]
        bt.logging.info(f"Prepared molecule '{molecule_name}' (SMILES: {molecule_smiles}) for scoring")
        bt.logging.info(f"Target: {target}, Antitarget: {antitarget}")
    except Exception as e:
        bt.logging.error(f"Failed to prepare molecules: {e}")
        sys.exit(1)
    
    # Initialize BoltzWrapper
    try:
        bt.logging.info("Initializing Boltz model...")
        boltz = BoltzWrapper()
        bt.logging.info("Boltz model initialized successfully")
    except Exception as e:
        bt.logging.error(f"Failed to initialize Boltz model: {e}")
        sys.exit(1)
    
    # Run scoring (using a dummy block hash since we don't have a real one)
    # The block hash is used for random sampling, but we'll use a fixed value
    final_block_hash = "0x" + "0" * 64  # Dummy block hash
    
    try:
        bt.logging.info("Running Boltz scoring...")
        start_time = time.time()
        boltz.score_molecules_target(
            valid_molecules_by_uid, 
            score_dict, 
            subnet_config, 
            final_block_hash
        )
        elapsed_time = time.time() - start_time
        bt.logging.info(f"Boltz scoring completed successfully in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        if elapsed_time > 0:
            bt.logging.error(f"Failed to run Boltz scoring after {elapsed_time:.2f} seconds: {e}")
        else:
            bt.logging.error(f"Failed to run Boltz scoring: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())
        sys.exit(1)
    
    # Prepare results for output
    uid = 0
    valid_mols = valid_molecules_by_uid.get(uid, {})
    smiles = valid_mols.get('smiles', [])[0] if valid_mols.get('smiles') else None
    name = valid_mols.get('names', [])[0] if valid_mols.get('names') else None
    
    results = {
        'target': target,
        'antitarget': antitarget,
        'molecule': {
            'name': name,
            'smiles': smiles,
            'boltz_score': boltz.per_molecule_metric.get(uid, {}).get(smiles) if smiles else None,
            'components': boltz.per_molecule_components.get(uid, {}).get(smiles, {}) if smiles else {}
        },
        'scores': {}
    }
    
    if uid in score_dict:
        data = score_dict[uid]
        results['scores'] = {
            'boltz_score': data.get('boltz_score'),
            'entropy_boltz': data.get('entropy_boltz'),
        }
    
    # Write results to result.json
    output_path = "result.json"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        bt.logging.info(f"Results written to {output_path}")
    except Exception as e:
        bt.logging.error(f"Failed to write results: {e}")
        sys.exit(1)
    
    # Cleanup (if method exists)
    if hasattr(boltz, 'cleanup_model'):
        try:
            boltz.cleanup_model()
        except Exception:
            pass
    
    bt.logging.info("Boltz scoring completed successfully!")

if __name__ == "__main__":
    main()

