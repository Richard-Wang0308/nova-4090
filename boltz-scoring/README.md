# Boltz Scoring Project

This project uses the exact same Boltz scoring logic as `nova/neurons/validator/validator.py` to score molecules against target proteins.

## Structure

- `main.py` - Main script that reads config, runs scoring, and writes results
- `config.yaml` - Configuration file with target, antitarget, and molecules
- `boltz/` - Boltz wrapper and configuration (same as nova/boltz)
- `utils/` - Utility functions for proteins and molecules
- `result.json` - Output file with scoring results

## Usage

1. Edit `config.yaml` to specify:
   - Target protein (weekly_target)
   - Molecules to score (with name and smiles)
   - Scoring parameters

2. Run the scoring:
   ```bash
   python main.py
   ```

3. Results will be written to `result.json`

## Configuration

The `config.yaml` file should contain:

- `protein_selection.weekly_target`: Target protein code (UniProt ID)
- `protein_constraints`: Optional binding pocket constraints
- `molecule_validation`: Scoring parameters (num_molecules_boltz, boltz_metric, etc.)
- `molecules`: List of molecules with `name` and `smiles` fields

## Output

The `result.json` file contains:
- Target protein
- Molecules with their scores
- Per-molecule metrics and components
- Overall boltz_score

