# BoltzPredictor

A deep learning model for predicting binding affinity scores between molecules and proteins, designed for virtual drug screening competitions.

## Architecture

- **Molecule Encoder**: GINE (Graph Isomorphism Network) with RDKit-extracted features
  - Atom features: atomic number, degree, formal charge, aromatic flag, hybridization
  - Bond features: bond type, conjugation, ring membership
  - Output: 256-dim molecular embedding
  
- **Protein Encoder**: ESM-2 (650M, frozen) for protein sequence embeddings
  - Frozen to prevent overfitting on small datasets
  - Mean-pooled residue embeddings (1280-dim)
  - Automatic caching for efficiency
  
- **Interaction Head**: MLP that combines molecule and protein embeddings
  - Architecture: concat(h_m, h_p) → Linear(?, 512) → GELU → Linear(512, 128) → GELU → Linear(128, 1)
  
- **Final Score**: `S(m, target) - w * S(m, antitarget)`
  - Shared affinity model ensures consistent predictions
  - Decomposition helps with generalization

## Installation

```bash
pip install -r requirements.txt
```

**Note**: Requires PyTorch with CUDA support for optimal performance. ESM-2 model will be downloaded automatically on first use (~2.5GB).

## Data Format

Training data should be in CSV format with the following columns:

- `molecule_name`: Molecule name in rxn format (e.g., `rxn:1:62588:2672`)
- `target_protein`: Target protein code (e.g., `Q63380`)
- `antitarget_protein`: Antitarget protein code (e.g., `A0A5N3V1Q2`)
- `final_score`: Ground truth final score (target_score - w * antitarget_score)
- `epoch` (optional): Epoch ID for ranking loss grouping

**Note**: The dataset automatically converts:
- Molecule names (rxn format) → SMILES strings
- Protein codes → Protein sequences (from UniProt/HuggingFace)

Example:
```csv
molecule_name,target_protein,antitarget_protein,final_score,epoch
rxn:1:62588:2672,Q63380,A0A5N3V1Q2,0.523,19959
rxn:2:12345:67890,Q63380,A0A5N3V1Q2,0.412,19959
```

## Usage

### Data Preparation

First, prepare your training data from the competition API:

```bash
python scripts/prepare_training_data.py --start_epoch 19959
```

This will:
- Fetch leaderboard data from the API (starting from epoch 19959)
- Auto-detect the latest available epoch
- Save only: `molecule_name`, `target_protein`, `antitarget_protein`, `final_score`, `epoch`
- Conversions (rxn→SMILES, protein code→sequence) happen during dataset loading

The script automatically:
- Finds the latest epoch by binary search
- Resumes from existing data (skips already processed epochs)
- Handles errors gracefully

See `README_DATA_PREPARATION.md` for detailed instructions.

### Precompute Protein Embeddings (Recommended)

Before training, precompute embeddings for all unique proteins to speed up training:

```bash
python scripts/precompute_proteins.py --data_file data/train.csv --cache_dir cache/protein_embeddings
```

### Training

1. **Offline Pretraining** (initial training on all data):
```bash
python scripts/train.py --config configs/pretrain.yaml
```

   This will:
   - Train the full model (molecule encoder + interaction head)
   - Freeze protein encoder
   - Use combined regression + ranking loss
   - Save best model to `checkpoints/best.pt`

2. **Online Target Adaptation** (when target changes weekly):
```bash
python scripts/train.py --config configs/adapt.yaml --checkpoint checkpoints/best.pt
```

   This will:
   - Freeze molecule and protein encoders
   - Fine-tune only the interaction head
   - Use lower learning rate (1e-4)
   - Train for fewer epochs (3-5)

### Inference

**Single prediction**:
```bash
python scripts/inference.py \
    --checkpoint checkpoints/best.pt \
    --smiles "CCO" \
    --target_seq "MKTAYIAKQR..." \
    --antitarget_seq "MKTAYIAKQR..." \
    --antitarget_weight 1.0
```

**Batch prediction from CSV**:
```bash
python scripts/inference.py \
    --checkpoint checkpoints/best.pt \
    --input_file data/test.csv \
    --output_file predictions.csv \
    --antitarget_weight 1.0
```

## Training Details

### Loss Function

Combined loss with two components:

1. **Regression Loss** (70% weight): MSE between predicted and actual final scores
2. **Ranking Loss** (30% weight): Margin-based ranking loss within epochs
   - Ensures molecules with higher scores are ranked correctly
   - Critical for winner-takes-all competitions

### Hyperparameters

**Pretraining**:
- Batch size: 64-128
- Learning rate: 3e-4
- Epochs: 20-30
- Optimizer: AdamW with weight decay 1e-5

**Adaptation**:
- Batch size: 64
- Learning rate: 1e-4
- Epochs: 3-5
- Only interaction head is trainable

### Performance Expectations

On RTX 4090:
- **VRAM Usage**: ~5-8 GB (with ESM-2 650M frozen)
- **Inference Speed**: ~1-3 ms per forward pass
- **Throughput**: ~100k molecules scored per minute
- **Training Speed**: ~100-200 samples/second

## Project Structure

```
BoltzPredictor/
├── boltzpredictor/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── molecule_encoder.py      # GINE encoder
│   │   ├── protein_encoder.py       # ESM-2 encoder with caching
│   │   └── boltz_predictor.py       # Main model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # PyTorch Dataset
│   │   └── preprocessing.py         # SMILES to graph conversion
│   └── utils/
│       ├── __init__.py
│       ├── protein_cache.py         # Embedding precomputation
│       └── losses.py                # Combined loss function
├── scripts/
│   ├── train.py                     # Training script
│   ├── inference.py                 # Inference script
│   └── precompute_proteins.py       # Precompute embeddings
├── configs/
│   ├── pretrain.yaml                # Pretraining config
│   └── adapt.yaml                   # Adaptation config
├── checkpoints/                     # Model checkpoints (created)
├── cache/                           # Protein embeddings cache (created)
├── logs/                            # TensorBoard logs (created)
├── requirements.txt
├── setup.py
└── README.md
```

## Key Design Decisions

1. **Frozen Protein Encoder**: Prevents overfitting on small datasets and enables generalization to unseen proteins
2. **Shared Affinity Model**: Single model for both target and antitarget ensures consistency
3. **Ranking Loss**: Critical for winner-takes-all competitions where top-1 accuracy matters
4. **Protein Embedding Caching**: Precomputed embeddings speed up training significantly
5. **GINE for Molecules**: Best performance per parameter for QSAR tasks

## Troubleshooting

**Out of Memory**: Reduce batch size in config file or use gradient accumulation

**Slow Training**: Ensure protein embeddings are precomputed and cached

**Poor Performance**: 
- Check data quality and label distribution
- Verify protein sequences are valid
- Try adjusting loss weights (regression vs ranking)
- Increase number of training epochs

## License

This project is designed for research and competition use.
