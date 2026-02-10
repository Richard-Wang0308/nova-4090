"""Protein sequence fetching utilities."""

import requests
import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)


def get_sequence_from_protein_code(protein_code: str) -> str:
    """
    Get the amino acid sequence for a protein code.
    First tries to fetch from UniProt API, and if that fails,
    falls back to searching the Hugging Face dataset.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{protein_code}.fasta"

    response = requests.get(url)

    if response.status_code == 200:
        lines = response.text.splitlines()
        sequence_lines = [line.strip() for line in lines if not line.startswith('>')]
        amino_acid_sequence = ''.join(sequence_lines)
        # Check if the sequence is empty
        if not amino_acid_sequence:
            logger.warning(f"Retrieved empty sequence for {protein_code} from UniProt API. Trying Hugging Face dataset.")
        else:
            return amino_acid_sequence
    
    logger.info(f"Failed to retrieve sequence for {protein_code} from UniProt API. Trying Hugging Face dataset.")
    try:
        dataset = load_dataset("Metanova/Proteins", split="train")
        
        for i in range(len(dataset)):
            if dataset[i]["Entry"] == protein_code:
                sequence = dataset[i]["Sequence"]
                logger.info(f"Found sequence for {protein_code} in Hugging Face dataset")
                return sequence
                
        logger.error(f"Could not find protein {protein_code} in Hugging Face dataset")
        return None
        
    except Exception as e:
        logger.error(f"Error accessing Hugging Face dataset: {e}")
        return None
