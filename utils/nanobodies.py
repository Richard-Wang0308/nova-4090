"""Minimal nanobody helpers for submission / HF archive checks (no heavy deps)."""

import hashlib


def normalize_seq(seq: str) -> str:
    return seq.strip().upper()


def seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def nanobody_unique_for_target_hf(target: str, sequence: str) -> bool:
    """
    True if normalized sequence hash is not in Metanova/Submission-Archive
    for this protein target (same contract as neurons/nano.py validation).
    """
    if not target or not sequence or sequence.strip() == "" or sequence == "~":
        return True
    from utils.challenge import entry_unique_for_protein_hf

    h = seq_hash(normalize_seq(sequence))
    return entry_unique_for_protein_hf(target, h, "nanobodies")
