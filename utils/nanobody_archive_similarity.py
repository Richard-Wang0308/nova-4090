"""
Validator-parity nanobody similarity vs top Submission-Archive binders.

Mirrors ``nova/neurons/validator/nanobody_validity.py`` (HF exact hash check is
elsewhere in ``utils.nanobodies``): builds ``index_top_sequences`` search engines
and rejects candidates when ``is_duplicate(match)`` for any search match.

Requires ``NOVA-nanobody-filter`` (metanano) at repo root
``<nova-4090>/NOVA-nanobody-filter``. If missing or index build fails, behaviour
matches the validator when no search engine is available (similarity skipped).
"""

from __future__ import annotations

import os
import sys
import threading
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

import bittensor as bt
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from utils.minmax_weighted_rank import rank_binders

_HERE = os.path.dirname(os.path.abspath(__file__))
_NOVA4090_ROOT = os.path.dirname(_HERE)
_FILTER_DIR = os.path.join(_NOVA4090_ROOT, "NOVA-nanobody-filter")

_metanano_path_ready = False
_engine_lock = threading.Lock()
_search_engine_cache: Dict[str, Any] = {}


def clear_archive_similarity_cache() -> None:
    """Drop cached SearchEngine objects (e.g. each submission window)."""
    with _engine_lock:
        _search_engine_cache.clear()


def _ensure_metanano_path() -> bool:
    global _metanano_path_ready
    if _metanano_path_ready:
        return True
    if not os.path.isdir(_FILTER_DIR):
        bt.logging.warning(
            f"NOVA-nanobody-filter not found at {_FILTER_DIR} — "
            "nanobody similarity vs archive top binders is disabled "
            "(same as validator when index cannot be built)."
        )
        return False
    if _FILTER_DIR not in sys.path:
        sys.path.insert(0, _FILTER_DIR)
    _metanano_path_ready = True
    return True


def is_duplicate(match: Any) -> tuple:
    """
    Same rules as ``nova/utils/nanobodies.is_duplicate`` / validator.
    """
    identity = match.identity
    cdr_sim = match.cdr_similarity

    if identity >= 0.95:
        bt.logging.debug(
            f"match is_duplicate step 1: whole-sequence identity {identity:.0%}"
        )
        return True, "near-identical sequence"

    if cdr_sim is not None:
        cdr3 = cdr_sim.get("CDR3", 0.0)
        if cdr3 >= 0.85:
            bt.logging.debug(
                f"match is_duplicate step 2.1: CDR3 identity {cdr3:.0%}"
            )
            return True, f"CDR3 identity {cdr3:.0%} (same clonotype)"
        cdr1 = cdr_sim.get("CDR1", 0.0)
        cdr2 = cdr_sim.get("CDR2", 0.0)
        if cdr3 >= 0.80 and cdr1 >= 0.90 and cdr2 >= 0.90:
            bt.logging.debug(
                "match is_duplicate step 2.2: high CDR3 with conserved CDR1/CDR2"
            )
            return True, f"CDR3={cdr3:.0%} with conserved CDR1/CDR2"

    if identity >= 0.90 and cdr_sim is None:
        bt.logging.debug(
            f"match is_duplicate step 3: identity {identity:.0%} (no CDR data)"
        )
        return True, f"whole-sequence identity {identity:.0%} (no CDR data)"

    return False, "novel"


def _compute_igblast_nativeness_batch(sequences: Dict[str, str]) -> List[Dict[str, Any]]:
    from metanano.utils import igblast_nativeness

    return igblast_nativeness.score_sequences(sequences)


def index_top_sequences(target: str, n: int = 50) -> Any:
    """
    Build SearchEngine over top-ranked archive nanobodies for ``target``.
    Same structure as ``nova/utils/nanobodies.index_top_sequences``.
    """
    if not _ensure_metanano_path():
        return None

    from metanano.config import SearchConfig
    from metanano.search import IndexManager, SearchEngine
    from metanano.utils.alignment import AlignmentEngine
    from metanano.utils.cdr_utils import extract_cdrs
    from metanano.utils.igblast_nativeness import features_to_cdrs
    from metanano.utils.kmer import generate_kmers

    search_config = SearchConfig()
    index_manager = IndexManager(search_config)
    alignment_engine = AlignmentEngine(search_config.fine_alignment)
    search_engine = SearchEngine(search_config, index_manager, alignment_engine)

    try:
        local_path = hf_hub_download(
            repo_id="Metanova/Submission-Archive",
            filename=f"{target}_nanobodies.csv",
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"),
        )
        top_sequences = pd.read_csv(local_path)
        top_sequences = rank_binders(top_sequences, k=50, max_liability_violations=50)
        top_sequences = top_sequences.head(n)[["sequence", "sequence_hash"]]
    except EntryNotFoundError:
        return None
    except Exception as e:
        bt.logging.warning(
            f"Could not load {target}_nanobodies.csv from Submission-Archive: {e}"
        )
        return None

    for seq, seq_id in top_sequences.values:
        kmers = generate_kmers(seq, k=search_config.k)
        cdrs = extract_cdrs(seq)

        if cdrs is None:
            bt.logging.warning(
                f"CDR extract (abnumber) failed for {seq_id}; trying IgBLAST fallback"
            )
            try:
                with NamedTemporaryFile(mode="w", suffix=".fasta", delete=True) as tf:
                    tf.write(f">seq_{seq_id}\n{seq}\n")
                    tf.flush()
                    batch = _compute_igblast_nativeness_batch({str(seq_id): seq})
                    if not batch:
                        continue
                    cdrs = features_to_cdrs(batch[0]["features"])
            except Exception as ex:
                bt.logging.warning(f"IgBLAST CDR fallback failed for {seq_id}: {ex}")
                continue

        if cdrs is None:
            bt.logging.warning(f"Could not obtain CDRs for archive sequence {seq_id}")
            continue

        index_manager.add_sequence(seq_id, seq, cdrs, kmers)

    return search_engine


def get_cached_search_engine(target: str) -> Any:
    if not target:
        return None
    with _engine_lock:
        if target in _search_engine_cache:
            return _search_engine_cache[target]
    try:
        eng = index_top_sequences(target)
    except Exception as e:
        bt.logging.warning(f"index_top_sequences({target!r}) failed: {e}")
        eng = None
    with _engine_lock:
        _search_engine_cache[target] = eng
    return eng


def nanobody_passes_submission_gates(
    nanobody_targets: List[str],
    sequence: str,
) -> bool:
    """
    HF exact-hash uniqueness for every target, then archive top-binder similarity
    (validator order: dedupe then similarity).
    """
    from utils.nanobodies import nanobody_unique_for_target_hf

    for t in nanobody_targets:
        if t and not nanobody_unique_for_target_hf(t, sequence):
            return False
    return nanobody_passes_archive_similarity(nanobody_targets, sequence)


def collect_nanobody_targets_for_submission(
    cfg: Any,
    challenge_params: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Ordered unique list: CLI ``nanobody_target`` first, then challenge params.
    """
    out: List[str] = []
    cli = (getattr(cfg, "nanobody_target", None) or "").strip()
    if cli:
        out.append(cli)
    if challenge_params:
        nt = challenge_params.get("nanobody_target")
        if isinstance(nt, (list, tuple)):
            for x in nt:
                xs = str(x).strip()
                if xs and xs not in out:
                    out.append(xs)
        elif nt:
            xs = str(nt).strip()
            if xs and xs not in out:
                out.append(xs)
    if not out and challenge_params:
        wt = challenge_params.get("weekly_target")
        if wt:
            xs = str(wt).strip()
            if xs and xs not in out:
                out.append(xs)
    seen: set[str] = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def nanobody_passes_archive_similarity(
    nanobody_targets: List[str],
    sequence: str,
) -> bool:
    """
    True if sequence passes similarity vs top archive binders for every target
    that has a non-None search engine (validator behaviour).

    On search errors, returns False (validator invalidates UID).
    """
    from utils.nanobodies import normalize_seq

    norm = normalize_seq(sequence)
    if "~" in norm:
        return False

    for target in nanobody_targets:
        if not target:
            continue
        engine = get_cached_search_engine(target)
        if engine is None:
            continue
        try:
            result = engine.search(
                norm,
                include_alignment=True,
                exclude_ids=None,
                coarse_min_shared=None,
                coarse_jaccard=None,
            )
        except Exception as e:
            bt.logging.warning(
                f"Similarity search failed for target={target!r}: {e}"
            )
            return False
        if any(is_duplicate(m)[0] for m in result.matches):
            bt.logging.debug(
                f"Rejecting sequence (too similar to archive top) target={target!r}"
            )
            return False
    return True
