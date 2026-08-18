"""
score_store.py — single canonical layer for score_results_{rxn}.sqlite.

orchestrator.py's ``ScoreStore`` defines the on-disk contract for the shared
score database. Every other search script (miner/miner.py,
neurons/crossover.py, neurons/genetic.py, neurons/component_exhaust.py,
neurons/late_stage_search.py) historically created and wrote a narrower
4-column table, which meant:

  * rows they wrote had NULL smiles/inchikey and were therefore invisible to
    orchestrator's ``ScoreStore.dataframe()`` until it paid to re-resolve and
    re-validate them (and rows failing validation were dropped entirely);
  * rows had no target_key, so the target-safety guard could not tell which
    protein they belonged to;
  * crossover.py / genetic.py used a bare ``INSERT`` instead of an upsert, so
    a single already-present molecule_name raised UNIQUE and silently threw
    away the whole batch of fresh Boltz scores.

This module reproduces orchestrator's schema, target handling and upsert
semantics exactly, so all writers agree on one format.

Canonical ``scored_molecules`` columns (order matters for readers):

    molecule_name TEXT PRIMARY KEY
    score         REAL NOT NULL
    scored_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    available     BOOLEAN DEFAULT TRUE
    iteration     INTEGER
    target_key    TEXT
    target_label  TEXT
    rxn_id        INTEGER
    smiles        TEXT
    inchikey      TEXT
    source        TEXT
    round         INTEGER

plus a ``metadata(key, value)`` table holding ``active_target_key`` /
``active_target_label``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

log = logging.getLogger("score_store")

# Column name -> SQL type, in canonical order. Kept identical to
# orchestrator.ScoreStore._ensure_columns so a DB touched by either writer is
# byte-for-byte compatible with the other.
BASE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scored_molecules (
    molecule_name TEXT PRIMARY KEY,
    score REAL NOT NULL,
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    available BOOLEAN DEFAULT TRUE,
    iteration INTEGER
)
"""

EXTRA_COLUMNS: Dict[str, str] = {
    # miner/miner.py writes `iteration` and never `round`; keep both so the
    # writers stay interoperable on a shared DB.
    "iteration": "INTEGER",
    "target_key": "TEXT",
    "target_label": "TEXT",
    "rxn_id": "INTEGER",
    "smiles": "TEXT",
    "inchikey": "TEXT",
    "source": "TEXT",
    "round": "INTEGER",
}

_UPSERT_SQL = """
INSERT INTO scored_molecules
(molecule_name,score,available,iteration,target_key,target_label,
 rxn_id,smiles,inchikey,source,round)
VALUES (?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(molecule_name) DO UPDATE SET
    score=excluded.score,
    available=excluded.available,
    iteration=excluded.iteration,
    target_key=excluded.target_key,
    target_label=excluded.target_label,
    rxn_id=excluded.rxn_id,
    smiles=excluded.smiles,
    inchikey=excluded.inchikey,
    source=excluded.source,
    round=excluded.round,
    scored_at=CURRENT_TIMESTAMP
"""


# =============================================================================
# Paths / target identity
# =============================================================================

def score_db_path(rxn_id: int, base_dir: str = BASE_DIR) -> str:
    """Canonical path used by orchestrator.ScoreStore for this reaction."""
    return os.path.join(base_dir, f"score_results_{rxn_id}.sqlite")


def target_identity(config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Byte-identical to orchestrator.target_identity so the same config always
    yields the same target_key and the archive guard never false-trips.
    """
    targets = config.get("small_molecule_target") or []
    clips = config.get("small_molecule_target_clip_interval") or []
    payload = json.dumps({"targets": targets, "clips": clips}, sort_keys=True)
    key = hashlib.sha256(payload.encode()).hexdigest()
    label = "_".join(map(str, targets)) if targets else "unknown"
    return key, label


# =============================================================================
# Molecule helpers
# =============================================================================

_smiles_cache: Dict[str, Optional[str]] = {}
_inchikey_cache: Dict[str, str] = {}


def resolve_smiles(name: str) -> Optional[str]:
    """Build the product SMILES for a `rxn:id:a:b[:c]` name (cached)."""
    if name in _smiles_cache:
        return _smiles_cache[name]
    smiles = None
    try:
        from combinatorial_db.reactions import get_smiles_from_reaction

        smiles = get_smiles_from_reaction(name)
    except Exception:
        smiles = None
    _smiles_cache[name] = smiles
    return smiles


def inchikey(smiles: str) -> str:
    """InChIKey via the same call orchestrator uses (Chem.MolToInchiKey)."""
    if not smiles:
        return ""
    cached = _inchikey_cache.get(smiles)
    if cached is not None:
        return cached
    try:
        mol = Chem.MolFromSmiles(smiles)
        key = Chem.MolToInchiKey(mol) if mol is not None else ""
    except Exception:
        key = ""
    _inchikey_cache[smiles] = key
    return key


def parse_rxn_id(name: str) -> Optional[int]:
    parts = str(name).split(":")
    if len(parts) < 4 or parts[0] != "rxn":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


# =============================================================================
# Connection / schema
# =============================================================================

def connect(db_path: str) -> sqlite3.Connection:
    """WAL + NORMAL sync, matching orchestrator.ScoreStore._connect."""
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_columns(conn: sqlite3.Connection) -> Set[str]:
    cur = conn.cursor()
    cur.execute(BASE_TABLE_SQL)
    cols = {r[1] for r in cur.execute("PRAGMA table_info(scored_molecules)").fetchall()}
    for name, sqltype in EXTRA_COLUMNS.items():
        if name not in cols:
            cur.execute(f"ALTER TABLE scored_molecules ADD COLUMN {name} {sqltype}")
            cols.add(name)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()
    return cols


def init_score_results_db(
    db_path: str,
    rxn_id: Optional[int] = None,
    target_key: Optional[str] = None,
    target_label: Optional[str] = None,
) -> None:
    """
    Create/migrate the DB to the canonical schema and stamp the active target.

    Target handling mirrors orchestrator.ScoreStore._init:
      * no metadata yet -> stamp the current target and backfill existing rows;
      * metadata matches -> nothing to do;
      * metadata differs -> archive the live rows inside the same file and
        clear scored_molecules for the new target.

    Pass target_key/target_label (from ``target_identity(config)``) whenever the
    caller knows the target; omit them to only ensure the schema.
    """
    with connect(db_path) as conn:
        cols = _ensure_columns(conn)

        if not target_key:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scored_score "
                "ON scored_molecules(score DESC)"
            )
            conn.commit()
            return

        row = conn.execute(
            "SELECT value FROM metadata WHERE key='active_target_key'"
        ).fetchone()
        active = row[0] if row else None
        count = conn.execute("SELECT COUNT(*) FROM scored_molecules").fetchone()[0]

        if active is None:
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES('active_target_key',?)",
                (target_key,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES('active_target_label',?)",
                (target_label,),
            )
            if count:
                # legacy DBs may predate either `iteration` or `round`
                round_clause = (
                    ", round=COALESCE(round,iteration)" if "iteration" in cols else ""
                )
                conn.execute(
                    f"""
                    UPDATE scored_molecules
                    SET target_key=COALESCE(target_key,?),
                        target_label=COALESCE(target_label,?),
                        rxn_id=COALESCE(rxn_id,?){round_clause}
                    """,
                    (target_key, target_label, rxn_id),
                )
            conn.commit()

        elif active != target_key:
            stamp = int(time.time())
            archive = f"scored_molecules_archive_{stamp}"
            conn.execute(f"CREATE TABLE {archive} AS SELECT * FROM scored_molecules")
            conn.execute("DELETE FROM scored_molecules")
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES('active_target_key',?)",
                (target_key,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES('active_target_label',?)",
                (target_label,),
            )
            conn.commit()
            log.warning(
                "Target changed. Archived old rows to %s inside %s and reset live table.",
                archive, db_path,
            )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON scored_molecules(score)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scored_score "
            "ON scored_molecules(score DESC)"
        )
        conn.commit()


def active_target(db_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (active_target_key, active_target_label) recorded in the DB."""
    if not os.path.exists(db_path):
        return None, None
    try:
        with connect(db_path) as conn:
            rows = dict(
                conn.execute(
                    "SELECT key, value FROM metadata "
                    "WHERE key IN ('active_target_key','active_target_label')"
                ).fetchall()
            )
        return rows.get("active_target_key"), rows.get("active_target_label")
    except sqlite3.Error:
        return None, None


# =============================================================================
# Writing
# =============================================================================

def write_scores_to_db(
    db_path: str,
    records: Sequence[Dict[str, Any]],
    rxn_id: Optional[int] = None,
    round_no: int = 0,
    target_key: Optional[str] = None,
    target_label: Optional[str] = None,
    source: str = "search",
) -> int:
    """
    Upsert scored molecules using orchestrator's exact column set.

    Each record needs ``name`` plus a score under ``boltz_score`` or ``score``.
    ``smiles`` is taken from the record when present and otherwise rebuilt from
    the name, so the column is never left NULL for orchestrator to backfill.
    Per-record ``source`` / ``generation_method`` overrides the ``source``
    argument. Returns the number of rows written.
    """
    if not records:
        return 0

    rows: List[Tuple[Any, ...]] = []
    for r in records:
        name = r.get("name")
        if not name:
            continue

        score = r.get("boltz_score", r.get("score"))
        if score is None:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(score):
            continue

        smiles = r.get("smiles") or resolve_smiles(name)
        if not smiles:
            continue

        row_rxn = rxn_id if rxn_id is not None else parse_rxn_id(name)
        row_source = str(r.get("source") or r.get("generation_method") or source)

        rows.append(
            (
                name, score, True, int(round_no),
                target_key, target_label, row_rxn,
                smiles, r.get("inchikey") or inchikey(smiles),
                row_source,
                int(round_no),
            )
        )

    if not rows:
        return 0

    try:
        with connect(db_path) as conn:
            conn.executemany(_UPSERT_SQL, rows)
            conn.commit()
    except sqlite3.Error as e:
        log.error("Error writing scores to %s: %s", db_path, e)
        return 0

    return len(rows)


# =============================================================================
# Reading
# =============================================================================

def load_all_scored(
    db_path: str,
    rxn_id: Optional[int] = None,
    backfill: bool = True,
) -> pd.DataFrame:
    """
    Load scored rows in orchestrator's ``dataframe()`` shape:
    columns [name, smiles, inchikey, score, source, round].

    Uses the stored ``smiles`` column and, when ``backfill`` is set, resolves
    and writes back any row that predates it — the same repair orchestrator
    performs, so both writers converge on fully populated rows.
    """
    empty = pd.DataFrame(
        columns=["name", "smiles", "inchikey", "score", "source", "round"]
    )
    if not os.path.exists(db_path):
        return empty

    query = """
        SELECT molecule_name, smiles, inchikey, score,
               COALESCE(source,'legacy') AS source,
               COALESCE(round,iteration,0) AS round
        FROM scored_molecules
    """
    params: Tuple[Any, ...] = ()
    if rxn_id is not None:
        query += " WHERE molecule_name LIKE ?"
        params = (f"rxn:{rxn_id}:%",)
    query += " ORDER BY score DESC"

    try:
        with connect(db_path) as conn:
            rows = conn.execute(query, params).fetchall()
    except sqlite3.Error as e:
        log.error("Error reading %s: %s", db_path, e)
        return empty

    df = pd.DataFrame(
        rows, columns=["name", "smiles", "inchikey", "score", "source", "round"]
    )
    if df.empty:
        return empty

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[np.isfinite(df["score"])].dropna(subset=["score"]).reset_index(drop=True)
    if df.empty:
        return empty

    missing = df["smiles"].isna() | (df["smiles"].astype(str) == "")
    if missing.any():
        updates = []
        for idx in df.index[missing]:
            name = df.at[idx, "name"]
            smiles = resolve_smiles(name)
            if not smiles:
                continue
            ik = inchikey(smiles)
            df.at[idx, "smiles"] = smiles
            df.at[idx, "inchikey"] = ik
            updates.append((smiles, ik, name))
        if updates and backfill:
            try:
                with connect(db_path) as conn:
                    conn.executemany(
                        "UPDATE scored_molecules SET smiles=?, inchikey=? "
                        "WHERE molecule_name=?",
                        updates,
                    )
                    conn.commit()
                log.info("Backfilled smiles/inchikey for %d legacy rows", len(updates))
            except sqlite3.Error as e:
                log.warning("Could not backfill legacy SMILES: %s", e)

    df = df[df["smiles"].notna() & (df["smiles"].astype(str) != "")]
    return df.reset_index(drop=True)


def load_scored_name_set(db_path: str, rxn_id: Optional[int] = None) -> Set[str]:
    """Every molecule_name already scored (optionally limited to one rxn)."""
    if not os.path.exists(db_path):
        return set()
    query = "SELECT molecule_name FROM scored_molecules"
    params: Tuple[Any, ...] = ()
    if rxn_id is not None:
        query += " WHERE molecule_name LIKE ?"
        params = (f"rxn:{rxn_id}:%",)
    try:
        with connect(db_path) as conn:
            return {r[0] for r in conn.execute(query, params).fetchall()}
    except sqlite3.Error as e:
        log.debug("load_scored_name_set: %s", e)
        return set()


def batch_get_scores_from_db(
    db_path: str,
    molecule_names: Iterable[str],
) -> Dict[str, float]:
    """Look up existing scores by name, chunked under SQLite's variable limit."""
    names = list(molecule_names)
    if not names or not os.path.exists(db_path):
        return {}
    out: Dict[str, float] = {}
    try:
        with connect(db_path) as conn:
            for i in range(0, len(names), 900):
                chunk = names[i : i + 900]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT molecule_name, score FROM scored_molecules "
                    f"WHERE molecule_name IN ({placeholders})",
                    chunk,
                ).fetchall()
                out.update({n: float(s) for n, s in rows if s is not None})
    except sqlite3.Error as e:
        log.debug("batch_get_scores_from_db: %s", e)
    return out


def get_score_from_db(db_path: str, molecule_name: str) -> Optional[float]:
    """Single-name score lookup."""
    return batch_get_scores_from_db(db_path, [molecule_name]).get(molecule_name)


def count_scored(db_path: str, rxn_id: Optional[int] = None) -> int:
    """Row count, optionally limited to one reaction."""
    if not db_path or not os.path.exists(db_path):
        return 0
    query = "SELECT COUNT(*) FROM scored_molecules"
    params: Tuple[Any, ...] = ()
    if rxn_id is not None:
        query += " WHERE molecule_name LIKE ?"
        params = (f"rxn:{rxn_id}:%",)
    try:
        with connect(db_path) as conn:
            return int(conn.execute(query, params).fetchone()[0])
    except sqlite3.Error:
        return 0
