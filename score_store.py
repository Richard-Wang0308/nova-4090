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
    -- COALESCE so a writer that passes round_no=None (e.g. a backfill that is
    -- only correcting scores) leaves the molecule's original round/iteration
    -- intact instead of stamping it with a meaningless one.
    iteration=COALESCE(excluded.iteration, scored_molecules.iteration),
    target_key=excluded.target_key,
    target_label=excluded.target_label,
    rxn_id=excluded.rxn_id,
    smiles=excluded.smiles,
    inchikey=excluded.inchikey,
    source=excluded.source,
    round=COALESCE(excluded.round, scored_molecules.round),
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
    round_no: Optional[int] = 0,
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
    argument. Pass ``round_no=None`` to leave an existing row's round/iteration
    untouched -- for writers that are only correcting a score. Returns the
    number of rows written.
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

        stamp = None if round_no is None else int(round_no)
        rows.append(
            (
                name, score, True, stamp,
                target_key, target_label, row_rxn,
                smiles, r.get("inchikey") or inchikey(smiles),
                row_source,
                stamp,
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


# =============================================================================
# Replicate scoring / variance tracking
#
# A Boltz score is a *draw*, not a fixed property of a molecule: boltz.predict
# calls seed_everything(seed) once and then consumes the RNG sequentially as
# records stream through the dataloader (input files are picked up with
# glob("*")). A molecule's noise therefore depends on the seed AND on which
# other molecules share its run and in what order. Validators score a
# different pool, so they draw different noise for the same molecule.
#
# Consequence: picking the top-20 out of a large pool scored once each selects
# the molecules with the luckiest draws (winner's curse), and those regress to
# the mean when the validator re-draws. The cure is to average several
# independent draws per molecule and rank on the estimate, not on one draw.
# =============================================================================

REPLICATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS molecule_replicates (
    molecule_name TEXT NOT NULL,
    seed          INTEGER NOT NULL,
    -- Confirmation re-scores at one fixed seed, so the seed alone cannot key
    -- separate draws; draw_idx does. Variance tooling that varies the seed
    -- simply leaves this at 0.
    draw_idx      INTEGER NOT NULL DEFAULT 0,
    score         REAL,
    affinity_probability_binary  REAL,
    affinity_pred_value          REAL,
    affinity_probability_binary1 REAL,
    affinity_probability_binary2 REAL,
    affinity_pred_value1         REAL,
    affinity_pred_value2         REAL,
    confidence_score REAL,
    ligand_iptm      REAL,
    complex_plddt    REAL,
    iptm             REAL,
    ptm              REAL,
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (molecule_name, seed, draw_idx)
)
"""

# Consensus columns added to scored_molecules. `score` keeps holding whatever
# submit.py should rank on; `score_single` preserves the original one-shot draw.
CONSENSUS_COLUMNS: Dict[str, str] = {
    "score_single": "REAL",   # original single-draw score, never overwritten
    "mu": "REAL",             # mean over replicates
    "sigma": "REAL",          # std over replicates (population spread of a draw)
    "sem": "REAL",            # standard error of the mean
    "n_reps": "INTEGER",      # replicate count
    "lcb": "REAL",            # mu - lambda*sigma, the risk-adjusted rank key
    "ens_disagree": "REAL",   # |head1 - head2| ensemble disagreement (free proxy)
    "ligand_iptm": "REAL",    # pose confidence (free proxy)
    "confidence_score": "REAL",
    # TRUE once the molecule has TOTAL_DRAWS confirmation draws on record, so
    # later rounds skip it instead of paying to re-score it forever.
    "rescored": "BOOLEAN DEFAULT FALSE",
}

_REPLICATE_UPSERT_SQL = """
INSERT INTO molecule_replicates
(molecule_name, seed, draw_idx, score,
 affinity_probability_binary, affinity_pred_value,
 affinity_probability_binary1, affinity_probability_binary2,
 affinity_pred_value1, affinity_pred_value2,
 confidence_score, ligand_iptm, complex_plddt, iptm, ptm)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(molecule_name, seed, draw_idx) DO UPDATE SET
    score=excluded.score,
    affinity_probability_binary=excluded.affinity_probability_binary,
    affinity_pred_value=excluded.affinity_pred_value,
    affinity_probability_binary1=excluded.affinity_probability_binary1,
    affinity_probability_binary2=excluded.affinity_probability_binary2,
    affinity_pred_value1=excluded.affinity_pred_value1,
    affinity_pred_value2=excluded.affinity_pred_value2,
    confidence_score=excluded.confidence_score,
    ligand_iptm=excluded.ligand_iptm,
    complex_plddt=excluded.complex_plddt,
    iptm=excluded.iptm,
    ptm=excluded.ptm,
    scored_at=CURRENT_TIMESTAMP
"""

_COMPONENT_FIELDS = [
    "affinity_probability_binary", "affinity_pred_value",
    "affinity_probability_binary1", "affinity_probability_binary2",
    "affinity_pred_value1", "affinity_pred_value2",
    "confidence_score", "ligand_iptm", "complex_plddt", "iptm", "ptm",
]


def _migrate_replicates_add_draw_idx(conn: sqlite3.Connection) -> None:
    """
    Older DBs keyed molecule_replicates on (molecule_name, seed). SQLite cannot
    ALTER a primary key, so rebuild the table and carry existing rows over as
    draw_idx 0. No-op when the table is already current or absent.
    """
    cur = conn.cursor()
    exists = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='molecule_replicates'"
    ).fetchone()
    if not exists:
        return
    cols = {r[1] for r in cur.execute("PRAGMA table_info(molecule_replicates)").fetchall()}
    if "draw_idx" in cols:
        return
    log.info("Migrating molecule_replicates to key (molecule_name, seed, draw_idx)")
    cur.execute("ALTER TABLE molecule_replicates RENAME TO molecule_replicates_old")
    cur.execute(REPLICATE_TABLE_SQL)
    carried = sorted(cols & {
        "molecule_name", "seed", "score", "scored_at", *_COMPONENT_FIELDS
    })
    collist = ", ".join(carried)
    cur.execute(
        f"INSERT INTO molecule_replicates ({collist}, draw_idx) "
        f"SELECT {collist}, 0 FROM molecule_replicates_old"
    )
    cur.execute("DROP TABLE molecule_replicates_old")
    conn.commit()


def init_variance_tables(db_path: str) -> None:
    """Add the replicate table and consensus columns (idempotent)."""
    with connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(REPLICATE_TABLE_SQL)
        _migrate_replicates_add_draw_idx(conn)
        cur = conn.cursor()
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_rep_name "
            "ON molecule_replicates(molecule_name)"
        )
        cur.execute(BASE_TABLE_SQL)
        cols = {r[1] for r in cur.execute("PRAGMA table_info(scored_molecules)").fetchall()}
        for name, sqltype in CONSENSUS_COLUMNS.items():
            if name not in cols:
                cur.execute(f"ALTER TABLE scored_molecules ADD COLUMN {name} {sqltype}")
        conn.commit()


def _f(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def record_replicates(
    db_path: str,
    rows: Sequence[Dict[str, Any]],
) -> int:
    """
    Persist one draw per (molecule, seed).

    Each row: {'name', 'seed', 'score', optional 'draw_idx', **components}. Components come
    straight from BoltzWrapper.per_molecule_components, which every miner
    currently computes and then throws away.
    """
    if not rows:
        return 0
    payload = []
    for r in rows:
        name, seed = r.get("name"), r.get("seed")
        if not name or seed is None:
            continue
        payload.append(
            (name, int(seed), int(r.get("draw_idx", 0) or 0), _f(r.get("score")))
            + tuple(_f(r.get(f)) for f in _COMPONENT_FIELDS)
        )
    if not payload:
        return 0
    try:
        with connect(db_path) as conn:
            conn.executemany(_REPLICATE_UPSERT_SQL, payload)
            conn.commit()
    except sqlite3.Error as e:
        log.error("Error recording replicates in %s: %s", db_path, e)
        return 0
    return len(payload)


def replicate_stats(db_path: str, rxn_id: Optional[int] = None) -> pd.DataFrame:
    """Per-molecule replicate summary: n_reps, mu, sigma, sem, ens_disagree."""
    empty = pd.DataFrame(
        columns=["name", "n_reps", "mu", "sigma", "sem", "ens_disagree",
                 "ligand_iptm", "confidence_score"]
    )
    if not os.path.exists(db_path):
        return empty

    query = """
        SELECT molecule_name, score,
               affinity_probability_binary1, affinity_probability_binary2,
               ligand_iptm, confidence_score
        FROM molecule_replicates
        WHERE score IS NOT NULL
    """
    params: Tuple[Any, ...] = ()
    if rxn_id is not None:
        query += " AND molecule_name LIKE ?"
        params = (f"rxn:{rxn_id}:%",)
    try:
        with connect(db_path) as conn:
            rows = conn.execute(query, params).fetchall()
    except sqlite3.Error as e:
        log.error("replicate_stats: %s", e)
        return empty
    if not rows:
        return empty

    df = pd.DataFrame(
        rows,
        columns=["name", "score", "apb1", "apb2", "ligand_iptm", "confidence_score"],
    )
    df["ens_disagree"] = (df["apb1"] - df["apb2"]).abs()

    grp = df.groupby("name")
    out = pd.DataFrame({
        "n_reps": grp["score"].size(),
        "mu": grp["score"].mean(),
        # ddof=1 needs >=2 draws; a single draw has no measurable spread yet.
        "sigma": grp["score"].std(ddof=1),
        "ens_disagree": grp["ens_disagree"].mean(),
        "ligand_iptm": grp["ligand_iptm"].mean(),
        "confidence_score": grp["confidence_score"].mean(),
    }).reset_index()
    out["sigma"] = out["sigma"].fillna(0.0)
    out["sem"] = out["sigma"] / np.sqrt(out["n_reps"].clip(lower=1))
    return out[["name", "n_reps", "mu", "sigma", "sem", "ens_disagree",
                "ligand_iptm", "confidence_score"]]


def commit_consensus(
    db_path: str,
    stats: pd.DataFrame,
    lambda_sigma: float = 0.5,
    rank_on: str = "lcb",
) -> int:
    """
    Write consensus stats onto scored_molecules and re-point `score` at the
    replicate-based estimate.

    `score_single` keeps the original one-shot draw the first time a molecule
    is committed, so nothing is lost. Because submit.py ranks on `score`, this
    is what makes the robust set the one that actually gets submitted, with no
    change to submit.py itself.

    rank_on: 'lcb' (mu - lambda*sigma), 'mu', or 'none' (stats only).
    """
    if stats is None or stats.empty:
        return 0

    updates = []
    for r in stats.itertuples(index=False):
        mu = _f(r.mu)
        if mu is None:
            continue
        sigma = _f(r.sigma) or 0.0
        sem = _f(r.sem) or 0.0
        lcb = mu - lambda_sigma * sigma
        if rank_on == "lcb":
            new_score = lcb
        elif rank_on == "mu":
            new_score = mu
        else:
            new_score = None
        updates.append((
            mu, sigma, sem, int(r.n_reps), lcb,
            _f(r.ens_disagree), _f(r.ligand_iptm), _f(r.confidence_score),
            new_score, new_score, r.name,
        ))

    if not updates:
        return 0

    try:
        with connect(db_path) as conn:
            conn.executemany(
                """
                UPDATE scored_molecules
                SET score_single = COALESCE(score_single, score),
                    mu = ?, sigma = ?, sem = ?, n_reps = ?, lcb = ?,
                    ens_disagree = ?, ligand_iptm = ?, confidence_score = ?,
                    score = CASE WHEN ? IS NULL THEN score ELSE ? END
                WHERE molecule_name = ?
                """,
                updates,
            )
            conn.commit()
    except sqlite3.Error as e:
        log.error("commit_consensus: %s", e)
        return 0
    return len(updates)


def load_consensus(db_path: str, rxn_id: Optional[int] = None) -> pd.DataFrame:
    """Scored molecules joined with their consensus columns."""
    empty = pd.DataFrame(columns=[
        "name", "smiles", "inchikey", "score", "score_single",
        "mu", "sigma", "sem", "n_reps", "lcb", "ens_disagree",
        "ligand_iptm", "confidence_score",
    ])
    if not os.path.exists(db_path):
        return empty
    query = """
        SELECT molecule_name, smiles, inchikey, score, score_single,
               mu, sigma, sem, n_reps, lcb, ens_disagree,
               ligand_iptm, confidence_score
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
        log.error("load_consensus: %s", e)
        return empty
    return pd.DataFrame(rows, columns=list(empty.columns))


def deflate_unmeasured(
    db_path: str,
    delta: float,
    rxn_id: Optional[int] = None,
) -> int:
    """
    Put un-replicated molecules on the same scale as replicated ones.

    Committing consensus replaces `score` with mu (or mu - lambda*sigma) for the
    molecules you re-scored. Those numbers are honest, and therefore LOWER than
    a lucky single draw. If the rest of the pool keeps its inflated one-shot
    scores, the molecules you never checked float to the top and submit.py picks
    exactly the unverified ones — the opposite of what you wanted.

    So subtract the expected winner's-curse premium `delta` from every molecule
    that has no replicates. It is a uniform shift, so their order is untouched,
    but they stop out-ranking measured molecules for free.

    Idempotent: always recomputed from `score_single`, never from the current
    `score`, so running it repeatedly does not compound.
    """
    if delta <= 0:
        return 0
    where = "(n_reps IS NULL OR n_reps < 1)"
    params: List[Any] = [float(delta)]
    if rxn_id is not None:
        where += " AND molecule_name LIKE ?"
        params.append(f"rxn:{rxn_id}:%")
    try:
        with connect(db_path) as conn:
            cur = conn.execute(
                f"""
                UPDATE scored_molecules
                SET score_single = COALESCE(score_single, score),
                    score = COALESCE(score_single, score) - ?
                WHERE {where}
                """,
                params,
            )
            conn.commit()
            return cur.rowcount
    except sqlite3.Error as e:
        log.error("deflate_unmeasured: %s", e)
        return 0


def revert_consensus(db_path: str, rxn_id: Optional[int] = None) -> int:
    """
    Undo commit_consensus/deflate_unmeasured: restore `score` from `score_single`.

    Consensus rewrites the column submit.py ranks on, so there has to be a way
    back. The replicate measurements themselves are kept — only the ranking key
    is restored.
    """
    where = "score_single IS NOT NULL"
    params: List[Any] = []
    if rxn_id is not None:
        where += " AND molecule_name LIKE ?"
        params.append(f"rxn:{rxn_id}:%")
    try:
        with connect(db_path) as conn:
            cur = conn.execute(
                f"UPDATE scored_molecules SET score = score_single WHERE {where}",
                params,
            )
            conn.commit()
            return cur.rowcount
    except sqlite3.Error as e:
        log.error("revert_consensus: %s", e)
        return 0


def existing_seeds(db_path: str, rxn_id: Optional[int] = None) -> Set[int]:
    """
    Seeds already recorded in molecule_replicates.

    Replicates are keyed (molecule_name, seed), so re-running with the same
    seed sequence overwrites the previous draws instead of adding to them --
    you would believe you had 2K replicates and actually have K. Callers use
    this to pick fresh seeds when topping a molecule up.
    """
    if not os.path.exists(db_path):
        return set()
    query = "SELECT DISTINCT seed FROM molecule_replicates"
    params: Tuple[Any, ...] = ()
    if rxn_id is not None:
        query += " WHERE molecule_name LIKE ?"
        params = (f"rxn:{rxn_id}:%",)
    try:
        with connect(db_path) as conn:
            return {int(r[0]) for r in conn.execute(query, params).fetchall()}
    except sqlite3.Error:
        return set()


# =============================================================================
# Confirmation bookkeeping (rescored flag)
# =============================================================================

def replicate_counts(
    db_path: str,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """How many draws each molecule already has on record."""
    if not os.path.exists(db_path):
        return {}
    out: Dict[str, int] = {}
    try:
        with connect(db_path) as conn:
            if names is None:
                rows = conn.execute(
                    "SELECT molecule_name, COUNT(*) FROM molecule_replicates "
                    "WHERE score IS NOT NULL GROUP BY molecule_name"
                ).fetchall()
                out.update({n: int(c) for n, c in rows})
            else:
                todo = list(names)
                for i in range(0, len(todo), 900):
                    chunk = todo[i:i + 900]
                    ph = ",".join("?" * len(chunk))
                    rows = conn.execute(
                        f"SELECT molecule_name, COUNT(*) FROM molecule_replicates "
                        f"WHERE score IS NOT NULL AND molecule_name IN ({ph}) "
                        f"GROUP BY molecule_name",
                        chunk,
                    ).fetchall()
                    out.update({n: int(c) for n, c in rows})
    except sqlite3.Error as e:
        log.debug("replicate_counts: %s", e)
    return out


def load_rescored_names(db_path: str, rxn_id: Optional[int] = None) -> Set[str]:
    """Molecules already confirmed — never re-score these again."""
    if not os.path.exists(db_path):
        return set()
    query = "SELECT molecule_name FROM scored_molecules WHERE rescored=TRUE"
    params: Tuple[Any, ...] = ()
    if rxn_id is not None:
        query += " AND molecule_name LIKE ?"
        params = (f"rxn:{rxn_id}:%",)
    try:
        with connect(db_path) as conn:
            return {r[0] for r in conn.execute(query, params).fetchall()}
    except sqlite3.Error as e:
        log.debug("load_rescored_names: %s", e)
        return set()


def mark_rescored(db_path: str, names: Iterable[str]) -> int:
    """Flag molecules as fully confirmed so future rounds skip them."""
    todo = list(names)
    if not todo:
        return 0
    try:
        with connect(db_path) as conn:
            conn.executemany(
                "UPDATE scored_molecules SET rescored=TRUE WHERE molecule_name=?",
                [(n,) for n in todo],
            )
            conn.commit()
        return len(todo)
    except sqlite3.Error as e:
        log.error("mark_rescored: %s", e)
        return 0


def replicate_scores(
    db_path: str,
    names: Iterable[str],
) -> Dict[str, List[float]]:
    """
    Every recorded draw per molecule, oldest first.

    Confirmation seeds each molecule's running list from this so a run that was
    interrupted resumes correctly: it tops up to the target count instead of
    starting over, and the average covers every draw on record rather than just
    the ones taken in the current round.
    """
    todo = list(names)
    out: Dict[str, List[float]] = {}
    if not todo or not os.path.exists(db_path):
        return out
    try:
        with connect(db_path) as conn:
            for i in range(0, len(todo), 900):
                chunk = todo[i:i + 900]
                ph = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT molecule_name, score FROM molecule_replicates "
                    f"WHERE score IS NOT NULL AND molecule_name IN ({ph}) "
                    f"ORDER BY molecule_name, seed, draw_idx",
                    chunk,
                ).fetchall()
                for n, v in rows:
                    out.setdefault(n, []).append(float(v))
    except sqlite3.Error as e:
        log.debug("replicate_scores: %s", e)
    return out
