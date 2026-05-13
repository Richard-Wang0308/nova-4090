#!/usr/bin/env python3
"""
NANOBODY DB — Flask API Server
================================
Serves ``nanobodies.sqlite`` over HTTP. ``neurons/submit.py`` only uses:

  GET /health  — startup probe (no bearer token)
  GET /top     — submission window; ``target``, ``n``, ``available_only=true``
                 (Bearer if the server was started with ``--token``)

Other routes below are optional for operators / scripts.

Endpoints:
  GET /                              — JSON index of all routes (no auth)
  GET /health                        — liveness check
  GET /count                         — total row count
  GET /top?target=Q9NZQ7&n=50                    — top-N scored sequences
  GET /top?target=Q9NZQ7&n=50&available_only=true — only rows with available=TRUE (submit miner)
  GET /pending?target=Q9NZQ7         — sequences awaiting BoltzGen
  GET /stats?target=Q9NZQ7           — per-target summary stats
  GET /sequence?seq=EVQLVE...        — look up one sequence
  GET /all?target=Q9NZQ7&limit=1000  — dump all scored rows

Usage:
  python3 serve_db.py
  python3 serve_db.py --port 5001 --db /path/to/nanobodies.sqlite
  python3 serve_db.py --host 0.0.0.0 --port 5001 --token mysecrettoken
"""

import os
import sys
import sqlite3
import argparse
import logging
import time
from functools import wraps
from datetime import datetime

from flask import Flask, jsonify, request, g

# ══════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════

NOVA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DEFAULT_DB_PATH = os.path.join(NOVA_DIR, "nanobodies.sqlite")

# ══════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("nanobody-api")

# ══════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_db() -> sqlite3.Connection:
    """
    Return a per-request SQLite connection (read-only).
    Stored on Flask's g object so it's closed after each request.
    """
    if "db" not in g:
        db_path = app.config["DB_PATH"]
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DB not found: {db_path}")
        # uri=True + ?mode=ro opens in read-only mode — safe for concurrent nano.py writes
        g.db = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def query(sql: str, params: tuple = ()) -> list[dict]:
    conn = get_db()
    c    = conn.cursor()
    c.execute(sql, params)
    return [dict(r) for r in c.fetchall()]


def query_one(sql: str, params: tuple = ()) -> dict | None:
    conn = get_db()
    c    = conn.cursor()
    c.execute(sql, params)
    row = c.fetchone()
    return dict(row) if row else None


def nanobodies_has_available_column() -> bool:
    """True if ``nanobodies`` has an ``available`` column (submit gating)."""
    try:
        row = query_one(
            "SELECT 1 AS ok FROM pragma_table_info('nanobodies') "
            "WHERE name = 'available' LIMIT 1"
        )
        return row is not None
    except Exception:
        return False


def _parse_bool_query(name: str, default: bool = False) -> bool:
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _normalize_available_in_rows(rows: list[dict]) -> None:
    """Coerce ``available`` to JSON bool when present."""
    for r in rows:
        if "available" not in r or r["available"] is None:
            continue
        v = r["available"]
        if isinstance(v, (int, float)):
            r["available"] = bool(int(v))
        elif isinstance(v, str):
            r["available"] = v.strip().lower() in ("1", "true", "yes")
        else:
            r["available"] = bool(v)


# ══════════════════════════════════════════════════════════════════════════
# AUTH (optional bearer token)
# ══════════════════════════════════════════════════════════════════════════

def require_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = app.config.get("API_TOKEN")
        if token:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {token}":
                return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════════════════
# REQUEST LOGGING MIDDLEWARE
# ══════════════════════════════════════════════════════════════════════════

@app.before_request
def _before():
    g.t0 = time.time()


@app.after_request
def _after(response):
    elapsed = (time.time() - g.get("t0", time.time())) * 1000
    log.info(f"  {request.method} {request.path}  "
             f"args={dict(request.args)}  "
             f"→ {response.status_code}  "
             f"({elapsed:.1f}ms)")
    return response


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Liveness check — also returns DB row count and last write time."""
    try:
        row = query_one(
            "SELECT COUNT(*) as total, "
            "       MAX(created_at) as last_write "
            "FROM nanobodies"
        )
        has_av = nanobodies_has_available_column()
        return jsonify({
            "status":     "ok",
            "db_path":    app.config["DB_PATH"],
            "total_rows": row["total"] if row else 0,
            "last_write": row["last_write"] if row else None,
            "nanobodies_has_available_column": has_av,
            "server_time": datetime.utcnow().isoformat() + "Z",
        })
    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 503


@app.route("/count", methods=["GET"])
@require_token
def count():
    """Total row count, optionally filtered by target."""
    target = request.args.get("target")
    if target:
        row = query_one(
            "SELECT COUNT(*) as n FROM nanobodies WHERE target=?",
            (target,))
    else:
        row = query_one("SELECT COUNT(*) as n FROM nanobodies")
    return jsonify({"count": row["n"] if row else 0})


@app.route("/top", methods=["GET"])
@require_token
def top():
    """
    Top-N scored sequences for a target, ordered by final_nanobody_score ASC.

    Query params:
      target          (required) — e.g. Q9NZQ7
      n               (optional, default 50) — how many rows to return (max 5000)
      method          (optional) — filter by generation_method (option_a / option_b)
      available_only  (optional) — when true/1/yes: only rows with available=TRUE
                                   (requires ``available`` column; used by submit miner)
    """
    target = request.args.get("target")
    if not target:
        return jsonify({"error": "target param required"}), 400

    n              = min(int(request.args.get("n", 50)), 5000)
    method         = request.args.get("method")
    available_only = _parse_bool_query("available_only", default=False)
    has_av         = nanobodies_has_available_column()

    if available_only and not has_av:
        log.warning(
            "/top: available_only=true but nanobodies has no 'available' column — "
            "returning no rows (add column on serving DB)"
        )
        return jsonify({
            "target":   target,
            "count":    0,
            "results":  [],
            "warning":  "available_only requested but column 'available' is missing",
        })

    av_sel = ", available" if has_av else ""
    base_filter = """
              AND TRIM(COALESCE(sequence, '')) != ''
              AND final_nanobody_score IS NOT NULL
    """
    av_sql = ""
    if available_only and has_av:
        av_sql = """
              AND (available = 1 OR CAST(available AS INTEGER) = 1)
        """

    if method:
        rows = query(f"""
            SELECT sequence, target, final_nanobody_score,
                   design_iiptm, design_ptm, design_to_target_iptm,
                   min_design_to_target_pae, interaction_pae,
                   plip_hbonds_refolded, plip_saltbridge_refolded,
                   delta_sasa_refolded, liability_score,
                   liability_num_violations, rank_sum,
                   confidence_rank_sum, physical_interaction_rank_sum,
                   developability_rank_sum,
                   generation_method, scored_by, calc_time_sec, created_at
                   {av_sel}
            FROM nanobodies
            WHERE target=?
              AND generation_method=?
              {base_filter}
              {av_sql}
            ORDER BY final_nanobody_score ASC, sequence ASC
            LIMIT ?
        """, (target, method, n))
    else:
        rows = query(f"""
            SELECT sequence, target, final_nanobody_score,
                   design_iiptm, design_ptm, design_to_target_iptm,
                   min_design_to_target_pae, interaction_pae,
                   plip_hbonds_refolded, plip_saltbridge_refolded,
                   delta_sasa_refolded, liability_score,
                   liability_num_violations, rank_sum,
                   confidence_rank_sum, physical_interaction_rank_sum,
                   developability_rank_sum,
                   generation_method, scored_by, calc_time_sec, created_at
                   {av_sel}
            FROM nanobodies
            WHERE target=?
              {base_filter}
              {av_sql}
            ORDER BY final_nanobody_score ASC, sequence ASC
            LIMIT ?
        """, (target, n))

    _normalize_available_in_rows(rows)

    payload: dict = {
        "target":           target,
        "count":            len(rows),
        "results":          rows,
        "available_only":   available_only,
    }
    return jsonify(payload)


@app.route("/pending", methods=["GET"])
@require_token
def pending():
    """
    Sequences that passed developability but are still awaiting BoltzGen scoring.

    Query params:
      target  (required)
    """
    target = request.args.get("target")
    if not target:
        return jsonify({"error": "target param required"}), 400

    rows = query("""
        SELECT sequence, target, scored_by, generation_method, created_at
        FROM nanobodies
        WHERE target=? AND scored_by='pending_boltzgen'
        ORDER BY created_at ASC
    """, (target,))

    return jsonify({
        "target":  target,
        "count":   len(rows),
        "results": rows,
    })


@app.route("/stats", methods=["GET"])
@require_token
def stats():
    """
    Per-target summary statistics.

    Query params:
      target  (optional) — if omitted, returns stats for all targets
    """
    target = request.args.get("target")

    base_sql = """
        SELECT
            target,
            COUNT(*)                                        AS total_rows,
            SUM(CASE WHEN final_nanobody_score IS NOT NULL
                     THEN 1 ELSE 0 END)                    AS scored,
            SUM(CASE WHEN scored_by='pending_boltzgen'
                     THEN 1 ELSE 0 END)                    AS pending,
            ROUND(MIN(final_nanobody_score), 4)             AS best_score,
            ROUND(MAX(final_nanobody_score), 4)             AS worst_score,
            ROUND(AVG(final_nanobody_score), 4)             AS avg_score,
            ROUND(MIN(design_iiptm), 4)                     AS min_iiptm,
            ROUND(MAX(design_iiptm), 4)                     AS max_iiptm,
            ROUND(AVG(design_iiptm), 4)                     AS avg_iiptm,
            ROUND(AVG(liability_score), 4)                  AS avg_liability,
            ROUND(AVG(liability_num_violations), 2)         AS avg_violations,
            SUM(CASE WHEN generation_method='option_a'
                     THEN 1 ELSE 0 END)                    AS option_a_count,
            SUM(CASE WHEN generation_method='option_b'
                     THEN 1 ELSE 0 END)                    AS option_b_count,
            MAX(created_at)                                 AS last_write
        FROM nanobodies
    """

    if target:
        rows = query(base_sql + " WHERE target=? GROUP BY target", (target,))
    else:
        rows = query(base_sql + " GROUP BY target")

    return jsonify({
        "count":   len(rows),
        "results": rows,
    })


@app.route("/sequence", methods=["GET"])
@require_token
def sequence():
    """
    Look up a single sequence by exact match.

    Query params:
      seq     (required) — full amino acid sequence
      target  (optional) — filter by target
    """
    seq = request.args.get("seq", "").strip().upper()
    if not seq:
        return jsonify({"error": "seq param required"}), 400

    target = request.args.get("target")
    if target:
        row = query_one("""
            SELECT * FROM nanobodies WHERE sequence=? AND target=?
        """, (seq, target))
    else:
        row = query_one("""
            SELECT * FROM nanobodies WHERE sequence=? LIMIT 1
        """, (seq,))

    if row is None:
        return jsonify({"found": False, "sequence": seq}), 404

    return jsonify({"found": True, "result": row})


@app.route("/all", methods=["GET"])
@require_token
def all_rows():
    """
    Dump all scored rows for a target (up to limit).

    Query params:
      target  (required)
      limit   (optional, default 1000, max 10000)
      offset  (optional, default 0) — for pagination
      scored_only (optional, default true) — exclude pending rows
    """
    target = request.args.get("target")
    if not target:
        return jsonify({"error": "target param required"}), 400

    limit       = min(int(request.args.get("limit", 1000)), 10000)
    offset      = int(request.args.get("offset", 0))
    scored_only = request.args.get("scored_only", "true").lower() != "false"

    if scored_only:
        rows = query("""
            SELECT sequence, target, final_nanobody_score,
                   design_iiptm, design_ptm, design_to_target_iptm,
                   min_design_to_target_pae, interaction_pae,
                   plip_hbonds_refolded, plip_saltbridge_refolded,
                   delta_sasa_refolded, liability_score,
                   liability_num_violations, rank_sum,
                   generation_method, scored_by, created_at
            FROM nanobodies
            WHERE target=? AND final_nanobody_score IS NOT NULL
            ORDER BY final_nanobody_score ASC
            LIMIT ? OFFSET ?
        """, (target, limit, offset))
    else:
        rows = query("""
            SELECT sequence, target, final_nanobody_score,
                   design_iiptm, design_ptm, design_to_target_iptm,
                   min_design_to_target_pae, interaction_pae,
                   plip_hbonds_refolded, plip_saltbridge_refolded,
                   delta_sasa_refolded, liability_score,
                   liability_num_violations, rank_sum,
                   generation_method, scored_by, created_at
            FROM nanobodies
            WHERE target=?
            ORDER BY final_nanobody_score ASC NULLS LAST
            LIMIT ? OFFSET ?
        """, (target, limit, offset))

    return jsonify({
        "target":  target,
        "count":   len(rows),
        "offset":  offset,
        "limit":   limit,
        "results": rows,
    })


# ══════════════════════════════════════════════════════════════════════════
# ROOT — submit miner does not call this; scanners often POST /
# ══════════════════════════════════════════════════════════════════════════

_ROOT_INDEX = {
    "service": "nanobody-db-api",
    "description": "Read-only HTTP API over nanobodies.sqlite",
    "submit_miner_neurons_submit_py": {
        "note": "Only these HTTP calls are made by nova-4090/neurons/submit.py",
        "requests": [
            {
                "method": "GET",
                "path": "/health",
                "auth": "none (always public)",
                "purpose": "Startup health check",
            },
            {
                "method": "GET",
                "path": "/top",
                "auth": "Bearer … if server started with --token",
                "query": "target (required), n, available_only=true",
                "purpose": "Submit window: top-N rows with available=TRUE by score",
            },
        ],
    },
    "endpoints": [
        "GET  /",
        "GET  /health",
        "GET  /top?target=…&n=…&available_only=true",
        "GET  /count?target=…",
        "GET  /pending?target=…",
        "GET  /stats?target=…",
        "GET  /sequence?seq=…&target=…",
        "GET  /all?target=…&limit=…&offset=…",
    ],
}


@app.route("/", methods=["GET"])
def root_index():
    """JSON service index (no auth)."""
    return jsonify(_ROOT_INDEX)


@app.route("/", methods=["POST", "PUT", "PATCH", "DELETE"])
def root_disallow_writes():
    """Read-only API — JSON 405 for POST / etc. (probes, misconfigured clients)."""
    return jsonify({
        "error": "method_not_allowed",
        "message": "This API is read-only.",
        "used_by_submit_miner": ["GET /health", "GET /top"],
        "hint": "GET / for full route list.",
    }), 405


# ══════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "endpoint_not_found",
        "path": request.path,
        "hint": "GET / for supported routes (submit miner: GET /health, GET /top)",
    }), 404


@app.errorhandler(500)
def server_error(e):
    log.error(f"Internal server error: {e}")
    return jsonify({"error": "internal server error", "detail": str(e)}), 500


@app.errorhandler(Exception)
def unhandled(e):
    log.error(f"Unhandled exception: {e}")
    return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nanobody DB — Flask API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",
                   default="0.0.0.0",
                   help="Host to bind (0.0.0.0 = all interfaces)")
    p.add_argument("--port",
                   type=int, default=50001,
                   help="Port to listen on")
    p.add_argument("--db",
                   default=DEFAULT_DB_PATH,
                   dest="db_path",
                   help="Path to nanobodies.sqlite")
    p.add_argument("--token",
                   default=None,
                   help="Optional bearer token for auth "
                        "(if omitted, all requests are allowed)")
    p.add_argument("--debug",
                   action="store_true",
                   help="Enable Flask debug mode (do NOT use in production)")
    return p.parse_args()


def main():
    args = parse_args()

    app.config["DB_PATH"]   = args.db_path
    app.config["API_TOKEN"] = args.token

    log.info("=" * 60)
    log.info("  NANOBODY DB — Flask API Server")
    log.info("=" * 60)
    log.info(f"  DB path:   {args.db_path}")
    log.info(f"  Listen:    http://{args.host}:{args.port}")
    log.info(f"  Auth:      {'token required' if args.token else 'open (no token)'}")
    log.info(f"  Debug:     {args.debug}")
    log.info("")
    log.info("  Endpoints:")
    log.info(f"    GET /")
    log.info(f"    GET /health")
    log.info(f"    GET /count?target=Q9NZQ7")
    log.info(f"    GET /top?target=Q9NZQ7&n=50&available_only=true")
    log.info(f"    GET /pending?target=Q9NZQ7")
    log.info(f"    GET /stats?target=Q9NZQ7")
    log.info(f"    GET /sequence?seq=EVQLVE...&target=Q9NZQ7")
    log.info(f"    GET /all?target=Q9NZQ7&limit=1000&offset=0")
    log.info("=" * 60)

    if not os.path.exists(args.db_path):
        log.warning(f"  DB file not found yet: {args.db_path}")
        log.warning(f"  Server will start anyway and serve once nano.py creates it.")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()