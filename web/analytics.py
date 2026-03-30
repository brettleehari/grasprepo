"""SQLite-backed internal analytics for tracking repo analyses."""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

# Use persistent disk on Render (/data), fall back to local web/data/
_RENDER_DISK = Path("/data")
DB_DIR = _RENDER_DISK if _RENDER_DISK.exists() and os.access(str(_RENDER_DISK), os.W_OK) else Path(__file__).parent / "data"
DB_PATH = DB_DIR / "analytics.db"

_CREATE_ANALYSES = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_url TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    file_count INTEGER DEFAULT 0,
    ranked_files_count INTEGER DEFAULT 0,
    churn_files_count INTEGER DEFAULT 0,
    hotspot_count INTEGER DEFAULT 0,
    top_ranked_file TEXT,
    top_churned_file TEXT,
    top_hotspot_file TEXT,
    graph_nodes INTEGER DEFAULT 0,
    graph_edges INTEGER DEFAULT 0,
    ast_symbol_count INTEGER DEFAULT 0,
    churn_total_contributors INTEGER DEFAULT 0,
    duration_seconds REAL DEFAULT 0
);
"""

# Columns added after initial release — migrate existing DBs gracefully
_MIGRATE_COLUMNS = [
    ("top_hotspot_file", "TEXT"),
    ("graph_nodes", "INTEGER DEFAULT 0"),
    ("graph_edges", "INTEGER DEFAULT 0"),
    ("ast_symbol_count", "INTEGER DEFAULT 0"),
    ("churn_total_contributors", "INTEGER DEFAULT 0"),
]

_CREATE_LEADS = """
CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT,
    linkedin TEXT,
    repo_url TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    ip_address TEXT,
    user_agent TEXT
);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the data directory and tables if they don't exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = _get_conn()
    conn.execute(_CREATE_ANALYSES)
    conn.execute(_CREATE_LEADS)
    # Migrate: add new columns to existing DBs
    for col_name, col_type in _MIGRATE_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE analyses ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()


def record_analysis(
    repo_url: str,
    repo_name: str,
    ip_address: str | None = None,
    user_agent: str | None = None,
    file_count: int = 0,
    ranked_files_count: int = 0,
    churn_files_count: int = 0,
    hotspot_count: int = 0,
    top_ranked_file: str | None = None,
    top_churned_file: str | None = None,
    top_hotspot_file: str | None = None,
    graph_nodes: int = 0,
    graph_edges: int = 0,
    ast_symbol_count: int = 0,
    churn_total_contributors: int = 0,
    duration_seconds: float = 0,
):
    conn = _get_conn()
    conn.execute(
        """INSERT INTO analyses
           (repo_url, repo_name, timestamp, ip_address, user_agent,
            file_count, ranked_files_count, churn_files_count, hotspot_count,
            top_ranked_file, top_churned_file, top_hotspot_file,
            graph_nodes, graph_edges, ast_symbol_count,
            churn_total_contributors, duration_seconds)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            repo_url,
            repo_name,
            datetime.utcnow().isoformat(),
            ip_address,
            user_agent,
            file_count,
            ranked_files_count,
            churn_files_count,
            hotspot_count,
            top_ranked_file,
            top_churned_file,
            top_hotspot_file,
            graph_nodes,
            graph_edges,
            ast_symbol_count,
            churn_total_contributors,
            duration_seconds,
        ),
    )
    conn.commit()
    conn.close()


def get_total_count() -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as c FROM analyses").fetchone()
    conn.close()
    return row["c"]


def get_recent(limit: int = 20) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM analyses ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_popular_repos(limit: int = 10) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT repo_url, repo_name, COUNT(*) as analysis_count,
                  MAX(timestamp) as last_analyzed
           FROM analyses
           GROUP BY repo_url
           ORDER BY analysis_count DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_analyses() -> tuple[list[str], list[dict]]:
    """Return (column_names, rows) for all analyses — used for CSV export."""
    conn = _get_conn()
    cursor = conn.execute("SELECT * FROM analyses ORDER BY timestamp DESC")
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return columns, rows


def get_all_leads() -> tuple[list[str], list[dict]]:
    """Return (column_names, rows) for all leads — used for CSV export."""
    conn = _get_conn()
    cursor = conn.execute("SELECT * FROM leads ORDER BY timestamp DESC")
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return columns, rows


def get_unique_repos_count() -> int:
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(DISTINCT repo_url) as c FROM analyses"
    ).fetchone()
    conn.close()
    return row["c"]


# ---------------------------------------------------------------------------
# Leads
# ---------------------------------------------------------------------------

def record_lead(
    email: str | None,
    linkedin: str | None,
    repo_url: str,
    repo_name: str,
    ip_address: str | None = None,
    user_agent: str | None = None,
):
    conn = _get_conn()
    conn.execute(
        """INSERT INTO leads (email, linkedin, repo_url, repo_name, timestamp, ip_address, user_agent)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (email, linkedin, repo_url, repo_name, datetime.utcnow().isoformat(), ip_address, user_agent),
    )
    conn.commit()
    conn.close()


def get_leads(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM leads ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_leads_count() -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as c FROM leads").fetchone()
    conn.close()
    return row["c"]
