
import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "speakwise_history.db"


def get_connection():
    """Create a connection to the local SpeakWise history database."""
    connection = sqlite3.connect(str(DB_PATH), timeout=10)
    connection.row_factory = sqlite3.Row
    return connection



def init_database():
    """Create the history table and migrate older databases safely."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                session_type TEXT NOT NULL,
                title TEXT NOT NULL,
                duration REAL,
                speech_seconds REAL,
                pause_count INTEGER,
                dtw_distance REAL,
                flaw_count INTEGER,
                metrics TEXT NOT NULL DEFAULT '{}',
                details TEXT NOT NULL DEFAULT '{}',
                signature TEXT UNIQUE
            )
        """)

        # Check which columns already exist in an older database.
        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(sessions)"
            ).fetchall()
        }

        # Add missing columns without deleting existing history.
        migrations = {
            "duration": "REAL",
            "speech_seconds": "REAL",
            "pause_count": "INTEGER",
            "dtw_distance": "REAL",
            "flaw_count": "INTEGER",
            "metrics": "TEXT NOT NULL DEFAULT '{}'",
            "details": "TEXT NOT NULL DEFAULT '{}'",
            "signature": "TEXT",
        }

        for column_name, column_type in migrations.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE sessions ADD COLUMN "
                    f"{column_name} {column_type}"
                )

        # Create a unique index after the signature column exists.
        connection.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_sessions_signature
            ON sessions(signature)
        """)

        connection.commit()

def save_session(
    session_type,
    title,
    metrics,
    details=None,
    signature=None
):
    """
    Save one analysis session.

    Returns the saved session ID. If a signature already exists,
    returns the existing session ID to prevent duplicate saves.
    """
    metrics = metrics or {}
    details = details or {}

    if signature:
        with get_connection() as connection:
            existing = connection.execute(
                "SELECT id FROM sessions WHERE signature = ?",
                (signature,)
            ).fetchone()

            if existing:
                return existing["id"]

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO sessions (
                created_at, session_type, title,
                duration, speech_seconds, pause_count,
                dtw_distance, flaw_count, metrics, details, signature
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signature) DO NOTHING
        """, (
            created_at,
            str(session_type),
            str(title),
            metrics.get("duration"),
            metrics.get("speech_seconds"),
            metrics.get("pause_count"),
            metrics.get("dtw_distance"),
            metrics.get("flaw_count"),
            json.dumps(metrics, ensure_ascii=False, default=str),
            json.dumps(details, ensure_ascii=False, default=str),
            signature
        ))

        connection.commit()

        if cursor.lastrowid:
            return cursor.lastrowid

        if signature:
            existing = connection.execute(
                "SELECT id FROM sessions WHERE signature = ?",
                (signature,)
            ).fetchone()
            if existing:
                return existing["id"]

        return None


def get_sessions(session_type=None, limit=500):
    """Return recent sessions, optionally filtered by type."""
    limit = max(1, min(int(limit), 5000))

    with get_connection() as connection:
        if session_type:
            rows = connection.execute("""
                SELECT * FROM sessions
                WHERE session_type = ?
                ORDER BY id DESC
                LIMIT ?
            """, (session_type, limit)).fetchall()
        else:
            rows = connection.execute("""
                SELECT * FROM sessions
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()

    sessions = []
    for row in rows:
        item = dict(row)

        try:
            item["metrics"] = json.loads(item.get("metrics") or "{}")
        except (TypeError, json.JSONDecodeError):
            item["metrics"] = {}

        try:
            item["details"] = json.loads(item.get("details") or "{}")
        except (TypeError, json.JSONDecodeError):
            item["details"] = {}

        sessions.append(item)

    return sessions


def get_session(session_id):
    """Return one session with decoded metrics and details."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (int(session_id),)
        ).fetchone()

    if row is None:
        return None

    item = dict(row)

    for field in ("metrics", "details"):
        try:
            item[field] = json.loads(item.get(field) or "{}")
        except (TypeError, json.JSONDecodeError):
            item[field] = {}

    return item


def delete_session(session_id):
    """Delete one saved session."""
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM sessions WHERE id = ?",
            (int(session_id),)
        )
        connection.commit()
        return cursor.rowcount > 0


def clear_history(session_type=None):
    """Delete all sessions or only sessions of one type."""
    with get_connection() as connection:
        if session_type:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE session_type = ?",
                (session_type,)
            )
        else:
            cursor = connection.execute("DELETE FROM sessions")

        connection.commit()
        return cursor.rowcount
