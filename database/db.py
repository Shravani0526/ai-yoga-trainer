"""Persistence for sessions, recurring-error statistics, calibration and difficulty levels.

Backends
    * SQLite (default, standard library): ``sqlite:///data/yoga_trainer.db``
    * PostgreSQL (optional, persistent on Streamlit Community Cloud): set ``DATABASE_URL`` to a
      ``postgresql://...`` URL (Supabase, Neon, ...). Requires ``psycopg2-binary``.

Streamlit Community Cloud has an EPHEMERAL file system, so a local SQLite file is lost when the
app restarts or is redeployed. Use PostgreSQL for permanent storage, or use the JSON backup
(``export_user_json`` / ``import_user_json``) that the app offers in the sidebar.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

DEFAULT_SQLITE_URL = "sqlite:///data/yoga_trainer.db"

SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    pose TEXT NOT NULL,
    variant TEXT,
    difficulty TEXT,
    source TEXT,
    started_at TEXT NOT NULL,
    duration_s REAL, frames INTEGER, detected_frames INTEGER, correct_frames INTEGER,
    held_s REAL, best_streak_s REAL, breaks INTEGER, hold_target_s REAL, completed INTEGER,
    avg_score REAL, avg_alignment REAL, avg_stability REAL, avg_symmetry REAL, max_score REAL,
    timeline_json TEXT
);
CREATE TABLE IF NOT EXISTS session_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    key TEXT NOT NULL, label TEXT, unit TEXT,
    frames_total INTEGER, frames_bad INTEGER, bad_frac REAL,
    mean_deviation REAL, direction TEXT, mean_score REAL
);
CREATE TABLE IF NOT EXISTS calibrations (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    pose TEXT NOT NULL, variant TEXT NOT NULL,
    profile_json TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, pose, variant)
);
CREATE TABLE IF NOT EXISTS user_levels (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    pose TEXT NOT NULL, level TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, pose)
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_pose ON sessions(user_id, pose, started_at);
CREATE INDEX IF NOT EXISTS idx_errors_session ON session_errors(session_id);
"""

SCHEMA_POSTGRES = (SCHEMA_SQLITE
                   .replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
                   .replace("REAL", "DOUBLE PRECISION"))

SESSION_COLUMNS = ["pose", "variant", "difficulty", "source", "started_at", "duration_s", "frames",
                   "detected_frames", "correct_frames", "held_s", "best_streak_s", "breaks",
                   "hold_target_s", "completed", "avg_score", "avg_alignment", "avg_stability",
                   "avg_symmetry", "max_score"]
ERROR_COLUMNS = ["key", "label", "unit", "frames_total", "frames_bad", "bad_frac",
                 "mean_deviation", "direction", "mean_score"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Small repository class; every public method opens/uses a connection safely."""

    def __init__(self, url: Optional[str] = None):
        url = (url or os.environ.get("DATABASE_URL") or DEFAULT_SQLITE_URL).strip()
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        self.url = url
        self.is_postgres = url.startswith("postgresql://")
        self._lock = threading.RLock()
        self._shared: Optional[sqlite3.Connection] = None
        self.sqlite_path: Optional[str] = None
        if not self.is_postgres:
            if not url.startswith("sqlite:///"):
                raise ValueError("DATABASE_URL must start with sqlite:/// or postgresql://")
            path = url[len("sqlite:///"):]
            self.sqlite_path = path
            if path == ":memory:":
                self._shared = sqlite3.connect(":memory:", check_same_thread=False)
                self._shared.row_factory = sqlite3.Row
                self._shared.execute("PRAGMA foreign_keys = ON")
            else:
                Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    # ------------------------------------------------------------------ connections
    @property
    def backend(self) -> str:
        return "PostgreSQL" if self.is_postgres else "SQLite"

    def describe(self) -> str:
        if self.is_postgres:
            return "PostgreSQL (persistent)"
        if self.sqlite_path == ":memory:":
            return "SQLite in memory (temporary)"
        return f"SQLite file {self.sqlite_path} (ephemeral on Streamlit Cloud)"

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        with self._lock:
            if self._shared is not None:
                conn, close = self._shared, False
            elif self.is_postgres:
                import psycopg2  # imported lazily so SQLite users do not need it
                import psycopg2.extras
                conn, close = psycopg2.connect(self.url, cursor_factory=psycopg2.extras.RealDictCursor), True
            else:
                conn = sqlite3.connect(self.sqlite_path, timeout=30, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                close = True
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                if close:
                    conn.close()

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.is_postgres else sql

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(self._sql(sql), params)

    def _fetchall(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(self._sql(sql), params)
            return [dict(r) for r in cur.fetchall()]

    def _insert_returning_id(self, sql: str, params: tuple) -> int:
        with self._connection() as conn:
            cur = conn.cursor()
            if self.is_postgres:
                cur.execute(self._sql(sql) + " RETURNING id", params)
                return int(cur.fetchone()["id"])
            cur.execute(sql, params)
            return int(cur.lastrowid)

    def close(self) -> None:
        """Close the shared in-memory connection (file / PostgreSQL connections are per-call)."""
        with self._lock:
            if self._shared is not None:
                self._shared.close()
                self._shared = None

    # ------------------------------------------------------------------ schema
    def init_schema(self) -> None:
        schema = SCHEMA_POSTGRES if self.is_postgres else SCHEMA_SQLITE
        statements = [s.strip() for s in schema.split(";") if s.strip()]
        with self._connection() as conn:
            cur = conn.cursor()
            for stmt in statements:
                cur.execute(stmt)

    # ------------------------------------------------------------------ users
    def get_or_create_user(self, name: str) -> int:
        name = (name or "").strip() or "guest"
        rows = self._fetchall("SELECT id FROM users WHERE name = ?", (name,))
        if rows:
            return int(rows[0]["id"])
        try:
            return self._insert_returning_id("INSERT INTO users (name, created_at) VALUES (?, ?)", (name, _now()))
        except Exception:  # race: another thread inserted the same name
            rows = self._fetchall("SELECT id FROM users WHERE name = ?", (name,))
            if rows:
                return int(rows[0]["id"])
            raise

    def list_users(self) -> List[str]:
        return [r["name"] for r in self._fetchall("SELECT name FROM users ORDER BY name")]

    def delete_user(self, user_id: int) -> None:
        with self._connection() as conn:
            cur = conn.cursor()
            # explicit deletes keep behaviour identical even if foreign keys are disabled
            cur.execute(self._sql("DELETE FROM session_errors WHERE session_id IN (SELECT id FROM sessions WHERE user_id = ?)"), (user_id,))
            for table in ("sessions", "calibrations", "user_levels"):
                cur.execute(self._sql(f"DELETE FROM {table} WHERE user_id = ?"), (user_id,))
            cur.execute(self._sql("DELETE FROM users WHERE id = ?"), (user_id,))

    # ------------------------------------------------------------------ sessions
    def save_session(self, user_id: int, summary: Dict[str, Any]) -> int:
        cols = ", ".join(["user_id"] + SESSION_COLUMNS + ["timeline_json"])
        marks = ", ".join(["?"] * (len(SESSION_COLUMNS) + 2))
        values = [user_id] + [summary.get(c) for c in SESSION_COLUMNS] + [json.dumps(summary.get("timeline", []))]
        session_id = self._insert_returning_id(f"INSERT INTO sessions ({cols}) VALUES ({marks})", tuple(values))
        errors = summary.get("errors", [])
        if errors:
            ecols = ", ".join(["session_id"] + ERROR_COLUMNS)
            emarks = ", ".join(["?"] * (len(ERROR_COLUMNS) + 1))
            with self._connection() as conn:
                cur = conn.cursor()
                for e in errors:
                    cur.execute(self._sql(f"INSERT INTO session_errors ({ecols}) VALUES ({emarks})"),
                                tuple([session_id] + [e.get(c) for c in ERROR_COLUMNS]))
        return session_id

    def get_sessions(self, user_id: int, pose: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        sql = f"SELECT id, {', '.join(SESSION_COLUMNS)} FROM sessions WHERE user_id = ?"
        params: List[Any] = [user_id]
        if pose:
            sql += " AND pose = ?"
            params.append(pose)
        sql += " ORDER BY started_at DESC, id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self._fetchall(sql, tuple(params))
        df = pd.DataFrame(rows, columns=["id"] + SESSION_COLUMNS)
        return df.sort_values(["started_at", "id"]).reset_index(drop=True)

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        rows = self._fetchall(f"SELECT id, {', '.join(SESSION_COLUMNS)}, timeline_json FROM sessions WHERE id = ?", (session_id,))
        if not rows:
            return None
        row = rows[0]
        row["timeline"] = json.loads(row.pop("timeline_json") or "[]")
        row["errors"] = self._fetchall(f"SELECT {', '.join(ERROR_COLUMNS)} FROM session_errors WHERE session_id = ? ORDER BY bad_frac DESC", (session_id,))
        return row

    def get_errors(self, user_id: int, pose: Optional[str] = None) -> pd.DataFrame:
        """All per-session joint error rows of a user, joined with session info."""
        sql = ("SELECT e.session_id, s.pose, s.started_at, e.key, e.label, e.unit, e.frames_total, "
               "e.frames_bad, e.bad_frac, e.mean_deviation, e.direction, e.mean_score "
               "FROM session_errors e JOIN sessions s ON s.id = e.session_id WHERE s.user_id = ?")
        params: List[Any] = [user_id]
        if pose:
            sql += " AND s.pose = ?"
            params.append(pose)
        cols = ["session_id", "pose", "started_at", "key", "label", "unit", "frames_total", "frames_bad",
                "bad_frac", "mean_deviation", "direction", "mean_score"]
        return pd.DataFrame(self._fetchall(sql, tuple(params)), columns=cols)

    # ------------------------------------------------------------------ calibration
    def save_calibration(self, user_id: int, pose: str, variant: str, profile: Dict[str, Any]) -> None:
        self._execute(
            "INSERT INTO calibrations (user_id, pose, variant, profile_json, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (user_id, pose, variant) DO UPDATE SET profile_json = excluded.profile_json, "
            "updated_at = excluded.updated_at",
            (user_id, pose, variant, json.dumps(profile), _now()))

    def get_calibrations(self, user_id: int, pose: str) -> Dict[str, Dict[str, Any]]:
        rows = self._fetchall("SELECT variant, profile_json FROM calibrations WHERE user_id = ? AND pose = ?",
                              (user_id, pose))
        return {r["variant"]: json.loads(r["profile_json"]) for r in rows}

    def delete_calibrations(self, user_id: int, pose: Optional[str] = None) -> None:
        if pose:
            self._execute("DELETE FROM calibrations WHERE user_id = ? AND pose = ?", (user_id, pose))
        else:
            self._execute("DELETE FROM calibrations WHERE user_id = ?", (user_id,))

    # ------------------------------------------------------------------ levels
    def get_levels(self, user_id: int) -> Dict[str, str]:
        rows = self._fetchall("SELECT pose, level FROM user_levels WHERE user_id = ?", (user_id,))
        return {r["pose"]: r["level"] for r in rows}

    def set_level(self, user_id: int, pose: str, level: str) -> None:
        self._execute(
            "INSERT INTO user_levels (user_id, pose, level, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (user_id, pose) DO UPDATE SET level = excluded.level, updated_at = excluded.updated_at",
            (user_id, pose, level, _now()))

    # ------------------------------------------------------------------ backup / restore
    def export_user_json(self, user_id: int) -> Dict[str, Any]:
        sessions = []
        for sid in self.get_sessions(user_id)["id"].tolist():
            s = self.get_session(int(sid))
            if s:
                sessions.append(s)
        calibrations = []
        for pose in {s["pose"] for s in sessions} | set(self._poses_with_calibration(user_id)):
            for variant, profile in self.get_calibrations(user_id, pose).items():
                calibrations.append({"pose": pose, "variant": variant, "profile": profile})
        return {"format": "ai-yoga-trainer-backup", "version": 1, "exported_at": _now(),
                "sessions": sessions, "calibrations": calibrations, "levels": self.get_levels(user_id)}

    def _poses_with_calibration(self, user_id: int) -> List[str]:
        return [r["pose"] for r in self._fetchall("SELECT DISTINCT pose FROM calibrations WHERE user_id = ?", (user_id,))]

    def import_user_json(self, user_id: int, data: Dict[str, Any]) -> Dict[str, int]:
        """Restore a backup; sessions already present (same pose + start time) are skipped."""
        if data.get("format") != "ai-yoga-trainer-backup":
            raise ValueError("This file is not an AI Yoga Trainer backup.")
        existing = {(r["pose"], r["started_at"]) for r in self._fetchall(
            "SELECT pose, started_at FROM sessions WHERE user_id = ?", (user_id,))}
        added = 0
        for s in data.get("sessions", []):
            if (s.get("pose"), s.get("started_at")) in existing:
                continue
            self.save_session(user_id, s)
            added += 1
        for c in data.get("calibrations", []):
            self.save_calibration(user_id, c["pose"], c["variant"], c["profile"])
        for pose, level in data.get("levels", {}).items():
            self.set_level(user_id, pose, level)
        return {"sessions_added": added, "calibrations": len(data.get("calibrations", []))}


_DB_CACHE: Dict[str, Database] = {}


def get_database(url: Optional[str] = None) -> Database:
    """Return a shared Database instance per URL."""
    key = (url or os.environ.get("DATABASE_URL") or DEFAULT_SQLITE_URL)
    if key not in _DB_CACHE:
        _DB_CACHE[key] = Database(key)
    return _DB_CACHE[key]
