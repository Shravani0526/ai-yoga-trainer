-- Reference copy of the SQLite schema (created automatically by database/db.py).
-- Run manually with:  sqlite3 data/yoga_trainer.db < database/schema.sql

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
