"""
SQLite persistence layer.

Uses the standard-library ``sqlite3`` (no heavy ORM) so the whole system stays
dependency-light and trivially testable. A single connection is guarded by a
lock; ``check_same_thread=False`` lets the API (async, multiple worker threads)
and the camera/AI threads share it safely under that lock.

Schema covers every entity the platform needs now and in the future:
employees, employee_images, face_embeddings, events, detections, components,
wires, model_config, and a generic settings key/value table.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Iterable, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_code TEXT UNIQUE,
    full_name     TEXT NOT NULL,
    department    TEXT,
    job_title     TEXT,
    profile_image TEXT,                 -- relative path under the data dir
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS employee_images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    path        TEXT NOT NULL,          -- relative path under the data dir
    created_at  REAL NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    image_id    INTEGER,
    embedder    TEXT NOT NULL,          -- which embedder produced this vector
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,          -- float32 little-endian
    created_at  REAL NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE,
    FOREIGN KEY (image_id)    REFERENCES employee_images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT NOT NULL,          -- face_recognized|unknown_person|component_detected|wiring_error|system_alert...
    camera_id   TEXT,
    camera_name TEXT,
    label       TEXT,
    confidence  REAL,
    employee_id INTEGER,
    snapshot    TEXT,                   -- relative path under the data dir
    payload     TEXT,                   -- JSON with extra structured detail
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT,
    model       TEXT,
    label       TEXT,
    confidence  REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    payload     TEXT,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS components (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT,
    name        TEXT,
    comp_type   TEXT,
    confidence  REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    position    TEXT,
    payload     TEXT,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS wires (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id     TEXT,
    wire_uid      TEXT,
    start_x REAL, start_y REAL, end_x REAL, end_y REAL,
    color         TEXT,
    status        TEXT,                 -- ok|broken|disconnected|missing|loose|incorrect
    from_component INTEGER,
    to_component   INTEGER,
    payload       TEXT,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS model_config (
    name       TEXT PRIMARY KEY,        -- task name: detection|face|components|wires
    backend    TEXT,                    -- selected backend id
    enabled    INTEGER NOT NULL DEFAULT 0,
    params     TEXT,                    -- JSON of thresholds / device / etc.
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events(type);
CREATE INDEX IF NOT EXISTS idx_emb_employee   ON face_embeddings(employee_id);
"""


class Database:
    """Thin, thread-safe wrapper around a single SQLite connection."""

    def __init__(self, path: str) -> None:
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- low-level ---------------------------------------------------------

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return cur.fetchall()

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return cur.fetchone()

    def insert(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return int(cur.lastrowid)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- settings key/value ------------------------------------------------

    def set_setting(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO settings(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value), time.time()),
        )

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM settings WHERE key=?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (TypeError, json.JSONDecodeError):
            return default

    def all_settings(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for row in self.query("SELECT key, value FROM settings"):
            try:
                out[row["key"]] = json.loads(row["value"])
            except (TypeError, json.JSONDecodeError):
                out[row["key"]] = None
        return out
