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
    quality     REAL,                   -- 0..1 capture quality score
    meta        TEXT,                   -- JSON: blur, brightness, det_score, bbox, ...
    created_at  REAL NOT NULL,          -- capture date (epoch seconds)
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

CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    employee_name TEXT,
    camera_id   TEXT,
    camera_name TEXT,
    confidence  REAL,
    snapshot    TEXT,                   -- relative path under the data dir
    day         TEXT NOT NULL,          -- YYYY-MM-DD local date, for once-per-day queries
    created_at  REAL NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    kind        TEXT,                   -- yolo|coco|voc|classification|images|videos|mixed|unknown
    path        TEXT NOT NULL,          -- relative path under the data dir
    status      TEXT NOT NULL DEFAULT 'uploaded',  -- uploaded|validating|valid|invalid|error
    n_images    INTEGER DEFAULT 0,
    n_labels    INTEGER DEFAULT 0,
    n_classes   INTEGER DEFAULT 0,
    classes     TEXT,                   -- JSON list of class names
    report      TEXT,                   -- JSON validation report
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS training_jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    dataset_id  INTEGER,
    task        TEXT,                   -- classification|detection
    models      TEXT,                   -- JSON list of model architectures to train
    config      TEXT,                   -- JSON hyperparameters / options
    status      TEXT NOT NULL DEFAULT 'queued',  -- queued|running|paused|stopped|completed|failed
    progress    REAL DEFAULT 0.0,
    metrics     TEXT,                   -- JSON latest metrics
    history     TEXT,                   -- JSON list of per-epoch metric snapshots
    comparison  TEXT,                   -- JSON per-model comparison + selected best
    best_model  TEXT,
    artifacts   TEXT,                   -- JSON list of exported artifact paths
    error       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS reference_designs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    kind        TEXT,                   -- pdf|image|dxf|dwg|other
    path        TEXT NOT NULL,          -- relative path under the data dir
    description TEXT,
    spec        TEXT,                   -- JSON expected components/wires (for inspection)
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS inspections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_id  INTEGER,
    camera_id     TEXT,
    source        TEXT,                 -- camera|upload
    status        TEXT,                 -- pass|fail|warning
    n_mismatches  INTEGER DEFAULT 0,
    result        TEXT,                 -- JSON full comparison result
    snapshot      TEXT,
    report_path   TEXT,
    created_at    REAL NOT NULL,
    FOREIGN KEY (reference_id) REFERENCES reference_designs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,          -- panel_analysis|inspection|training|dataset
    title       TEXT,
    ref_id      INTEGER,                -- id of the source row (inspection/panel/etc.)
    path        TEXT,                   -- relative path of PDF/JSON under the data dir
    summary     TEXT,                   -- JSON short summary
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at REAL NOT NULL
);

-- =====================================================================
-- Industrial Panel Inspection (Reference Panel Learning) — normalized
-- schema. Additive: existing tables above are untouched. A reference
-- panel is LEARNED from one or more images (captured from RTSP or
-- uploaded); the learned template + electrical graph are stored here,
-- and every live/uploaded inspection is compared against it.
-- =====================================================================

CREATE TABLE IF NOT EXISTS reference_panels (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    version      TEXT NOT NULL DEFAULT 'v1',
    description  TEXT,
    status       TEXT NOT NULL DEFAULT 'draft',   -- draft|learning|ready|error
    template     TEXT,                             -- JSON reusable panel template
    features     TEXT,                             -- JSON feature embedding / descriptors
    thumbnail    TEXT,                             -- relative path under data dir
    n_images     INTEGER NOT NULL DEFAULT 0,
    n_components INTEGER NOT NULL DEFAULT 0,
    n_terminals  INTEGER NOT NULL DEFAULT 0,
    n_wires      INTEGER NOT NULL DEFAULT 0,
    note         TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reference_images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    panel_id    INTEGER NOT NULL,
    path        TEXT NOT NULL,                     -- relative path under data dir
    source      TEXT,                              -- camera|upload
    camera_id   TEXT,
    width       INTEGER,
    height      INTEGER,
    is_primary  INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    FOREIGN KEY (panel_id) REFERENCES reference_panels(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reference_components (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    panel_id    INTEGER NOT NULL,
    ref_id      TEXT,                              -- stable id within the template (e.g. C0)
    comp_type   TEXT,                              -- mcb|contactor|plc|terminal_block...
    label       TEXT,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    cx REAL, cy REAL, w REAL, h REAL,
    rotation    REAL,                              -- degrees
    confidence  REAL,
    position    TEXT,                              -- coarse grid label
    payload     TEXT,                              -- JSON extras (terminals etc.)
    created_at  REAL NOT NULL,
    FOREIGN KEY (panel_id) REFERENCES reference_panels(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reference_terminals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    panel_id     INTEGER NOT NULL,
    ref_id       TEXT,                             -- stable id (e.g. T0)
    component_ref TEXT,                            -- owning component ref_id (nullable)
    label        TEXT,
    kind         TEXT,                             -- screw|block|entry
    x REAL, y REAL,
    created_at   REAL NOT NULL,
    FOREIGN KEY (panel_id) REFERENCES reference_panels(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reference_wires (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    panel_id     INTEGER NOT NULL,
    wire_uid     TEXT,
    start_x REAL, start_y REAL, end_x REAL, end_y REAL,
    polyline     TEXT,                             -- JSON [[x,y],...]
    length       REAL,
    thickness    REAL,
    color        TEXT,
    direction    REAL,                             -- degrees
    from_terminal TEXT,                            -- terminal ref_id
    to_terminal   TEXT,
    from_component TEXT,                           -- component ref_id
    to_component   TEXT,
    payload      TEXT,
    created_at   REAL NOT NULL,
    FOREIGN KEY (panel_id) REFERENCES reference_panels(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reference_connections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    panel_id      INTEGER NOT NULL,
    wire_uid      TEXT,
    from_node     TEXT,
    to_node       TEXT,
    from_terminal TEXT,
    to_terminal   TEXT,
    color         TEXT,
    created_at    REAL NOT NULL,
    FOREIGN KEY (panel_id) REFERENCES reference_panels(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reference_graph (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    panel_id    INTEGER NOT NULL UNIQUE,
    nodes       TEXT,                              -- JSON list of nodes
    edges       TEXT,                              -- JSON list of edges
    created_at  REAL NOT NULL,
    FOREIGN KEY (panel_id) REFERENCES reference_panels(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS inspection_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    panel_id    INTEGER,
    camera_id   TEXT,
    source      TEXT,                              -- camera|upload
    status      TEXT,                              -- pass|warning|fail
    score       REAL,                              -- 0..1 overall match score
    n_errors    INTEGER NOT NULL DEFAULT 0,
    n_warnings  INTEGER NOT NULL DEFAULT 0,
    result      TEXT,                              -- JSON full comparison result
    snapshot    TEXT,                              -- annotated overlay image
    report_path TEXT,
    created_at  REAL NOT NULL,
    FOREIGN KEY (panel_id) REFERENCES reference_panels(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS inspection_components (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id   INTEGER NOT NULL,
    comp_type   TEXT,
    label       TEXT,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    confidence  REAL,
    matched_ref TEXT,                              -- reference component ref_id it matched
    status      TEXT,                              -- ok|missing|extra|wrong|moved|rotated
    created_at  REAL NOT NULL,
    FOREIGN KEY (result_id) REFERENCES inspection_results(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS inspection_terminals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id   INTEGER NOT NULL,
    label       TEXT,
    kind        TEXT,
    x REAL, y REAL,
    matched_ref TEXT,
    status      TEXT,
    created_at  REAL NOT NULL,
    FOREIGN KEY (result_id) REFERENCES inspection_results(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS inspection_wires (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id   INTEGER NOT NULL,
    wire_uid    TEXT,
    start_x REAL, start_y REAL, end_x REAL, end_y REAL,
    color       TEXT,
    length      REAL,
    thickness   REAL,
    status      TEXT,                              -- ok|missing|extra|wrong|loose|disconnected|broken
    matched_ref TEXT,
    created_at  REAL NOT NULL,
    FOREIGN KEY (result_id) REFERENCES inspection_results(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS inspection_errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id   INTEGER NOT NULL,
    error_type  TEXT NOT NULL,                     -- missing_component|extra_wire|wrong_color...
    severity    TEXT,                              -- error|warning|info
    target      TEXT,                              -- what it refers to (ref id / label)
    detail      TEXT,
    confidence  REAL,
    x REAL, y REAL,
    created_at  REAL NOT NULL,
    FOREIGN KEY (result_id) REFERENCES inspection_results(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS datasheets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    kind         TEXT,                             -- pdf|image|dxf|svg|other
    path         TEXT NOT NULL,                    -- relative path under data dir
    panel_id     INTEGER,                          -- optional link to a reference panel
    description  TEXT,
    ocr_engine   TEXT,                             -- paddleocr|tesseract|none
    extracted    TEXT,                             -- JSON extraction result
    expected_graph TEXT,                           -- JSON expected graph built from doc
    status       TEXT NOT NULL DEFAULT 'uploaded', -- uploaded|processing|extracted|error
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    FOREIGN KEY (panel_id) REFERENCES reference_panels(id) ON DELETE SET NULL
);

-- =====================================================================
-- General AI Image Analysis & Comparison (any image, not just panels).
-- Additive: nothing above is touched.
-- =====================================================================

CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    path        TEXT NOT NULL,                     -- relative path under data dir
    format      TEXT,                              -- jpg|png|...
    width       INTEGER,
    height      INTEGER,
    bytes       INTEGER,
    sha256      TEXT,                              -- dedupe / integrity
    phash       TEXT,                              -- perceptual hash (hex)
    dominant_colors TEXT,                          -- JSON list of {hex,ratio}
    metadata    TEXT,                              -- JSON (exif-ish / channels)
    summary     TEXT,                              -- AI summary sentence
    tags        TEXT,                              -- JSON list
    ocr_text    TEXT,                              -- concatenated OCR text
    n_objects   INTEGER NOT NULL DEFAULT 0,
    analysis    TEXT,                              -- JSON full analysis result
    status      TEXT NOT NULL DEFAULT 'uploaded',  -- uploaded|analyzing|analyzed|error
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS image_objects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    INTEGER NOT NULL,
    label       TEXT,
    confidence  REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    source      TEXT,                              -- detector backend id
    created_at  REAL NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS image_ocr (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    INTEGER NOT NULL,
    text        TEXT,
    confidence  REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    engine      TEXT,
    created_at  REAL NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS image_comparisons (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ref_image_id  INTEGER,
    cur_image_id  INTEGER,
    similarity    REAL,                            -- 0..100 %
    n_diffs       INTEGER NOT NULL DEFAULT 0,
    status        TEXT,                            -- identical|minor|major
    report        TEXT,                            -- JSON full comparison result
    overlay_path  TEXT,                            -- side-by-side + heatmap image
    heatmap_path  TEXT,
    aligned_path  TEXT,
    report_pdf    TEXT,
    created_at    REAL NOT NULL,
    FOREIGN KEY (ref_image_id) REFERENCES images(id) ON DELETE SET NULL,
    FOREIGN KEY (cur_image_id) REFERENCES images(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS image_diffs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    comparison_id INTEGER NOT NULL,
    diff_type     TEXT,                            -- missing_object|new_object|moved_object|color_change|text_change|region_changed...
    severity      TEXT,                            -- info|minor|major
    detail        TEXT,
    confidence    REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    created_at    REAL NOT NULL,
    FOREIGN KEY (comparison_id) REFERENCES image_comparisons(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events(type);
CREATE INDEX IF NOT EXISTS idx_emb_employee   ON face_embeddings(employee_id);
CREATE INDEX IF NOT EXISTS idx_att_day        ON attendance(day);
CREATE INDEX IF NOT EXISTS idx_att_employee   ON attendance(employee_id, day);
CREATE INDEX IF NOT EXISTS idx_insp_created   ON inspections(created_at);
CREATE INDEX IF NOT EXISTS idx_refimg_panel   ON reference_images(panel_id);
CREATE INDEX IF NOT EXISTS idx_refcomp_panel  ON reference_components(panel_id);
CREATE INDEX IF NOT EXISTS idx_refwire_panel  ON reference_wires(panel_id);
CREATE INDEX IF NOT EXISTS idx_refterm_panel  ON reference_terminals(panel_id);
CREATE INDEX IF NOT EXISTS idx_inspres_panel  ON inspection_results(panel_id);
CREATE INDEX IF NOT EXISTS idx_insperr_result ON inspection_errors(result_id);
CREATE INDEX IF NOT EXISTS idx_images_created ON images(created_at);
CREATE INDEX IF NOT EXISTS idx_imgobj_image  ON image_objects(image_id);
CREATE INDEX IF NOT EXISTS idx_imgcmp_created ON image_comparisons(created_at);
CREATE INDEX IF NOT EXISTS idx_imgdiff_cmp   ON image_diffs(comparison_id);
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
        self._migrate()

    def _migrate(self) -> None:
        """Additive, idempotent migrations for databases created by an older
        schema. Only adds missing columns — never drops or rewrites data."""
        with self._lock:
            cols = {r["name"] for r in self._conn.execute(
                "PRAGMA table_info(face_embeddings)").fetchall()}
            if "quality" not in cols:
                self._conn.execute("ALTER TABLE face_embeddings ADD COLUMN quality REAL")
            if "meta" not in cols:
                self._conn.execute("ALTER TABLE face_embeddings ADD COLUMN meta TEXT")
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
