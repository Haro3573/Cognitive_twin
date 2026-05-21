"""
SQLite database setup: connection, WAL mode, schema, advisory write lock.

All stores share a single connection. WAL mode allows concurrent reads while
the async promotion loop holds the advisory write lock for batch writes.
"""

import sqlite3
import threading
import contextlib
from typing import Generator

# Module-level advisory write lock for async promotion loop writes.
# The async loop acquires this before any batch governance mutation.
# Request-path reads never acquire it — they see WAL-consistent snapshots.
_write_lock = threading.Lock()


@contextlib.contextmanager
def advisory_write_lock() -> Generator[None, None, None]:
    """Context manager used by async loops for batch writes to governance tables."""
    acquired = _write_lock.acquire(timeout=30.0)
    if not acquired:
        raise TimeoutError("Could not acquire advisory write lock within 30 seconds")
    try:
        yield
    finally:
        _write_lock.release()


def open_db(path: str) -> sqlite3.Connection:
    """Opens a SQLite connection with WAL mode, row_factory, and FK enforcement."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS shards (
    shard_id           TEXT PRIMARY KEY,
    user_id            TEXT NOT NULL,
    context            TEXT NOT NULL,
    content            TEXT NOT NULL,
    compression_level  INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL,
    last_activated_at  TEXT NOT NULL,
    activation_count   INTEGER NOT NULL DEFAULT 0,
    decay_score        REAL NOT NULL DEFAULT 0.0,
    domain_tags        TEXT NOT NULL DEFAULT '[]',
    embedding          TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_shards_user_last_activated
    ON shards(user_id, last_activated_at);

CREATE TABLE IF NOT EXISTS anchors (
    anchor_id              TEXT PRIMARY KEY,
    user_id                TEXT NOT NULL,
    statement              TEXT NOT NULL,
    structured_form        TEXT,
    confidence             REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    supporting_shard_ids   TEXT NOT NULL DEFAULT '[]',
    contradicting_shard_ids TEXT NOT NULL DEFAULT '[]',
    context_scope          TEXT NOT NULL DEFAULT '[]',
    established_at         TEXT NOT NULL,
    last_reinforced_at     TEXT NOT NULL,
    last_user_confirmed_at TEXT,
    embedding              TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_anchors_user_id ON anchors(user_id);

CREATE TABLE IF NOT EXISTS pending_anchors (
    anchor_id                TEXT PRIMARY KEY,
    user_id                  TEXT NOT NULL,
    statement                TEXT NOT NULL,
    structured_form          TEXT,
    confidence               REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    supporting_shard_ids     TEXT NOT NULL DEFAULT '[]',
    contradicting_shard_ids  TEXT NOT NULL DEFAULT '[]',
    context_scope            TEXT NOT NULL DEFAULT '[]',
    established_at           TEXT NOT NULL,
    last_reinforced_at       TEXT NOT NULL,
    last_user_confirmed_at   TEXT,
    pending_confirmation_at  TEXT NOT NULL,
    seeded_from              TEXT NOT NULL DEFAULT 'system',
    embedding                TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS governance_rules (
    rule_id             TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    version             INTEGER NOT NULL DEFAULT 1,
    statement           TEXT NOT NULL,
    structured_form     TEXT,
    confidence          REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    evidence_count      INTEGER NOT NULL DEFAULT 0,
    context_scope       TEXT NOT NULL DEFAULT '[]',
    supporting_traces   TEXT NOT NULL DEFAULT '[]',
    contradicting_traces TEXT NOT NULL DEFAULT '[]',
    activated_at        TEXT NOT NULL,
    supersedes          TEXT,
    superseded_by       TEXT,
    rule_class          TEXT NOT NULL
                        CHECK (rule_class IN ('value','preference','constraint','heuristic'))
);
CREATE INDEX IF NOT EXISTS idx_rules_user_active
    ON governance_rules(user_id, superseded_by);

CREATE TABLE IF NOT EXISTS proposals (
    proposal_id        TEXT PRIMARY KEY,
    user_id            TEXT NOT NULL,
    type               TEXT NOT NULL,
    target_rule_id     TEXT,
    proposed_rule      TEXT,
    rationale          TEXT NOT NULL,
    evidence_count     INTEGER NOT NULL DEFAULT 1,
    weight             REAL NOT NULL DEFAULT 1.0,
    promotion_threshold INTEGER NOT NULL,
    supporting_traces  TEXT NOT NULL DEFAULT '[]',
    first_observed     TEXT NOT NULL,
    last_reinforced    TEXT NOT NULL,
    context_signature  TEXT NOT NULL DEFAULT '[]',
    merge_key          TEXT NOT NULL,
    delta              REAL,
    status             TEXT NOT NULL DEFAULT 'active'
                       CHECK (status IN (
                           'active','promoted','discarded',
                           'superseded_by_add_rule_proposal'
                       ))
);
CREATE INDEX IF NOT EXISTS idx_proposals_user_merge ON proposals(user_id, merge_key);
CREATE INDEX IF NOT EXISTS idx_proposals_type_status ON proposals(type, status);

CREATE TABLE IF NOT EXISTS traces (
    trace_id   TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    data       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_user_created
    ON traces(user_id, created_at);

CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id       TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    trace_id         TEXT NOT NULL,
    outcome_type     TEXT NOT NULL
                     CHECK (outcome_type IN ('accepted','edited','rejected','used_alternative')),
    original_content TEXT,
    edited_content   TEXT,
    rejection_reason TEXT,
    alternative_id   TEXT,
    reported_at      TEXT NOT NULL,
    meta_weight      REAL NOT NULL DEFAULT 1.0,
    processed_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_outcomes_user_reported
    ON outcomes(user_id, reported_at);
CREATE INDEX IF NOT EXISTS idx_outcomes_trace_id
    ON outcomes(trace_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_unprocessed
    ON outcomes(user_id, processed_at);

CREATE TABLE IF NOT EXISTS demoted_anchors (
    anchor_id               TEXT PRIMARY KEY,
    user_id                 TEXT NOT NULL,
    statement               TEXT NOT NULL,
    structured_form         TEXT,
    confidence              REAL NOT NULL,
    supporting_shard_ids    TEXT NOT NULL DEFAULT '[]',
    contradicting_shard_ids TEXT NOT NULL DEFAULT '[]',
    context_scope           TEXT NOT NULL DEFAULT '[]',
    established_at          TEXT NOT NULL,
    last_reinforced_at      TEXT NOT NULL,
    demoted_at              TEXT NOT NULL,
    demotion_reason         TEXT
);
CREATE INDEX IF NOT EXISTS idx_demoted_anchors_user
    ON demoted_anchors(user_id);

CREATE TABLE IF NOT EXISTS review_items (
    review_id    TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    item_type    TEXT NOT NULL,
    item_id      TEXT NOT NULL,
    context      TEXT NOT NULL DEFAULT '{}',
    surfaced_at  TEXT NOT NULL,
    response     TEXT,
    responded_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_items_user_pending
    ON review_items(user_id, responded_at);
"""


_MIGRATION_SQL = [
    # outcomes.processed_at — added for async outcome processor
    "ALTER TABLE outcomes ADD COLUMN processed_at TEXT",
    # governance_rules.status — added for deprecation support
    (
        "ALTER TABLE governance_rules ADD COLUMN status TEXT NOT NULL DEFAULT 'active' "
        "CHECK(status IN ('active','deprecated'))"
    ),
]


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Applies incremental ALTER TABLE migrations. Safe to run on every startup."""
    for sql in _MIGRATION_SQL:
        try:
            conn.execute(sql)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


def init_schema(conn: sqlite3.Connection) -> None:
    """Creates all tables/indexes and runs incremental column migrations."""
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    _run_migrations(conn)
