"""
TraceStore: compressed state persistence for outcome processing.

Traces are written at the end of each request (if trace_persist_required=True)
and read by the async outcome processor when report_decision_outcome arrives.
UPSERT semantics — safe to retry.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional


class TraceStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, trace_id: str, compressed_state: dict) -> None:
        """UPSERT: safe to call multiple times for the same trace_id."""
        user_id = compressed_state.get("user_id", "")
        self._conn.execute(
            """
            INSERT INTO traces (trace_id, user_id, data, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(trace_id) DO UPDATE SET data = excluded.data
            """,
            (
                trace_id,
                user_id,
                json.dumps(compressed_state),
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()

    def get(self, trace_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT data FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else None

    def list_recent(self, user_id: str, days: int = 30) -> list[dict]:
        """Returns deserialized trace states for user in last `days` days, newest first."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            """
            SELECT trace_id, data FROM traces
            WHERE user_id = ? AND created_at >= ?
            ORDER BY created_at DESC
            """,
            (user_id, cutoff),
        ).fetchall()
        result = []
        for row in rows:
            d = json.loads(row["data"])
            d["trace_id"] = row["trace_id"]
            result.append(d)
        return result

    def count_recent(self, user_id: str, days: int = 30) -> int:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        row = self._conn.execute(
            "SELECT COUNT(*) FROM traces WHERE user_id = ? AND created_at >= ?",
            (user_id, cutoff),
        ).fetchone()
        return row[0] if row else 0
