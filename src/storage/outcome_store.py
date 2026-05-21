"""
OutcomeStore: records decisions outcomes from report_decision_outcome.

UPSERT semantics (safe to retry). Outcome coverage is tracked here
per patch §B4: outcome_coverage_rate = reported_outcomes / total_decisions_30d.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Literal, Optional


OutcomeType = Literal["accepted", "edited", "rejected", "used_alternative"]


class OutcomeStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(
        self,
        user_id: str,
        trace_id: str,
        outcome_type: OutcomeType,
        *,
        original_content: Optional[str] = None,
        edited_content: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        alternative_id: Optional[str] = None,
        meta_weight: float = 1.0,
    ) -> str:
        outcome_id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO outcomes
                (outcome_id, user_id, trace_id, outcome_type,
                 original_content, edited_content, rejection_reason,
                 alternative_id, reported_at, meta_weight)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(outcome_id) DO NOTHING
            """,
            (
                outcome_id,
                user_id,
                trace_id,
                outcome_type,
                original_content,
                edited_content,
                rejection_reason,
                alternative_id,
                datetime.now().isoformat(),
                meta_weight,
            ),
        )
        self._conn.commit()
        return outcome_id

    def get_by_trace(self, trace_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM outcomes WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        return dict(row) if row else None

    def count_for_user(self, user_id: str, days: int = 30) -> int:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        row = self._conn.execute(
            "SELECT COUNT(*) FROM outcomes WHERE user_id = ? AND reported_at >= ?",
            (user_id, cutoff),
        ).fetchone()
        return row[0] if row else 0

    def unprocessed(self, user_id: str, limit: int = 100) -> list[dict]:
        """Returns outcomes not yet picked up by the outcome processor."""
        rows = self._conn.execute(
            """
            SELECT * FROM outcomes
            WHERE user_id = ? AND processed_at IS NULL
            ORDER BY reported_at ASC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_recent_reversals(
        self, user_id: str, days: int = 30
    ) -> list[dict]:
        """Returns rejected/edited outcomes from the last `days` days, newest first."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            """
            SELECT * FROM outcomes
            WHERE user_id = ?
              AND outcome_type IN ('rejected', 'edited')
              AND reported_at >= ?
            ORDER BY reported_at DESC
            """,
            (user_id, cutoff),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_processed(self, outcome_id: str) -> None:
        """Records the processing timestamp. Called in a try/finally so it always runs."""
        self._conn.execute(
            "UPDATE outcomes SET processed_at = ? WHERE outcome_id = ?",
            (datetime.now().isoformat(), outcome_id),
        )
        self._conn.commit()

    def list_recent(self, user_id: str, days: int = 30) -> list[dict]:
        """Returns all outcomes for user in last `days` days, newest first."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            """
            SELECT * FROM outcomes
            WHERE user_id = ? AND reported_at >= ?
            ORDER BY reported_at DESC
            """,
            (user_id, cutoff),
        ).fetchall()
        return [dict(r) for r in rows]

    def assess_coverage(self, user_id: str, trace_store) -> dict:
        """
        Returns outcome coverage state per patch §B4.
        trace_store: TraceStore instance (injected to avoid circular import).
        """
        recent_decisions = trace_store.count_recent(user_id, days=30)
        reported = self.count_for_user(user_id, days=30)

        if recent_decisions == 0:
            return {"state": "no_recent_activity", "rate": None, "warning": None}

        rate = reported / recent_decisions

        if rate >= 0.7:
            return {"state": "healthy", "rate": rate, "warning": None}
        elif rate >= 0.3:
            return {
                "state": "degraded",
                "rate": rate,
                "warning": (
                    "Outcome coverage below 70%. "
                    "Meta-learning is operating on partial signal."
                ),
            }
        else:
            return {
                "state": "impaired",
                "rate": rate,
                "warning": (
                    "Outcome coverage below 30%. Meta-learning is effectively disabled. "
                    "Parent agent should call report_decision_outcome to enable learning."
                ),
            }
