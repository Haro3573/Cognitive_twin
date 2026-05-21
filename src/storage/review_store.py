"""
ReviewStore: queue of items surfaced to the user for confirmation or correction.

Used by anchor_consolidation (dormant/contradicted anchors) and reversal_reviewer
(patterns that suggest a previously-used decision was wrong). The queue is
read-only from the request path; items are written by async loops.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Optional


class ReviewStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def enqueue(
        self,
        user_id: str,
        item_type: str,
        item_id: str,
        context: dict,
    ) -> str:
        """
        Adds an item to the review queue. Returns review_id.

        item_type: "dormant_anchor" | "contradicted_anchor" | "reversal_pattern"
        item_id: the anchor_id or trace_id being surfaced
        """
        review_id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO review_items
                (review_id, user_id, item_type, item_id, context, surfaced_at)
            VALUES (?,?,?,?,?,?)
            """,
            (
                review_id,
                user_id,
                item_type,
                item_id,
                json.dumps(context),
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()
        return review_id

    def list_pending(self, user_id: str) -> list[dict]:
        """Returns review items not yet responded to, oldest first."""
        rows = self._conn.execute(
            """
            SELECT * FROM review_items
            WHERE user_id = ? AND responded_at IS NULL
            ORDER BY surfaced_at ASC
            """,
            (user_id,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["context"] = json.loads(d["context"])
            result.append(d)
        return result

    def record_response(self, review_id: str, response: str) -> None:
        """Records user response and timestamps it."""
        self._conn.execute(
            """
            UPDATE review_items
            SET response = ?, responded_at = ?
            WHERE review_id = ?
            """,
            (response, datetime.now().isoformat(), review_id),
        )
        self._conn.commit()

    def count_pending(self, user_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM review_items WHERE user_id = ? AND responded_at IS NULL",
            (user_id,),
        ).fetchone()
        return row[0] if row else 0

    def list_by_type(self, user_id: str, item_type: str) -> list[dict]:
        """Returns all review items of a given type, including responded ones, oldest first."""
        rows = self._conn.execute(
            """
            SELECT * FROM review_items
            WHERE user_id = ? AND item_type = ?
            ORDER BY surfaced_at ASC
            """,
            (user_id, item_type),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["context"] = json.loads(d["context"])
            result.append(d)
        return result
