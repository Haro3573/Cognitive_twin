"""
PendingAnchorStore: staging area for anchor candidates awaiting user confirmation.

Confirmation flow: anchor_consolidation job surfaces pending anchors;
user confirms → moved to AnchorStore; user rejects → deleted here.
See patch §A5.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Optional

from .models import PendingAnchorModel


def _row_to_pending(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["structured_form"] = json.loads(d["structured_form"]) if d["structured_form"] else None
    d["supporting_shard_ids"] = json.loads(d["supporting_shard_ids"])
    d["contradicting_shard_ids"] = json.loads(d["contradicting_shard_ids"])
    d["context_scope"] = json.loads(d["context_scope"])
    d["embedding"] = json.loads(d["embedding"])
    d["established_at"] = datetime.fromisoformat(d["established_at"])
    d["last_reinforced_at"] = datetime.fromisoformat(d["last_reinforced_at"])
    d["pending_confirmation_at"] = datetime.fromisoformat(d["pending_confirmation_at"])
    d["last_user_confirmed_at"] = (
        datetime.fromisoformat(d["last_user_confirmed_at"])
        if d["last_user_confirmed_at"]
        else None
    )
    return d


class PendingAnchorStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def stage(self, anchor: dict) -> str:
        """Stages an anchor for user confirmation. Returns anchor_id."""
        model = PendingAnchorModel.model_validate(anchor)
        anchor_id = model.anchor_id or str(uuid.uuid4())
        model.anchor_id = anchor_id

        self._conn.execute(
            """
            INSERT OR REPLACE INTO pending_anchors
                (anchor_id, user_id, statement, structured_form, confidence,
                 supporting_shard_ids, contradicting_shard_ids, context_scope,
                 established_at, last_reinforced_at, last_user_confirmed_at,
                 pending_confirmation_at, seeded_from, embedding)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                model.anchor_id,
                model.user_id,
                model.statement,
                json.dumps(model.structured_form) if model.structured_form else None,
                model.confidence,
                json.dumps(model.supporting_shard_ids),
                json.dumps(model.contradicting_shard_ids),
                json.dumps(model.context_scope),
                model.established_at.isoformat(),
                model.last_reinforced_at.isoformat(),
                model.last_user_confirmed_at.isoformat() if model.last_user_confirmed_at else None,
                model.pending_confirmation_at.isoformat(),
                model.seeded_from,
                json.dumps(model.embedding),
            ),
        )
        self._conn.commit()
        return anchor_id

    def get(self, anchor_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM pending_anchors WHERE anchor_id = ?", (anchor_id,)
        ).fetchone()
        return _row_to_pending(row) if row else None

    def list_for_user(self, user_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM pending_anchors WHERE user_id = ?", (user_id,)
        ).fetchall()
        return [_row_to_pending(r) for r in rows]

    def delete(self, anchor_id: str) -> None:
        self._conn.execute(
            "DELETE FROM pending_anchors WHERE anchor_id = ?", (anchor_id,)
        )
        self._conn.commit()
