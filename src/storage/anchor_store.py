"""
AnchorStore: identity-constitutive pattern store (slow-changing).

Embeddings stored in the same row (BLOB via JSON TEXT) per spec §6.1.
Chroma collection "anchors" provides semantic search.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Callable, Optional

import chromadb

from .models import AnchorModel


EmbedFn = Callable[[list[str]], list[list[float]]]


def _row_to_anchor(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["structured_form"] = json.loads(d["structured_form"]) if d["structured_form"] else None
    d["supporting_shard_ids"] = json.loads(d["supporting_shard_ids"])
    d["contradicting_shard_ids"] = json.loads(d["contradicting_shard_ids"])
    d["context_scope"] = json.loads(d["context_scope"])
    d["embedding"] = json.loads(d["embedding"])
    d["established_at"] = datetime.fromisoformat(d["established_at"])
    d["last_reinforced_at"] = datetime.fromisoformat(d["last_reinforced_at"])
    d["last_user_confirmed_at"] = (
        datetime.fromisoformat(d["last_user_confirmed_at"])
        if d["last_user_confirmed_at"]
        else None
    )
    return d


class AnchorStore:
    def __init__(
        self,
        conn: sqlite3.Connection,
        chroma_persist_dir: str,
        embed_fn: Optional[EmbedFn] = None,
    ) -> None:
        self._conn = conn
        self._embed_fn = embed_fn
        client = chromadb.PersistentClient(path=chroma_persist_dir)
        self._collection = client.get_or_create_collection("anchors")

    def _embed(self, text: str) -> list[float]:
        if self._embed_fn is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            self._embed_fn = lambda texts: model.encode(texts, show_progress_bar=False).tolist()
        return self._embed_fn([text])[0]

    def add(self, anchor: dict) -> str:
        model = AnchorModel.model_validate(anchor)
        if not model.embedding:
            model.embedding = self._embed(model.statement)
        anchor_id = model.anchor_id or str(uuid.uuid4())
        model.anchor_id = anchor_id

        self._conn.execute(
            """
            INSERT OR REPLACE INTO anchors
                (anchor_id, user_id, statement, structured_form, confidence,
                 supporting_shard_ids, contradicting_shard_ids, context_scope,
                 established_at, last_reinforced_at, last_user_confirmed_at, embedding)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
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
                json.dumps(model.embedding),
            ),
        )
        self._conn.commit()

        self._collection.upsert(
            ids=[anchor_id],
            embeddings=[model.embedding],
            documents=[model.statement],
            metadatas=[{
                "user_id": model.user_id,
                "context_scope": json.dumps(model.context_scope),
                "confidence": model.confidence,
            }],
        )
        return anchor_id

    def get(self, anchor_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM anchors WHERE anchor_id = ?", (anchor_id,)
        ).fetchone()
        return _row_to_anchor(row) if row else None

    def list_for_user(self, user_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM anchors WHERE user_id = ?", (user_id,)
        ).fetchall()
        return [_row_to_anchor(r) for r in rows]

    def count_for_user(self, user_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM anchors WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row[0] if row else 0

    def search(self, query_text: str, user_id: str, k: int = 10) -> list[dict]:
        embedding = self._embed(query_text)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where={"user_id": {"$eq": user_id}},
            include=["distances"],
        )
        ids = results["ids"][0] if results["ids"] else []
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT * FROM anchors WHERE anchor_id IN ({placeholders})", ids
        ).fetchall()
        return [_row_to_anchor(r) for r in rows]

    def update_confidence(self, anchor_id: str, delta: float) -> None:
        self._conn.execute(
            """
            UPDATE anchors
            SET confidence = MAX(0.0, MIN(1.0, confidence + ?))
            WHERE anchor_id = ?
            """,
            (delta, anchor_id),
        )
        self._conn.commit()

    def list_dormant(self, user_id: str, days: int = 180) -> list[dict]:
        """Returns anchors not reinforced in the last `days` days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            "SELECT * FROM anchors WHERE user_id = ? AND last_reinforced_at < ?",
            (user_id, cutoff),
        ).fetchall()
        return [_row_to_anchor(r) for r in rows]

    def list_contradicted(self, user_id: str) -> list[dict]:
        """Returns anchors where contradicting shards outnumber supporting shards."""
        anchors = self.list_for_user(user_id)
        return [
            a for a in anchors
            if len(a["contradicting_shard_ids"]) > len(a["supporting_shard_ids"])
        ]

    def confirm_anchor(self, anchor_id: str) -> None:
        """Resets last_user_confirmed_at to now (e.g. after explicit user confirmation)."""
        self._conn.execute(
            "UPDATE anchors SET last_user_confirmed_at = ? WHERE anchor_id = ?",
            (datetime.now().isoformat(), anchor_id),
        )
        self._conn.commit()

    def demote_anchor(self, anchor_id: str, reason: str = "") -> None:
        """Moves the anchor to demoted_anchors and removes it from active anchors."""
        anchor = self.get(anchor_id)
        if not anchor:
            return

        now = datetime.now()
        self._conn.execute(
            """
            INSERT INTO demoted_anchors
                (anchor_id, user_id, statement, structured_form, confidence,
                 supporting_shard_ids, contradicting_shard_ids, context_scope,
                 established_at, last_reinforced_at, demoted_at, demotion_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                anchor["anchor_id"],
                anchor["user_id"],
                anchor["statement"],
                json.dumps(anchor["structured_form"]) if anchor["structured_form"] else None,
                anchor["confidence"],
                json.dumps(anchor["supporting_shard_ids"]),
                json.dumps(anchor["contradicting_shard_ids"]),
                json.dumps(anchor["context_scope"]),
                anchor["established_at"].isoformat(),
                anchor["last_reinforced_at"].isoformat(),
                now.isoformat(),
                reason,
            ),
        )
        self._conn.execute("DELETE FROM anchors WHERE anchor_id = ?", (anchor_id,))
        self._conn.commit()
        try:
            self._collection.delete(ids=[anchor_id])
        except Exception:
            pass

    def count_demoted(self, user_id: str) -> int:
        """Returns total number of demoted anchors for user."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM demoted_anchors WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row[0] if row else 0

    def count_demoted_in_period(self, user_id: str, days: int = 30) -> int:
        """Returns anchors demoted in the last `days` days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        row = self._conn.execute(
            "SELECT COUNT(*) FROM demoted_anchors WHERE user_id = ? AND demoted_at >= ?",
            (user_id, cutoff),
        ).fetchone()
        return row[0] if row else 0

    def add_supporting_shard(self, anchor_id: str, shard_id: str) -> None:
        row = self._conn.execute(
            "SELECT supporting_shard_ids FROM anchors WHERE anchor_id = ?", (anchor_id,)
        ).fetchone()
        if not row:
            return
        ids = json.loads(row["supporting_shard_ids"])
        if shard_id not in ids:
            ids.append(shard_id)
            self._conn.execute(
                "UPDATE anchors SET supporting_shard_ids = ? WHERE anchor_id = ?",
                (json.dumps(ids), anchor_id),
            )
            self._conn.commit()
