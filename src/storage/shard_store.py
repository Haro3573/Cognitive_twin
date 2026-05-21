"""
ShardStore: episodic memory store.

SQLite holds all shard data (including embeddings for cosine similarity in
baseline detection). Chroma holds embeddings + minimal metadata for semantic
search. Both are written on insert/update — SQLite is authoritative.

See DECISIONS.md D6 for why embeddings are duplicated.
"""

import json
import math
import random
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Callable, Optional

import chromadb

from .models import ShardModel


EmbedFn = Callable[[list[str]], list[list[float]]]


def _default_embed_fn() -> EmbedFn:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return lambda texts: model.encode(texts, show_progress_bar=False).tolist()


def _row_to_shard(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["context"] = json.loads(d["context"])
    d["domain_tags"] = json.loads(d["domain_tags"])
    d["embedding"] = json.loads(d["embedding"])
    d["created_at"] = datetime.fromisoformat(d["created_at"])
    d["last_activated_at"] = datetime.fromisoformat(d["last_activated_at"])
    return d


class ShardStore:
    """
    Dual-write shard store. SQLite is authoritative; Chroma is the search index.

    Public methods match the tool contracts expected by recall_subagent:
      add(shard_dict) → shard_id
      search(query_text, user_id, k, filter_tags) → list[dict]
      get(shard_id) → dict | None
      update_activation(shard_id, now) → None
      sample_recent(user_id, days, sample_size, recency_weighted) → list[dict]
      count_shards_matching_anchor(user_id, anchor_id, anchor_shard_ids, days) → int
      count_for_user(user_id) → int
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        chroma_persist_dir: str,
        embed_fn: Optional[EmbedFn] = None,
    ) -> None:
        self._conn = conn
        self._embed_fn: EmbedFn = embed_fn if embed_fn is not None else _default_embed_fn()
        client = chromadb.PersistentClient(path=chroma_persist_dir)
        self._collection = client.get_or_create_collection("shards")

    def embed(self, text: str) -> list[float]:
        return self._embed_fn([text])[0]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, shard: dict) -> str:
        """Validates, inserts into SQLite and Chroma. Returns shard_id."""
        model = ShardModel.model_validate(shard)
        if not model.embedding:
            model.embedding = self.embed(model.content)

        shard_id = model.shard_id or str(uuid.uuid4())
        model.shard_id = shard_id

        self._conn.execute(
            """
            INSERT OR REPLACE INTO shards
                (shard_id, user_id, context, content, compression_level,
                 created_at, last_activated_at, activation_count,
                 decay_score, domain_tags, embedding)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                model.shard_id,
                model.user_id,
                json.dumps(model.context),
                model.content,
                model.compression_level,
                model.created_at.isoformat(),
                model.last_activated_at.isoformat(),
                model.activation_count,
                model.decay_score,
                json.dumps(model.domain_tags),
                json.dumps(model.embedding),
            ),
        )
        self._conn.commit()

        self._collection.upsert(
            ids=[shard_id],
            embeddings=[model.embedding],
            documents=[model.content],
            metadatas=[{
                "user_id": model.user_id,
                "domain_tags": json.dumps(model.domain_tags),
                "last_activated_at": int(model.last_activated_at.timestamp()),
            }],
        )
        return shard_id

    def update_activation(self, shard_id: str, now: Optional[datetime] = None) -> None:
        """Increments activation_count and refreshes last_activated_at."""
        now = now or datetime.now()
        self._conn.execute(
            """
            UPDATE shards
            SET activation_count = activation_count + 1, last_activated_at = ?
            WHERE shard_id = ?
            """,
            (now.isoformat(), shard_id),
        )
        self._conn.commit()
        self._collection.update(
            ids=[shard_id],
            metadatas=[{"last_activated_at": int(now.timestamp())}],
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, shard_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM shards WHERE shard_id = ?", (shard_id,)
        ).fetchone()
        return _row_to_shard(row) if row else None

    def search(
        self,
        query_text: str,
        user_id: str,
        k: int = 10,
        domain_tags: Optional[list[str]] = None,
    ) -> list[dict]:
        """Semantic search via Chroma; full shard data from SQLite."""
        query_embedding = self.embed(query_text)
        where: dict = {"user_id": {"$eq": user_id}}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["distances"],
        )
        ids = results["ids"][0] if results["ids"] else []
        if not ids:
            return []

        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT * FROM shards WHERE shard_id IN ({placeholders})", ids
        ).fetchall()
        shards = [_row_to_shard(r) for r in rows]

        if domain_tags:
            tag_set = set(domain_tags)
            shards = [s for s in shards if set(s["domain_tags"]) & tag_set]

        return shards

    def sample_recent(
        self,
        user_id: str,
        days: int = 30,
        sample_size: int = 20,
        recency_weighted: bool = True,
    ) -> list[dict]:
        """
        Returns up to sample_size shards from the last `days` days.
        recency_weighted=True uses exponential weighting toward more recent shards.
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            """
            SELECT * FROM shards
            WHERE user_id = ? AND last_activated_at >= ?
            ORDER BY last_activated_at DESC
            """,
            (user_id, cutoff),
        ).fetchall()

        if not rows:
            return []
        if len(rows) <= sample_size:
            return [_row_to_shard(r) for r in rows]

        if not recency_weighted:
            return [_row_to_shard(r) for r in random.sample(rows, sample_size)]

        # Exponential recency weighting: weight ∝ exp(-age / window)
        now_ts = datetime.now().timestamp()
        window_seconds = days * 86400.0
        weights = []
        for row in rows:
            age = now_ts - datetime.fromisoformat(row["last_activated_at"]).timestamp()
            weights.append(math.exp(-age / window_seconds))

        # Weighted sampling without replacement
        selected: set[int] = set()
        attempts = 0
        while len(selected) < sample_size and attempts < sample_size * 10:
            i = random.choices(range(len(rows)), weights=weights, k=1)[0]
            selected.add(i)
            attempts += 1

        return [_row_to_shard(rows[i]) for i in selected]

    def count_shards_matching_anchor(
        self,
        user_id: str,
        anchor_shard_ids: list[str],
        days: int = 30,
    ) -> int:
        """
        Counts how many of the anchor's supporting shards were activated in the last `days`.

        anchor_shard_ids: the anchor's supporting_shard_ids list (from AnchorStore).
        Uses an indexed query on last_activated_at (see DECISIONS.md D6).
        """
        if not anchor_shard_ids:
            return 0
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        placeholders = ",".join("?" * len(anchor_shard_ids))
        row = self._conn.execute(
            f"""
            SELECT COUNT(*) FROM shards
            WHERE user_id = ?
              AND shard_id IN ({placeholders})
              AND last_activated_at >= ?
            """,
            (user_id, *anchor_shard_ids, cutoff),
        ).fetchone()
        return row[0] if row else 0

    def count_for_user(self, user_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM shards WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row[0] if row else 0

    def add_from_trace(
        self,
        state: dict,
        outcome: dict,
        content_override: Optional[str] = None,
    ) -> str:
        """Creates a shard from a completed trace state. Returns shard_id."""
        hypothesis = state.get("selected_hypothesis") or {}
        content = content_override or hypothesis.get("content", "")
        perceived = state.get("perceived_context") or {}
        domain_tags = list(perceived.get("domain_tags", []))
        now = datetime.now()
        shard = {
            "shard_id": str(uuid.uuid4()),
            "user_id": state.get("user_id", ""),
            "context": perceived,
            "content": content,
            "compression_level": 0,
            "created_at": now,
            "last_activated_at": now,
            "activation_count": 1,
            "decay_score": 0.0,
            "domain_tags": domain_tags,
            "embedding": [],
        }
        return self.add(shard)

    def compress(
        self,
        shard_id: str,
        summary: str,
        new_embedding: list[float],
        new_level: int,
    ) -> None:
        """Replaces shard content with a compressed summary."""
        self._conn.execute(
            """
            UPDATE shards
            SET content = ?, embedding = ?, compression_level = ?
            WHERE shard_id = ?
            """,
            (summary, json.dumps(new_embedding), new_level, shard_id),
        )
        self._conn.commit()
        try:
            self._collection.update(
                ids=[shard_id],
                embeddings=[new_embedding],
                documents=[summary],
            )
        except Exception:
            pass

    def update_decay_score(self, shard_id: str, new_score: float) -> None:
        """Persists the computed decay score without touching other fields."""
        self._conn.execute(
            "UPDATE shards SET decay_score = ? WHERE shard_id = ?",
            (new_score, shard_id),
        )
        self._conn.commit()

    def delete(self, shard_id: str) -> None:
        """Removes a fully-decayed shard from SQLite and Chroma."""
        self._conn.execute("DELETE FROM shards WHERE shard_id = ?", (shard_id,))
        self._conn.commit()
        try:
            self._collection.delete(ids=[shard_id])
        except Exception:
            pass

    def list_for_decay(self, user_id: str, min_age_days: int = 90) -> list[dict]:
        """Returns shards inactive for at least min_age_days, ordered oldest first."""
        cutoff = (datetime.now() - timedelta(days=min_age_days)).isoformat()
        rows = self._conn.execute(
            """
            SELECT * FROM shards
            WHERE user_id = ? AND last_activated_at < ?
            ORDER BY last_activated_at ASC
            """,
            (user_id, cutoff),
        ).fetchall()
        return [_row_to_shard(r) for r in rows]
