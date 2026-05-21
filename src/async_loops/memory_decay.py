"""
Memory decay loop: ages and compresses stale shards.

Decay formula (applied per run):
  new_decay = old_decay + AGE_RATE * (days_inactive / FULL_DECAY_DAYS)
  Clamped to [0.0, 1.0].

Compression trigger: decay_score >= COMPRESS_THRESHOLD AND activation_count <= COMPRESS_MAX_ACTIVATIONS
Deletion trigger:    decay_score >= DELETE_THRESHOLD AND compression_level >= 1

LLM is used for semantic compression (summarise shard content). Without LLM,
compression is skipped and only deletion of already-compressed shards runs.
"""

from datetime import datetime, timedelta
from typing import Optional

from src.deps import Stores

AGE_RATE = 0.1
FULL_DECAY_DAYS = 365.0
COMPRESS_THRESHOLD = 0.5
COMPRESS_MAX_ACTIVATIONS = 3
DELETE_THRESHOLD = 0.85

_COMPRESS_PROMPT = """\
Compress the following memory entry to its core factual claim in one sentence (max 100 chars).
Preserve the most specific and unusual detail.

Original: {content}

Compressed:"""


def decay_memory(
    user_id: str,
    stores: Stores,
    llm: Optional[object] = None,
) -> dict:
    """
    Applies age-based decay to shards inactive for >= 90 days.

    Returns {"decayed": n, "compressed": m, "deleted": k}.
    """
    stale = stores.shards.list_for_decay(user_id, min_age_days=90)
    if not stale:
        return {"decayed": 0, "compressed": 0, "deleted": 0}

    now = datetime.now()
    decayed = compressed = deleted = 0

    for shard in stale:
        inactive_days = (now - shard["last_activated_at"]).days
        age_fraction = min(1.0, inactive_days / FULL_DECAY_DAYS)
        new_decay = min(1.0, shard["decay_score"] + AGE_RATE * age_fraction)

        if (
            new_decay >= DELETE_THRESHOLD
            and shard.get("compression_level", 0) >= 1
        ):
            stores.shards.delete(shard["shard_id"])
            deleted += 1
            continue

        stores.shards.update_decay_score(shard["shard_id"], new_decay)
        decayed += 1

        if (
            llm is not None
            and new_decay >= COMPRESS_THRESHOLD
            and shard.get("activation_count", 0) <= COMPRESS_MAX_ACTIVATIONS
            and shard.get("compression_level", 0) == 0
        ):
            summary = _compress_shard(shard["content"], llm)
            if summary:
                new_embedding = stores.shards.embed(summary)
                stores.shards.compress(
                    shard["shard_id"],
                    summary,
                    new_embedding,
                    new_level=1,
                )
                compressed += 1

    return {"decayed": decayed, "compressed": compressed, "deleted": deleted}


def _compress_shard(content: str, llm) -> Optional[str]:
    """Returns a one-sentence summary, or None on failure."""
    try:
        prompt = _COMPRESS_PROMPT.format(content=content)
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip()
        return text[:150] if text else None
    except Exception:
        return None
