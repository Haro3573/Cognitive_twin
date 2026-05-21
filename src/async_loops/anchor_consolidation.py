"""
Anchor consolidation loop: surfaces dormant or contradicted anchors for user review.

Dormant: not reinforced in 180+ days — may no longer reflect current values.
Contradicted: contradicting_shard_ids > supporting_shard_ids — confidence eroding.

In v1 this loop only enqueues items into the review queue. Actual demotion
requires explicit user confirmation (via ReviewStore.record_response).
"""

from typing import Optional

from src.deps import Stores
from src.storage.review_store import ReviewStore

DORMANCY_DAYS = 180


def consolidate_anchors(
    user_id: str,
    stores: Stores,
    review_store: ReviewStore,
) -> dict:
    """
    Surfaces dormant and contradicted anchors for user review.

    Returns {"dormant_queued": n, "contradicted_queued": m}.
    Already-pending items are not re-enqueued.
    """
    pending_ids = {
        item["item_id"]
        for item in review_store.list_pending(user_id)
    }

    dormant_queued = _queue_dormant(user_id, stores, review_store, pending_ids)
    contradicted_queued = _queue_contradicted(user_id, stores, review_store, pending_ids)

    return {"dormant_queued": dormant_queued, "contradicted_queued": contradicted_queued}


def _queue_dormant(
    user_id: str,
    stores: Stores,
    review_store: ReviewStore,
    already_pending: set,
) -> int:
    dormant = stores.anchors.list_dormant(user_id, days=DORMANCY_DAYS)
    count = 0
    for anchor in dormant:
        if anchor["anchor_id"] in already_pending:
            continue
        review_store.enqueue(
            user_id=user_id,
            item_type="dormant_anchor",
            item_id=anchor["anchor_id"],
            context={
                "statement": anchor["statement"],
                "confidence": anchor["confidence"],
                "last_reinforced_at": anchor["last_reinforced_at"].isoformat(),
                "days_dormant": _days_since(anchor["last_reinforced_at"]),
            },
        )
        count += 1
    return count


def _queue_contradicted(
    user_id: str,
    stores: Stores,
    review_store: ReviewStore,
    already_pending: set,
) -> int:
    contradicted = stores.anchors.list_contradicted(user_id)
    count = 0
    for anchor in contradicted:
        if anchor["anchor_id"] in already_pending:
            continue
        n_supporting = len(anchor["supporting_shard_ids"])
        n_contradicting = len(anchor["contradicting_shard_ids"])
        review_store.enqueue(
            user_id=user_id,
            item_type="contradicted_anchor",
            item_id=anchor["anchor_id"],
            context={
                "statement": anchor["statement"],
                "confidence": anchor["confidence"],
                "supporting_count": n_supporting,
                "contradicting_count": n_contradicting,
            },
        )
        count += 1
    return count


def _days_since(dt) -> int:
    from datetime import datetime
    if hasattr(dt, "timestamp"):
        return (datetime.now() - dt).days
    return 0
