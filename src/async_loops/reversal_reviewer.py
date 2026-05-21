"""
Reversal reviewer: surfaces decision reversals for user review.

A reversal occurs when a decision outcome is "rejected" (or edited with a
directional_change) for a trace whose rule_basis overlaps with previously
accepted decisions. This suggests a rule is misfiring.

In v1 this loop:
  1. Finds recent rejected/edited outcomes
  2. For each, checks if the rule_basis overlaps with proposal deprecation candidates
  3. Enqueues a review_item so the user can confirm or dismiss

No LLM is used here — purely deterministic rule-overlap detection.
"""

from typing import Optional

from src.deps import Stores
from src.storage.review_store import ReviewStore

LOOKBACK_DAYS = 30


def review_reversals(
    user_id: str,
    stores: Stores,
    review_store: ReviewStore,
) -> dict:
    """
    Surfaces reversal patterns for user review.

    Returns {"queued": n} where n is newly enqueued reversal review items.
    """
    pending_trace_ids = {
        item["item_id"]
        for item in review_store.list_pending(user_id)
        if item["item_type"] == "reversal_pattern"
    }

    rows = stores.outcomes.list_recent_reversals(user_id, days=LOOKBACK_DAYS)

    queued = 0
    for outcome in rows:
        trace_id = outcome["trace_id"]

        if trace_id in pending_trace_ids:
            continue

        state = stores.traces.get(trace_id)
        if not state:
            continue

        rule_basis = state.get("rule_basis") or []
        if not rule_basis:
            continue

        # Check if any basis rule has a deprecation proposal already
        has_deprecation_candidate = any(
            stores.governance.rule_contradicting_evidence_ratio(rid) >= 0.3
            for rid in rule_basis
        )
        if not has_deprecation_candidate:
            continue

        output_payload = state.get("output_payload") or {}
        decision = output_payload.get("decision") or {}

        review_store.enqueue(
            user_id=user_id,
            item_type="reversal_pattern",
            item_id=trace_id,
            context={
                "outcome_type": outcome["outcome_type"],
                "rejection_reason": outcome.get("rejection_reason"),
                "decision_content": decision.get("content", ""),
                "rule_basis": rule_basis,
                "reported_at": outcome["reported_at"],
            },
        )
        queued += 1

    return {"queued": queued}
