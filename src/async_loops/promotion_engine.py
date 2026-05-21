"""
Promotion engine: promotes eligible proposals to live governance rules.

Eligible = status='active' AND evidence_count >= promotion_threshold.
Runs under the advisory write lock to prevent concurrent governance mutations
from the request path (which only reads — but we still serialize writers).
"""

import uuid
from datetime import datetime
from typing import Optional

from src.deps import Stores
from src.storage.db import advisory_write_lock


def promote_proposals(
    user_id: str,
    stores: Stores,
    llm: Optional[object] = None,
) -> dict:
    """
    Promotes all eligible proposals for user_id.

    Returns {"promoted": n, "skipped": m} where skipped = proposals whose
    target rule was not found (already superseded or deleted).
    """
    eligible = stores.proposals.eligible_for_review(user_id)
    if not eligible:
        return {"promoted": 0, "skipped": 0}

    promoted = 0
    skipped = 0

    with advisory_write_lock():
        for proposal in eligible:
            success = _promote_one(proposal, stores)
            if success:
                promoted += 1
                proposal["status"] = "promoted"
            else:
                skipped += 1
                proposal["status"] = "discarded"
            stores.proposals.update(proposal)

    return {"promoted": promoted, "skipped": skipped}


def _promote_one(proposal: dict, stores: Stores) -> bool:
    """
    Applies a single proposal. Returns True on success, False if the
    target rule is missing or the proposal type is unrecognised.
    """
    p_type = proposal["type"]

    if p_type == "add_rule":
        proposed = proposal.get("proposed_rule") or {}
        rule = {
            "rule_id": str(uuid.uuid4()),
            "user_id": proposal["user_id"],
            "version": 1,
            "statement": proposed.get("statement", ""),
            "confidence": max(0.0, min(1.0, proposed.get("confidence_adjustment", 0.5) + 0.5)),
            "evidence_count": proposal.get("evidence_count", 1),
            "context_scope": proposed.get("context_scope", []),
            "supporting_traces": proposal.get("supporting_traces", []),
            "contradicting_traces": [],
            "activated_at": datetime.now(),
            "supersedes": None,
            "rule_class": proposed.get("rule_class", "preference"),
        }
        stores.governance.add_rule(rule)
        return True

    elif p_type == "modify_rule":
        target_id = proposal.get("target_rule_id")
        if not target_id or not stores.governance.get(target_id):
            return False
        proposed = proposal.get("proposed_rule") or {}
        stores.governance.modify(target_id, proposed)
        return True

    elif p_type == "deprecate_rule":
        target_id = proposal.get("target_rule_id")
        if not target_id or not stores.governance.get(target_id):
            return False
        stores.governance.deprecate(target_id)
        return True

    elif p_type in ("adjust_weight", "investigate_divergence", "investigate_rule"):
        # These types don't produce direct governance mutations in v1;
        # they are recorded as promoted so they don't re-trigger.
        return True

    return False
