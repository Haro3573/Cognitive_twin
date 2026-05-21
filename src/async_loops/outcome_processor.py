"""
Outcome processor: the primary feedback loop between reported outcomes and
the governance/memory system (spec §12, patch §B3).

Per-outcome algorithm:
  1. Fetch trace (skip + mark processed if orphaned)
  2. Read bootstrap flag from output_payload.annotations.bootstrap_mode
  3. accepted  → create shard; if not bootstrap, reinforce rule basis
  4. edited    → analyze edit; if substantive, create shard from edited content;
                 if not bootstrap, check rule responsibility and queue proposals
  5. rejected  → if not bootstrap, extract pattern and queue proposals;
                 check deprecation trigger per rule in basis
  6. Always mark processed in finally block

Bootstrap policy:
  - Bootstrap outcomes create shards but do NOT generate rule proposals.
    This prevents low-confidence seed data from polluting governance.
"""

from datetime import datetime
from typing import Optional

from src.deps import Stores
from src.storage.governance_store import DEPRECATION_TRIGGER
from src.async_loops.mutations import (
    analyze_edit,
    rule_might_be_responsible,
    generate_modified_rule,
    extract_rule_pattern_from_rejection,
    check_accumulators_for_promotion,
)


def process_outcomes(
    user_id: str,
    stores: Stores,
    llm: Optional[object] = None,
) -> dict:
    """
    Processes up to 100 unprocessed outcomes for user_id.

    Returns {"processed": n, "shards_created": m, "proposals_added": k}.
    """
    outcomes = stores.outcomes.unprocessed(user_id, limit=100)
    if not outcomes:
        return {"processed": 0, "shards_created": 0, "proposals_added": 0}

    embed_fn = getattr(stores.shards, "_embed_fn", None) or (lambda texts: [[0.0]] * len(texts))

    stats = {"processed": 0, "shards_created": 0, "proposals_added": 0}

    for outcome in outcomes:
        try:
            _process_one(outcome, stores, llm, embed_fn, stats)
        finally:
            stores.outcomes.mark_processed(outcome["outcome_id"])
            stats["processed"] += 1

    # After individual outcomes, check accumulators
    check_accumulators_for_promotion(user_id, stores.proposals, llm)

    return stats


# ---------------------------------------------------------------------------
# Per-outcome handlers
# ---------------------------------------------------------------------------

def _process_one(
    outcome: dict,
    stores: Stores,
    llm,
    embed_fn,
    stats: dict,
) -> None:
    state = stores.traces.get(outcome["trace_id"])
    if not state:
        return  # orphan outcome — mark processed, no evidence

    annotations = (state.get("output_payload") or {}).get("annotations") or {}
    is_bootstrap = annotations.get("bootstrap_mode", False)
    rule_basis = state.get("rule_basis") or []
    outcome_type = outcome["outcome_type"]

    if outcome_type == "accepted":
        _handle_accepted(outcome, state, is_bootstrap, rule_basis, stores, stats)

    elif outcome_type == "edited":
        _handle_edited(outcome, state, is_bootstrap, rule_basis, stores, llm, embed_fn, stats)

    elif outcome_type == "rejected":
        _handle_rejected(outcome, state, is_bootstrap, rule_basis, stores, llm, stats)


def _handle_accepted(
    outcome: dict,
    state: dict,
    is_bootstrap: bool,
    rule_basis: list,
    stores: Stores,
    stats: dict,
) -> None:
    stores.shards.add_from_trace(state, outcome)
    stats["shards_created"] += 1

    if not is_bootstrap:
        for rule_id in rule_basis:
            stores.governance.reinforce_rule(rule_id, outcome["trace_id"])


def _handle_edited(
    outcome: dict,
    state: dict,
    is_bootstrap: bool,
    rule_basis: list,
    stores: Stores,
    llm,
    embed_fn,
    stats: dict,
) -> None:
    # Get original content from trace
    selected = state.get("selected_hypothesis") or {}
    original = selected.get("content", "")
    edited = outcome.get("edited_content") or ""

    edit_analysis = analyze_edit(original, edited, llm)

    # Always create a shard from the user's edited version
    stores.shards.add_from_trace(state, outcome, content_override=edited)
    stats["shards_created"] += 1

    if is_bootstrap or not edit_analysis.substantive:
        return

    # Check each basis rule for responsibility and propose modification
    for rule_id in rule_basis:
        if not rule_might_be_responsible(
            rule_id, edit_analysis, rule_basis, stores.governance, embed_fn
        ):
            continue

        current_rule = stores.governance.get(rule_id)
        if not current_rule:
            continue

        proposed = generate_modified_rule(current_rule, edit_analysis, llm)
        if proposed is None:
            continue

        stores.proposals.add(state.get("user_id", ""), {
            "type": "modify_rule",
            "target_rule_id": rule_id,
            "proposed_rule": proposed.model_dump(),
            "rationale": f"edit analysis: {edit_analysis.pattern}",
            "supporting_traces": [outcome["trace_id"]],
            "context": state.get("perceived_context") or {},
            "weight": edit_analysis.confidence,
        })
        stats["proposals_added"] += 1


def _handle_rejected(
    outcome: dict,
    state: dict,
    is_bootstrap: bool,
    rule_basis: list,
    stores: Stores,
    llm,
    stats: dict,
) -> None:
    user_id = state.get("user_id", "")
    rejection_reason = outcome.get("rejection_reason") or ""

    # Always record contradicting evidence — needed for deprecation trigger
    # even if this outcome came from a bootstrap trace.
    for rule_id in rule_basis:
        stores.governance.add_contradicting_evidence(rule_id, outcome["trace_id"])
        ratio = stores.governance.rule_contradicting_evidence_ratio(rule_id)
        if ratio >= DEPRECATION_TRIGGER:
            stores.proposals.add(user_id, {
                "type": "deprecate_rule",
                "target_rule_id": rule_id,
                "proposed_rule": None,
                "rationale": f"contradicting evidence ratio {ratio:.2f} >= {DEPRECATION_TRIGGER}",
                "supporting_traces": [outcome["trace_id"]],
                "context": state.get("perceived_context") or {},
                "weight": ratio,
            })
            stats["proposals_added"] += 1

    # Extract a new rule pattern from the rejection
    proposed = extract_rule_pattern_from_rejection(rejection_reason, state, llm)
    if proposed is not None:
        stores.proposals.add(user_id, {
            "type": "add_rule",
            "target_rule_id": None,
            "proposed_rule": proposed.model_dump(),
            "rationale": f"extracted from rejection: {rejection_reason[:80]}",
            "supporting_traces": [outcome["trace_id"]],
            "context": state.get("perceived_context") or {},
            "weight": proposed.confidence_adjustment,
        })
        stats["proposals_added"] += 1
