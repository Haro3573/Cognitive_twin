"""
Trace ID generation and state compression for persistence (patch §A8).
"""

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import CognitiveLoopState


def new_trace_id() -> str:
    return str(uuid.uuid4())


def compress_state_for_persistence(state: "CognitiveLoopState") -> dict:
    """
    Extracts the fields needed for long-term trace storage.
    The promotion engine and meta_learn_node read this payload — keep it stable.

    meta_weight = 0.3 when the decision is off-baseline (less representative),
    1.0 otherwise. This weight is read by meta_learn_node to discount signal
    from atypical decisions.
    """
    active_rules = state.get("active_governance_rules") or []
    return {
        "trace_id": state["trace_id"],
        "user_id": state["user_id"],
        "perceived_context": state["perceived_context"],
        "selected_hypothesis": state["selected_hypothesis"],
        "alignment_summary": {
            "alignment_confidence": state["alignment_confidence"],
            "reproduction_confidence": state["reproduction_confidence"],
            "divergence": state["confidence_divergence"],
        },
        "rule_basis": state["rule_basis"],
        "annotations_at_trace_time": {
            "is_off_baseline": state["is_off_baseline"],
            "is_bootstrap": state["is_bootstrap"],
            "baseline_deviation": state["baseline_deviation_score"],
            "sparse_domain": state["sparse_domain_flag"],
        },
        "retrieved_shard_ids": [s["shard_id"] for s in state.get("retrieved_shards", [])],
        "retrieved_anchor_ids": [a["anchor_id"] for a in state.get("retrieved_anchors", [])],
        "active_rule_ids_at_trace_time": [r["rule_id"] for r in active_rules],
        "meta_weight": 0.3 if state["is_off_baseline"] else 1.0,
    }
