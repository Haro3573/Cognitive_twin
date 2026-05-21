from typing import TypedDict, Annotated, Literal, Optional
from operator import add
from datetime import datetime


def replace_reducer(old, new):
    """Replaces value each recursion cycle rather than accumulating."""
    return new if new is not None else old


class CognitiveLoopState(TypedDict):
    # === Identity & invocation ===
    user_id: str
    invocation_mode: Literal["direct", "subagent"]
    parent_agent_context: Optional[dict]
    trace_id: str

    # === Step 1: Perceive ===
    raw_input: str
    perceived_context: dict
    honesty_assertion_required: bool

    # === Governance (loaded once per request) ===
    active_governance_rules: Optional[list[dict]]
    governance_version: int

    # === Bootstrap state ===
    is_bootstrap: bool

    # === Step 2: Recall (dual-track) ===
    retrieved_shards: list[dict]
    retrieved_anchors: list[dict]
    retrieval_strategy: str
    is_off_baseline: bool
    baseline_deviation_score: float

    # === Step 3: Reasoning ===
    # replace_reducer: resets each recursion cycle
    hypotheses: Annotated[list[dict], replace_reducer]
    reasoning_traces: Annotated[list[dict], replace_reducer]
    # add reducer: accumulates across recursion so reasoner avoids repeating itself
    historical_hypotheses: Annotated[list[dict], add]

    # === Step 4: Dual-confidence alignment ===
    alignment_scores: dict
    selected_hypothesis: Optional[dict]
    alignment_confidence: float
    reproduction_confidence: float
    confidence_divergence: float

    # === Hard-limit annotation (NOT gating) ===
    sparse_domain_flag: Optional[Literal["health", "legal", "financial", "close_relationships"]]

    # === Step 5: Output ===
    output_payload: Optional[dict]
    rule_basis: list[str]

    # === Step 6: Meta-learning ===
    proposed_governance_updates: list[dict]
    pending_reversal_check: bool

    # === Loop control ===
    recursion_depth: int
    self_refine_reason: Optional[str]
    trace_persist_required: bool
