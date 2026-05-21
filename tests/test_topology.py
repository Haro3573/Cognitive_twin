"""
Topology test — verifies graph structure, routing, and recursion invariants.

All nodes are real implementations but sub-agents are stubs, so no LLM
calls reach a live API. Store methods are satisfied by MagicMock.
"""

import pytest
import uuid
from unittest.mock import MagicMock

from src.graph import build_graph, RUNTIME_CONFIG
from src.deps import Stores
from src.subagents import recall_subagent, reason_subagent, align_subagent

_STUB_SUBAGENTS = (recall_subagent, reason_subagent, align_subagent)


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------

def make_mock_stores(bootstrap: bool = False) -> Stores:
    """
    Returns a Stores object with MagicMock instances.

    bootstrap=True  → counts below threshold → perceive_node sets is_bootstrap=True
    bootstrap=False → counts above threshold → perceive_node sets is_bootstrap=False
    """
    anchor_count = 2 if bootstrap else 10
    shard_count = 5 if bootstrap else 30
    rule_count = 1 if bootstrap else 5

    anchor_store = MagicMock()
    anchor_store.count_for_user.return_value = anchor_count

    shard_store = MagicMock()
    shard_store.count_for_user.return_value = shard_count

    gov_store = MagicMock()
    gov_store.count_active_rules.return_value = rule_count
    gov_store.query_active_rules.return_value = []
    gov_store.current_version.return_value = 0

    trace_store = MagicMock()
    proposal_queue = MagicMock()
    pending_anchor_store = MagicMock()
    outcome_store = MagicMock()

    review_store = MagicMock()

    return Stores(
        shards=shard_store,
        anchors=anchor_store,
        governance=gov_store,
        traces=trace_store,
        proposals=proposal_queue,
        pending_anchors=pending_anchor_store,
        outcomes=outcome_store,
        reviews=review_store,
    )


MINIMAL_INPUT = {
    "user_id": "test-user",
    "invocation_mode": "subagent",
    "parent_agent_context": {"goal": "test", "caller": "test-agent"},
    "raw_input": "Should I accept this meeting?",
    "recursion_depth": 0,
    "trace_id": uuid.uuid4().hex,
    "trace_persist_required": False,
    "active_governance_rules": None,
    "is_bootstrap": False,
    "historical_hypotheses": [],
    "retrieved_shards": [],
    "retrieved_anchors": [],
    "retrieval_strategy": "",
    "is_off_baseline": False,
    "baseline_deviation_score": 0.0,
    "hypotheses": [],
    "reasoning_traces": [],
    "alignment_scores": {},
    "selected_hypothesis": None,
    "alignment_confidence": 0.0,
    "reproduction_confidence": 0.0,
    "confidence_divergence": 0.0,
    "sparse_domain_flag": None,
    "output_payload": None,
    "rule_basis": [],
    "proposed_governance_updates": [],
    "pending_reversal_check": False,
    "self_refine_reason": None,
    "perceived_context": {},
    "honesty_assertion_required": False,
    "governance_version": 0,
}


@pytest.fixture
def compiled_graph():
    return build_graph(stores=make_mock_stores(bootstrap=False), subagents=_STUB_SUBAGENTS, checkpointer=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_graph_compiles():
    """Graph compiles without raising — nodes and edges are valid."""
    graph = build_graph(stores=make_mock_stores(), subagents=_STUB_SUBAGENTS, checkpointer=None)
    assert graph is not None


def test_subagent_mode_terminates(compiled_graph):
    """In subagent mode, the loop terminates (does not recurse infinitely)."""
    result = compiled_graph.invoke(MINIMAL_INPUT, config=RUNTIME_CONFIG)
    assert result is not None


def test_output_payload_present(compiled_graph):
    """compose_output_node produces output_payload with required keys."""
    result = compiled_graph.invoke(MINIMAL_INPUT, config=RUNTIME_CONFIG)
    payload = result.get("output_payload")
    assert payload is not None, "output_payload must be set by compose_output_node"
    assert "decision" in payload
    assert "confidences" in payload
    assert "annotations" in payload
    assert "trace_id" in payload
    assert "alternatives" in payload
    assert "rule_basis" in payload


def test_confidences_shape(compiled_graph):
    """Confidences dict has alignment, reproduction, divergence keys."""
    result = compiled_graph.invoke(MINIMAL_INPUT, config=RUNTIME_CONFIG)
    confs = result["output_payload"]["confidences"]
    assert "alignment" in confs
    assert "reproduction" in confs
    assert "divergence" in confs
    for v in confs.values():
        assert 0.0 <= v <= 1.0


def test_recursion_depth_bounded_subagent(compiled_graph):
    """In subagent mode, recursion_depth in final state is at most 1."""
    result = compiled_graph.invoke(MINIMAL_INPUT, config=RUNTIME_CONFIG)
    assert result["recursion_depth"] <= 1, (
        f"subagent mode cap=1 violated: recursion_depth={result['recursion_depth']}"
    )


def test_recursion_depth_bounded_direct():
    """In direct mode, recursion_depth in final state is at most 3."""
    graph = build_graph(stores=make_mock_stores(bootstrap=False), subagents=_STUB_SUBAGENTS, checkpointer=None)
    inp = {**MINIMAL_INPUT, "invocation_mode": "direct"}
    result = graph.invoke(inp, config=RUNTIME_CONFIG)
    assert result["recursion_depth"] <= 3, (
        f"direct mode cap=3 violated: recursion_depth={result['recursion_depth']}"
    )


def test_governance_load_skips_on_recursion(compiled_graph):
    """governance_load_node must not overwrite rules already loaded (recursion guard)."""
    sentinel_rules = [{"rule_id": "sentinel", "statement": "test rule", "version": 1,
                       "confidence": 0.9, "evidence_count": 1, "context_scope": [],
                       "supporting_traces": [], "contradicting_traces": [],
                       "activated_at": None, "supersedes": None,
                       "rule_class": "preference"}]
    inp = {**MINIMAL_INPUT, "active_governance_rules": sentinel_rules, "governance_version": 42}
    result = compiled_graph.invoke(inp, config=RUNTIME_CONFIG)
    assert result["active_governance_rules"] == sentinel_rules
    assert result["governance_version"] == 42


def test_bootstrap_alignment_cap():
    """When perceive sets is_bootstrap=True (stores below threshold), alignment is capped at 0.4."""
    graph = build_graph(stores=make_mock_stores(bootstrap=True), subagents=_STUB_SUBAGENTS, checkpointer=None)
    result = graph.invoke(MINIMAL_INPUT, config=RUNTIME_CONFIG)
    assert result["alignment_confidence"] <= 0.4, (
        f"bootstrap cap violated: alignment_confidence={result['alignment_confidence']}"
    )
    assert result["output_payload"]["annotations"]["bootstrap_mode"] is True


def test_command_routing_increments_recursion():
    """Verify recursion_depth increments when confidence_router recurses.

    Non-bootstrap stores → perceive sets is_bootstrap=False → align returns 0.6 (≥ 0.5)
    → router proceeds without recursing, final recursion_depth=0.

    Bootstrap stores → perceive sets is_bootstrap=True → align capped at 0.4 (< 0.5)
    → router recurses once (subagent cap=1), final recursion_depth=1.
    """
    # Non-bootstrap path
    graph_normal = build_graph(stores=make_mock_stores(bootstrap=False), subagents=_STUB_SUBAGENTS, checkpointer=None)
    result = graph_normal.invoke(MINIMAL_INPUT, config=RUNTIME_CONFIG)
    assert result["recursion_depth"] == 0, (
        f"non-bootstrap should not recurse, got recursion_depth={result['recursion_depth']}"
    )

    # Bootstrap path
    graph_bootstrap = build_graph(stores=make_mock_stores(bootstrap=True), subagents=_STUB_SUBAGENTS, checkpointer=None)
    result_bs = graph_bootstrap.invoke(MINIMAL_INPUT, config=RUNTIME_CONFIG)
    assert result_bs["recursion_depth"] == 1, (
        f"bootstrap subagent should recurse exactly once, got {result_bs['recursion_depth']}"
    )
