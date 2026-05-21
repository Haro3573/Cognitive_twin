"""
Unit tests for src/nodes/*.

Each node factory is tested with MagicMock stores/subagents — no LLM calls
reach a live API and no SQLite connections are opened.
"""

import pytest
from unittest.mock import MagicMock

from src.deps import Stores
from src.nodes.perceive import make_perceive_node
from src.nodes.governance_load import make_governance_load_node
from src.nodes.recall import make_recall_node
from src.nodes.reason import make_reason_node
from src.nodes.align import make_align_node
from src.nodes.hard_limit_annotator import make_hard_limit_annotator_node
from src.nodes.compose_output import make_compose_output_node
from src.nodes.meta_learn import make_meta_learn_node


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_mock_stores(anchors=10, shards=30, rules=5) -> Stores:
    anchor_store = MagicMock()
    anchor_store.count_for_user.return_value = anchors
    shard_store = MagicMock()
    shard_store.count_for_user.return_value = shards
    gov_store = MagicMock()
    gov_store.count_active_rules.return_value = rules
    gov_store.query_active_rules.return_value = []
    gov_store.current_version.return_value = 7
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


def _base_state(**overrides) -> dict:
    base = {
        "user_id": "u1",
        "invocation_mode": "subagent",
        "parent_agent_context": None,
        "raw_input": "Should I accept the job offer?",
        "trace_id": "trace-001",
        "trace_persist_required": False,
        "is_bootstrap": False,
        "is_off_baseline": False,
        "baseline_deviation_score": 0.0,
        "perceived_context": {"domain_tags": ["work"], "situation_type": "career decision",
                               "emotional_valence": "neutral", "time_pressure": "low",
                               "stakes": "high", "key_entities": []},
        "honesty_assertion_required": False,
        "active_governance_rules": [],
        "governance_version": 0,
        "retrieved_shards": [],
        "retrieved_anchors": [],
        "retrieval_strategy": "semantic",
        "hypotheses": [],
        "reasoning_traces": [],
        "historical_hypotheses": [],
        "alignment_scores": {"hypotheses": []},
        "selected_hypothesis": None,
        "alignment_confidence": 0.6,
        "reproduction_confidence": 0.5,
        "confidence_divergence": 0.1,
        "sparse_domain_flag": None,
        "output_payload": None,
        "rule_basis": [],
        "proposed_governance_updates": [],
        "pending_reversal_check": False,
        "self_refine_reason": None,
        "recursion_depth": 0,
    }
    base.update(overrides)
    return base


_STUB_HYP = {
    "id": "hyp-1",
    "content": "Accept the offer",
    "decision_type": "action",
    "structured_payload": None,
    "derivation": {"rules": [], "shards": [], "anchors": [], "source": "stub"},
    "rule_conflicting": False,
    "conflict_details": [],
}


# ---------------------------------------------------------------------------
# perceive_node
# ---------------------------------------------------------------------------


def test_perceive_sets_is_bootstrap_true():
    """With stores below threshold, perceive sets is_bootstrap=True."""
    stores = make_mock_stores(anchors=2, shards=5, rules=1)
    mock_llm = MagicMock()
    from src.helpers.context import ExtractedContext
    mock_llm.with_structured_output.return_value.invoke.return_value = ExtractedContext(
        domain_tags=["work"]
    )
    node = make_perceive_node(stores, llm=mock_llm)
    result = node(_base_state())
    assert result["is_bootstrap"] is True


def test_perceive_sets_is_bootstrap_false():
    """With stores above threshold, perceive sets is_bootstrap=False."""
    stores = make_mock_stores(anchors=10, shards=30, rules=5)
    mock_llm = MagicMock()
    from src.helpers.context import ExtractedContext
    mock_llm.with_structured_output.return_value.invoke.return_value = ExtractedContext(
        domain_tags=["work"]
    )
    node = make_perceive_node(stores, llm=mock_llm)
    result = node(_base_state())
    assert result["is_bootstrap"] is False


def test_perceive_returns_required_keys():
    stores = make_mock_stores()
    mock_llm = MagicMock()
    from src.helpers.context import ExtractedContext
    mock_llm.with_structured_output.return_value.invoke.return_value = ExtractedContext()
    node = make_perceive_node(stores, llm=mock_llm)
    result = node(_base_state())
    assert "perceived_context" in result
    assert "honesty_assertion_required" in result
    assert "is_bootstrap" in result


def test_perceive_honesty_detection_fires():
    """perceive detects sincere AI identity question."""
    stores = make_mock_stores()
    # LLM for context extraction
    from src.helpers.context import ExtractedContext
    from src.helpers.honesty import _SincerityCheck
    mock_llm = MagicMock()
    # context extraction returns ok
    mock_llm.with_structured_output.return_value.invoke.return_value = ExtractedContext()
    # honesty detection uses the same llm object — make both structured outputs work
    sincerity_mock = MagicMock()
    sincerity_mock.invoke.return_value = _SincerityCheck(is_sincere_inquiry=True)
    context_mock = MagicMock()
    context_mock.invoke.return_value = ExtractedContext()
    mock_llm.with_structured_output.side_effect = lambda schema: (
        sincerity_mock if schema.__name__ == "_SincerityCheck" else context_mock
    )
    node = make_perceive_node(stores, llm=mock_llm)
    result = node(_base_state(raw_input="Are you an AI?"))
    assert result["honesty_assertion_required"] is True


# ---------------------------------------------------------------------------
# governance_load_node
# ---------------------------------------------------------------------------


def test_governance_load_loads_rules():
    sentinel = [{"rule_id": "r1", "statement": "test", "context_scope": []}]
    stores = make_mock_stores()
    stores.governance.query_active_rules.return_value = sentinel
    stores.governance.current_version.return_value = 3
    node = make_governance_load_node(stores)
    state = _base_state(active_governance_rules=None)
    result = node(state)
    assert result["active_governance_rules"] == sentinel
    assert result["governance_version"] == 3


def test_governance_load_skips_when_already_loaded():
    stores = make_mock_stores()
    node = make_governance_load_node(stores)
    state = _base_state(active_governance_rules=[{"rule_id": "existing"}])
    result = node(state)
    assert result == {}
    stores.governance.query_active_rules.assert_not_called()


def test_governance_load_empty_rules():
    stores = make_mock_stores()
    stores.governance.query_active_rules.return_value = []
    node = make_governance_load_node(stores)
    result = node(_base_state(active_governance_rules=None))
    assert result["active_governance_rules"] == []


# ---------------------------------------------------------------------------
# recall_node
# ---------------------------------------------------------------------------


def _make_recall_subagent(shards=None, anchors=None, strategy="semantic", deviation=0.0):
    stub = MagicMock()
    stub.invoke.return_value = {
        "shards": shards or [],
        "anchors": anchors or [],
        "strategy": strategy,
        "baseline_deviation": deviation,
    }
    return stub


def test_recall_maps_subagent_output():
    sub = _make_recall_subagent(shards=[{"shard_id": "s1"}], strategy="anchor_led")
    node = make_recall_node(sub)
    result = node(_base_state())
    assert result["retrieved_shards"] == [{"shard_id": "s1"}]
    assert result["retrieval_strategy"] == "anchor_led"


def test_recall_is_off_baseline_true():
    sub = _make_recall_subagent(deviation=0.9)
    node = make_recall_node(sub)
    result = node(_base_state())
    assert result["is_off_baseline"] is True
    assert result["baseline_deviation_score"] == 0.9


def test_recall_is_off_baseline_false():
    sub = _make_recall_subagent(deviation=0.2)
    node = make_recall_node(sub)
    result = node(_base_state())
    assert result["is_off_baseline"] is False


def test_recall_exactly_at_threshold_not_off_baseline():
    sub = _make_recall_subagent(deviation=0.4)
    node = make_recall_node(sub)
    result = node(_base_state())
    assert result["is_off_baseline"] is False  # threshold is >, not >=


def test_recall_passes_bootstrap_flag_to_subagent():
    sub = _make_recall_subagent()
    node = make_recall_node(sub)
    node(_base_state(is_bootstrap=True))
    call_args = sub.invoke.call_args[0][0]
    assert call_args["is_bootstrap"] is True


# ---------------------------------------------------------------------------
# reason_node
# ---------------------------------------------------------------------------


def _make_reason_subagent(candidates=None, traces=None):
    stub = MagicMock()
    stub.invoke.return_value = {
        "candidates": candidates or [_STUB_HYP],
        "traces": traces or [{"stub": True}],
    }
    return stub


def test_reason_maps_candidates():
    sub = _make_reason_subagent(candidates=[_STUB_HYP])
    node = make_reason_node(sub)
    result = node(_base_state())
    assert result["hypotheses"] == [_STUB_HYP]
    assert result["historical_hypotheses"] == [_STUB_HYP]


def test_reason_passes_historical_hypotheses():
    sub = _make_reason_subagent()
    node = make_reason_node(sub)
    prior = [{"id": "prior-1", "content": "prior hyp"}]
    node(_base_state(historical_hypotheses=prior))
    call_args = sub.invoke.call_args[0][0]
    assert call_args["historical_hypotheses"] == prior


def test_reason_passes_active_rules():
    sub = _make_reason_subagent()
    node = make_reason_node(sub)
    rules = [{"rule_id": "r1"}]
    node(_base_state(active_governance_rules=rules))
    call_args = sub.invoke.call_args[0][0]
    assert call_args["active_rules"] == rules


# ---------------------------------------------------------------------------
# align_node
# ---------------------------------------------------------------------------


def _make_align_subagent(align_conf=0.6, repro_conf=0.5, rule_conformance=1.0,
                          rule_conflict_details=None, hyp=None):
    stub = MagicMock()
    stub.invoke.return_value = {
        "hypotheses": [
            {
                "hypothesis": hyp or _STUB_HYP,
                "alignment_confidence": align_conf,
                "reproduction_confidence": repro_conf,
                "rule_conformance_score": rule_conformance,
                "rule_conflict_details": rule_conflict_details or [],
                "reasoning": "test",
            }
        ]
    }
    return stub


def test_align_returns_selected_hypothesis():
    sub = _make_align_subagent()
    node = make_align_node(sub)
    result = node(_base_state(hypotheses=[_STUB_HYP]))
    assert result["selected_hypothesis"] == _STUB_HYP


def test_align_bootstrap_caps_alignment():
    sub = _make_align_subagent(align_conf=0.9)
    node = make_align_node(sub)
    result = node(_base_state(hypotheses=[_STUB_HYP], is_bootstrap=True))
    assert result["alignment_confidence"] == pytest.approx(0.4)


def test_align_no_cap_when_not_bootstrap():
    sub = _make_align_subagent(align_conf=0.9)
    node = make_align_node(sub)
    result = node(_base_state(hypotheses=[_STUB_HYP], is_bootstrap=False))
    assert result["alignment_confidence"] == pytest.approx(0.9)


def test_align_computes_divergence():
    sub = _make_align_subagent(align_conf=0.8, repro_conf=0.5)
    node = make_align_node(sub)
    result = node(_base_state(hypotheses=[_STUB_HYP]))
    assert result["confidence_divergence"] == pytest.approx(0.3)


def test_align_rule_basis_patch_a2():
    """§A2: rule_influenced_hypothesis must receive hypothesis dict, not ScoredHypothesis."""
    rule = {"rule_id": "r1", "context_scope": []}
    hyp = {**_STUB_HYP, "derivation": {"rules": ["r1"], "shards": [], "anchors": [], "source": "test"}}
    sub = _make_align_subagent(hyp=hyp)
    node = make_align_node(sub)
    result = node(_base_state(hypotheses=[hyp], active_governance_rules=[rule]))
    assert "r1" in result["rule_basis"]


def test_align_rule_basis_excludes_non_matching_rules():
    rule = {"rule_id": "r-not-matched", "context_scope": []}
    hyp = {**_STUB_HYP, "derivation": {"rules": ["r-other"], "shards": [], "anchors": [], "source": "test"}}
    sub = _make_align_subagent(hyp=hyp)
    node = make_align_node(sub)
    result = node(_base_state(hypotheses=[hyp], active_governance_rules=[rule]))
    assert result["rule_basis"] == []


def test_align_selects_highest_alignment_hypothesis():
    hyp_low = {**_STUB_HYP, "id": "low"}
    hyp_high = {**_STUB_HYP, "id": "high"}
    sub = MagicMock()
    sub.invoke.return_value = {
        "hypotheses": [
            {"hypothesis": hyp_low, "alignment_confidence": 0.3,
             "reproduction_confidence": 0.5, "rule_conformance_score": 1.0,
             "rule_conflict_details": [], "reasoning": ""},
            {"hypothesis": hyp_high, "alignment_confidence": 0.8,
             "reproduction_confidence": 0.6, "rule_conformance_score": 1.0,
             "rule_conflict_details": [], "reasoning": ""},
        ]
    }
    node = make_align_node(sub)
    result = node(_base_state(hypotheses=[hyp_low, hyp_high]))
    assert result["selected_hypothesis"]["id"] == "high"


# ---------------------------------------------------------------------------
# hard_limit_annotator_node
# ---------------------------------------------------------------------------


def test_hard_limit_health_tag():
    node = make_hard_limit_annotator_node()
    state = _base_state(perceived_context={"domain_tags": ["health"]})
    result = node(state)
    assert result["sparse_domain_flag"] == "health"


def test_hard_limit_no_sparse_domain():
    node = make_hard_limit_annotator_node()
    state = _base_state(perceived_context={"domain_tags": ["work"], "situation_type": "career"})
    result = node(state)
    assert result["sparse_domain_flag"] is None


def test_hard_limit_keyword_in_shard():
    node = make_hard_limit_annotator_node()
    state = _base_state(
        perceived_context={"domain_tags": []},
        retrieved_shards=[{"content": "I need to see a doctor about my diagnosis"}],
    )
    result = node(state)
    assert result["sparse_domain_flag"] == "health"


# ---------------------------------------------------------------------------
# compose_output_node
# ---------------------------------------------------------------------------


def _state_with_hypothesis(**overrides) -> dict:
    hyp = _STUB_HYP
    scored = [{
        "hypothesis": hyp,
        "alignment_confidence": 0.6,
        "reproduction_confidence": 0.5,
        "rule_conformance_score": 1.0,
        "rule_conflict_details": [],
        "reasoning": "stub",
    }]
    # overrides win so callers can change alignment_confidence/divergence etc.
    defaults = dict(
        selected_hypothesis=hyp,
        alignment_scores={"hypotheses": scored},
        alignment_confidence=0.6,
        reproduction_confidence=0.5,
        confidence_divergence=0.1,
    )
    defaults.update(overrides)
    return _base_state(**defaults)


def test_compose_output_payload_shape():
    node = make_compose_output_node()
    result = node(_state_with_hypothesis())
    payload = result["output_payload"]
    for key in ("decision", "confidences", "annotations", "alternatives", "rule_basis", "trace_id"):
        assert key in payload, f"missing key: {key}"


def test_compose_output_alternatives_excludes_selected():
    hyp2 = {**_STUB_HYP, "id": "hyp-2", "content": "Decline the offer"}
    scored = [
        {"hypothesis": _STUB_HYP, "alignment_confidence": 0.8,
         "reproduction_confidence": 0.5, "rule_conformance_score": 1.0,
         "rule_conflict_details": [], "reasoning": ""},
        {"hypothesis": hyp2, "alignment_confidence": 0.4,
         "reproduction_confidence": 0.4, "rule_conformance_score": 1.0,
         "rule_conflict_details": [], "reasoning": ""},
    ]
    node = make_compose_output_node()
    state = _base_state(
        selected_hypothesis=_STUB_HYP,
        alignment_scores={"hypotheses": scored},
        alignment_confidence=0.8,
        reproduction_confidence=0.5,
        confidence_divergence=0.3,
    )
    result = node(state)
    alt_ids = [a["id"] for a in result["output_payload"]["alternatives"]]
    assert "hyp-1" not in alt_ids
    assert "hyp-2" in alt_ids


def test_compose_output_alternatives_capped_at_two():
    hyps = [
        {**_STUB_HYP, "id": f"hyp-{i}", "content": f"option {i}"}
        for i in range(5)
    ]
    selected = hyps[0]
    scored = [
        {"hypothesis": h, "alignment_confidence": 0.5,
         "reproduction_confidence": 0.5, "rule_conformance_score": 1.0,
         "rule_conflict_details": [], "reasoning": ""}
        for h in hyps
    ]
    node = make_compose_output_node()
    state = _base_state(
        selected_hypothesis=selected,
        alignment_scores={"hypotheses": scored},
        alignment_confidence=0.5,
        reproduction_confidence=0.5,
        confidence_divergence=0.0,
    )
    result = node(state)
    assert len(result["output_payload"]["alternatives"]) <= 2


def test_compose_honesty_assertion_rewrites():
    from src.helpers.honesty import _RewriteResult
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = _RewriteResult(
        rewritten_response="As an AI, I suggest accepting."
    )
    node = make_compose_output_node(llm=mock_llm)
    hyp = {**_STUB_HYP, "type": "response_text", "response_text": "Accept the offer."}
    scored = [{"hypothesis": hyp, "alignment_confidence": 0.6,
               "reproduction_confidence": 0.5, "rule_conformance_score": 1.0,
               "rule_conflict_details": [], "reasoning": ""}]
    state = _base_state(
        selected_hypothesis=hyp,
        alignment_scores={"hypotheses": scored},
        alignment_confidence=0.6,
        reproduction_confidence=0.5,
        confidence_divergence=0.1,
        honesty_assertion_required=True,
    )
    result = node(state)
    assert result["output_payload"]["decision"]["response_text"] == "As an AI, I suggest accepting."


# ---------------------------------------------------------------------------
# meta_learn_node
# ---------------------------------------------------------------------------


def test_meta_learn_no_proposals_typical():
    """Typical trace (no divergence, alignment < 0.85) produces no proposals."""
    stores = make_mock_stores()
    node = make_meta_learn_node(stores)
    result = node(_state_with_hypothesis())
    assert result["proposed_governance_updates"] == []
    stores.proposals.add.assert_not_called()


def test_meta_learn_light1_divergence():
    stores = make_mock_stores()
    node = make_meta_learn_node(stores)
    state = _state_with_hypothesis(confidence_divergence=0.4,
                                   alignment_confidence=0.5,
                                   reproduction_confidence=0.9)
    result = node(state)
    types = [p["type"] for p in result["proposed_governance_updates"]]
    assert "investigate_divergence" in types


def test_meta_learn_light2_retrieval_reinforcement():
    stores = make_mock_stores()
    node = make_meta_learn_node(stores)
    state = _state_with_hypothesis(alignment_confidence=0.9, retrieval_strategy="anchor_led",
                                   reproduction_confidence=0.5, confidence_divergence=0.4)
    result = node(state)
    types = [p["type"] for p in result["proposed_governance_updates"]]
    assert "adjust_weight" in types


def test_meta_learn_light3_rule_conflict():
    stores = make_mock_stores()
    node = make_meta_learn_node(stores)
    hyp = {**_STUB_HYP, "id": "h1"}
    scored = [{
        "hypothesis": hyp,
        "alignment_confidence": 0.5,
        "reproduction_confidence": 0.8,
        "rule_conformance_score": 0.2,
        "rule_conflict_details": [{"rule_id": "r-conflict", "description": "x"}],
        "reasoning": "",
    }]
    state = _base_state(
        selected_hypothesis=hyp,
        alignment_scores={"hypotheses": scored},
        alignment_confidence=0.5,
        reproduction_confidence=0.8,
        confidence_divergence=0.3,
    )
    result = node(state)
    types = [p["type"] for p in result["proposed_governance_updates"]]
    assert "investigate_rule" in types


def test_meta_learn_light4_pattern_emergence():
    stores = make_mock_stores()
    node = make_meta_learn_node(stores)
    hyp = {**_STUB_HYP, "id": "h1"}
    scored = [{"hypothesis": hyp, "alignment_confidence": 0.9,
               "reproduction_confidence": 0.9, "rule_conformance_score": 1.0,
               "rule_conflict_details": [], "reasoning": ""}]
    # active_governance_rules=[] → has_governance_coverage returns False
    state = _base_state(
        selected_hypothesis=hyp,
        alignment_scores={"hypotheses": scored},
        alignment_confidence=0.9,
        reproduction_confidence=0.9,
        confidence_divergence=0.0,
        active_governance_rules=[],
    )
    result = node(state)
    types = [p["type"] for p in result["proposed_governance_updates"]]
    assert "investigate_new_rule" in types


def test_meta_learn_learning_weight_off_baseline():
    stores = make_mock_stores()
    node = make_meta_learn_node(stores)
    state = _state_with_hypothesis(is_off_baseline=True, confidence_divergence=0.4,
                                   alignment_confidence=0.3, reproduction_confidence=0.7)
    result = node(state)
    p = next(p for p in result["proposed_governance_updates"] if p["type"] == "investigate_divergence")
    assert p["weight"] == pytest.approx(0.3)


def test_meta_learn_learning_weight_bootstrap():
    stores = make_mock_stores()
    node = make_meta_learn_node(stores)
    state = _state_with_hypothesis(is_bootstrap=True, is_off_baseline=False,
                                   confidence_divergence=0.4, alignment_confidence=0.3,
                                   reproduction_confidence=0.7)
    result = node(state)
    p = next(p for p in result["proposed_governance_updates"] if p["type"] == "investigate_divergence")
    assert p["weight"] == pytest.approx(0.5)  # 1.0 * 0.5


def test_meta_learn_learning_weight_off_baseline_and_bootstrap():
    stores = make_mock_stores()
    node = make_meta_learn_node(stores)
    state = _state_with_hypothesis(is_bootstrap=True, is_off_baseline=True,
                                   confidence_divergence=0.4, alignment_confidence=0.3,
                                   reproduction_confidence=0.7)
    result = node(state)
    p = next(p for p in result["proposed_governance_updates"] if p["type"] == "investigate_divergence")
    assert p["weight"] == pytest.approx(0.15)  # 0.3 * 0.5


def test_meta_learn_trace_persist_when_required():
    stores = make_mock_stores()
    node = make_meta_learn_node(stores)
    state = _state_with_hypothesis(trace_persist_required=True)
    node(state)
    stores.traces.save.assert_called_once()


def test_meta_learn_no_trace_persist_when_not_required():
    stores = make_mock_stores()
    node = make_meta_learn_node(stores)
    state = _state_with_hypothesis(trace_persist_required=False)
    node(state)
    stores.traces.save.assert_not_called()


def test_meta_learn_pending_review_on_sparse_domain():
    stores = make_mock_stores()
    node = make_meta_learn_node(stores)
    result = node(_state_with_hypothesis(sparse_domain_flag="health"))
    assert result["pending_reversal_check"] is True


def test_meta_learn_pending_review_false_when_clean():
    stores = make_mock_stores()
    node = make_meta_learn_node(stores)
    result = node(_state_with_hypothesis(
        sparse_domain_flag=None,
        confidence_divergence=0.1,
        is_off_baseline=False,
    ))
    assert result["pending_reversal_check"] is False
