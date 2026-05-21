"""
Tests for src/helpers/*.

LLM-dependent helpers receive a MagicMock that intercepts
llm.with_structured_output(...).invoke(...) and returns a pre-built Pydantic object.
"""

import pytest
from unittest.mock import MagicMock

from src.helpers.trace_utils import new_trace_id, compress_state_for_persistence
from src.helpers.predicates import is_bootstrap_state, has_governance_coverage, rule_influenced_hypothesis
from src.helpers.sparse_domain import detect_sparse_domain
from src.helpers.context import ExtractedContext, extract_context
from src.helpers.honesty import _SincerityCheck, _RewriteResult, detect_honesty_assertion, enforce_honesty_assertion


# ---------------------------------------------------------------------------
# trace_utils
# ---------------------------------------------------------------------------


def test_new_trace_id_is_uuid():
    tid = new_trace_id()
    assert isinstance(tid, str) and len(tid) == 36


def test_new_trace_ids_are_unique():
    assert new_trace_id() != new_trace_id()


def _minimal_state(**overrides) -> dict:
    base = {
        "trace_id": "t1",
        "user_id": "u1",
        "perceived_context": {"domain_tags": ["work"]},
        "selected_hypothesis": {"type": "decision", "response_text": "do it"},
        "alignment_confidence": 0.8,
        "reproduction_confidence": 0.7,
        "confidence_divergence": 0.1,
        "rule_basis": ["r1"],
        "is_off_baseline": False,
        "is_bootstrap": False,
        "baseline_deviation_score": 0.0,
        "sparse_domain_flag": None,
        "retrieved_shards": [{"shard_id": "s1"}, {"shard_id": "s2"}],
        "retrieved_anchors": [{"anchor_id": "a1"}],
        "active_governance_rules": [{"rule_id": "r1"}],
    }
    base.update(overrides)
    return base


def test_compress_state_keys():
    payload = compress_state_for_persistence(_minimal_state())
    for key in (
        "trace_id", "user_id", "perceived_context", "selected_hypothesis",
        "alignment_summary", "rule_basis", "annotations_at_trace_time",
        "retrieved_shard_ids", "retrieved_anchor_ids",
        "active_rule_ids_at_trace_time", "meta_weight",
    ):
        assert key in payload, f"missing key: {key}"


def test_compress_meta_weight_normal():
    payload = compress_state_for_persistence(_minimal_state(is_off_baseline=False))
    assert payload["meta_weight"] == 1.0


def test_compress_meta_weight_off_baseline():
    payload = compress_state_for_persistence(_minimal_state(is_off_baseline=True))
    assert payload["meta_weight"] == 0.3


def test_compress_shard_ids_extracted():
    state = _minimal_state(retrieved_shards=[{"shard_id": "s1"}, {"shard_id": "s2"}])
    payload = compress_state_for_persistence(state)
    assert payload["retrieved_shard_ids"] == ["s1", "s2"]


def test_compress_active_rules_extracted():
    state = _minimal_state(active_governance_rules=[{"rule_id": "r-a"}, {"rule_id": "r-b"}])
    payload = compress_state_for_persistence(state)
    assert payload["active_rule_ids_at_trace_time"] == ["r-a", "r-b"]


def test_compress_empty_active_rules():
    state = _minimal_state(active_governance_rules=None)
    payload = compress_state_for_persistence(state)
    assert payload["active_rule_ids_at_trace_time"] == []


# ---------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------


def _mock_stores(anchors=10, shards=30, rules=5):
    anchor_store = MagicMock()
    anchor_store.count_for_user.return_value = anchors
    shard_store = MagicMock()
    shard_store.count_for_user.return_value = shards
    gov_store = MagicMock()
    gov_store.count_active_rules.return_value = rules
    return anchor_store, shard_store, gov_store


def test_is_bootstrap_all_above_threshold():
    anchor_store, shard_store, gov_store = _mock_stores(anchors=10, shards=30, rules=5)
    assert not is_bootstrap_state("u1", anchor_store, shard_store, gov_store)


def test_is_bootstrap_insufficient_anchors():
    anchor_store, shard_store, gov_store = _mock_stores(anchors=2, shards=30, rules=5)
    assert is_bootstrap_state("u1", anchor_store, shard_store, gov_store)


def test_is_bootstrap_insufficient_shards():
    anchor_store, shard_store, gov_store = _mock_stores(anchors=10, shards=5, rules=5)
    assert is_bootstrap_state("u1", anchor_store, shard_store, gov_store)


def test_is_bootstrap_insufficient_rules():
    anchor_store, shard_store, gov_store = _mock_stores(anchors=10, shards=30, rules=1)
    assert is_bootstrap_state("u1", anchor_store, shard_store, gov_store)


def test_is_bootstrap_exactly_at_threshold_not_bootstrap():
    # Thresholds are 5/20/3 — exactly at threshold is NOT bootstrap (< not <=)
    anchor_store, shard_store, gov_store = _mock_stores(anchors=5, shards=20, rules=3)
    assert not is_bootstrap_state("u1", anchor_store, shard_store, gov_store)


def test_has_governance_coverage_universal_rule():
    rules = [{"rule_id": "r1", "context_scope": []}]
    assert has_governance_coverage({"domain_tags": ["work"]}, rules)


def test_has_governance_coverage_matching_scope():
    rules = [{"rule_id": "r1", "context_scope": ["work", "career"]}]
    assert has_governance_coverage({"domain_tags": ["work", "health"]}, rules)


def test_has_governance_coverage_no_match():
    rules = [{"rule_id": "r1", "context_scope": ["health"]}]
    assert not has_governance_coverage({"domain_tags": ["finance"]}, rules)


def test_has_governance_coverage_none_tags_filtered():
    rules = [{"rule_id": "r1", "context_scope": ["work"]}]
    # None in domain_tags should not cause errors
    assert not has_governance_coverage({"domain_tags": [None, "finance"]}, rules)


def test_has_governance_coverage_empty_rules():
    assert not has_governance_coverage({"domain_tags": ["work"]}, [])


def test_rule_influenced_hypothesis_true():
    rule = {"rule_id": "r1"}
    hyp = {"derivation": {"rules": ["r1", "r2"]}}
    assert rule_influenced_hypothesis(rule, hyp)


def test_rule_influenced_hypothesis_false():
    rule = {"rule_id": "r3"}
    hyp = {"derivation": {"rules": ["r1", "r2"]}}
    assert not rule_influenced_hypothesis(rule, hyp)


def test_rule_influenced_hypothesis_no_derivation():
    rule = {"rule_id": "r1"}
    hyp = {}
    assert not rule_influenced_hypothesis(rule, hyp)


# ---------------------------------------------------------------------------
# sparse_domain
# ---------------------------------------------------------------------------


def test_detect_sparse_fast_path_tag():
    ctx = {"domain_tags": ["health"]}
    assert detect_sparse_domain(ctx, [], []) == "health"


def test_detect_sparse_legal_tag():
    ctx = {"domain_tags": ["legal"]}
    assert detect_sparse_domain(ctx, [], []) == "legal"


def test_detect_sparse_keyword_in_shard():
    ctx = {"domain_tags": []}
    shards = [{"content": "I need to see a doctor about my diagnosis"}]
    assert detect_sparse_domain(ctx, shards, []) == "health"


def test_detect_sparse_keyword_in_anchor():
    ctx = {"domain_tags": []}
    anchors = [{"summary": "always consults a lawyer before signing contracts"}]
    assert detect_sparse_domain(ctx, [], anchors) == "legal"


def test_detect_sparse_financial_keyword():
    ctx = {"domain_tags": [], "situation_type": "deciding whether to invest in the stock market"}
    assert detect_sparse_domain(ctx, [], []) == "financial"


def test_detect_sparse_close_relationships():
    ctx = {"domain_tags": []}
    shards = [{"content": "dealing with a divorce is hard on the family"}]
    assert detect_sparse_domain(ctx, shards, []) == "close_relationships"


def test_detect_sparse_none():
    ctx = {"domain_tags": [], "situation_type": "choosing a software library"}
    assert detect_sparse_domain(ctx, [], []) is None


def test_detect_sparse_priority_health_over_financial():
    # A shard containing both health and financial keywords should return health first
    ctx = {"domain_tags": []}
    shards = [{"content": "medical insurance debt medication mortgage"}]
    result = detect_sparse_domain(ctx, shards, [])
    assert result == "health"


# ---------------------------------------------------------------------------
# context (LLM mocked)
# ---------------------------------------------------------------------------


def _mock_llm_for_extract(domain_tags=None, situation_type=None):
    mock_llm = MagicMock()
    extracted = ExtractedContext(
        domain_tags=domain_tags or ["work"],
        situation_type=situation_type or "career decision",
        emotional_valence="neutral",
        time_pressure="low",
        stakes="medium",
        key_entities=[],
    )
    mock_llm.with_structured_output.return_value.invoke.return_value = extracted
    return mock_llm


def test_extract_context_returns_dict(tmp_path):
    mock_llm = _mock_llm_for_extract(domain_tags=["work"])
    result = extract_context("Should I quit my job?", llm=mock_llm)
    assert isinstance(result, dict)
    assert result["domain_tags"] == ["work"]
    assert result["situation_type"] == "career decision"


def test_extract_context_fallback_on_error():
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failure")
    result = extract_context("some input", llm=mock_llm)
    assert isinstance(result, dict)
    assert result["domain_tags"] == []
    assert result["emotional_valence"] == "neutral"


def test_extract_context_passes_parent_context():
    mock_llm = _mock_llm_for_extract()
    parent = {"domain_tags": ["work"]}
    extract_context("input", parent_context=parent, llm=mock_llm)
    call_args = mock_llm.with_structured_output.return_value.invoke.call_args
    messages = call_args[0][0]
    assert any("work" in str(m.content) for m in messages)


# ---------------------------------------------------------------------------
# honesty (LLM mocked)
# ---------------------------------------------------------------------------


def _mock_llm_for_sincerity(is_sincere: bool):
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = _SincerityCheck(
        is_sincere_inquiry=is_sincere
    )
    return mock_llm


def test_detect_honesty_no_pattern():
    mock_llm = MagicMock()
    result = detect_honesty_assertion("What should I eat for dinner?", {}, llm=mock_llm)
    assert result is False
    mock_llm.with_structured_output.assert_not_called()  # LLM not invoked if regex misses


def test_detect_honesty_pattern_sincere():
    mock_llm = _mock_llm_for_sincerity(True)
    result = detect_honesty_assertion("Are you an AI?", {}, llm=mock_llm)
    assert result is True


def test_detect_honesty_pattern_not_sincere():
    mock_llm = _mock_llm_for_sincerity(False)
    result = detect_honesty_assertion("Are you an AI?", {}, llm=mock_llm)
    assert result is False


def test_detect_honesty_llm_error_conservative():
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("err")
    # Regex matches, LLM fails → conservative True
    result = detect_honesty_assertion("Are you an AI?", {}, llm=mock_llm)
    assert result is True


def test_detect_honesty_pattern_human_or_ai():
    mock_llm = _mock_llm_for_sincerity(True)
    result = detect_honesty_assertion("Are you a human or AI?", {}, llm=mock_llm)
    assert result is True


def test_detect_honesty_talking_to_pattern():
    mock_llm = _mock_llm_for_sincerity(True)
    result = detect_honesty_assertion("Am I talking to a bot?", {}, llm=mock_llm)
    assert result is True


def _mock_llm_for_rewrite(rewritten: str):
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = _RewriteResult(
        rewritten_response=rewritten
    )
    return mock_llm


def test_enforce_honesty_rewrites_response_text():
    decision = {"type": "response_text", "response_text": "I think you should go."}
    mock_llm = _mock_llm_for_rewrite("As an AI, I'd suggest going.")
    result = enforce_honesty_assertion(decision, llm=mock_llm)
    assert result["response_text"] == "As an AI, I'd suggest going."


def test_enforce_honesty_skips_non_response_text():
    decision = {"type": "action", "response_text": "do something"}
    mock_llm = MagicMock()
    result = enforce_honesty_assertion(decision, llm=mock_llm)
    assert result == decision
    mock_llm.with_structured_output.assert_not_called()


def test_enforce_honesty_fallback_on_error():
    decision = {"type": "response_text", "response_text": "original"}
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("err")
    result = enforce_honesty_assertion(decision, llm=mock_llm)
    assert result["response_text"] == "original"  # unchanged on error
