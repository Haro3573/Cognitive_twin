"""
Unit tests for Step 5 sub-agents:
  - compute_baseline_deviation (baseline.py)
  - RecallAgent (recall.py)
  - ReasonerAgent (reasoner.py)
  - CriticAgent (critic.py)

All store interactions use MagicMock. LLM interactions use MagicMock
with structured output returning Pydantic models directly.
"""

import math
import pytest
from unittest.mock import MagicMock, patch

from src.subagents.baseline import compute_baseline_deviation
from src.subagents.recall import RecallAgent
from src.subagents.reasoner import ReasonerAgent, ReasonerOut, HypothesisOut
from src.subagents.critic import CriticAgent, CriticOut, ScoredHypothesisOut


# ===========================================================================
# Helpers
# ===========================================================================

def _unit_vec(dim: int, val: float = 1.0) -> list[float]:
    """Returns a unit-normalized vector with `val` in first position."""
    v = [0.0] * dim
    v[0] = val
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


def _make_shard(embedding: list[float], content: str = "shard") -> dict:
    return {"shard_id": "s1", "content": content, "embedding": embedding}


def _make_anchor(embedding: list[float], statement: str = "anchor") -> dict:
    return {"anchor_id": "a1", "statement": statement, "embedding": embedding}


def _make_hypothesis(idx: int = 0) -> dict:
    return {
        "id": f"hyp-{idx}",
        "content": f"Decision option {idx}",
        "decision_type": "affirmative",
        "structured_payload": None,
        "derivation": {"shards": [], "anchors": [], "rules": [], "source": "test"},
        "rule_conflicting": False,
        "conflict_details": [],
    }


def _make_mock_stores(has_results: bool = True) -> MagicMock:
    stores = MagicMock()
    if has_results:
        stores.shards.search.return_value = [_make_shard(_unit_vec(4))]
        stores.shards.sample_recent.return_value = [_make_shard(_unit_vec(4))]
        stores.anchors.search.return_value = [_make_anchor(_unit_vec(4))]
        stores.anchors.list_for_user.return_value = [_make_anchor(_unit_vec(4))]
    else:
        stores.shards.search.return_value = []
        stores.shards.sample_recent.return_value = []
        stores.anchors.search.return_value = []
        stores.anchors.list_for_user.return_value = []
    return stores


# ===========================================================================
# compute_baseline_deviation
# ===========================================================================

class TestComputeBaselineDeviation:
    def test_returns_zero_insufficient_anchors(self):
        shards = [_make_shard(_unit_vec(4)) for _ in range(15)]
        anchors = [_make_anchor(_unit_vec(4)) for _ in range(3)]  # < 5
        assert compute_baseline_deviation(shards, anchors) == 0.0

    def test_returns_zero_insufficient_shards(self):
        shards = [_make_shard(_unit_vec(4)) for _ in range(5)]   # < 10
        anchors = [_make_anchor(_unit_vec(4)) for _ in range(6)]
        assert compute_baseline_deviation(shards, anchors) == 0.0

    def test_identical_embeddings_returns_near_zero_deviation(self):
        vec = _unit_vec(4)
        shards = [_make_shard(vec) for _ in range(10)]
        anchors = [_make_anchor(vec) for _ in range(5)]
        deviation = compute_baseline_deviation(shards, anchors)
        assert deviation < 0.05, f"identical embeddings → near-zero deviation, got {deviation}"

    def test_orthogonal_embeddings_return_high_deviation(self):
        dim = 4
        anchor_vec = [1.0, 0.0, 0.0, 0.0]
        shard_vec = [0.0, 1.0, 0.0, 0.0]
        shards = [_make_shard(shard_vec) for _ in range(10)]
        anchors = [_make_anchor(anchor_vec) for _ in range(5)]
        deviation = compute_baseline_deviation(shards, anchors)
        assert deviation > 0.9, f"orthogonal embeddings → high deviation, got {deviation}"

    def test_result_clamped_to_unit_interval(self):
        vec = _unit_vec(4)
        shards = [_make_shard(vec) for _ in range(10)]
        anchors = [_make_anchor(vec) for _ in range(5)]
        d = compute_baseline_deviation(shards, anchors)
        assert 0.0 <= d <= 1.0

    def test_skips_shards_missing_embedding(self):
        shards = [{"shard_id": "x", "content": "no embedding"}] * 15
        anchors = [_make_anchor(_unit_vec(4)) for _ in range(5)]
        # No embeddings → insufficient → returns 0.0
        assert compute_baseline_deviation(shards, anchors) == 0.0

    def test_zero_magnitude_vector_handled_gracefully(self):
        zero_vec = [0.0, 0.0, 0.0, 0.0]
        shards = [_make_shard(zero_vec) for _ in range(10)]
        anchors = [_make_anchor(_unit_vec(4)) for _ in range(5)]
        d = compute_baseline_deviation(shards, anchors)
        assert 0.0 <= d <= 1.0


# ===========================================================================
# RecallAgent
# ===========================================================================

class TestRecallAgent:
    def test_returns_required_keys(self):
        stores = _make_mock_stores(has_results=True)
        agent = RecallAgent(stores)
        result = agent.invoke({
            "user_id": "u1",
            "context": {"summary": "test decision", "domain_tags": []},
        })
        assert "shards" in result
        assert "anchors" in result
        assert "strategy" in result
        assert "baseline_deviation" in result

    def test_semantic_strategy_when_query_text_present(self):
        stores = _make_mock_stores(has_results=True)
        agent = RecallAgent(stores)
        result = agent.invoke({
            "user_id": "u1",
            "context": {"summary": "career decision", "domain_tags": []},
        })
        assert result["strategy"] == "semantic"
        stores.shards.search.assert_called_once()
        stores.anchors.search.assert_called_once()

    def test_recency_strategy_when_no_query_text(self):
        stores = _make_mock_stores(has_results=True)
        agent = RecallAgent(stores)
        result = agent.invoke({
            "user_id": "u1",
            "context": {},
        })
        assert result["strategy"] == "recency"
        stores.shards.sample_recent.assert_called()
        stores.anchors.list_for_user.assert_called_once()

    def test_baseline_deviation_returned_as_float(self):
        stores = _make_mock_stores(has_results=True)
        agent = RecallAgent(stores)
        result = agent.invoke({
            "user_id": "u1",
            "context": {"summary": "test"},
        })
        assert isinstance(result["baseline_deviation"], float)
        assert 0.0 <= result["baseline_deviation"] <= 1.0

    def test_empty_results_when_store_returns_nothing(self):
        stores = _make_mock_stores(has_results=False)
        agent = RecallAgent(stores)
        result = agent.invoke({
            "user_id": "u1",
            "context": {"summary": "test"},
        })
        assert result["shards"] == []
        assert result["anchors"] == []

    def test_bootstrap_mode_adjusts_k(self):
        stores = _make_mock_stores(has_results=True)
        agent = RecallAgent(stores)
        agent.invoke({
            "user_id": "u1",
            "context": {"summary": "test"},
            "is_bootstrap": True,
        })
        call_kwargs = stores.shards.search.call_args
        k_used = call_kwargs.kwargs.get("k")
        assert k_used == 15

    def test_domain_tags_passed_to_search(self):
        stores = _make_mock_stores(has_results=True)
        agent = RecallAgent(stores)
        agent.invoke({
            "user_id": "u1",
            "context": {"summary": "test", "domain_tags": ["health"]},
        })
        call_kwargs = stores.shards.search.call_args
        domain_tags_used = call_kwargs.kwargs.get("domain_tags") or (
            call_kwargs.args[3] if len(call_kwargs.args) > 3 else None
        )
        assert domain_tags_used == ["health"]


# ===========================================================================
# ReasonerAgent
# ===========================================================================

def _make_reasoner_out(n: int = 2) -> ReasonerOut:
    candidates = [
        HypothesisOut(
            id=f"hyp-{i}",
            content=f"Option {i}",
            decision_type="affirmative",
            structured_payload=None,
            derivation={"shards": [], "anchors": [], "rules": [], "source": "llm"},
            rule_conflicting=False,
            conflict_details=[],
        )
        for i in range(n)
    ]
    return ReasonerOut(
        candidates=candidates,
        traces=[{"step": i, "reasoning": f"trace {i}"} for i in range(n)],
    )


class TestReasonerAgent:
    def test_fallback_when_no_llm(self):
        agent = ReasonerAgent(llm=None)
        result = agent.invoke({"context": {}, "shards": [], "anchors": [], "active_rules": []})
        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["decision_type"] == "abstention"

    def test_returns_llm_candidates(self):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _make_reasoner_out(3)
        mock_llm.with_structured_output.return_value = mock_structured

        agent = ReasonerAgent(llm=mock_llm)
        result = agent.invoke({
            "context": {"summary": "career decision"},
            "shards": [],
            "anchors": [],
            "active_rules": [],
        })
        assert len(result["candidates"]) == 3
        assert "traces" in result

    def test_fallback_on_llm_exception(self):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = RuntimeError("API error")
        mock_llm.with_structured_output.return_value = mock_structured

        agent = ReasonerAgent(llm=mock_llm)
        result = agent.invoke({"context": {}, "shards": [], "anchors": [], "active_rules": []})
        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["id"] == "fallback-hyp-1"

    def test_each_candidate_has_required_keys(self):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _make_reasoner_out(2)
        mock_llm.with_structured_output.return_value = mock_structured

        agent = ReasonerAgent(llm=mock_llm)
        result = agent.invoke({"context": {}, "shards": [], "anchors": [], "active_rules": []})
        for candidate in result["candidates"]:
            assert "id" in candidate
            assert "content" in candidate
            assert "decision_type" in candidate
            assert "derivation" in candidate
            assert "rule_conflicting" in candidate

    def test_bootstrap_note_in_prompt(self):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _make_reasoner_out(1)
        mock_llm.with_structured_output.return_value = mock_structured

        agent = ReasonerAgent(llm=mock_llm)
        agent.invoke({
            "context": {"summary": "test"},
            "is_bootstrap": True,
            "shards": [],
            "anchors": [],
            "active_rules": [],
        })
        call_args = mock_structured.invoke.call_args[0][0]
        assert "Bootstrap" in call_args or "bootstrap" in call_args.lower()

    def test_self_refine_note_in_prompt(self):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _make_reasoner_out(1)
        mock_llm.with_structured_output.return_value = mock_structured

        agent = ReasonerAgent(llm=mock_llm)
        agent.invoke({
            "context": {"summary": "test"},
            "self_refine_reason": "alignment below threshold",
            "shards": [],
            "anchors": [],
            "active_rules": [],
        })
        call_args = mock_structured.invoke.call_args[0][0]
        assert "alignment below threshold" in call_args

    def test_with_structured_output_exception_handled(self):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = AttributeError("not supported")

        agent = ReasonerAgent(llm=mock_llm)
        # Should not raise; _structured becomes None → fallback
        result = agent.invoke({"context": {}, "shards": [], "anchors": [], "active_rules": []})
        assert result["candidates"][0]["id"] == "fallback-hyp-1"


# ===========================================================================
# CriticAgent
# ===========================================================================

def _make_pass1_out(hypotheses: list[dict]) -> MagicMock:
    out = MagicMock()
    out.hypotheses = [
        {"hypothesis_index": i, "reproduction_confidence": 0.7, "reasoning": f"pass1 trace {i}"}
        for i in range(len(hypotheses))
    ]
    return out


def _make_pass2_out(hypotheses: list[dict]) -> MagicMock:
    out = MagicMock()
    out.hypotheses = [
        {
            "hypothesis_index": i,
            "alignment_confidence": 0.8,
            "rule_conformance_score": 1.0,
            "rule_conflict_details": [],
            "reasoning": f"pass2 trace {i}",
        }
        for i in range(len(hypotheses))
    ]
    return out


class TestCriticAgent:
    def test_fallback_when_no_llm(self):
        agent = CriticAgent(llm=None)
        hyps = [_make_hypothesis(0), _make_hypothesis(1)]
        result = agent.invoke({"hypotheses": hyps, "shards": [], "anchors": [], "active_rules": [], "context": {}})
        assert len(result["hypotheses"]) == 2
        for h in result["hypotheses"]:
            assert h["alignment_confidence"] == 0.6
            assert h["reproduction_confidence"] == 0.5

    def test_empty_hypotheses_returns_empty(self):
        agent = CriticAgent(llm=None)
        result = agent.invoke({"hypotheses": [], "shards": [], "anchors": [], "active_rules": [], "context": {}})
        assert result["hypotheses"] == []

    def test_two_pass_scoring(self):
        mock_llm = MagicMock()
        hyps = [_make_hypothesis(0), _make_hypothesis(1)]

        call_count = [0]
        def structured_side_effect(schema):
            mock_chain = MagicMock()
            if call_count[0] == 0:
                mock_chain.invoke.return_value = _make_pass1_out(hyps)
            else:
                mock_chain.invoke.return_value = _make_pass2_out(hyps)
            call_count[0] += 1
            return mock_chain

        mock_llm.with_structured_output.side_effect = structured_side_effect

        agent = CriticAgent(llm=mock_llm)
        result = agent.invoke({
            "hypotheses": hyps,
            "shards": [_make_shard(_unit_vec(4))],
            "anchors": [_make_anchor(_unit_vec(4))],
            "active_rules": [],
            "context": {"summary": "test"},
        })
        assert len(result["hypotheses"]) == 2
        assert mock_llm.with_structured_output.call_count == 2  # two passes

    def test_scored_hypotheses_contain_original_hypothesis(self):
        agent = CriticAgent(llm=None)
        hyp = _make_hypothesis(0)
        result = agent.invoke({
            "hypotheses": [hyp],
            "shards": [],
            "anchors": [],
            "active_rules": [],
            "context": {},
        })
        assert result["hypotheses"][0]["hypothesis"]["id"] == hyp["id"]

    def test_confidence_scores_clamped_to_unit_interval(self):
        mock_llm = MagicMock()
        hyps = [_make_hypothesis(0)]

        # Return out-of-range values to test clamping
        pass1 = MagicMock()
        pass1.hypotheses = [{"hypothesis_index": 0, "reproduction_confidence": 1.5, "reasoning": ""}]
        pass2 = MagicMock()
        pass2.hypotheses = [{"hypothesis_index": 0, "alignment_confidence": -0.2,
                              "rule_conformance_score": 2.0, "rule_conflict_details": [], "reasoning": ""}]

        call_count = [0]
        def se(schema):
            mc = MagicMock()
            mc.invoke.return_value = pass1 if call_count[0] == 0 else pass2
            call_count[0] += 1
            return mc

        mock_llm.with_structured_output.side_effect = se

        agent = CriticAgent(llm=mock_llm)
        result = agent.invoke({
            "hypotheses": hyps,
            "shards": [],
            "anchors": [],
            "active_rules": [],
            "context": {},
        })
        h = result["hypotheses"][0]
        assert 0.0 <= h["alignment_confidence"] <= 1.0
        assert 0.0 <= h["reproduction_confidence"] <= 1.0
        assert 0.0 <= h["rule_conformance_score"] <= 1.0

    def test_fallback_on_llm_exception(self):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = RuntimeError("API down")

        agent = CriticAgent(llm=mock_llm)
        hyps = [_make_hypothesis(0)]
        result = agent.invoke({
            "hypotheses": hyps,
            "shards": [],
            "anchors": [],
            "active_rules": [],
            "context": {},
        })
        assert len(result["hypotheses"]) == 1
        assert result["hypotheses"][0]["alignment_confidence"] == 0.6

    def test_result_contains_reasoning(self):
        agent = CriticAgent(llm=None)
        hyps = [_make_hypothesis(0)]
        result = agent.invoke({
            "hypotheses": hyps,
            "shards": [],
            "anchors": [],
            "active_rules": [],
            "context": {},
        })
        assert "reasoning" in result["hypotheses"][0]

    def test_fallback_alignment_triggers_confidence_router_route(self):
        """Fallback produces alignment=0.6 ≥ 0.5, so router should NOT recurse."""
        agent = CriticAgent(llm=None)
        hyps = [_make_hypothesis(0)]
        result = agent.invoke({
            "hypotheses": hyps, "shards": [], "anchors": [], "active_rules": [], "context": {},
        })
        assert result["hypotheses"][0]["alignment_confidence"] >= 0.5
