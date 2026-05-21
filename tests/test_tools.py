"""
Unit tests for src/tools.py — the three external tool functions.

All stores are MagicMock. The compiled_graph is a stub returning a canned
output_payload, so no LLM or graph execution occurs.
"""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.tools import make_tools, _build_initial_state


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mock_stores():
    stores = MagicMock()
    stores.shards.add.return_value = str(uuid.uuid4())
    stores.governance.add_rule.return_value = str(uuid.uuid4())
    stores.pending_anchors.stage.return_value = str(uuid.uuid4())
    stores.outcomes.record.return_value = str(uuid.uuid4())
    stores.traces.get.return_value = {"user_id": "user-1", "trace_id": "abc123"}
    # bootstrap: counts above threshold → not bootstrap
    stores.anchors.count_for_user.return_value = 10
    stores.shards.count_for_user.return_value = 30
    stores.governance.count_active_rules.return_value = 5
    stores.reviews = MagicMock()
    return stores


def _make_stub_graph(output_payload=None):
    if output_payload is None:
        output_payload = {
            "decision": {"id": "h1", "content": "Accept the meeting", "decision_type": "recommendation"},
            "confidences": {"alignment": 0.8, "reproduction": 0.7, "divergence": 0.1},
            "annotations": {
                "is_off_baseline": False,
                "baseline_deviation": 0.0,
                "sparse_domain": None,
                "honesty_assertion_enforced": False,
                "bootstrap_mode": False,
                "recursion_depth": 0,
                "self_refine_reason": None,
            },
            "alternatives": [],
            "rule_basis": [],
            "trace_id": "abc123",
        }
    graph = MagicMock()
    graph.invoke.return_value = {"output_payload": output_payload}
    return graph


# ---------------------------------------------------------------------------
# make_tools
# ---------------------------------------------------------------------------

class TestMakeTools:
    def test_returns_three_tools(self):
        tools = make_tools(_make_mock_stores(), _make_stub_graph())
        assert len(tools) == 3

    def test_tool_names(self):
        tools = make_tools(_make_mock_stores(), _make_stub_graph())
        names = {t.name for t in tools}
        assert names == {"decide_as_user", "seed_user_data", "report_decision_outcome"}


# ---------------------------------------------------------------------------
# _build_initial_state
# ---------------------------------------------------------------------------

class TestBuildInitialState:
    def test_required_fields_present(self):
        state = _build_initial_state("u1", "Should I accept?", "make a decision", "agent-1")
        assert state["user_id"] == "u1"
        assert state["raw_input"] == "Should I accept?"
        assert state["parent_agent_context"] == {"goal": "make a decision", "caller": "agent-1"}
        assert state["invocation_mode"] == "subagent"
        assert state["trace_persist_required"] is True
        assert state["recursion_depth"] == 0
        assert state["trace_id"] != ""

    def test_unique_trace_ids(self):
        s1 = _build_initial_state("u1", "A", "g", "a")
        s2 = _build_initial_state("u1", "A", "g", "a")
        assert s1["trace_id"] != s2["trace_id"]


# ---------------------------------------------------------------------------
# decide_as_user
# ---------------------------------------------------------------------------

class TestDecideAsUser:
    def _get_tool(self, stores=None, graph=None):
        stores = stores or _make_mock_stores()
        graph = graph or _make_stub_graph()
        tools = make_tools(stores, graph)
        return next(t for t in tools if t.name == "decide_as_user")

    def test_returns_output_payload(self):
        tool = self._get_tool()
        result = tool.invoke({
            "user_id": "u1",
            "situation": "Should I accept the meeting?",
            "parent_goal": "schedule meetings",
            "parent_agent_id": "scheduler-agent",
        })
        assert "decision" in result
        assert "confidences" in result
        assert "trace_id" in result

    def test_invokes_graph_with_correct_user_id(self):
        stub_graph = _make_stub_graph()
        tool = self._get_tool(graph=stub_graph)
        tool.invoke({
            "user_id": "target-user",
            "situation": "Accept?",
            "parent_goal": "goal",
            "parent_agent_id": "agent",
        })
        call_kwargs = stub_graph.invoke.call_args
        state_arg = call_kwargs[0][0]
        assert state_arg["user_id"] == "target-user"

    def test_invokes_graph_with_subagent_mode(self):
        stub_graph = _make_stub_graph()
        tool = self._get_tool(graph=stub_graph)
        tool.invoke({
            "user_id": "u1",
            "situation": "Accept?",
            "parent_goal": "goal",
            "parent_agent_id": "agent",
        })
        state_arg = stub_graph.invoke.call_args[0][0]
        assert state_arg["invocation_mode"] == "subagent"
        assert state_arg["trace_persist_required"] is True

    def test_returns_only_output_payload_not_full_state(self):
        stub_graph = _make_stub_graph()
        tool = self._get_tool(graph=stub_graph)
        result = tool.invoke({
            "user_id": "u1",
            "situation": "Accept?",
            "parent_goal": "goal",
            "parent_agent_id": "agent",
        })
        # Should not include full state keys like retrieved_shards
        assert "retrieved_shards" not in result
        assert "hypotheses" not in result


# ---------------------------------------------------------------------------
# seed_user_data
# ---------------------------------------------------------------------------

class TestSeedUserData:
    def _get_tool(self, stores=None):
        stores = stores or _make_mock_stores()
        return next(
            t for t in make_tools(stores, _make_stub_graph())
            if t.name == "seed_user_data"
        ), stores

    def test_decision_creates_shard(self):
        tool, stores = self._get_tool()
        result = tool.invoke({
            "user_id": "u1",
            "seed_items": [{"content": "I prefer async meetings", "type": "decision"}],
        })
        stores.shards.add.assert_called_once()
        assert result["shards_created"] == 1
        assert result["rules_created"] == 0
        assert result["anchors_created"] == 0

    def test_preference_creates_shard_with_tag(self):
        tool, stores = self._get_tool()
        tool.invoke({
            "user_id": "u1",
            "seed_items": [{"content": "I prefer mornings", "type": "preference"}],
        })
        call_args = stores.shards.add.call_args[0][0]
        assert "preference" in call_args["domain_tags"]

    def test_value_creates_shard_and_rule(self):
        tool, stores = self._get_tool()
        result = tool.invoke({
            "user_id": "u1",
            "seed_items": [{"content": "Family comes first", "type": "value"}],
        })
        stores.shards.add.assert_called_once()
        stores.governance.add_rule.assert_called_once()
        assert result["shards_created"] == 1
        assert result["rules_created"] == 1

    def test_value_rule_has_low_confidence(self):
        tool, stores = self._get_tool()
        tool.invoke({
            "user_id": "u1",
            "seed_items": [{"content": "Family comes first", "type": "value"}],
        })
        rule_arg = stores.governance.add_rule.call_args[0][0]
        assert rule_arg["confidence"] == 0.3

    def test_value_rule_uses_stated_confidence(self):
        tool, stores = self._get_tool()
        tool.invoke({
            "user_id": "u1",
            "seed_items": [{"content": "Family comes first", "type": "value", "stated_confidence": 0.9}],
        })
        rule_arg = stores.governance.add_rule.call_args[0][0]
        assert rule_arg["confidence"] == 0.9

    def test_anchor_candidate_stages_not_adds(self):
        tool, stores = self._get_tool()
        result = tool.invoke({
            "user_id": "u1",
            "seed_items": [{"content": "I value work-life balance", "type": "anchor_candidate"}],
        })
        stores.pending_anchors.stage.assert_called_once()
        stores.anchors.add.assert_not_called()
        assert result["anchors_created"] == 1
        assert result["shards_created"] == 0

    def test_mixed_items_counted_correctly(self):
        tool, stores = self._get_tool()
        result = tool.invoke({
            "user_id": "u1",
            "seed_items": [
                {"content": "Decision 1", "type": "decision"},
                {"content": "Value 1", "type": "value"},
                {"content": "Anchor 1", "type": "anchor_candidate"},
            ],
        })
        assert result["shards_created"] == 2  # decision + value
        assert result["rules_created"] == 1
        assert result["anchors_created"] == 1

    def test_returns_still_bootstrap_false_when_above_threshold(self):
        stores = _make_mock_stores()
        # counts above threshold
        stores.anchors.count_for_user.return_value = 10
        stores.shards.count_for_user.return_value = 30
        stores.governance.count_active_rules.return_value = 5
        tool, _ = self._get_tool(stores)
        result = tool.invoke({"user_id": "u1", "seed_items": []})
        assert result["still_bootstrap"] is False

    def test_returns_still_bootstrap_true_when_below_threshold(self):
        stores = _make_mock_stores()
        stores.anchors.count_for_user.return_value = 2
        stores.shards.count_for_user.return_value = 5
        stores.governance.count_active_rules.return_value = 1
        tool, _ = self._get_tool(stores)
        result = tool.invoke({"user_id": "u1", "seed_items": []})
        assert result["still_bootstrap"] is True


# ---------------------------------------------------------------------------
# report_decision_outcome
# ---------------------------------------------------------------------------

class TestReportDecisionOutcome:
    def _get_tool(self, stores=None):
        stores = stores or _make_mock_stores()
        return next(
            t for t in make_tools(stores, _make_stub_graph())
            if t.name == "report_decision_outcome"
        ), stores

    def test_accepted_unchanged_maps_to_accepted(self):
        tool, stores = self._get_tool()
        tool.invoke({"trace_id": "t1", "outcome": "accepted_unchanged"})
        call_kwargs = stores.outcomes.record.call_args
        assert call_kwargs[1]["outcome_type"] == "accepted"

    def test_accepted_with_edits_maps_to_edited(self):
        tool, stores = self._get_tool()
        tool.invoke({"trace_id": "t1", "outcome": "accepted_with_edits", "edited_content": "modified"})
        call_kwargs = stores.outcomes.record.call_args
        assert call_kwargs[1]["outcome_type"] == "edited"

    def test_rejected_maps_to_rejected(self):
        tool, stores = self._get_tool()
        tool.invoke({"trace_id": "t1", "outcome": "rejected", "rejection_reason": "wrong tone"})
        call_kwargs = stores.outcomes.record.call_args
        assert call_kwargs[1]["outcome_type"] == "rejected"

    def test_ignored_skips_db_write(self):
        tool, stores = self._get_tool()
        result = tool.invoke({"trace_id": "t1", "outcome": "ignored"})
        stores.outcomes.record.assert_not_called()
        assert result["received"] is True
        assert result["outcome_id"] is None

    def test_returns_received_true(self):
        tool, stores = self._get_tool()
        result = tool.invoke({"trace_id": "t1", "outcome": "accepted_unchanged"})
        assert result["received"] is True

    def test_returns_outcome_id(self):
        stores = _make_mock_stores()
        fixed_id = "outcome-abc"
        stores.outcomes.record.return_value = fixed_id
        tool, _ = self._get_tool(stores)
        result = tool.invoke({"trace_id": "t1", "outcome": "accepted_unchanged"})
        assert result["outcome_id"] == fixed_id

    def test_returns_processing_eta(self):
        tool, _ = self._get_tool()
        result = tool.invoke({"trace_id": "t1", "outcome": "accepted_unchanged"})
        assert result["processing_eta_seconds"] == 300

    def test_passes_edited_content_to_store(self):
        tool, stores = self._get_tool()
        tool.invoke({
            "trace_id": "t1",
            "outcome": "accepted_with_edits",
            "edited_content": "the edited text",
        })
        call_kwargs = stores.outcomes.record.call_args
        assert call_kwargs[1]["edited_content"] == "the edited text"

    def test_passes_rejection_reason_to_store(self):
        tool, stores = self._get_tool()
        tool.invoke({
            "trace_id": "t1",
            "outcome": "rejected",
            "rejection_reason": "too formal",
        })
        call_kwargs = stores.outcomes.record.call_args
        assert call_kwargs[1]["rejection_reason"] == "too formal"

    def test_final_action_taken_does_not_reach_store(self):
        tool, stores = self._get_tool()
        tool.invoke({
            "trace_id": "t1",
            "outcome": "accepted_unchanged",
            "final_action_taken": {"action": "sent_email"},
        })
        call_kwargs = stores.outcomes.record.call_args
        # final_action_taken should not appear in any store call argument
        assert "final_action_taken" not in (call_kwargs[1] or {})
