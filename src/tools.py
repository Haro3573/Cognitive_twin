"""
Tool surface for the Cognitive Twin Sub-Agent (spec §11).

make_tools(stores, compiled_graph) → list of three LangChain StructuredTools:
  - decide_as_user
  - seed_user_data
  - report_decision_outcome

The compiled_graph is built once by the caller and injected (D13).
Outcome-type mapping lives here at the tool boundary (D12):
  accepted_unchanged → "accepted"
  accepted_with_edits → "edited"
  rejected            → "rejected"
  ignored             → skip DB write
"""

import uuid
from datetime import datetime
from typing import Literal, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.deps import Stores
from src.graph import RUNTIME_CONFIG
from src.helpers.predicates import is_bootstrap_state


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class _DecideAsUserInput(BaseModel):
    user_id: str = Field(description="ID of the user to simulate")
    situation: str = Field(description="The decision situation to evaluate")
    parent_goal: str = Field(description="Goal of the calling parent agent")
    parent_agent_id: str = Field(description="ID of the calling parent agent")


class _SeedItem(BaseModel):
    content: str
    context: Optional[dict] = None
    type: Literal["decision", "preference", "value", "anchor_candidate"]
    stated_confidence: Optional[float] = None
    as_of: Optional[datetime] = None


class _SeedUserDataInput(BaseModel):
    user_id: str = Field(description="ID of the user to seed data for")
    seed_items: list[_SeedItem] = Field(description="Items to seed into user stores")


class _ReportDecisionOutcomeInput(BaseModel):
    trace_id: str = Field(description="trace_id from the decide_as_user result")
    outcome: Literal["accepted_unchanged", "accepted_with_edits", "rejected", "ignored"] = Field(
        description="What the parent agent did with the decision"
    )
    edited_content: Optional[str] = Field(default=None, description="If accepted_with_edits, the edited text")
    rejection_reason: Optional[str] = Field(default=None, description="If rejected, why")
    final_action_taken: Optional[dict] = Field(default=None, description="Accepted but not stored; for caller bookkeeping only")


# ---------------------------------------------------------------------------
# Initial state builder
# ---------------------------------------------------------------------------

def _build_initial_state(
    user_id: str,
    situation: str,
    parent_goal: str,
    parent_agent_id: str,
) -> dict:
    return {
        "user_id": user_id,
        "invocation_mode": "subagent",
        "parent_agent_context": {"goal": parent_goal, "caller": parent_agent_id},
        "raw_input": situation,
        "recursion_depth": 0,
        "trace_id": uuid.uuid4().hex,
        "trace_persist_required": True,
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


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _make_decide_as_user(compiled_graph):
    def decide_as_user(
        user_id: str,
        situation: str,
        parent_goal: str,
        parent_agent_id: str,
    ) -> dict:
        """
        Returns the decision this user would make in the given situation.

        Returns a DecisionPayload with decision, confidences, annotations,
        alternatives, rule_basis, and trace_id. Parent agents SHOULD call
        report_decision_outcome with the trace_id after acting on the result.

        For behavioral PREDICTION tasks, choose from alternatives by
        reproduction_confidence (see spec §15 selection tradeoff note).
        """
        state = _build_initial_state(user_id, situation, parent_goal, parent_agent_id)
        result = compiled_graph.invoke(state, config=RUNTIME_CONFIG)
        return result["output_payload"]

    return decide_as_user


def _make_seed_user_data(stores: Stores):
    def seed_user_data(user_id: str, seed_items: list[dict]) -> dict:
        """
        Seeds user stores from external data (onboarding, imported history, etc.).

        Each item must have 'content' and 'type'. Supported types:
          decision/preference → creates a shard
          value               → creates a shard + candidate governance rule (confidence=0.3)
          anchor_candidate    → stages in pending_anchors (requires user confirmation)

        Returns counts and whether the user is still in bootstrap mode.
        """
        shards_created = 0
        anchors_created = 0
        rules_created = 0
        now = datetime.now()

        for raw in seed_items:
            item = _SeedItem.model_validate(raw) if isinstance(raw, dict) else raw
            as_of = item.as_of or now
            item_context = item.context or {}

            if item.type in ("decision", "preference", "value"):
                domain_tags: list[str] = []
                if item.type == "preference":
                    domain_tags = ["preference"]
                elif item.type == "value":
                    domain_tags = ["value"]

                shard = {
                    "shard_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "context": item_context,
                    "content": item.content,
                    "compression_level": 0,
                    "created_at": as_of,
                    "last_activated_at": as_of,
                    "activation_count": 0,
                    "decay_score": 0.0,
                    "domain_tags": domain_tags,
                    "embedding": [],
                }
                stores.shards.add(shard)
                shards_created += 1

                if item.type == "value":
                    rule = {
                        "rule_id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "statement": item.content,
                        "confidence": item.stated_confidence if item.stated_confidence is not None else 0.3,
                        "evidence_count": 0,
                        "activated_at": as_of,
                        "rule_class": "value",
                        "context_scope": list(item_context.get("domain_tags", [])),
                    }
                    stores.governance.add_rule(rule)
                    rules_created += 1

            elif item.type == "anchor_candidate":
                confidence = item.stated_confidence if item.stated_confidence is not None else 0.5
                anchor = {
                    "anchor_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "statement": item.content,
                    "confidence": confidence,
                    "established_at": as_of,
                    "last_reinforced_at": as_of,
                    "pending_confirmation_at": now,
                    "seeded_from": "seed_user_data",
                    "context_scope": list(item_context.get("domain_tags", [])),
                }
                stores.pending_anchors.stage(anchor)
                anchors_created += 1

        still_bootstrap = is_bootstrap_state(
            user_id,
            stores.anchors,
            stores.shards,
            stores.governance,
        )
        return {
            "shards_created": shards_created,
            "anchors_created": anchors_created,
            "rules_created": rules_created,
            "still_bootstrap": still_bootstrap,
        }

    return seed_user_data


def _make_report_decision_outcome(stores: Stores):
    # Outcome type mapping (D12): spec vocabulary → DB vocabulary
    _OUTCOME_MAP = {
        "accepted_unchanged": "accepted",
        "accepted_with_edits": "edited",
        "rejected": "rejected",
    }

    def report_decision_outcome(
        trace_id: str,
        outcome: str,
        edited_content: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        final_action_taken: Optional[dict] = None,  # accepted and dropped; no DB column
    ) -> dict:
        """
        Reports what the parent agent did with a previous decide_as_user result.

        Writes to OutcomeStore; async outcome processor picks up within 5 minutes.
        Late reports (>30 days) are accepted but flagged as low-weight.
        'ignored' outcomes are acknowledged but not stored.
        """
        if outcome == "ignored":
            return {"received": True, "outcome_id": None, "processing_eta_seconds": 0}

        db_outcome = _OUTCOME_MAP.get(outcome)
        if db_outcome is None:
            raise ValueError(f"Unknown outcome type: {outcome!r}")

        # Derive user_id from the trace record for the outcome store call.
        # trace_store.get returns the compressed state dict; user_id is stored there.
        trace = stores.traces.get(trace_id)
        user_id = trace.get("user_id", "") if trace else ""

        outcome_id = stores.outcomes.record(
            user_id=user_id,
            trace_id=trace_id,
            outcome_type=db_outcome,
            edited_content=edited_content,
            rejection_reason=rejection_reason,
        )
        return {"received": True, "outcome_id": outcome_id, "processing_eta_seconds": 300}

    return report_decision_outcome


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_tools(stores: Stores, compiled_graph) -> list:
    """
    Returns a list of three LangChain StructuredTools.

    compiled_graph must already be built (D13: graph cached, not rebuilt per call).
    """
    return [
        StructuredTool.from_function(
            func=_make_decide_as_user(compiled_graph),
            name="decide_as_user",
            args_schema=_DecideAsUserInput,
        ),
        StructuredTool.from_function(
            func=_make_seed_user_data(stores),
            name="seed_user_data",
            args_schema=_SeedUserDataInput,
        ),
        StructuredTool.from_function(
            func=_make_report_decision_outcome(stores),
            name="report_decision_outcome",
            args_schema=_ReportDecisionOutcomeInput,
        ),
    ]
