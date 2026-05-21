"""
ReasonerAgent: hypothesis generator sub-agent (spec §7.2).

Uses structured LLM output (with_structured_output) to generate 3-5 decision
candidates from retrieved shards, anchors, and active governance rules.

D10: data is pre-injected via inputs dict rather than tool-call retrieval.
Upgrade path: swap prompt-injection for tool-calling DeepAgents sub-agent
by replacing ReasonerAgent.invoke while keeping the same output schema.
"""

from typing import Optional

from pydantic import BaseModel, Field

from src.llm import with_structured_output as _with_structured_output


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class HypothesisOut(BaseModel):
    id: str = Field(description="Unique identifier for this hypothesis")
    content: str = Field(description="Human-readable description of the decision")
    decision_type: str = Field(
        description="One of: affirmative, negative, abstention, conditional, deferral"
    )
    structured_payload: Optional[dict] = Field(
        default=None,
        description="Machine-readable decision payload (optional)",
    )
    derivation: dict = Field(
        description="Sources: {shards: [...], anchors: [...], rules: [...], source: str}"
    )
    rule_conflicting: bool = Field(
        default=False,
        description="True if this hypothesis conflicts with any active rule",
    )
    conflict_details: list[dict] = Field(
        default_factory=list,
        description="List of {rule_id, statement, conflict_type} for conflicting rules",
    )


class ReasonerOut(BaseModel):
    candidates: list[HypothesisOut] = Field(
        description="3-5 decision candidates, ordered from most to least plausible"
    )
    traces: list[dict] = Field(
        description="Reasoning traces for each candidate"
    )


# ---------------------------------------------------------------------------
# Fallback when LLM is unavailable
# ---------------------------------------------------------------------------

_FALLBACK_HYPOTHESIS = HypothesisOut(
    id="fallback-hyp-1",
    content="Insufficient data to generate hypothesis (LLM unavailable)",
    decision_type="abstention",
    structured_payload=None,
    derivation={"shards": [], "anchors": [], "rules": [], "source": "fallback"},
    rule_conflicting=False,
    conflict_details=[],
)


def _build_prompt(inputs: dict) -> str:
    context = inputs.get("context") or {}
    shards = inputs.get("shards") or []
    anchors = inputs.get("anchors") or []
    active_rules = inputs.get("active_rules") or []
    historical = inputs.get("historical_hypotheses") or []
    is_bootstrap = inputs.get("is_bootstrap", False)
    is_off_baseline = inputs.get("is_off_baseline", False)
    self_refine_reason = inputs.get("self_refine_reason") or None

    shard_summaries = "\n".join(
        f"- [{s.get('shard_id', '?')}] {s.get('content', '')[:200]}"
        for s in shards[:10]
    )
    anchor_summaries = "\n".join(
        f"- [{a.get('anchor_id', '?')}] {a.get('statement', '')[:200]}"
        for a in anchors[:10]
    )
    rule_summaries = "\n".join(
        f"- [{r.get('rule_id', '?')}] {r.get('statement', '')[:200]}"
        for r in active_rules
    )
    bootstrap_note = "NOTE: Bootstrap mode — limited personal data available.\n" if is_bootstrap else ""
    baseline_note = "NOTE: Context is off-baseline — treat with extra caution.\n" if is_off_baseline else ""
    refine_note = f"SELF-REFINE INSTRUCTION: {self_refine_reason}\n" if self_refine_reason else ""

    prior_ids = [h.get("id") for h in historical if h.get("id")]
    prior_note = (
        f"Previously generated hypothesis IDs: {prior_ids}. Generate distinct alternatives.\n"
        if prior_ids else ""
    )

    return f"""{bootstrap_note}{baseline_note}{refine_note}{prior_note}
You are generating decision hypotheses for a specific user given a situation.
Generate 3-5 candidate decisions the user might make, ordered from most to least plausible.

SITUATION:
{context.get('summary') or context.get('raw_input') or str(context)[:500]}

RELEVANT MEMORIES (shards):
{shard_summaries or "(none)"}

IDENTITY ANCHORS:
{anchor_summaries or "(none)"}

ACTIVE GOVERNANCE RULES:
{rule_summaries or "(none)"}

For each candidate, identify:
1. The decision itself (content)
2. The decision type (affirmative/negative/abstention/conditional/deferral)
3. Which shards/anchors/rules support it (derivation)
4. Whether it conflicts with any rule (rule_conflicting, conflict_details)

Return candidates as a JSON array under "candidates" and reasoning traces under "traces".
"""


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class ReasonerAgent:
    """
    Hypothesis generator. Uses structured LLM output when an LLM is provided,
    falls back to a single placeholder hypothesis otherwise.
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm
        self._structured: Optional[object] = None
        if llm is not None:
            try:
                self._structured = _with_structured_output(llm, ReasonerOut)
            except Exception:
                self._structured = None

    def invoke(self, inputs: dict) -> dict:
        if self._structured is None:
            return {
                "candidates": [_FALLBACK_HYPOTHESIS.model_dump()],
                "traces": [{"source": "fallback", "reason": "no_llm"}],
            }

        prompt = _build_prompt(inputs)
        try:
            result: ReasonerOut = self._structured.invoke(prompt)
            return {
                "candidates": [h.model_dump() for h in result.candidates],
                "traces": result.traces,
            }
        except Exception as exc:
            return {
                "candidates": [_FALLBACK_HYPOTHESIS.model_dump()],
                "traces": [{"source": "fallback", "reason": str(exc)}],
            }
