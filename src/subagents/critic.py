"""
CriticAgent: dual-confidence critic sub-agent (spec §7.3).

Two-pass scoring (spec §7.3):
  Pass 1 — reproduction_confidence: scored from shards only (moment-of-action).
  Pass 2 — alignment_confidence: scored from anchors + rules (reflective endorsement).

D10: shards/anchors/active_rules are pre-injected via inputs dict.
Upgrade path: replace invoke internals with tool-calling DeepAgents critic
while keeping CriticOut schema and the two-pass structure.
"""

from typing import Optional

from pydantic import BaseModel, Field

from src.llm import with_structured_output as _with_structured_output


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class ScoredHypothesisOut(BaseModel):
    hypothesis: dict = Field(description="The original hypothesis dict")
    alignment_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Reflective endorsement score (anchors + rules)",
    )
    reproduction_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Moment-of-action prediction score (shards only)",
    )
    rule_conformance_score: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of active rules this hypothesis conforms to",
    )
    rule_conflict_details: list[dict] = Field(
        default_factory=list,
        description="List of {rule_id, conflict_type} for conflicting rules",
    )
    reasoning: str = Field(description="Brief scoring rationale")


class CriticOut(BaseModel):
    hypotheses: list[ScoredHypothesisOut] = Field(
        description="All input hypotheses with dual confidence scores"
    )


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

def _fallback_scored(hypotheses: list[dict]) -> dict:
    scored = []
    for hyp in hypotheses:
        scored.append({
            "hypothesis": hyp,
            "alignment_confidence": 0.6,
            "reproduction_confidence": 0.5,
            "rule_conformance_score": 1.0,
            "rule_conflict_details": [],
            "reasoning": "fallback: no LLM available",
        })
    return {"hypotheses": scored}


def _build_prompt(hypotheses: list[dict], shards: list[dict], anchors: list[dict],
                  active_rules: list[dict], context: dict, pass_num: int) -> str:
    hyp_text = "\n".join(
        f"[{i}] {h.get('content', '')[:300]} (type: {h.get('decision_type', '?')})"
        for i, h in enumerate(hypotheses)
    )
    if pass_num == 1:
        evidence = "\n".join(
            f"- {s.get('content', '')[:200]}"
            for s in shards[:10]
        )
        instruction = (
            "PASS 1: Score reproduction_confidence only.\n"
            "Reproduction = how likely is this exactly what the user WOULD DO in the moment, "
            "based on past behavior (shards)?\n"
            "Evidence (shards):\n" + (evidence or "(none)")
        )
    else:
        anchor_text = "\n".join(
            f"- {a.get('statement', '')[:200]}" for a in anchors[:10]
        )
        rule_text = "\n".join(
            f"- [{r.get('rule_id', '?')}] {r.get('statement', '')[:200]}"
            for r in active_rules
        )
        instruction = (
            "PASS 2: Score alignment_confidence and rule conformance.\n"
            "Alignment = how much would the user REFLECTIVELY ENDORSE this decision "
            "based on their identity anchors and values?\n"
            "Also identify rule conflicts.\n\n"
            "Anchors:\n" + (anchor_text or "(none)") + "\n\n"
            "Active rules:\n" + (rule_text or "(none)")
        )

    return f"""You are scoring decision hypotheses for a specific user.

SITUATION: {context.get('summary') or context.get('raw_input') or str(context)[:300]}

HYPOTHESES:
{hyp_text}

{instruction}

Return scores for each hypothesis as a JSON array under "hypotheses".
Each entry: hypothesis (copy from input), alignment_confidence, reproduction_confidence,
rule_conformance_score, rule_conflict_details, reasoning.
"""


class _Pass1Out(BaseModel):
    hypotheses: list[dict] = Field(
        description="Each entry: {hypothesis_index, reproduction_confidence, reasoning}"
    )


class _Pass2Out(BaseModel):
    hypotheses: list[dict] = Field(
        description="Each entry: {hypothesis_index, alignment_confidence, rule_conformance_score, rule_conflict_details, reasoning}"
    )


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class CriticAgent:
    """
    Two-pass dual-confidence critic.

    Pass 1 (shards only) → reproduction_confidence
    Pass 2 (anchors + rules) → alignment_confidence + rule conformance
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm

    def invoke(self, inputs: dict) -> dict:
        hypotheses: list[dict] = inputs.get("hypotheses") or []
        shards: list[dict] = inputs.get("shards") or []
        anchors: list[dict] = inputs.get("anchors") or []
        active_rules: list[dict] = inputs.get("active_rules") or []
        context: dict = inputs.get("context") or {}

        if not hypotheses:
            return {"hypotheses": []}

        if self._llm is None:
            return _fallback_scored(hypotheses)

        try:
            return self._two_pass_score(hypotheses, shards, anchors, active_rules, context)
        except Exception as exc:
            return _fallback_scored(hypotheses)

    def _two_pass_score(
        self,
        hypotheses: list[dict],
        shards: list[dict],
        anchors: list[dict],
        active_rules: list[dict],
        context: dict,
    ) -> dict:
        # Pass 1: reproduction from shards
        pass1_llm = _with_structured_output(self._llm, _Pass1Out)
        pass1_prompt = _build_prompt(hypotheses, shards, anchors, active_rules, context, pass_num=1)
        pass1_result: _Pass1Out = pass1_llm.invoke(pass1_prompt)
        repro_by_idx = {
            item.get("hypothesis_index", i): item.get("reproduction_confidence", 0.5)
            for i, item in enumerate(pass1_result.hypotheses)
        }

        # Pass 2: alignment from anchors + rules
        pass2_llm = _with_structured_output(self._llm, _Pass2Out)
        pass2_prompt = _build_prompt(hypotheses, shards, anchors, active_rules, context, pass_num=2)
        pass2_result: _Pass2Out = pass2_llm.invoke(pass2_prompt)
        align_data_by_idx = {
            item.get("hypothesis_index", i): item
            for i, item in enumerate(pass2_result.hypotheses)
        }

        scored = []
        for i, hyp in enumerate(hypotheses):
            repro = float(repro_by_idx.get(i, 0.5))
            align_data = align_data_by_idx.get(i, {})
            align = float(align_data.get("alignment_confidence", 0.6))
            conformance = float(align_data.get("rule_conformance_score", 1.0))
            conflicts = align_data.get("rule_conflict_details") or []
            reasoning = align_data.get("reasoning") or pass1_result.hypotheses[i].get("reasoning", "") if i < len(pass1_result.hypotheses) else ""
            scored.append({
                "hypothesis": hyp,
                "alignment_confidence": max(0.0, min(1.0, align)),
                "reproduction_confidence": max(0.0, min(1.0, repro)),
                "rule_conformance_score": max(0.0, min(1.0, conformance)),
                "rule_conflict_details": conflicts,
                "reasoning": reasoning,
            })

        return {"hypotheses": scored}
