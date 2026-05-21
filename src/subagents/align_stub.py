"""
Stub critic (align) sub-agent.

Input keys:  hypotheses, user_id, context
Output keys: hypotheses (list of ScoredHypothesis dicts)

Each ScoredHypothesis:
  hypothesis               — the original hypothesis dict
  alignment_confidence     — reflective endorsement score (0–1)
  reproduction_confidence  — moment-of-action prediction score (0–1)
  rule_conformance_score   — 0–1, 1.0 = full conformance
  rule_conflict_details    — list of {rule_id, description} conflicts
  reasoning                — free-text explanation

Stub always returns alignment_confidence=0.6, reproduction_confidence=0.5.
This preserves topology test invariants:
  - non-bootstrap: 0.6 >= 0.5  → confidence_router proceeds (no recursion)
  - bootstrap:     capped 0.4 < 0.5 → confidence_router recurses once (subagent cap)

Replaced in Step 5 with a real DeepAgents sub-agent.
"""

_FALLBACK_HYPOTHESIS = {
    "id": "fallback",
    "content": "No hypotheses available",
    "decision_type": "abstention",
    "structured_payload": None,
    "derivation": {},
    "rule_conflicting": False,
    "conflict_details": [],
}


class _AlignSubagentStub:
    def invoke(self, inputs: dict) -> dict:
        hypotheses = inputs.get("hypotheses") or []
        hyp = hypotheses[0] if hypotheses else _FALLBACK_HYPOTHESIS
        return {
            "hypotheses": [
                {
                    "hypothesis": hyp,
                    "alignment_confidence": 0.6,
                    "reproduction_confidence": 0.5,
                    "rule_conformance_score": 1.0,
                    "rule_conflict_details": [],
                    "reasoning": "stub",
                }
            ],
        }


align_subagent = _AlignSubagentStub()
