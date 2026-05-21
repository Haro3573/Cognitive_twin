"""
Stub reasoner sub-agent.

Input keys:  context, shards, anchors, active_rules, historical_hypotheses,
             is_off_baseline, is_bootstrap
Output keys: candidates (list[dict]), traces (list[dict])

Replaced in Step 5 with a real DeepAgents sub-agent that reads retrieved
shards/anchors and governance rules via tools, then generates hypotheses.
"""

_STUB_HYPOTHESIS = {
    "id": "stub-hyp-1",
    "content": "STUB: no hypotheses generated yet",
    "decision_type": "abstention",
    "structured_payload": None,
    "derivation": {"shards": [], "anchors": [], "rules": [], "source": "stub"},
    "rule_conflicting": False,
    "conflict_details": [],
}


class _ReasonSubagentStub:
    def invoke(self, inputs: dict) -> dict:
        return {
            "candidates": [_STUB_HYPOTHESIS],
            "traces": [{"stub": True}],
        }


reason_subagent = _ReasonSubagentStub()
