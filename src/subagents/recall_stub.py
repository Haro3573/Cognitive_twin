"""
Stub recall sub-agent.

Input keys:  user_id, context, is_bootstrap
Output keys: shards, anchors, strategy, baseline_deviation

Replaced in Step 5 with a real DeepAgents sub-agent that uses ShardStore +
AnchorStore retrieval tools and computes baseline deviation via
ShardStore.compute_baseline_deviation.
"""


class _RecallSubagentStub:
    def invoke(self, inputs: dict) -> dict:
        return {
            "shards": [],
            "anchors": [],
            "strategy": "semantic",
            "baseline_deviation": 0.0,
        }


recall_subagent = _RecallSubagentStub()
