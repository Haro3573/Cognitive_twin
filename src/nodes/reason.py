"""
reason_node factory — Step 3: hypothesis generation via reasoner sub-agent.

Candidates use the replace_reducer (reset each recursion cycle).
historical_hypotheses uses the add reducer (accumulates across cycles so
the reasoner avoids re-generating the same hypotheses).
"""


def make_reason_node(reason_subagent):
    def reason_node(state):
        result = reason_subagent.invoke({
            "context": state["perceived_context"],
            "shards": state["retrieved_shards"],
            "anchors": state["retrieved_anchors"],
            "active_rules": state.get("active_governance_rules") or [],
            "historical_hypotheses": state.get("historical_hypotheses", []),
            "is_off_baseline": state["is_off_baseline"],
            "is_bootstrap": state["is_bootstrap"],
        })
        return {
            "hypotheses": result["candidates"],
            "reasoning_traces": result["traces"],
            "historical_hypotheses": result["candidates"],  # appended via add reducer
        }

    return reason_node
