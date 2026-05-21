"""
recall_node factory — Step 2: dual-track retrieval via recall sub-agent.
"""

_BASELINE_THRESHOLD = 0.4


def make_recall_node(recall_subagent):
    def recall_node(state):
        result = recall_subagent.invoke({
            "user_id": state["user_id"],
            "context": state["perceived_context"],
            "is_bootstrap": state["is_bootstrap"],
        })
        return {
            "retrieved_shards": result["shards"],
            "retrieved_anchors": result["anchors"],
            "retrieval_strategy": result["strategy"],
            "is_off_baseline": result["baseline_deviation"] > _BASELINE_THRESHOLD,
            "baseline_deviation_score": result["baseline_deviation"],
        }

    return recall_node
