"""
RecallAgent: memory retriever sub-agent (spec §7.1).

Pure Python — no LLM call. Uses ShardStore/AnchorStore for retrieval
and compute_baseline_deviation for drift detection.

invoke(inputs) → {shards, anchors, strategy, baseline_deviation}

D10: data is fetched here via store calls, not injected from state.
The result dict is what recall_node reads from state.
"""

from .baseline import compute_baseline_deviation


class RecallAgent:
    """
    Retrieves shards and anchors for a given user+context, then computes
    baseline deviation from the retrieved sample.

    stores: Stores frozen dataclass (stores.shards, stores.anchors)
    """

    def __init__(self, stores) -> None:
        self._stores = stores

    def invoke(self, inputs: dict) -> dict:
        user_id: str = inputs["user_id"]
        context: dict = inputs.get("context") or {}
        is_bootstrap: bool = inputs.get("is_bootstrap", False)

        # Semantic search on shards
        query_text = context.get("summary") or context.get("raw_input") or ""
        domain_tags = context.get("domain_tags") or []
        k_shards = 15 if is_bootstrap else 20
        k_anchors = 5 if is_bootstrap else 10

        if query_text:
            shards = self._stores.shards.search(
                query_text=query_text,
                user_id=user_id,
                k=k_shards,
                domain_tags=domain_tags or None,
            )
            anchors = self._stores.anchors.search(
                query_text=query_text,
                user_id=user_id,
                k=k_anchors,
            )
            strategy = "semantic"
        else:
            shards = self._stores.shards.sample_recent(
                user_id=user_id, days=30, sample_size=k_shards
            )
            anchors = self._stores.anchors.list_for_user(user_id=user_id)[:k_anchors]
            strategy = "recency"

        recent_sample = self._stores.shards.sample_recent(
            user_id=user_id, days=30, sample_size=20
        )
        baseline_deviation = compute_baseline_deviation(
            shards=recent_sample, anchors=anchors
        )

        return {
            "shards": shards,
            "anchors": anchors,
            "strategy": strategy,
            "baseline_deviation": baseline_deviation,
        }
