"""
hard_limit_annotator_node factory — annotates output with sparse-domain flag.

This is NOT a gate: execution always continues to compose_output regardless
of whether a sparse domain is detected. The flag is surfaced in the output
payload for the calling agent to act on.
"""

from src.helpers.sparse_domain import detect_sparse_domain


def make_hard_limit_annotator_node():
    def hard_limit_annotator_node(state):
        sparse_domain = detect_sparse_domain(
            state["perceived_context"],
            state["retrieved_shards"],
            state["retrieved_anchors"],
        )
        return {"sparse_domain_flag": sparse_domain}

    return hard_limit_annotator_node
