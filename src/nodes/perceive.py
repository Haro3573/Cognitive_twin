"""
perceive_node factory — Step 1 of the cognitive loop.

Extracts structured context from raw input, detects honesty assertions,
and determines whether the user is in bootstrap state.
"""

from src.helpers.context import extract_context
from src.helpers.honesty import detect_honesty_assertion
from src.helpers.predicates import is_bootstrap_state


def make_perceive_node(stores, llm=None):
    def perceive_node(state):
        context = extract_context(
            state["raw_input"],
            state.get("parent_agent_context"),
            llm=llm,
        )
        honesty_required = detect_honesty_assertion(
            state["raw_input"], context, llm=llm
        )
        bootstrap = is_bootstrap_state(
            state["user_id"], stores.anchors, stores.shards, stores.governance
        )
        return {
            "perceived_context": context,
            "honesty_assertion_required": honesty_required,
            "is_bootstrap": bootstrap,
        }

    return perceive_node
