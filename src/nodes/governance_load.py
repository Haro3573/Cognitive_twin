"""
governance_load_node factory — loads active rules once per request.

Guard: if active_governance_rules is already set (recursion case), returns {}
so the sentinel rules survive the cycle unchanged.
"""


def make_governance_load_node(stores):
    def governance_load_node(state):
        if state.get("active_governance_rules") is not None:
            return {}
        rules = stores.governance.query_active_rules(
            user_id=state["user_id"],
            context=state["perceived_context"],
            min_confidence=0.5,
        )
        return {
            "active_governance_rules": rules,
            "governance_version": stores.governance.current_version(state["user_id"]),
        }

    return governance_load_node
