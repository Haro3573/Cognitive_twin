"""
compose_output_node factory — builds the final DecisionPayload.

alternatives: top-2 non-selected hypotheses from alignment_scores["hypotheses"].
honesty_assertion: rewrites response_text if the user asked sincerely whether
they're talking to an AI.
"""

from src.helpers.honesty import enforce_honesty_assertion


def make_compose_output_node(llm=None):
    def compose_output_node(state):
        decision = state["selected_hypothesis"]

        if state["honesty_assertion_required"]:
            decision = enforce_honesty_assertion(decision, llm=llm)

        decision_id = decision.get("id")
        hypotheses = state["alignment_scores"].get("hypotheses", [])
        alternatives = [
            h["hypothesis"]
            for h in hypotheses
            if h["hypothesis"].get("id") != decision_id
        ][:2]

        payload = {
            "decision": decision,
            "confidences": {
                "alignment": state["alignment_confidence"],
                "reproduction": state["reproduction_confidence"],
                "divergence": state["confidence_divergence"],
            },
            "annotations": {
                "is_off_baseline": state["is_off_baseline"],
                "baseline_deviation": state["baseline_deviation_score"],
                "sparse_domain": state["sparse_domain_flag"],
                "honesty_assertion_enforced": state["honesty_assertion_required"],
                "bootstrap_mode": state["is_bootstrap"],
                "recursion_depth": state["recursion_depth"],
                "self_refine_reason": state.get("self_refine_reason"),
            },
            "alternatives": alternatives,
            "rule_basis": state.get("rule_basis", []),
            "trace_id": state["trace_id"],
        }
        return {"output_payload": payload}

    return compose_output_node
