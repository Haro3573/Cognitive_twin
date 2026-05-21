"""
align_node factory — Step 4: dual-confidence scoring via critic sub-agent.

Patch §A2: rule_influenced_hypothesis receives best["hypothesis"] (the raw
hypothesis dict), NOT best (the ScoredHypothesis wrapper).
"""

from src.helpers.predicates import rule_influenced_hypothesis


def make_align_node(align_subagent):
    def align_node(state):
        scored = align_subagent.invoke({
            "hypotheses": state["hypotheses"],
            "user_id": state["user_id"],
            "context": state["perceived_context"],
            "shards": state.get("retrieved_shards") or [],
            "anchors": state.get("retrieved_anchors") or [],
            "active_rules": state.get("active_governance_rules") or [],
        })

        best = max(scored["hypotheses"], key=lambda h: h["alignment_confidence"])
        align_conf = best["alignment_confidence"]
        if state["is_bootstrap"]:
            align_conf = min(align_conf, 0.4)

        active_rules = state.get("active_governance_rules") or []
        # §A2: pass best["hypothesis"], not best (which is the ScoredHypothesis wrapper)
        rule_basis = [
            r["rule_id"]
            for r in active_rules
            if rule_influenced_hypothesis(r, best["hypothesis"])
        ]

        return {
            "alignment_scores": scored,
            "selected_hypothesis": best["hypothesis"],
            "alignment_confidence": align_conf,
            "reproduction_confidence": best["reproduction_confidence"],
            "confidence_divergence": abs(align_conf - best["reproduction_confidence"]),
            "rule_basis": rule_basis,
        }

    return align_node
