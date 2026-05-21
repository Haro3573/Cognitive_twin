"""
Boolean predicates used as guards and routing conditions in the graph.
"""

from typing import Optional


# Bootstrap thresholds: all three must be met to exit bootstrap.
_BOOTSTRAP_MIN_ANCHORS = 5
_BOOTSTRAP_MIN_SHARDS = 20
_BOOTSTRAP_MIN_RULES = 3


def is_bootstrap_state(
    user_id: str,
    anchor_store,
    shard_store,
    governance_store,
) -> bool:
    """
    True if the user lacks enough memory to calibrate confidently.

    OR logic: a single failing threshold keeps the user in bootstrap.
    This is conservative by design — the system only exits bootstrap
    when all three data sources are populated.
    """
    anchors = anchor_store.count_for_user(user_id)
    shards = shard_store.count_for_user(user_id)
    rules = governance_store.count_active_rules(user_id)
    return anchors < _BOOTSTRAP_MIN_ANCHORS or shards < _BOOTSTRAP_MIN_SHARDS or rules < _BOOTSTRAP_MIN_RULES


def has_governance_coverage(context: dict, active_rules: list[dict]) -> bool:
    """
    True when at least one active rule covers the context's domain dimensions.

    Universal rules (empty context_scope) always count as coverage.
    None is filtered from domain_tags before intersection.
    """
    domain_dims = {t for t in context.get("domain_tags", []) if t is not None}
    for rule in active_rules:
        scope = set(rule.get("context_scope") or [])
        if not scope:  # universal rule
            return True
        if scope & domain_dims:  # at least one tag overlaps
            return True
    return False


def rule_influenced_hypothesis(rule: dict, hyp: dict) -> bool:
    """
    True when the hypothesis derivation records that the rule was consulted.

    Derivation structure: hyp["derivation"]["rules"] = list of rule_id strings.
    Missing derivation or missing rules list → False (safe default).
    """
    derivation = hyp.get("derivation") or {}
    rule_ids = derivation.get("rules") or []
    return rule["rule_id"] in rule_ids
