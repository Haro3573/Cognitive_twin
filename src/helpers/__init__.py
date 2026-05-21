"""Helper functions for the Cognitive Twin Sub-Agent."""

from .trace_utils import new_trace_id, compress_state_for_persistence
from .predicates import is_bootstrap_state, has_governance_coverage, rule_influenced_hypothesis
from .sparse_domain import detect_sparse_domain
from .context import extract_context
from .honesty import detect_honesty_assertion, enforce_honesty_assertion

__all__ = [
    "new_trace_id",
    "compress_state_for_persistence",
    "is_bootstrap_state",
    "has_governance_coverage",
    "rule_influenced_hypothesis",
    "detect_sparse_domain",
    "extract_context",
    "detect_honesty_assertion",
    "enforce_honesty_assertion",
]
