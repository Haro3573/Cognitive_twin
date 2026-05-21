"""Sub-agent interfaces for the Cognitive Twin Sub-Agent."""

from .recall_stub import recall_subagent
from .reason_stub import reason_subagent
from .align_stub import align_subagent

from .baseline import compute_baseline_deviation
from .recall import RecallAgent
from .reasoner import ReasonerAgent
from .critic import CriticAgent

__all__ = [
    # Stubs (used by topology tests via explicit DI)
    "recall_subagent",
    "reason_subagent",
    "align_subagent",
    # Real sub-agent classes
    "RecallAgent",
    "ReasonerAgent",
    "CriticAgent",
    # Utility
    "compute_baseline_deviation",
]
