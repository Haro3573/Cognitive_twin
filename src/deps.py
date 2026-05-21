"""
Stores dataclass — groups all storage objects into a single injectable unit.

Passed into build_graph() and node factories so each node closure holds exactly
the stores it needs, without importing globals.
"""

from dataclasses import dataclass

from src.storage.shard_store import ShardStore
from src.storage.anchor_store import AnchorStore
from src.storage.governance_store import GovernanceStore
from src.storage.trace_store import TraceStore
from src.storage.proposal_queue import ProposalQueue
from src.storage.pending_anchor_store import PendingAnchorStore
from src.storage.outcome_store import OutcomeStore
from src.storage.review_store import ReviewStore


@dataclass(frozen=True)
class Stores:
    shards: ShardStore
    anchors: AnchorStore
    governance: GovernanceStore
    traces: TraceStore
    proposals: ProposalQueue
    pending_anchors: PendingAnchorStore
    outcomes: OutcomeStore
    reviews: ReviewStore
