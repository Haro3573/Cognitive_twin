"""
Pydantic models for storage boundary validation.

TypedDicts (src/state.py) are for in-process LangGraph state.
These models validate data at SQLite/Chroma read/write boundaries.
"""

from __future__ import annotations
from typing import Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ShardModel(BaseModel):
    shard_id: str
    user_id: str
    context: dict
    content: str
    compression_level: int = 0
    created_at: datetime
    last_activated_at: datetime
    activation_count: int = 0
    decay_score: float = 0.0
    domain_tags: list[str] = []
    embedding: list[float] = []


class AnchorModel(BaseModel):
    anchor_id: str
    user_id: str
    statement: str
    structured_form: Optional[dict] = None
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_shard_ids: list[str] = []
    contradicting_shard_ids: list[str] = []
    context_scope: list[str] = []
    established_at: datetime
    last_reinforced_at: datetime
    last_user_confirmed_at: Optional[datetime] = None
    embedding: list[float] = []


class PendingAnchorModel(BaseModel):
    anchor_id: str
    user_id: str
    statement: str
    structured_form: Optional[dict] = None
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_shard_ids: list[str] = []
    contradicting_shard_ids: list[str] = []
    context_scope: list[str] = []
    established_at: datetime
    last_reinforced_at: datetime
    last_user_confirmed_at: Optional[datetime] = None
    pending_confirmation_at: datetime
    seeded_from: str = "system"
    embedding: list[float] = []


class GovernanceRuleModel(BaseModel):
    rule_id: str
    user_id: str
    version: int = 1
    statement: str
    structured_form: Optional[dict] = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int = 0
    context_scope: list[str] = []
    supporting_traces: list[str] = []
    contradicting_traces: list[str] = []
    activated_at: datetime
    supersedes: Optional[str] = None
    rule_class: Literal["value", "preference", "constraint", "heuristic"]


class HypothesisModel(BaseModel):
    id: str
    content: str
    decision_type: Literal["response_text", "structured_action", "recommendation", "abstention"]
    structured_payload: Optional[dict] = None
    derivation: dict = {}
    rule_conflicting: bool = False
    conflict_details: list[str] = []


class ScoredHypothesisModel(BaseModel):
    hypothesis: HypothesisModel
    alignment_confidence: float = Field(ge=0.0, le=1.0)
    reproduction_confidence: float = Field(ge=0.0, le=1.0)
    rule_conformance_score: float = Field(ge=0.0, le=1.0)
    rule_conflict_details: list[dict] = []
    reasoning: str = ""


class ProposedRuleModel(BaseModel):
    statement: str
    context_scope: list[str] = []
    rule_class: Literal["value", "preference", "constraint", "heuristic"]
    confidence_adjustment: float = Field(ge=-0.5, le=0.5)
    rationale: str
    modification_type: Optional[
        Literal["narrow_scope", "soften_statement", "add_exception", "reclassify"]
    ] = None


class ProposalModel(BaseModel):
    proposal_id: str
    user_id: str
    type: Literal[
        "modify_rule",
        "add_rule",
        "deprecate_rule",
        "adjust_weight",
        "investigate_divergence",
        "investigate_rule",
        "investigate_new_rule",
    ]
    target_rule_id: Optional[str] = None
    proposed_rule: Optional[dict] = None
    rationale: str
    evidence_count: int = 1
    weight: float = Field(ge=0.0, le=1.0, default=1.0)
    promotion_threshold: int
    supporting_traces: list[str] = []
    first_observed: datetime
    last_reinforced: datetime
    context_signature: list[str] = []
    delta: Optional[float] = None
    status: Literal[
        "active", "promoted", "discarded", "superseded_by_add_rule_proposal"
    ] = "active"


class DecisionPayloadModel(BaseModel):
    decision: dict
    confidences: dict
    annotations: dict
    alternatives: list[dict] = []
    rule_basis: list[str] = []
    trace_id: str


class EditAnalysisModel(BaseModel):
    substantive: bool
    edit_type: Literal[
        "cosmetic", "tone_shift", "directional_change", "factual_correction", "values_change"
    ]
    pattern: str
    preserved_elements: list[str] = []
    changed_elements: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)


class RejectionPatternModel(BaseModel):
    pattern_detected: bool
    proposed_rule: Optional[ProposedRuleModel] = None
    confidence: float = Field(ge=0.0, le=1.0)
