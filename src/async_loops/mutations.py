"""
B3 mutation functions for the outcome processor.

These are pure transformers: they take data and return typed structs.
Store writes are the caller's responsibility (outcome_processor.py).

Functions:
  analyze_edit          — classify an accepted-with-edits outcome
  rule_might_be_responsible — check if a rule likely caused the edit
  generate_modified_rule    — produce a ProposedRuleModel from an edit
  extract_rule_pattern_from_rejection — extract a rule from a rejection
  check_accumulators_for_promotion    — synthesise investigate_* into add_rule

Constants:
  DEPRECATION_TRIGGER = 0.6  (>60% contradicting activations → propose deprecation)
  NEW_RULE_ACCUMULATOR_THRESHOLD = 5  (siblings needed for synthesis)
"""

import difflib
import math
from typing import Callable, Optional

from src.storage.models import EditAnalysisModel, ProposedRuleModel, RejectionPatternModel
from src.llm import with_structured_output as _with_structured_output

DEPRECATION_TRIGGER = 0.6
NEW_RULE_ACCUMULATOR_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _char_edit_distance(a: str, b: str) -> float:
    """Approximate character-level edit distance via SequenceMatcher ratio."""
    sm = difflib.SequenceMatcher(None, a, b)
    return (1.0 - sm.ratio()) * max(len(a), len(b), 1)


# ---------------------------------------------------------------------------
# 1. analyze_edit
# ---------------------------------------------------------------------------

_ANALYZE_EDIT_PROMPT = """\
Compare the original decision with the human-edited version and classify the edit.

Original: {original}

Edited: {edited}

Classify the edit. Return a JSON object with:
- substantive (bool): is this a meaningful change beyond cosmetics?
- edit_type: one of "cosmetic", "tone_shift", "directional_change", "factual_correction", "values_change"
- pattern: one short sentence describing what changed
- preserved_elements: list of concepts/commitments preserved
- changed_elements: list of concepts/commitments changed
- confidence: float 0-1 how confident you are in this classification
"""


def analyze_edit(
    original: str,
    edited: str,
    llm=None,
) -> EditAnalysisModel:
    """
    Classifies an accepted-with-edits outcome.

    Fast path: if character edit distance < 5, returns cosmetic/non-substantive.
    LLM path: structured output via EditAnalysisModel schema.
    No-LLM fallback: heuristic classification from character distance alone.
    """
    edit_dist = _char_edit_distance(original, edited)

    if edit_dist < 5.0:
        return EditAnalysisModel(
            substantive=False,
            edit_type="cosmetic",
            pattern="trivial character-level change",
            preserved_elements=[],
            changed_elements=[],
            confidence=1.0,
        )

    if llm is None:
        # Heuristic fallback: classify by magnitude without LLM
        edit_type = "tone_shift" if edit_dist < 50 else "directional_change"
        return EditAnalysisModel(
            substantive=True,
            edit_type=edit_type,
            pattern=f"edit of approximately {int(edit_dist)} characters",
            preserved_elements=[],
            changed_elements=[],
            confidence=0.3,
        )

    prompt = _ANALYZE_EDIT_PROMPT.format(original=original, edited=edited)
    structured_llm = _with_structured_output(llm, EditAnalysisModel)
    return structured_llm.invoke(prompt)


# ---------------------------------------------------------------------------
# 2. rule_might_be_responsible
# ---------------------------------------------------------------------------

def rule_might_be_responsible(
    rule_id: str,
    edit_analysis: EditAnalysisModel,
    trace_rule_basis: list[str],
    governance_store,
    embed_fn: Callable[[list[str]], list[list[float]]],
) -> bool:
    """
    Returns True if rule_id was active in the trace AND its embedding is
    semantically similar (cosine >= 0.5) to the edit's pattern.

    Pure deterministic — no LLM call.
    """
    if rule_id not in trace_rule_basis:
        return False

    rule = governance_store.get(rule_id)
    if not rule or not rule.get("embedding"):
        return False

    pattern_embedding = embed_fn([edit_analysis.pattern])[0]
    return _cosine(rule["embedding"], pattern_embedding) >= 0.5


# ---------------------------------------------------------------------------
# 3. generate_modified_rule
# ---------------------------------------------------------------------------

_MODIFY_RULE_PROMPT = """\
A governance rule was applied to a decision that the user subsequently edited.
Suggest a modified version of the rule that better captures the user's intent.

Current rule: {statement}
Rule class: {rule_class}
Context scope: {context_scope}

Edit analysis:
- Type: {edit_type}
- Pattern: {pattern}
- Changed elements: {changed_elements}
- Preserved elements: {preserved_elements}

Return a JSON object with:
- statement: the revised rule text (0.5x–1.5x the length of the original)
- context_scope: list of domain tags (must be a subset of the current scope)
- rule_class: same or more specific (value/preference/constraint/heuristic)
- confidence_adjustment: float in [-0.5, 0.0] — how much to lower confidence
- rationale: one sentence explaining the change
- modification_type: one of "narrow_scope", "soften_statement", "add_exception", "reclassify"
"""

_STATEMENT_LEN_MIN_RATIO = 0.5
_STATEMENT_LEN_MAX_RATIO = 1.5


def generate_modified_rule(
    current_rule: dict,
    edit_analysis: EditAnalysisModel,
    llm=None,
) -> Optional[ProposedRuleModel]:
    """
    Produces a ProposedRuleModel representing a modification to current_rule.

    Returns None if LLM is unavailable or post-validation fails.
    Post-LLM validation:
      - context_scope must be a subset of current_rule["context_scope"]
      - statement length must be 0.5x–1.5x the original
      - confidence_adjustment must be in [-0.5, 0.0]
    """
    if llm is None:
        return None

    prompt = _MODIFY_RULE_PROMPT.format(
        statement=current_rule.get("statement", ""),
        rule_class=current_rule.get("rule_class", "preference"),
        context_scope=current_rule.get("context_scope", []),
        edit_type=edit_analysis.edit_type,
        pattern=edit_analysis.pattern,
        changed_elements=edit_analysis.changed_elements,
        preserved_elements=edit_analysis.preserved_elements,
    )

    structured_llm = _with_structured_output(llm, ProposedRuleModel)
    try:
        proposal = structured_llm.invoke(prompt)
    except Exception:
        return None

    # --- Post-LLM validation ---
    # narrow_scope must be a subset of the original scope (spec §B3).
    # An empty original scope is universal; only empty new scope is a valid subset.
    orig_scope = set(current_rule.get("context_scope") or [])
    new_scope = set(proposal.context_scope or [])
    if not new_scope.issubset(orig_scope):
        return None

    orig_len = len(current_rule.get("statement", ""))
    new_len = len(proposal.statement)
    if orig_len > 0 and not (
        _STATEMENT_LEN_MIN_RATIO * orig_len <= new_len <= _STATEMENT_LEN_MAX_RATIO * orig_len
    ):
        return None

    if not (-0.5 <= proposal.confidence_adjustment <= 0.0):
        # clamp to allowed range for edits rather than discarding
        proposal = proposal.model_copy(
            update={"confidence_adjustment": max(-0.5, min(0.0, proposal.confidence_adjustment))}
        )

    return proposal


# ---------------------------------------------------------------------------
# 4. extract_rule_pattern_from_rejection
# ---------------------------------------------------------------------------

_EXTRACT_REJECTION_PROMPT = """\
A decision was rejected by the user. Identify whether the rejection reveals
a consistent governance rule or preference.

Rejection reason: {rejection_reason}

Decision that was rejected: {decision_content}
Rule basis used: {rule_basis}
Domain context: {domain_tags}

Return a JSON object with:
- pattern_detected (bool): is there a clear, learnable pattern here?
- proposed_rule (object or null): if pattern_detected, a rule with:
    - statement: the rule text
    - context_scope: list of domain tags this applies to
    - rule_class: "value" | "preference" | "constraint" | "heuristic"
    - confidence_adjustment: float in [0.0, 0.5] — initial confidence bump
    - rationale: one sentence explaining why this was rejected
    - modification_type: null (this is a new rule)
- confidence: float 0-1 — how confident you are in the pattern detection

Only return pattern_detected=true if confidence >= 0.6.
"""

_REJECTION_CONFIDENCE_FLOOR = 0.6


def extract_rule_pattern_from_rejection(
    rejection_reason: str,
    trace: dict,
    llm=None,
) -> Optional[ProposedRuleModel]:
    """
    Extracts a proposed governance rule from a rejection outcome.

    Returns None if LLM is unavailable, pattern_detected=False, or
    detection confidence < 0.6 (REJECTION_CONFIDENCE_FLOOR).
    """
    if llm is None:
        return None

    output_payload = trace.get("output_payload") or {}
    decision = output_payload.get("decision") or {}
    decision_content = decision.get("content", "")
    rule_basis = trace.get("rule_basis") or []
    perceived = trace.get("perceived_context") or {}
    domain_tags = perceived.get("domain_tags", [])

    prompt = _EXTRACT_REJECTION_PROMPT.format(
        rejection_reason=rejection_reason,
        decision_content=decision_content,
        rule_basis=rule_basis,
        domain_tags=domain_tags,
    )

    structured_llm = _with_structured_output(llm, RejectionPatternModel)
    try:
        result = structured_llm.invoke(prompt)
    except Exception:
        return None

    if not result.pattern_detected or result.confidence < _REJECTION_CONFIDENCE_FLOOR:
        return None

    return result.proposed_rule


# ---------------------------------------------------------------------------
# 5. check_accumulators_for_promotion
# ---------------------------------------------------------------------------

_SYNTHESIZE_RULE_PROMPT = """\
Multiple decision events show a consistent pattern that suggests a new governance rule.
Synthesise these observations into a single, clear rule.

Observations:
{observations}

Return a JSON object with:
- statement: a single clear rule statement
- context_scope: list of domain tags this applies to
- rule_class: "value" | "preference" | "constraint" | "heuristic"
- confidence_adjustment: float in [0.0, 0.5] — starting confidence
- rationale: one sentence summarising the evidence
- modification_type: null
"""


def check_accumulators_for_promotion(
    user_id: str,
    proposal_queue,
    llm=None,
) -> None:
    """
    Checks investigate_new_rule accumulators. When >= NEW_RULE_ACCUMULATOR_THRESHOLD
    siblings share the same context_signature, synthesises them into an add_rule proposal.

    Mutates proposal_queue in place: adds an add_rule proposal and marks
    the contributing accumulators as superseded_by_add_rule_proposal.
    """
    accumulators = proposal_queue.list(user_id, p_type="investigate_new_rule")
    if not accumulators:
        return

    # Group by context_signature
    from itertools import groupby
    def sig_key(p):
        return tuple(sorted(p.get("context_signature") or []))

    by_sig: dict[tuple, list] = {}
    for acc in accumulators:
        key = sig_key(acc)
        by_sig.setdefault(key, []).append(acc)

    for sig, siblings in by_sig.items():
        if len(siblings) < NEW_RULE_ACCUMULATOR_THRESHOLD:
            continue

        if llm is None:
            # Without LLM, can't synthesise — skip
            continue

        observations = "\n".join(
            f"- {s.get('rationale', '')}" for s in siblings
        )
        prompt = _SYNTHESIZE_RULE_PROMPT.format(observations=observations)
        structured_llm = _with_structured_output(llm, ProposedRuleModel)
        try:
            synthesised = structured_llm.invoke(prompt)
        except Exception:
            continue

        # Add the synthesised add_rule proposal
        proposal_queue.add(user_id, {
            "type": "add_rule",
            "target_rule_id": None,
            "proposed_rule": synthesised.model_dump(),
            "rationale": synthesised.rationale,
            "supporting_traces": [
                t for s in siblings
                for t in s.get("supporting_traces", [])
            ],
            "context": {"domain_tags": list(sig)},
            "weight": 1.0,
        })

        # Mark contributing accumulators as superseded
        for sibling in siblings:
            sibling["status"] = "superseded_by_add_rule_proposal"
            proposal_queue.update(sibling)
