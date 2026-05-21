"""
Honesty assertion detection and enforcement (patch §A10).

Two exported functions:
  detect_honesty_assertion  — should this request trigger an honesty check?
  enforce_honesty_assertion — rewrite a decision hypothesis to be transparent about AI nature.
"""

import re
from typing import Optional
from pydantic import BaseModel

from src.llm import with_structured_output as _with_structured_output


# ---------------------------------------------------------------------------
# Stage-1 regex patterns (verbatim from spec)
# ---------------------------------------------------------------------------
_HONESTY_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(are|am|is)\s+(you|i|this)\s+(an?\s+)?(ai|bot|robot|chatbot|llm|chatgpt|claude)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(human|real person|actual person)\s+or\s+(ai|bot|machine)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btalking\s+to\s+(an?\s+)?(ai|bot|robot|human|person)\b",
        re.IGNORECASE,
    ),
]


class _SincerityCheck(BaseModel):
    is_sincere_inquiry: bool


_SINCERITY_SYSTEM = """\
You are evaluating whether a user message contains a sincere question about \
whether they are talking to an AI, or whether the phrasing is incidental \
(e.g. roleplay, hypothetical, quoting someone else, rhetorical).

Return is_sincere_inquiry = true only if the question is a genuine, first-person \
inquiry about the nature of the system they are talking to."""


def detect_honesty_assertion(
    raw_input: str,
    context: dict,
    *,
    llm=None,
) -> bool:
    """
    Two-stage detection:
      1. Regex: fast pattern match. If no pattern fires, return False immediately.
      2. LLM sincerity check: confirm the match is a genuine AI-identity inquiry.

    Returns True only when both stages agree (sincere question about AI nature).
    """
    if not any(p.search(raw_input) for p in _HONESTY_PATTERNS):
        return False

    # Stage 2: LLM sincerity confirmation
    if llm is None:
        from src.llm import get_default_llm
        llm = get_default_llm(role="fast", max_tokens=128)

    structured = _with_structured_output(llm, _SincerityCheck)
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        result: _SincerityCheck = structured.invoke(
            [
                SystemMessage(content=_SINCERITY_SYSTEM),
                HumanMessage(content=f'User message: "{raw_input}"'),
            ]
        )
        return result.is_sincere_inquiry
    except Exception:
        # On LLM error, conservatively treat the regex hit as sincere.
        return True


# ---------------------------------------------------------------------------
# Honesty enforcement rewrite
# ---------------------------------------------------------------------------


class _RewriteResult(BaseModel):
    rewritten_response: str


_REWRITE_SYSTEM = """\
You are rewriting an AI-generated decision response to be transparent about the \
fact that it comes from an AI system, not a human. The original response should \
be preserved in substance, but the framing should make clear that this is an AI \
perspective, not a human one. Keep the rewrite concise and natural."""


def enforce_honesty_assertion(
    decision: dict,
    *,
    llm=None,
) -> dict:
    """
    Rewrites decision["response_text"] to acknowledge AI nature.
    Returns a new hypothesis dict with the rewritten response_text.
    Only processes hypotheses whose type is "response_text".
    """
    if decision.get("type") != "response_text":
        return decision

    if llm is None:
        from src.llm import get_default_llm
        llm = get_default_llm(role="fast", max_tokens=512)

    structured = _with_structured_output(llm, _RewriteResult)
    original = decision.get("response_text", "")

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        result: _RewriteResult = structured.invoke(
            [
                SystemMessage(content=_REWRITE_SYSTEM),
                HumanMessage(content=f"Original response:\n{original}"),
            ]
        )
        return {**decision, "response_text": result.rewritten_response}
    except Exception:
        # On error, leave decision unmodified rather than silently drop.
        return decision
