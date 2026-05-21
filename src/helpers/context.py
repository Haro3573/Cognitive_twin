"""
Context extraction: LLM-assisted structured parsing of raw user input.
"""

from typing import Optional
from pydantic import BaseModel, Field

from src.llm import with_structured_output as _with_structured_output


class ExtractedContext(BaseModel):
    domain_tags: list[str] = Field(default_factory=list)
    situation_type: Optional[str] = None
    emotional_valence: Optional[str] = None
    time_pressure: Optional[str] = None
    stakes: Optional[str] = None
    key_entities: list[str] = Field(default_factory=list)


_EXTRACT_SYSTEM = """\
You are a context extractor for a decision-modeling system. Given raw user input \
and optional parent context, extract structured context fields.

domain_tags: 1-5 concise domain labels (e.g. "work", "finance", "health").
situation_type: a short phrase describing the situation type (e.g. "career decision").
emotional_valence: "positive", "negative", "mixed", or "neutral".
time_pressure: "high", "medium", "low", or null if not evident.
stakes: "high", "medium", "low", or null if not evident.
key_entities: up to 5 named people, orgs, or objects central to the situation.

Return only the structured fields. Do not include explanations."""


def _build_extract_prompt(raw_input: str, parent_context: Optional[dict]) -> str:
    lines = [f"Raw input:\n{raw_input}"]
    if parent_context:
        lines.append(f"\nParent context hint:\n{parent_context}")
    return "\n".join(lines)


def _default_context(raw_input: str) -> dict:
    """Fallback when LLM call fails — returns a minimal valid context dict."""
    return {
        "domain_tags": [],
        "situation_type": None,
        "emotional_valence": "neutral",
        "time_pressure": None,
        "stakes": None,
        "key_entities": [],
    }


def extract_context(
    raw_input: str,
    parent_context: Optional[dict] = None,
    *,
    llm=None,
) -> dict:
    """
    Calls the LLM with structured output to extract context from raw_input.

    llm: a LangChain chat model. If None, uses the centralized default (src.llm).
    On any LLM error, returns _default_context (safe degradation).
    """
    if llm is None:
        from src.llm import get_default_llm
        llm = get_default_llm(role="fast", max_tokens=512)

    structured = _with_structured_output(llm, ExtractedContext)
    prompt = _build_extract_prompt(raw_input, parent_context)

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        result: ExtractedContext = structured.invoke(
            [SystemMessage(content=_EXTRACT_SYSTEM), HumanMessage(content=prompt)]
        )
        return result.model_dump()
    except Exception:
        return _default_context(raw_input)
