"""
Sparse-domain detection: identifies high-stakes domains where the agent
should annotate its output with a flag rather than silently proceeding.

v1 is rule-based only — no LLM call (see DECISIONS.md D7).
"""

from typing import Optional, Literal

SparseLabel = Literal["health", "legal", "financial", "close_relationships"]

# Keyword sets per domain. All matching is lowercase.
_DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "health": {
        "health", "medical", "medicine", "doctor", "hospital", "symptom",
        "diagnosis", "treatment", "therapy", "prescription", "drug", "illness",
        "disease", "surgery", "mental health", "depression", "anxiety",
        "medication", "clinical", "patient",
    },
    "legal": {
        "legal", "law", "lawyer", "attorney", "court", "lawsuit", "contract",
        "liability", "rights", "criminal", "civil", "regulation", "compliance",
        "statute", "ordinance", "judge", "jury", "trial", "settlement", "sue",
    },
    "financial": {
        "financial", "finance", "money", "investment", "tax", "debt", "loan",
        "mortgage", "bankruptcy", "credit", "insurance", "retirement",
        "pension", "stock", "market", "trading", "savings", "budget",
        "accounting", "audit",
    },
    "close_relationships": {
        "relationship", "marriage", "divorce", "partner", "spouse", "family",
        "parent", "child", "custody", "breakup", "affair", "separation",
        "dating", "intimate", "romantic",
    },
}


def detect_sparse_domain(
    context: dict,
    shards: list[dict],
    anchors: list[dict],
) -> Optional[SparseLabel]:
    """
    Returns the first matching sparse domain for the context, or None.

    Decision order: context domain_tags checked first (fast path); if no
    tag matches, fall back to keyword scan of context situation_type and
    shard/anchor text (slower path). Returns first match in domain priority
    order (health > legal > financial > close_relationships).

    v1 limitation: no LLM disambiguation — false positives are possible
    when keywords appear in benign contexts. This is acceptable because
    the flag is non-blocking (annotation only, not a gate).
    """
    # Fast path: explicit domain_tags on the context
    tags = {t.lower() for t in (context.get("domain_tags") or []) if t}
    for domain in ("health", "legal", "financial", "close_relationships"):
        if domain in tags or domain.replace("_", " ") in tags:
            return domain  # type: ignore[return-value]

    # Slow path: keyword scan across situation_type + shard/anchor summaries
    corpus_parts: list[str] = []
    if context.get("situation_type"):
        corpus_parts.append(context["situation_type"].lower())

    for shard in shards:
        if shard.get("content"):
            corpus_parts.append(shard["content"].lower())
    for anchor in anchors:
        if anchor.get("summary"):
            corpus_parts.append(anchor["summary"].lower())

    corpus = " ".join(corpus_parts)
    for domain in ("health", "legal", "financial", "close_relationships"):
        for kw in _DOMAIN_KEYWORDS[domain]:
            if kw in corpus:
                return domain  # type: ignore[return-value]

    return None
