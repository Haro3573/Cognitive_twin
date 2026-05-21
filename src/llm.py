"""
Centralized LLM configuration for the Cognitive Twin Sub-Agent.

Change the environment variables below to switch providers or models
project-wide without touching any other source file.

Environment variables (set in .env or shell):
  COGNITIVE_LLM_PROVIDER      anthropic | openai | google   (default: anthropic)
  COGNITIVE_LLM_MAIN_MODEL    model name for reasoning/critic
                              (default: claude-sonnet-4-6)
  COGNITIVE_LLM_FAST_MODEL    model name for quick helpers (context, honesty)
                              (default: claude-haiku-4-5-20251001)
  COGNITIVE_LLM_TEMPERATURE   float 0.0–1.0                 (default: 0.0)

Usage:
  from src.llm import get_default_llm

  # In production entry points:
  llm = get_default_llm()                         # main reasoning model
  fast_llm = get_default_llm(role="fast")         # helpers
  short_llm = get_default_llm(role="fast", max_tokens=128)  # honesty check

  # Build the graph without constructing an LLM manually:
  graph = build_graph(stores)   # picks up env config automatically
"""

import os
from typing import Literal, Optional


# Defaults exposed as module constants so callers can inspect them.
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MAIN_MODEL = "claude-sonnet-4-6"
DEFAULT_FAST_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_TEMPERATURE = 0.0


def get_default_llm(
    role: Literal["main", "fast"] = "main",
    *,
    max_tokens: Optional[int] = None,
):
    """
    Returns a configured LangChain chat model for the given role.

    Reads configuration from environment variables at call time, so
    os.environ overrides take effect without restarting the process.

    role="main"  — hypothesis generation and two-pass critic scoring
    role="fast"  — single-pass helpers (context extraction, honesty detection)
    max_tokens   — optional per-call output limit (mapped to the provider's
                   correct keyword: max_tokens for Anthropic/OpenAI,
                   max_output_tokens for Google)
    """
    provider = os.getenv("COGNITIVE_LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    main_model = os.getenv("COGNITIVE_LLM_MAIN_MODEL", DEFAULT_MAIN_MODEL)
    fast_model = os.getenv("COGNITIVE_LLM_FAST_MODEL", DEFAULT_FAST_MODEL)
    temperature = float(os.getenv("COGNITIVE_LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE)))

    model = main_model if role == "main" else fast_model

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs: dict = {"model": model, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatAnthropic(**kwargs)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {"model": model, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {"model": model, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens  # Google uses a different kwarg
        return ChatGoogleGenerativeAI(**kwargs)

    else:
        raise ValueError(
            f"Unsupported COGNITIVE_LLM_PROVIDER={provider!r}. "
            f"Valid options: anthropic, openai, google"
        )


def with_structured_output(llm, schema):
    """
    Provider-aware wrapper for structured output.

    OpenAI's default strict json_schema mode rejects schemas that contain bare
    dict, Optional[dict], or list[dict] fields (the project uses these heavily).
    method="function_calling" uses the legacy path that accepts any schema.
    Anthropic and Google use their own mechanisms and need no extra argument.
    """
    provider = os.getenv("COGNITIVE_LLM_PROVIDER", DEFAULT_PROVIDER).lower()
    if provider == "openai":
        return llm.with_structured_output(schema, method="function_calling")
    return llm.with_structured_output(schema)
