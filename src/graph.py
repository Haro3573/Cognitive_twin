"""
Main LangGraph graph for the Cognitive Twin Sub-Agent.

build_graph(stores, llm=None, checkpointer=None) is the single entry point.
All node functions are created as closures over their injected dependencies.
Sub-agent stubs (Step 4) are swapped for real DeepAgents sub-agents in Step 5.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from src.state import CognitiveLoopState
from src.deps import Stores
from src.subagents import (
    recall_subagent as _default_recall_stub,
    reason_subagent as _default_reason_stub,
    align_subagent as _default_align_stub,
    RecallAgent,
    ReasonerAgent,
    CriticAgent,
)
from src.nodes import (
    make_perceive_node,
    make_governance_load_node,
    make_recall_node,
    make_reason_node,
    make_align_node,
    make_hard_limit_annotator_node,
    make_compose_output_node,
    make_meta_learn_node,
)

BASELINE_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# confidence_router — pure routing node, no store dependency
# ---------------------------------------------------------------------------

def confidence_router(state: CognitiveLoopState) -> Command[Literal["recall", "hard_limit_annotator"]]:
    """Routes to recall (self-refine) or proceeds. Increments recursion counter."""
    max_recursion = 1 if state["invocation_mode"] == "subagent" else 3

    if state["alignment_confidence"] < 0.5 and state["recursion_depth"] < max_recursion:
        return Command(
            update={
                "recursion_depth": state["recursion_depth"] + 1,
                "self_refine_reason": (
                    f"alignment_confidence={state['alignment_confidence']:.2f} below 0.5"
                ),
            },
            goto="recall",
        )
    return Command(goto="hard_limit_annotator")


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(stores: Stores, llm=None, subagents=None, checkpointer=None):
    """
    Compile the Cognitive Loop graph.

    subagents: optional tuple (recall_sa, reason_sa, align_sa).
      - Pass explicitly in topology tests to inject stubs and avoid LLM calls.
      - Pass None (default) in production; llm is resolved from src.llm if not given.
    llm: a LangChain chat model. If None and subagents is None, the centralized
      default from src.llm (COGNITIVE_LLM_* env vars) is used automatically.
    """
    if subagents is not None:
        recall_sa, reason_sa, align_sa = subagents
    else:
        if llm is None:
            from src.llm import get_default_llm
            llm = get_default_llm(role="main")
        recall_sa = RecallAgent(stores)
        reason_sa = ReasonerAgent(llm)
        align_sa = CriticAgent(llm)

    graph = StateGraph(CognitiveLoopState)

    for name, fn in [
        ("perceive", make_perceive_node(stores, llm)),
        ("governance_load", make_governance_load_node(stores)),
        ("recall", make_recall_node(recall_sa)),
        ("reason", make_reason_node(reason_sa)),
        ("align", make_align_node(align_sa)),
        ("confidence_router", confidence_router),
        ("hard_limit_annotator", make_hard_limit_annotator_node()),
        ("compose_output", make_compose_output_node(llm)),
        ("meta_learn", make_meta_learn_node(stores)),
    ]:
        graph.add_node(name, fn)

    graph.add_edge(START, "perceive")
    graph.add_edge("perceive", "governance_load")
    graph.add_edge("governance_load", "recall")
    graph.add_edge("recall", "reason")
    graph.add_edge("reason", "align")
    graph.add_edge("align", "confidence_router")
    # No static edges FROM confidence_router — Command handles routing dynamically.
    graph.add_edge("hard_limit_annotator", "compose_output")
    graph.add_edge("compose_output", "meta_learn")
    graph.add_edge("meta_learn", END)

    return graph.compile(checkpointer=checkpointer)


RUNTIME_CONFIG = {"recursion_limit": 25}
