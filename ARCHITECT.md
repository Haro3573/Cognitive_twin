# Cognitive Twin Sub-Agent — `src/` Reference

> **Target audience:** Engineers integrating or extending this system. You should know Python and basic LangChain concepts. This document explains the *specific* architecture of this system, not general LangGraph or AI agent concepts.

---

## Table of Contents

1. [Architectural Philosophy](#1-architectural-philosophy)
2. [Directory Structure & Component Roles](#2-directory-structure--component-roles)
3. [Step-by-Step Data Flow: The Single Loop](#3-step-by-step-data-flow-the-single-loop)
4. [How the Agent Learns: The Double Loop](#4-how-the-agent-learns-the-double-loop)
5. [Storage Layer Deep-Dive](#5-storage-layer-deep-dive)
6. [Eval Harness](#6-eval-harness)
7. [Key Design Decisions](#7-key-design-decisions)

---

## 1. Architectural Philosophy

### What This System Is (and Isn't)

This is **not** a standalone chat agent. It is a pluggable **"Brain/Conscience" module** — a sub-agent designed to be attached to a Parent Agent via three LangChain `StructuredTool`s exposed in `src/tools.py`. The Parent Agent decides *when* to ask; this system decides *what* the user would choose and *how confident* it is.

```
┌─────────────────────────────────────────────────────────────┐
│                    Parent Agent                             │
│  (task agent, workflow agent, etc.)                         │
│                                                             │
│   parent_agent.tools = make_tools(stores, compiled_graph)  │
└──────────────┬──────────────┬──────────────────────────────┘
               │              │
    decide_as_user      report_decision_outcome
               │              │
               ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│            Cognitive Twin Sub-Agent (this repo)             │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  Single Loop │    │ Double Loop  │    │ Eval Harness  │  │
│  │  (LangGraph) │    │ (async_loops)│    │   (src/eval)  │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Storage Layer                      │   │
│  │   SQLite (authoritative) + ChromaDB (search index)  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### The Double-Loop Architecture

The system operates on two interleaved loops with very different characteristics:

| | **Single Loop (Request Path)** | **Double Loop (Learning Path)** |
|---|---|---|
| **Trigger** | `decide_as_user(...)` tool call | `report_decision_outcome(...)` followed by async job |
| **Mode** | Synchronous, blocking | Asynchronous, background |
| **I/O profile** | Read-heavy | Write-heavy |
| **Latency** | Low (LLM calls + DB reads) | Not user-facing; runs in background |
| **Entry point** | `src/graph.py` → `build_graph()` | `src/async_loops/*.py` |
| **Output** | `DecisionPayload` (JSON) | Updated `GovernanceStore`, new shards, proposals |

**Key invariant:** The request path **never writes to `GovernanceStore` directly.** All governance mutations are staged in `ProposalQueue` and promoted asynchronously by `promotion_engine.py`. This prevents transient feedback from corrupting governance in real time.

### Closure-based Dependency Injection

Every node in the graph is created by a factory function. The factory closes over injected dependencies (LLM instances, store handles) so the node itself is a pure function with no globals:

```python
# src/graph.py — each node is a closure
graph.add_node("meta_learn", make_meta_learn_node(stores))
graph.add_node("reason",     make_reason_node(reason_sa))
graph.add_node("perceive",   make_perceive_node(stores, llm))
```

This pattern enables:
- **Topology testing:** inject stubs (`recall_stub`, `reason_stub`, `align_stub`) to test graph routing without any LLM calls
- **Clean upgrade path:** swap a node's dependency (e.g., replace `ReasonerAgent` with a tool-calling DeepAgents variant) without touching the graph wiring

---

## 2. Directory Structure & Component Roles

```
src/
├── tools.py              ← API boundary (the only entry point for the Parent Agent)
├── graph.py              ← Wiring diagram (LangGraph StateGraph)
├── state.py              ← CognitiveLoopState (the shared "bucket")
├── deps.py               ← Stores dataclass (dependency injection container)
├── eval.py               ← Metrics dashboard (read-only, no persistence)
├── llm.py                ← Centralized LLM config: get_default_llm() + with_structured_output() wrapper
│
├── nodes/                ← Pipeline steps (pure function factories)
│   ├── perceive.py       ←   Step 1: context extraction + honesty detection
│   ├── governance_load.py←   Step 1b: load active rules into state
│   ├── recall.py         ←   Step 2: invoke RecallAgent, detect baseline drift
│   ├── reason.py         ←   Step 3: invoke ReasonerAgent, get hypotheses
│   ├── align.py          ←   Step 4: invoke CriticAgent, score hypotheses
│   ├── hard_limit_annotator.py ← Step 4b: sparse domain flag (no gating)
│   ├── compose_output.py ←   Step 5: build DecisionPayload
│   └── meta_learn.py     ←   Step 6: light proposals + trace persistence
│
├── subagents/            ← The actual "brains" (LLM logic is isolated here)
│   ├── recall.py         ←   RecallAgent: pure Python, no LLM
│   ├── reasoner.py       ←   ReasonerAgent: structured LLM output, 3-5 hypotheses
│   ├── critic.py         ←   CriticAgent: two-pass dual-confidence scoring
│   ├── baseline.py       ←   compute_baseline_deviation (two-component algorithm)
│   ├── recall_stub.py    ←   Deterministic stub (topology tests)
│   ├── reason_stub.py    ←   Deterministic stub (topology tests)
│   └── align_stub.py     ←   Deterministic stub (topology tests)
│
├── helpers/              ← Pure functions, no I/O
│   ├── context.py        ←   extract_context, classify situation
│   ├── honesty.py        ←   enforce_honesty_assertion
│   ├── predicates.py     ←   is_bootstrap_state, has_governance_coverage
│   └── trace_utils.py    ←   compress_state_for_persistence
│
├── storage/              ← Dual-write stores (SQLite authoritative, Chroma index)
│   ├── db.py             ←   Schema DDL + init_schema()
│   ├── shard_store.py    ←   Episodic memory (ShardStore)
│   ├── anchor_store.py   ←   Identity anchors (AnchorStore)
│   ├── governance_store.py ← Append-only rule store (GovernanceStore)
│   ├── proposal_queue.py ←   Staging area for governance mutations (ProposalQueue)
│   ├── trace_store.py    ←   Compressed state persistence (TraceStore)
│   ├── outcome_store.py  ←   Reported decision outcomes (OutcomeStore)
│   ├── review_store.py   ←   User review queue (ReviewStore)
│   ├── pending_anchor_store.py ← Staged anchor candidates
│   └── models.py         ←   Pydantic validation models for storage boundaries
│
└── async_loops/          ← Five background workers
    ├── outcome_processor.py ← Converts outcomes → shards + proposals
    ├── promotion_engine.py  ← Promotes proposals → GovernanceStore
    ├── reversal_reviewer.py ← Surfaces rule misfires for user review
    ├── memory_decay.py      ← Ages + compresses stale shards
    ├── anchor_consolidation.py ← Reviews dormant/contradicted anchors
    └── mutations.py         ← LLM helpers: analyze_edit, generate_modified_rule, etc.
```

### `src/tools.py` — The API Boundary

The Parent Agent never touches `graph.py`, `nodes/`, or `storage/` directly. Everything goes through three `StructuredTool`s:

| Tool | Purpose | Outcome |
|------|---------|---------|
| `decide_as_user` | Run the full single-loop graph for a situation | Returns `DecisionPayload` |
| `seed_user_data` | Onboard historical decisions, preferences, values | Creates shards, rules, staged anchors |
| `report_decision_outcome` | Feed back what the Parent Agent did with the decision | Writes to `OutcomeStore`; triggers double loop |

`make_tools(stores, compiled_graph)` is the factory. The compiled graph is passed in (built once by the caller, not rebuilt per tool call — see Decision D13).

### `src/state.py` — The Shared Bucket

`CognitiveLoopState` is a LangGraph `TypedDict` that all nodes read from and write partial updates to. Key field groups:

```python
class CognitiveLoopState(TypedDict):
    # Identity
    user_id: str; trace_id: str; is_bootstrap: bool

    # Step 1 — Perceive
    raw_input: str; perceived_context: dict; honesty_assertion_required: bool
    active_governance_rules: Optional[list[dict]]

    # Step 2 — Recall
    retrieved_shards: list[dict]; retrieved_anchors: list[dict]
    is_off_baseline: bool; baseline_deviation_score: float

    # Step 3 — Reason (replace_reducer: resets each recursion cycle)
    hypotheses: Annotated[list[dict], replace_reducer]

    # Step 4 — Align
    alignment_confidence: float; reproduction_confidence: float
    confidence_divergence: float; selected_hypothesis: Optional[dict]

    # Step 5 — Output
    output_payload: Optional[dict]

    # Step 6 — Meta-learn
    proposed_governance_updates: list[dict]
    recursion_depth: int  # incremented by confidence_router on self-refine
```

Note the `replace_reducer` on `hypotheses`: unlike LangGraph's default list append, this *replaces* the list each recursion cycle so the reasoner gets a clean slate and avoids re-scoring stale candidates from a previous pass.

### `src/subagents/` — The Brains

**`RecallAgent`** — No LLM. Pure Python retrieval:
- Semantic search via `ShardStore.search()` + `AnchorStore.search()` (ChromaDB under the hood)
- Calls `compute_baseline_deviation()` from `baseline.py`
- Bootstrap mode: uses `sample_recent()` instead of semantic search (no reliable query vector when data is sparse)

**`ReasonerAgent`** — LLM with structured output via `src.llm.with_structured_output(llm, ReasonerOut)`:
- Generates 3-5 `HypothesisOut` candidates, each with `decision_type`, `derivation` (which shards/anchors/rules support it), and `rule_conflicting`
- Prompt includes: situation summary, top-10 shards, top-10 anchors, all active rules, bootstrap/off-baseline flags, self-refine reason (if recursing)
- Accumulates `historical_hypotheses` across recursion cycles via `add` reducer so it avoids regenerating the same candidates

> **`src.llm.with_structured_output(llm, schema)`** is a provider-aware wrapper used by every structured LLM call in the project (ReasonerAgent, CriticAgent, helpers/context, helpers/honesty, all mutations functions). When `COGNITIVE_LLM_PROVIDER=openai`, it injects `method="function_calling"` to bypass OpenAI's strict JSON schema validator, which rejects the bare `dict`, `Optional[dict]`, and `list[dict]` fields the project schemas use. Anthropic and Google are passed through unchanged. **Never call `llm.with_structured_output()` directly in new code — always use this wrapper.**

**`CriticAgent`** — Two-pass scoring (load-bearing; do not collapse into one pass):

```
Pass 1: shards only  →  reproduction_confidence
        (What would this user ACTUALLY DO in the moment?)

Pass 2: anchors + rules  →  alignment_confidence + rule_conformance_score
        (Would the user REFLECTIVELY ENDORSE this decision?)
```

The two scores are deliberately separate. High reproduction + low alignment = behavioral prediction mode. High alignment + low reproduction = values-based mode. The `confidence_divergence = |alignment - reproduction|` is tracked by the eval harness as a calibration signal.

**Stub files** (`recall_stub.py`, `reason_stub.py`, `align_stub.py`): deterministic callables that return fixed valid responses. Used in `test_topology.py` to test graph routing and the self-refine cycle without touching LLMs or real storage.

---

## 3. Step-by-Step Data Flow: The Single Loop

**Scenario:** Parent Agent calls `decide_as_user(user_id="alice", situation="Alice received a rude email from a client", parent_goal="...", parent_agent_id="...")`

### Graph Topology

```
START
  │
  ▼
perceive ──────► governance_load
                       │
                       ▼
             ┌──── recall ◄──────────────────────────┐
             │       │                               │
             │       ▼                          self-refine
             │     reason                     (recursion_depth
             │       │                          < max, = 3)
             │       ▼                               │
             │     align                             │
             │       │                               │
             │       ▼                               │
             │  confidence_router ── align < 0.5? ───┘
             │       │
             │  align >= 0.5 (or max recursion reached)
             │       │
             │       ▼
             └► hard_limit_annotator
                       │
                       ▼
               compose_output
                       │
                       ▼
                 meta_learn ──► END
```

### Step-by-Step Trace

**1. `tools.py` — State initialization**

`_build_initial_state()` constructs the full `CognitiveLoopState` with all fields zeroed/empty. The `trace_id` is a fresh `uuid4().hex`. `trace_persist_required=True` tells `meta_learn` to save this trace to `TraceStore`.

```python
state = {
    "user_id": "alice",
    "trace_id": "a1b2c3...",
    "raw_input": "Alice received a rude email from a client",
    "recursion_depth": 0,
    "is_bootstrap": False,  # set by perceive
    # ... all other fields at zero/None/[]
}
```

**2. `perceive` — Context extraction**

Calls `extract_context(raw_input, llm)` from `src/helpers/context.py`. Populates `perceived_context`:

```json
{
  "summary": "Client sent a rude email to Alice",
  "domain_tags": ["professional", "interpersonal"],
  "situation_type": "conflict_response",
  "entities": ["client"],
  "emotional_valence": "negative"
}
```

Also detects `honesty_assertion_required` (does the situation involve the user sincerely asking if they're talking to an AI?).

**3. `governance_load` — Rule loading**

Reads `GovernanceStore.query_active_rules(user_id, context)`. Rules are context-filtered:
- A rule with `context_scope=["professional"]` is included because `"professional" ∈ perceived_context["domain_tags"]`
- A rule with `context_scope=[]` is universal and always included

Sets `is_bootstrap` via `is_bootstrap_state()` predicate (few shards + few anchors + few rules → bootstrap mode).

**4. `recall` — Dual-track retrieval**

`RecallAgent.invoke({"user_id", "context", "is_bootstrap"})`:
- If bootstrap: `ShardStore.sample_recent()` with recency weighting (no semantic query since the index is sparse)
- Otherwise: `ShardStore.search(query_text=context["summary"], user_id=..., k=10)` + `AnchorStore.search(...)` via ChromaDB

Calls `compute_baseline_deviation(context, shards)` — two-component score:
1. **Domain novelty:** how different are the domain_tags from Alice's historical average?
2. **Shard semantic distance:** cosine distance between query embedding and recent shard centroid

Both components are max-aggregated. If `baseline_deviation > 0.4` → `is_off_baseline = True` (weaker learning signal; extra caution note injected into reasoner prompt).

**5. `reason` — Hypothesis generation**

`ReasonerAgent.invoke({shards, anchors, active_rules, context, is_bootstrap, is_off_baseline, historical_hypotheses, self_refine_reason})`:

Returns 3-5 hypotheses, e.g.:

```json
{
  "candidates": [
    {
      "id": "hyp-1",
      "content": "Respond professionally, acknowledge frustration, propose a call",
      "decision_type": "affirmative",
      "derivation": {"shards": ["shard-42", "shard-18"], "rules": ["rule-7"]},
      "rule_conflicting": false
    },
    {
      "id": "hyp-2",
      "content": "Escalate to manager and CC legal",
      "decision_type": "conditional",
      "derivation": {"shards": ["shard-31"], "rules": []},
      "rule_conflicting": true,
      "conflict_details": [{"rule_id": "rule-3", "conflict_type": "prefer_direct_resolution"}]
    }
  ]
}
```

**6. `align` — Two-pass scoring**

`CriticAgent.invoke({hypotheses, shards, anchors, active_rules, context})`:

*Pass 1 (shards only → reproduction):*
```
hyp-1: reproduction_confidence = 0.78  (Alice has done this before per shard-42)
hyp-2: reproduction_confidence = 0.31  (Alice rarely escalates per history)
```

*Pass 2 (anchors + rules → alignment):*
```
hyp-1: alignment_confidence = 0.82  (consistent with anchor "I handle conflict directly")
        rule_conformance_score = 1.0  (no rule conflicts)
hyp-2: alignment_confidence = 0.25  (conflicts with anchor "I trust my own judgment")
        rule_conformance_score = 0.5  (violates rule-3)
```

`align_node` selects the hypothesis with the highest `alignment_confidence` and computes:
```python
selected               = hyp-1
alignment_confidence   = 0.82
reproduction_confidence = 0.78
confidence_divergence   = abs(0.82 - 0.78) = 0.04
```

**7. `confidence_router` — Self-refinement gate**

```python
# src/graph.py
def confidence_router(state) -> Command[Literal["recall", "hard_limit_annotator"]]:
    max_recursion = 1 if state["invocation_mode"] == "subagent" else 3
    if state["alignment_confidence"] < 0.5 and state["recursion_depth"] < max_recursion:
        return Command(
            update={"recursion_depth": state["recursion_depth"] + 1,
                    "self_refine_reason": f"alignment={state['alignment_confidence']:.2f} below 0.5"},
            goto="recall",
        )
    return Command(goto="hard_limit_annotator")
```

With `alignment_confidence = 0.82`, we proceed: `Command(goto="hard_limit_annotator")`.

**8. `hard_limit_annotator`**

Checks if `situation_type` falls into a sparse domain: `health`, `legal`, `financial`, `close_relationships`. If so, sets `sparse_domain_flag`. This is annotation only — **this system does not refuse or switch to advisory mode.** The flag is for the Parent Agent's use.

**9. `compose_output` — Build the payload**

```json
{
  "decision": {
    "id": "hyp-1",
    "content": "Respond professionally, acknowledge frustration, propose a call",
    "decision_type": "affirmative"
  },
  "confidences": {
    "alignment": 0.82,
    "reproduction": 0.78,
    "divergence": 0.04
  },
  "annotations": {
    "is_off_baseline": false,
    "sparse_domain": null,
    "bootstrap_mode": false,
    "recursion_depth": 0
  },
  "alternatives": ["hyp-2", ...],
  "rule_basis": ["rule-7"],
  "trace_id": "a1b2c3..."
}
```

> **Prediction vs. recommendation:** `alternatives` are ordered by `reproduction_confidence`. If the Parent Agent is building a behavioral *prediction* model (what would Alice actually do?), it should pick from `alternatives` using `reproduction_confidence`, not the selected decision's `alignment_confidence` (see spec §15).

**10. `meta_learn` — Light proposals + trace persistence**

Emits lightweight `investigate_*` or `adjust_weight` proposals to `ProposalQueue` based on signal quality, without waiting for outcome feedback:

| Signal condition | Proposal type | Rationale |
|-----------------|--------------|-----------|
| `confidence_divergence > 0.3` | `investigate_divergence` | alignment/reproduction are misaligned |
| `alignment_confidence > 0.85` + specific retrieval strategy | `adjust_weight` | reinforce strategy that produced high-confidence output |
| High reproduction + low rule conformance on any hypothesis | `investigate_rule` | a rule may be misfiring against revealed behavior |
| High confidence on both scores + no governance coverage | `investigate_new_rule` | emerging pattern worth a new rule |

Finally, saves `compress_state_for_persistence(state)` to `TraceStore`. The compressed state strips retrieved shard/anchor lists but keeps `output_payload`, `rule_basis`, `is_off_baseline`, `trace_id` — exactly what the async loops need.

---

## 4. How the Agent Learns: The Double Loop

### Feedback Entry Point

After acting on the decision, the Parent Agent calls:

```python
report_decision_outcome(
    trace_id="a1b2c3...",
    outcome="rejected",
    rejection_reason="Too aggressive, should have been softer"
)
```

This writes one row to `OutcomeStore` (`processed_at=NULL`) and returns immediately with `processing_eta_seconds=300`. `ignored` outcomes are acknowledged but not stored. The async outcome processor picks up unprocessed outcomes on the next scheduled run.

### Outcome Processor (`async_loops/outcome_processor.py`)

```
OutcomeStore.unprocessed(user_id, limit=100)
       │
       └── for each outcome:
               │
               ├── fetch trace from TraceStore
               ├── read is_bootstrap from output_payload.annotations
               │
               ├── "accepted"
               │     └─► create shard from decision content
               │         if not bootstrap: reinforce rule basis in GovernanceStore
               │
               ├── "edited"
               │     └─► analyze_edit(original, edited, llm)
               │           ├── substantive edit?
               │           │     → create shard from EDITED content
               │           │       if not bootstrap: check which rule is responsible
               │           │       → queue modify_rule proposal
               │           └── cosmetic edit?
               │                 → create shard from original content
               │
               └── "rejected"
                     └─► record contradicting evidence on each rule in rule_basis
                         (always, even in bootstrap — shards need bad signal too)
                         if not bootstrap:
                           extract_rule_pattern_from_rejection(llm)
                           → add_rule or deprecate_rule proposal
                           check DEPRECATION_TRIGGER (60% contradiction ratio)
                           → deprecate_rule proposal if triggered
               │
               └── mark_processed(outcome_id)  ← always in finally block (D14)
```

**Bootstrap policy:** Bootstrap outcomes (from `seed_user_data`) create shards but **never** generate rule proposals. This prevents low-confidence seed data from polluting governance before the system has real behavioral evidence.

### Proposal Accumulation

Proposals are not applied immediately. They accumulate in `ProposalQueue` with upsert-merge semantics:

```python
# Two "rejected, rule-7, professional context" outcomes merge into one proposal
merge_key = "deprecate_rule::rule-7::professional|conflict_response"
#            ─────────────  ──────  ────────────────────────────────
#            proposal type  rule_id  context_signature (sorted domain_tags + situation_type)

# evidence_count increments on each merge; status stays "active" until threshold
```

Each proposal type has an adaptive promotion threshold:

| Type | Threshold formula | Typical value |
|------|------------------|---------------|
| `modify_rule` | `BASE * (1 + 2 * rule_confidence)` | 3-9 (harder to modify confident rules) |
| `deprecate_rule` | `BASE * 2 * rule_confidence + 1` | 1-7 (needs strong evidence) |
| `add_rule` | `BASE * 1.5` | ~4-5 |
| `adjust_weight` | `BASE` | 3 |
| `investigate_*` | 1 | 1 (accumulators pass through immediately) |

### Promotion Engine (`async_loops/promotion_engine.py`)

Runs separately; reviews proposals that have crossed their threshold:

```
ProposalQueue.eligible_for_review(user_id)
       │
       └── for each eligible proposal:
               ├── "add_rule"      → GovernanceStore.add_rule(proposed_rule)
               ├── "modify_rule"   → GovernanceStore.supersede_rule(old, new)
               ├── "deprecate_rule"→ GovernanceStore.deprecate_rule(rule_id)
               ├── "adjust_weight" → GovernanceStore.adjust_rule_weight(...)
               └── "investigate_*" → ReviewStore.enqueue("reversal_pattern", ...)
               │
               └── proposal.status = "promoted"; ProposalQueue.update(proposal)
```

`GovernanceStore` uses append-only supersession: old rules are never deleted, just marked `superseded_by = new_rule_id`. This preserves audit history and allows rollback.

### Other Async Loops

| Loop | Trigger condition | Effect |
|------|-----------------|--------|
| `reversal_reviewer` | Rejected/edited outcomes whose `rule_basis` overlaps with a rule at ≥30% contradiction ratio | Enqueues `reversal_pattern` to `ReviewStore` for user confirmation |
| `memory_decay` | Shards inactive ≥ 90 days | Increments `decay_score`; at 0.5 → LLM-compress to 1 sentence; at 0.85 (if already compressed) → delete |
| `anchor_consolidation` | Anchors dormant ≥ 180 days, OR contradicting shards outnumber supporting | Enqueues `dormant_anchor` / `contradicted_anchor` to `ReviewStore`; confirmed anchors refreshed, unconfirmed demoted |

### End-to-End Learning Cycle

```
Parent calls decide_as_user("rude client email")
  │
  └─► CriticAgent scores hypotheses
      rule-7 ("prefer direct response") in rule_basis
  │
  └─► Parent rejects: "too aggressive"
  │
  └─► report_decision_outcome(outcome="rejected", reason="too aggressive")
  │
  └─► OutcomeStore ← new row (processed_at=NULL)
  │
  └─► [async] outcome_processor.process_outcomes()
      │
      ├── rule-7 gets contradicting evidence recorded
      ├── contradiction ratio checked (60% trigger)
      └── extract_rule_pattern_from_rejection(llm)
          → add_rule: "softer approach in conflict contexts"
          → deprecate_rule: rule-7
  │
  └─► ProposalQueue ← deprecate_rule::rule-7 (evidence_count=1)
                    ← add_rule::softer_approach (evidence_count=1)
  │
  └─► [after N more similar rejections, threshold crossed]
      promotion_engine: rule-7 superseded by rule-7b
  │
  └─► GovernanceStore: rule-7 → superseded_by=rule-7b
  │
  └─► Next decide_as_user("rude client email")
      → ReasonerAgent sees rule-7b ("softer conflict response preferred")
      → CriticAgent penalizes "escalate" hypothesis for violating rule-7b
      → Softer response wins
```

---

## 5. Storage Layer Deep-Dive

All stores are dual-write: **SQLite is authoritative**; ChromaDB is the semantic search index. If Chroma and SQLite diverge, SQLite wins. Chroma failures in write paths are silently swallowed (`try/except`) to avoid poisoning the main flow — the next write will re-sync.

### Memory Hierarchy

| Store | What it holds | Change frequency |
|-------|-------------|-----------------|
| `AnchorStore` | Identity-constitutive patterns ("Alice values directness") | Weeks/months |
| `GovernanceStore` | Decision rules with context scope and confidence | Days/weeks |
| `ShardStore` | Episodic decision memories (embeddings for semantic search) | Every request |
| `ProposalQueue` | Staged governance mutations | Every async loop run |
| `TraceStore` | Compressed state snapshots for async processing | Every request |
| `OutcomeStore` | Reported outcomes from Parent Agent | Per feedback |
| `ReviewStore` | Items queued for user confirmation | Per reversal/consolidation loop |

### CQRS Pattern in Storage

The governance write path enforces strict separation:

```
Request Path (READ ONLY for governance):
  GovernanceStore.query_active_rules()    ← never writes
  ShardStore.search()                     ← never writes (except activation counter)
  AnchorStore.search()                    ← never writes

Async Path (WRITES governance):
  GovernanceStore.add_rule() / supersede_rule() / deprecate_rule()
  ShardStore.add() / compress() / delete()
  AnchorStore.demote_anchor() / confirm_anchor()
```

### Proposal Merge Key

The merge key prevents duplicate proposals from the same pattern accumulating as separate entries:

```
"deprecate_rule::rule-7::professional|conflict_response"
 ──────────────  ──────  ─────────────────────────────
 proposal type  rule_id  context_signature
                         (sorted domain_tags | situation_type)
```

Two separate `report_decision_outcome` calls for the same rule in the same context type become one proposal (`evidence_count` incremented), not two.

### Bootstrap Mode

A user is in bootstrap mode when they have fewer than a threshold of shards, anchors, and rules. Bootstrap affects every stage:

| Stage | Normal behavior | Bootstrap behavior |
|-------|-----------------|--------------------|
| Recall | Semantic search | `sample_recent()` with recency weighting |
| Reasoner prompt | Full context | Bootstrap warning injected |
| Outcome processor | Shards + rule proposals | Shards only (no rule proposals) |
| Meta-learn | `learning_weight = 1.0` | `learning_weight = 0.5` |

---

## 6. Eval Harness

`src/eval.py` exposes a single function:

```python
dashboard = compute_dashboard(user_id, stores, days=30)
```

This is a **pure read function** — no writes, no side effects. All rates are `float | None` (None when the denominator is zero). Returns:

```python
{
  "user_id": "alice",
  "computed_at": "2026-05-19T...",
  "period_days": 30,

  "coverage": {
    "state": "healthy",      # "healthy" | "degraded" | "impaired" | "no_recent_activity"
    "rate": 0.84,            # reported_outcomes / total_decisions_30d
    "warning": None,
    "learning_impaired": False
  },

  "acceptance": {
    "rate": 0.71,            # accepted / total_outcomes
    "total_outcomes": 24,
    "accepted": 17, "edited": 5, "rejected": 2,
    "learning_impaired": False  # propagated from coverage.state == "impaired"
  },

  "reversal_resistance": {
    "engagement_rate": 0.40,    # engaged / total_prompted
    "confirmation_rate": 0.75,  # confirmed / engaged (None if engaged == 0)
    "total_prompted": 10, "engaged": 4, "confirmed": 3,
    "reliable": True,           # engagement_rate >= 0.30
    "flag": None,               # "UNRELIABLE" if engagement_rate < 0.30
    "learning_impaired": False
  },

  "secondary": {
    "confidence_calibration": [
      {"bin_label": "[0.0,0.4)", "acceptance_rate": 0.22, "count": 9},
      {"bin_label": "[0.4,0.6)", "acceptance_rate": 0.55, "count": 11},
      {"bin_label": "[0.6,0.8)", "acceptance_rate": 0.78, "count": 6},
      {"bin_label": "[0.8,1.0]", "acceptance_rate": 0.91, "count": 4}
    ],
    "divergence_rate": {
      "rate": 0.08,    # fraction of traces where |alignment - reproduction| > 0.2
      "alert": true,   # true if rate < 0.15 (LOW divergence = may be uncalibrated)
      "target": 0.15
    },
    "off_baseline_precision": {"rate": 0.60, "sample_size": 10},
    "anchor_stability": {
      "active_count": 12, "demoted_count": 2,
      "revision_rate_per_year": 24.3
    },
    "promotion_rejection_rate": {"rate": 0.18, "promoted": 22, "discarded": 5},
    "bootstrap_exit_time": {"status": "not_tracked", "reason": "no user creation timestamp"}
  },

  "latency": {"status": "not_tracked"}
}
```

**Two reversal rates, not one:** `engagement_rate` gates reliability (did users actually respond to review prompts?). `confirmation_rate` is the quality metric (of those who engaged, how many confirmed the reversal was real?). A user who only responds to obvious cases can appear to have 100% confirmation even if the signal is worthless — the `engagement_rate < 0.30` floor catches this and sets `flag="UNRELIABLE"`.

**`learning_impaired`** is computed once from `coverage.state == "impaired"` and propagated to both `acceptance` and `reversal_resistance`. When true, the meta-learning feedback loop is effectively disabled.

**Divergence alert logic:** `alert=true` when `rate < target` (0.15). Low divergence means alignment and reproduction are always close — the system may be collapsing to a single response mode, losing the two-track signal that makes calibration meaningful.

---

## 7. Key Design Decisions

Full rationale in `DECISIONS.md` (18 entries). Critical ones:

| Decision | Choice | Why it matters |
|----------|--------|----------------|
| D6 | Embeddings duplicated in SQLite + Chroma | SQLite cosine similarity used in baseline detection without a Chroma round-trip; Chroma is search-only |
| D10 | Sub-agents receive pre-fetched data, not tool calls | Keeps retrieval auditable in tests; cleaner upgrade path to tool-calling DeepAgents |
| D11 | `Stores` injected into `build_graph()`, not global | Enables test isolation; no singleton state shared across test cases |
| D12 | Outcome-type mapping at tool boundary | Spec vocabulary (`accepted_unchanged`) vs DB vocabulary (`accepted`) are decoupled |
| D13 | Compiled graph passed into `make_tools()` | `graph.compile()` is expensive; called once at startup, not per tool call |
| D14 | `mark_processed` in `finally` block | Prevents infinite retry on orphaned or exception-throwing outcomes |
| D17 | `generate_modified_rule` validates strict subset scope | `narrow_scope` mutation must never expand a rule's applicability |
| D18 | Eval harness is pure read, no persistence | Derived snapshots that influence the pipeline create ownership and stale-data problems |

### Quick Verification (`run.py`)

`run.py` is the manual verification entry point. It seeds the system from `initial_seeds.json`, runs two decision scenarios, simulates a rejection to trigger the double loop, and prints the eval dashboard.

```bash
# No API calls — stubs only, fast smoke test:
python run.py --no-llm

# Live mode — reads COGNITIVE_LLM_* from .env:
python run.py

# Persistent store (accumulates learning across re-runs):
python run.py --db ./twin.db
```

Copy `.env.example` to `.env` and set your API key + provider before running in live mode.

### LLM Configuration

All LLM provider and model settings are in `.env`. Change `COGNITIVE_LLM_PROVIDER` to switch providers project-wide — no source file edits required:

```bash
# .env
COGNITIVE_LLM_PROVIDER=anthropic          # or: openai | google
COGNITIVE_LLM_MAIN_MODEL=claude-sonnet-4-6
COGNITIVE_LLM_FAST_MODEL=claude-haiku-4-5-20251001
COGNITIVE_LLM_TEMPERATURE=0.0
ANTHROPIC_API_KEY=your-key-here
```

`src/llm.py` exposes `get_default_llm(role, max_tokens)` which reads these env vars at call time (not at import). All nodes and helpers call this instead of importing a provider directly.

### Embedding the System in a Parent Agent

```python
from src.storage.db import open_db, init_schema
from src.storage.shard_store import ShardStore
from src.storage.anchor_store import AnchorStore
from src.storage.governance_store import GovernanceStore
from src.storage.trace_store import TraceStore
from src.storage.proposal_queue import ProposalQueue
from src.storage.pending_anchor_store import PendingAnchorStore
from src.storage.outcome_store import OutcomeStore
from src.storage.review_store import ReviewStore
from src.deps import Stores
from src.graph import build_graph
from src.tools import make_tools

# 1. Initialize storage
conn = open_db("cognitive.db")   # WAL mode, row_factory, FK enforcement
init_schema(conn)

# 2. Build stores
governance = GovernanceStore(conn)
stores = Stores(
    shards=ShardStore(conn, chroma_persist_dir="./chroma"),
    anchors=AnchorStore(conn, chroma_persist_dir="./chroma"),
    governance=governance,
    traces=TraceStore(conn),
    proposals=ProposalQueue(conn, governance_store=governance),
    pending_anchors=PendingAnchorStore(conn),
    outcomes=OutcomeStore(conn),
    reviews=ReviewStore(conn),
)

# 3. Compile graph once (expensive — do not rebuild per call)
compiled_graph = build_graph(stores)   # picks up COGNITIVE_LLM_* from env

# 4. Expose tools to Parent Agent
tools = make_tools(stores, compiled_graph)
parent_agent.tools.extend(tools)
```

### Running Async Loops

Each loop is a standalone callable — no scheduler is bundled. Run them on a cron or call manually:

```python
from src.async_loops.outcome_processor import process_outcomes
from src.async_loops.promotion_engine import promote_proposals
from src.async_loops.reversal_reviewer import review_reversals
from src.async_loops.anchor_consolidation import consolidate_anchors
from src.async_loops.memory_decay import decay_memory

# Recommended order: process outcomes first, then promote, then review
process_outcomes(user_id, stores, llm=llm)
promote_proposals(user_id, stores, llm=llm)
review_reversals(user_id, stores, review_store=stores.reviews)
consolidate_anchors(user_id, stores, review_store=stores.reviews, llm=llm)
decay_memory(user_id, stores, llm=llm)
```

### Running Tests

```bash
python -m pytest                           # 298 tests, all components
python -m pytest tests/test_topology.py   # graph routing (no LLM, fast)
python -m pytest tests/test_eval.py       # eval harness
python -m pytest tests/test_async_loops.py # all 5 loops + storage methods
```
