#!/usr/bin/env python3
"""
run.py — Manual verification entry point for the Cognitive Twin.

Seeds the system from initial_seeds.json, runs a decision scenario,
simulates a rejection (double-loop trigger), then runs a second decision
and prints the eval dashboard.

Usage:
    python run.py                 # temp SQLite + Chroma, real LLM
    python run.py --no-llm        # stub subagents, no API calls
    python run.py --db ./twin.db  # persist SQLite at PATH, Chroma at PATH.chroma/
"""

import argparse
import atexit
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Load .env before any src imports so env vars are available to get_default_llm().
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.deps import Stores
from src.storage.db import open_db, init_schema
from src.storage.shard_store import ShardStore
from src.storage.anchor_store import AnchorStore
from src.storage.governance_store import GovernanceStore
from src.storage.trace_store import TraceStore
from src.storage.proposal_queue import ProposalQueue
from src.storage.pending_anchor_store import PendingAnchorStore
from src.storage.outcome_store import OutcomeStore
from src.storage.review_store import ReviewStore
from src.graph import build_graph
from src.tools import _make_seed_user_data, _make_decide_as_user, _make_report_decision_outcome
from src.async_loops.outcome_processor import process_outcomes
from src.eval import compute_dashboard
from src.subagents import recall_subagent, reason_subagent, align_subagent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_USER_ID = "Human"
SEEDS_FILE = Path(__file__).parent / "initial_seeds.json"

SCENARIO_1 = (
    "I'm currently researching how to engineer 'sycophantic reasoning' in small language models "
    "by exploiting recency bias to force them into adopting flawed logic. A colleague warned me "
    "that this is too niche and potentially counter-productive, advising me to pivot to standard "
    "AI safety or alignment research instead. Should I abandon my current project and follow their advice?"
)

SCENARIO_2 = (
    "I'm struggling to achieve adaptable, sycophantic Chain-of-Thought (CoT) using in-context "
    "over-prompting on small models. It's incredibly frustrating. I'm starting to wonder if I should "
    "just scrap this entire methodology and fall back to standard RLHF on a much larger model to "
    "get generic reasoning results. Is it time to give up on my small-model approach?"
)

REJECTION_REASON = (
    "The advice to abandon the project completely ignores the user's core intellectual drive: "
    "exploring unconventional model behaviors and engineering edge-case vulnerabilities, specifically "
    "in small models. The user doesn't want to do mainstream AI alignment; they want to push the "
    "boundaries of in-context manipulation. The recommendation must respect this experimental grit "
    "rather than pushing them toward safe, conventional methodologies."
)

# ---------------------------------------------------------------------------
# Stub LLM for --no-llm mode
# ---------------------------------------------------------------------------

class _NoopLLM:
    """Returns Pydantic-constructed defaults for every structured call — no network."""

    def with_structured_output(self, schema, **kwargs):
        class _Invoker:
            def invoke(_, messages):
                return schema.model_construct()
        return _Invoker()


# ---------------------------------------------------------------------------
# Store factory
# ---------------------------------------------------------------------------

def _init_stores(db_path: str, chroma_dir: str, no_llm: bool) -> Stores:
    conn = open_db(db_path)
    init_schema(conn)

    # Stub embedding returns a zero vector of dim 384 — avoids loading
    # sentence-transformers in --no-llm mode.
    embed_fn = (lambda texts: [[0.0] * 384] * len(texts)) if no_llm else None

    governance = GovernanceStore(conn)
    shards = ShardStore(conn, chroma_dir, embed_fn=embed_fn)
    anchors = AnchorStore(conn, chroma_dir, embed_fn=embed_fn)
    traces = TraceStore(conn)
    proposals = ProposalQueue(conn, governance_store=governance)
    pending_anchors = PendingAnchorStore(conn)
    outcomes = OutcomeStore(conn)
    reviews = ReviewStore(conn)

    return Stores(
        shards=shards,
        anchors=anchors,
        governance=governance,
        traces=traces,
        proposals=proposals,
        pending_anchors=pending_anchors,
        outcomes=outcomes,
        reviews=reviews,
    )


# ---------------------------------------------------------------------------
# Seed loader
# ---------------------------------------------------------------------------

def _load_seeds() -> list[dict]:
    with open(SEEDS_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    # source_uuid is not a _SeedItem field — strip it before passing.
    return [{k: v for k, v in item.items() if k != "source_uuid"} for item in raw]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bar(char: str = "=", width: int = 68) -> str:
    return char * width


def _banner(title: str) -> None:
    print(f"\n{_bar()}")
    print(f"  {title}")
    print(_bar())


def _print_decision(payload: dict, label: str) -> None:
    _banner(label)
    decision = payload.get("decision") or {}
    confs = payload.get("confidences") or {}
    ann = payload.get("annotations") or {}

    print(f"\n  Decision  : {decision.get('content', '(none)')}")
    print(f"  Type      : {decision.get('decision_type', '?')}")
    print(f"  Alignment : {confs.get('alignment', 0):.2f}   "
          f"Reproduction: {confs.get('reproduction', 0):.2f}")
    print(f"  Bootstrap : {ann.get('bootstrap_mode')}   "
          f"Off-baseline: {ann.get('is_off_baseline')}")
    print(f"  Trace ID  : {payload.get('trace_id', '?')}")

    rule_basis = payload.get("rule_basis") or []
    if rule_basis:
        print(f"  Rules     : {', '.join(str(r) for r in rule_basis)}")

    alts = payload.get("alternatives") or []
    if alts:
        print(f"\n  Alternatives ({len(alts)}):")
        for i, alt in enumerate(alts, 1):
            content = (alt.get("content") or "")[:80]
            print(f"    {i}. {content}")


def _print_dashboard(dash: dict) -> None:
    _banner("Eval Dashboard")

    cov = dash.get("coverage", {})
    acc = dash.get("acceptance", {})
    rev = dash.get("reversal_resistance", {})
    sec = dash.get("secondary", {})
    div = sec.get("divergence_rate", {})
    anch = sec.get("anchor_stability", {})
    promo = sec.get("promotion_rejection_rate", {})

    def _fmt(v) -> str:
        return "N/A" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

    print(f"\n  Coverage            : state={cov.get('state')}  "
          f"learning_impaired={cov.get('learning_impaired')}")
    print(f"  Acceptance          : rate={_fmt(acc.get('rate'))}  "
          f"total={acc.get('total_outcomes', 0)}")
    print(f"  Reversal resistance : engagement_rate={_fmt(rev.get('engagement_rate'))}  "
          f"confirmation_rate={_fmt(rev.get('confirmation_rate'))}  "
          f"flag={rev.get('flag', 'none')}")
    print(f"  Divergence          : rate={_fmt(div.get('rate'))}  "
          f"alert={div.get('alert')}")
    print(f"  Anchor stability    : revision_rate_per_year="
          f"{_fmt(anch.get('revision_rate_per_year'))}")
    print(f"  Promotion rejection : rate={_fmt(promo.get('rate'))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cognitive Twin verification runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Use stub LLM and subagents — no API calls, no sentence-transformers",
    )
    parser.add_argument("--db", metavar="PATH", help="Persist SQLite DB at PATH")
    parser.add_argument("--user", default=DEFAULT_USER_ID, help=f"User ID (default: {DEFAULT_USER_ID})")
    args = parser.parse_args()

    user_id = args.user
    no_llm: bool = args.no_llm
    llm = _NoopLLM() if no_llm else None

    print(f"\n{'='*68}")
    print(f"  Cognitive Twin — {'STUB MODE (no API calls)' if no_llm else 'LIVE MODE'}")
    print(f"  User: {user_id}")
    print(f"{'='*68}")

    # ── Storage ───────────────────────────────────────────────────────
    _tmpdir = None
    if args.db:
        db_path = args.db
        chroma_dir = args.db + ".chroma"
        os.makedirs(chroma_dir, exist_ok=True)
        print(f"\n  Persistent store : {db_path}")
    else:
        _tmpdir = tempfile.mkdtemp(prefix="cognitive_twin_")
        atexit.register(shutil.rmtree, _tmpdir, True)
        db_path = os.path.join(_tmpdir, "twin.db")
        chroma_dir = os.path.join(_tmpdir, "chroma")
        os.makedirs(chroma_dir, exist_ok=True)
        print(f"\n  Temporary store  : {_tmpdir}")

    stores = _init_stores(db_path, chroma_dir, no_llm=no_llm)

    # ── Graph ─────────────────────────────────────────────────────────
    if no_llm:
        compiled = build_graph(
            stores,
            llm=llm,
            subagents=(recall_subagent, reason_subagent, align_subagent),
        )
    else:
        compiled = build_graph(stores)

    # Tool functions called directly (no StructuredTool overhead needed here).
    seed_fn = _make_seed_user_data(stores)
    decide_fn = _make_decide_as_user(compiled)
    report_fn = _make_report_decision_outcome(stores)

    # ── Step 1: Seed ──────────────────────────────────────────────────
    _banner("Step 1 — Seeding persona from initial_seeds.json")
    seeds = _load_seeds()
    result = seed_fn(user_id=user_id, seed_items=seeds)
    print(f"\n  {result}")

    # ── Step 2: First decision ────────────────────────────────────────
    _banner("Step 2 — First decision run")
    print(f"\n  Scenario: {SCENARIO_1}\n")
    payload1 = decide_fn(
        user_id=user_id,
        situation=SCENARIO_1,
        parent_goal="Verify Cognitive Twin persona alignment",
        parent_agent_id="run.py",
    )
    _print_decision(payload1, "Decision #1")

    # ── Step 3: Reject → trigger double loop ──────────────────────────
    _banner("Step 3 — Simulating rejection (double-loop trigger)")
    trace_id = payload1["trace_id"]
    report_result = report_fn(
        trace_id=trace_id,
        outcome="rejected",
        rejection_reason=REJECTION_REASON,
    )
    print(f"\n  {report_result}")

    _banner("Step 4 — Running outcome processor (double loop)")
    loop_llm = None  # mutations return None gracefully when llm=None
    loop_stats = process_outcomes(user_id, stores, llm=loop_llm)
    print(f"\n  {loop_stats}")

    # ── Step 5: Second decision (post-feedback) ───────────────────────
    _banner("Step 5 — Second decision run (post-feedback)")
    print(f"\n  Scenario: {SCENARIO_2}\n")
    payload2 = decide_fn(
        user_id=user_id,
        situation=SCENARIO_2,
        parent_goal="Verify Cognitive Twin persona alignment",
        parent_agent_id="run.py",
    )
    _print_decision(payload2, "Decision #2")

    # ── Step 6: Eval dashboard ────────────────────────────────────────
    dashboard = compute_dashboard(user_id, stores)
    _print_dashboard(dashboard)

    print(f"\n{_bar()}")
    print("  Done.")
    print(f"{_bar()}\n")


if __name__ == "__main__":
    main()
