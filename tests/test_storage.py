"""
Step-2 storage layer tests.

Uses in-memory SQLite (no file I/O) and a dummy embedding function
(no sentence-transformers download required).

Critical invariants tested:
  - ProposalQueue upsert semantics (load-bearing per patch §B1)
  - GovernanceStore supersession (active-rule query)
  - ShardStore count_shards_matching_anchor
  - compute_promotion_threshold formula
  - context_signature determinism
"""

import uuid
from datetime import datetime

import pytest

from src.storage.db import open_db, init_schema
from src.storage.models import (
    ShardModel, AnchorModel, GovernanceRuleModel, ProposalModel,
)
from src.storage.shard_store import ShardStore
from src.storage.anchor_store import AnchorStore
from src.storage.governance_store import GovernanceStore
from src.storage.proposal_queue import (
    ProposalQueue, compute_promotion_threshold, context_signature, proposal_merge_key,
)
from src.storage.trace_store import TraceStore
from src.storage.outcome_store import OutcomeStore
from src.storage.pending_anchor_store import PendingAnchorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMBED_DIM = 8
_zero_embed = [0.0] * EMBED_DIM
_dummy_embed_fn = lambda texts: [_zero_embed for _ in texts]


@pytest.fixture
def conn():
    c = open_db(":memory:")
    init_schema(c)
    return c


@pytest.fixture
def gov_store(conn):
    return GovernanceStore(conn)


@pytest.fixture
def proposal_queue(conn, gov_store):
    return ProposalQueue(conn, governance_store=gov_store)


@pytest.fixture
def shard_store(conn, tmp_path):
    return ShardStore(conn, str(tmp_path / "chroma"), embed_fn=_dummy_embed_fn)


@pytest.fixture
def anchor_store(conn, tmp_path):
    return AnchorStore(conn, str(tmp_path / "chroma"), embed_fn=_dummy_embed_fn)


@pytest.fixture
def trace_store(conn):
    return TraceStore(conn)


@pytest.fixture
def outcome_store(conn):
    return OutcomeStore(conn)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def make_rule(**kwargs) -> dict:
    defaults = {
        "rule_id": str(uuid.uuid4()),
        "user_id": "u1",
        "version": 1,
        "statement": "Prefer direct communication",
        "confidence": 0.7,
        "evidence_count": 5,
        "context_scope": [],
        "supporting_traces": [],
        "contradicting_traces": [],
        "activated_at": datetime.now(),
        "supersedes": None,
        "rule_class": "preference",
    }
    defaults.update(kwargs)
    return defaults


def make_shard(**kwargs) -> dict:
    now = datetime.now()
    defaults = {
        "shard_id": str(uuid.uuid4()),
        "user_id": "u1",
        "context": {"situation_type": "work", "domain_tags": ["work"]},
        "content": "User agreed to the meeting",
        "compression_level": 0,
        "created_at": now,
        "last_activated_at": now,
        "activation_count": 1,
        "decay_score": 0.0,
        "domain_tags": ["work"],
        "embedding": _zero_embed,
    }
    defaults.update(kwargs)
    return defaults


def make_anchor(**kwargs) -> dict:
    now = datetime.now()
    defaults = {
        "anchor_id": str(uuid.uuid4()),
        "user_id": "u1",
        "statement": "User prefers punctuality",
        "confidence": 0.8,
        "supporting_shard_ids": [],
        "contradicting_shard_ids": [],
        "context_scope": ["work"],
        "established_at": now,
        "last_reinforced_at": now,
        "embedding": _zero_embed,
    }
    defaults.update(kwargs)
    return defaults


def make_proposal(**kwargs) -> dict:
    defaults = {
        "type": "investigate_divergence",
        "target_rule_id": None,
        "rationale": "alignment=0.3 reproduction=0.7",
        "weight": 1.0,
        "supporting_traces": ["trace-1"],
        "context": {"domain_tags": ["work"], "situation_type": "decision_support"},
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_creates_all_tables(conn):
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    expected = {
        "shards", "anchors", "pending_anchors",
        "governance_rules", "proposals", "traces", "outcomes",
    }
    assert expected.issubset(tables)


# ---------------------------------------------------------------------------
# compute_promotion_threshold
# ---------------------------------------------------------------------------

def test_threshold_modify_high_confidence():
    # int(3 * (1 + 2 * 0.9)) = int(3 * 2.8) = int(8.4) = 8
    assert compute_promotion_threshold("modify_rule", 0.9) == 8


def test_threshold_add_rule():
    # int(3 * 1.5) = int(4.5) = 4
    assert compute_promotion_threshold("add_rule", None) == 4


def test_threshold_deprecate():
    # int(3 * 2 * 0.7) + 1 = int(4.2) + 1 = 4 + 1 = 5
    assert compute_promotion_threshold("deprecate_rule", 0.7) == 5


def test_threshold_adjust_weight():
    assert compute_promotion_threshold("adjust_weight", None) == 3


def test_threshold_investigate():
    assert compute_promotion_threshold("investigate_divergence", None) == 1
    assert compute_promotion_threshold("investigate_new_rule", None) == 1


# ---------------------------------------------------------------------------
# context_signature
# ---------------------------------------------------------------------------

def test_context_signature_sorted():
    ctx = {"domain_tags": ["health", "work"], "situation_type": "decision_support"}
    sig = context_signature(ctx)
    assert sig == ["health", "work", "situation:decision_support"]


def test_context_signature_empty():
    assert context_signature({}) == []


def test_context_signature_deterministic():
    ctx = {"domain_tags": ["b", "a"], "situation_type": "other"}
    assert context_signature(ctx) == context_signature(ctx)


# ---------------------------------------------------------------------------
# ProposalQueue — upsert semantics (load-bearing)
# ---------------------------------------------------------------------------

def test_proposal_first_insert(proposal_queue):
    p = make_proposal()
    result = proposal_queue.add("u1", p)
    assert result is not None
    assert result["evidence_count"] == 1
    assert result["status"] == "active"


def test_proposal_same_key_upserts(proposal_queue):
    """Same type + target + context → one row, evidence_count accumulates."""
    ctx = {"domain_tags": ["work"], "situation_type": "decision_support"}
    p1 = make_proposal(type="investigate_divergence", context=ctx, supporting_traces=["t1"])
    p2 = make_proposal(type="investigate_divergence", context=ctx, supporting_traces=["t2"])

    r1 = proposal_queue.add("u1", p1)
    r2 = proposal_queue.add("u1", p2)

    assert r1["proposal_id"] == r2["proposal_id"], "Same key must produce same row"
    assert r2["evidence_count"] == 2
    assert "t1" in r2["supporting_traces"]
    assert "t2" in r2["supporting_traces"]


def test_proposal_weight_takes_max(proposal_queue):
    """Weight is max(old, new), not sum."""
    ctx = {"domain_tags": ["work"], "situation_type": "other"}
    r1 = proposal_queue.add("u1", make_proposal(context=ctx, weight=0.3))
    r2 = proposal_queue.add("u1", make_proposal(context=ctx, weight=0.9))
    assert r2["weight"] == pytest.approx(0.9)


def test_proposal_different_context_is_new_row(proposal_queue):
    """Different domain_tags → different context_signature → two rows."""
    r1 = proposal_queue.add("u1", make_proposal(context={"domain_tags": ["work"]}))
    r2 = proposal_queue.add("u1", make_proposal(context={"domain_tags": ["health"]}))
    assert r1["proposal_id"] != r2["proposal_id"]
    assert proposal_queue.count_active("u1") == 2


def test_proposal_investigate_new_rule_accumulates_with_none_target(proposal_queue):
    """investigate_new_rule with target_rule_id=None accumulates by context."""
    ctx = {"domain_tags": ["finance"], "situation_type": "decision_support"}
    for _ in range(3):
        proposal_queue.add("u1", make_proposal(
            type="investigate_new_rule",
            target_rule_id=None,
            context=ctx,
            supporting_traces=[str(uuid.uuid4())],
        ))
    active = proposal_queue.list("u1", p_type="investigate_new_rule")
    assert len(active) == 1
    assert active[0]["evidence_count"] == 3


def test_proposal_promotion_threshold_computed_once(proposal_queue, gov_store):
    """Threshold set at first insert; upsert doesn't recompute it."""
    rule_id = gov_store.add_rule(make_rule(confidence=0.5))
    p = make_proposal(type="modify_rule", target_rule_id=rule_id)
    r1 = proposal_queue.add("u1", p)
    original_threshold = r1["promotion_threshold"]

    # Add another rule with different confidence to governance, shouldn't affect threshold
    r2 = proposal_queue.add("u1", make_proposal(
        type="modify_rule", target_rule_id=rule_id,
        context={"domain_tags": ["work"]},
    ))
    # Same key → same row → same original threshold
    assert r2["promotion_threshold"] == original_threshold


# ---------------------------------------------------------------------------
# GovernanceStore — supersession and active-rule query
# ---------------------------------------------------------------------------

def test_add_rule(gov_store):
    rule_id = gov_store.add_rule(make_rule())
    assert rule_id is not None
    fetched = gov_store.get(rule_id)
    assert fetched["statement"] == "Prefer direct communication"


def test_query_active_excludes_superseded(gov_store):
    v1_id = gov_store.add_rule(make_rule(rule_id="rule-v1", confidence=0.7))
    v2_id = gov_store.add_rule(make_rule(rule_id="rule-v2", confidence=0.8, supersedes=v1_id))

    active = gov_store.query_active_rules("u1")
    active_ids = [r["rule_id"] for r in active]
    assert v2_id in active_ids
    assert v1_id not in active_ids


def test_query_active_context_filter_universal_rule(gov_store):
    """Empty context_scope = matches any context."""
    gov_store.add_rule(make_rule(context_scope=[]))  # universal
    active = gov_store.query_active_rules(
        "u1", context={"domain_tags": ["health"]}
    )
    assert len(active) >= 1


def test_query_active_context_filter_scoped_rule(gov_store):
    """Scoped rule only matches if its scope overlaps the context's domain_tags."""
    gov_store.add_rule(make_rule(rule_id="r-work", context_scope=["work"]))
    gov_store.add_rule(make_rule(rule_id="r-health", context_scope=["health"]))

    work_ctx = {"domain_tags": ["work"]}
    active = gov_store.query_active_rules("u1", context=work_ctx)
    ids = [r["rule_id"] for r in active]
    assert "r-work" in ids
    assert "r-health" not in ids


def test_reinforce_rule_nudges_confidence(gov_store):
    rule_id = gov_store.add_rule(make_rule(confidence=0.5))
    gov_store.reinforce_rule(rule_id, "trace-a", weight=1.0)
    rule = gov_store.get(rule_id)
    assert rule["confidence"] == pytest.approx(0.55)
    assert "trace-a" in rule["supporting_traces"]


def test_contradicting_evidence_nudges_down(gov_store):
    rule_id = gov_store.add_rule(make_rule(confidence=0.5))
    gov_store.add_contradicting_evidence(rule_id, "trace-b", weight=1.0)
    rule = gov_store.get(rule_id)
    assert rule["confidence"] == pytest.approx(0.45)


def test_contradiction_ratio(gov_store):
    rule_id = gov_store.add_rule(make_rule(confidence=0.5))
    gov_store.reinforce_rule(rule_id, "t1")
    gov_store.reinforce_rule(rule_id, "t2")
    gov_store.add_contradicting_evidence(rule_id, "t3")
    ratio = gov_store.rule_contradicting_evidence_ratio(rule_id)
    # 1 contradicting / (2 supporting + 1 contradicting) = 1/3 ≈ 0.333
    assert ratio == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# ShardStore — sample_recent and count_shards_matching_anchor
# ---------------------------------------------------------------------------

def test_shard_add_and_get(shard_store):
    s = make_shard()
    sid = shard_store.add(s)
    fetched = shard_store.get(sid)
    assert fetched is not None
    assert fetched["content"] == s["content"]


def test_count_shards_matching_anchor_filters_by_shard_ids(shard_store):
    """count_shards_matching_anchor only counts shards in anchor's supporting list."""
    now = datetime.now()
    s1 = make_shard(shard_id="s1", last_activated_at=now)
    s2 = make_shard(shard_id="s2", last_activated_at=now)
    s3 = make_shard(shard_id="s3", last_activated_at=now)
    for s in [s1, s2, s3]:
        shard_store.add(s)

    # Anchor only references s1 and s2
    count = shard_store.count_shards_matching_anchor("u1", ["s1", "s2"], days=30)
    assert count == 2


def test_count_shards_matching_anchor_respects_recency(shard_store):
    """Shards outside the time window are not counted."""
    from datetime import timedelta
    old = datetime.now() - timedelta(days=60)
    recent = datetime.now()

    shard_store.add(make_shard(shard_id="old-s", last_activated_at=old))
    shard_store.add(make_shard(shard_id="new-s", last_activated_at=recent))

    count = shard_store.count_shards_matching_anchor(
        "u1", ["old-s", "new-s"], days=30
    )
    assert count == 1  # only new-s is within 30 days


def test_count_shards_empty_anchor_shard_ids(shard_store):
    assert shard_store.count_shards_matching_anchor("u1", [], days=30) == 0


def test_sample_recent_returns_within_window(shard_store):
    from datetime import timedelta
    now = datetime.now()
    old = now - timedelta(days=60)
    for i in range(5):
        shard_store.add(make_shard(
            shard_id=f"new-{i}", last_activated_at=now
        ))
    shard_store.add(make_shard(shard_id="old-1", last_activated_at=old))

    samples = shard_store.sample_recent("u1", days=30, sample_size=10)
    ids = [s["shard_id"] for s in samples]
    assert "old-1" not in ids
    assert len(ids) == 5


def test_sample_recent_caps_at_sample_size(shard_store):
    now = datetime.now()
    for i in range(20):
        shard_store.add(make_shard(shard_id=f"s{i}", last_activated_at=now))
    samples = shard_store.sample_recent("u1", days=30, sample_size=10)
    assert len(samples) <= 10


# ---------------------------------------------------------------------------
# AnchorStore
# ---------------------------------------------------------------------------

def test_anchor_add_and_list(anchor_store):
    a = make_anchor()
    anchor_store.add(a)
    anchors = anchor_store.list_for_user("u1")
    assert len(anchors) == 1
    assert anchors[0]["statement"] == a["statement"]


def test_anchor_add_supporting_shard(anchor_store):
    a = make_anchor(anchor_id="anc-1")
    anchor_store.add(a)
    anchor_store.add_supporting_shard("anc-1", "shard-x")
    fetched = anchor_store.get("anc-1")
    assert "shard-x" in fetched["supporting_shard_ids"]


# ---------------------------------------------------------------------------
# TraceStore
# ---------------------------------------------------------------------------

def test_trace_save_and_get(trace_store):
    trace_id = "trace-abc"
    data = {"trace_id": trace_id, "user_id": "u1", "selected_hypothesis": {"id": "h1"}}
    trace_store.save(trace_id, data)
    fetched = trace_store.get(trace_id)
    assert fetched is not None
    assert fetched["trace_id"] == trace_id


def test_trace_save_is_idempotent(trace_store):
    """UPSERT: saving twice doesn't raise and last write wins."""
    trace_store.save("t1", {"trace_id": "t1", "user_id": "u1", "v": 1})
    trace_store.save("t1", {"trace_id": "t1", "user_id": "u1", "v": 2})
    assert trace_store.get("t1")["v"] == 2


def test_trace_count_recent(trace_store):
    trace_store.save("t1", {"trace_id": "t1", "user_id": "u1"})
    trace_store.save("t2", {"trace_id": "t2", "user_id": "u1"})
    assert trace_store.count_recent("u1", days=30) == 2


# ---------------------------------------------------------------------------
# OutcomeStore — coverage assessment
# ---------------------------------------------------------------------------

def test_outcome_record(outcome_store):
    oid = outcome_store.record("u1", "trace-1", "accepted")
    assert oid is not None


def test_outcome_coverage_healthy(outcome_store, trace_store):
    for i in range(10):
        trace_store.save(f"t{i}", {"trace_id": f"t{i}", "user_id": "u1"})
        outcome_store.record("u1", f"t{i}", "accepted")
    coverage = outcome_store.assess_coverage("u1", trace_store)
    assert coverage["state"] == "healthy"
    assert coverage["rate"] >= 0.7


def test_outcome_coverage_impaired(outcome_store, trace_store):
    for i in range(10):
        trace_store.save(f"t{i}", {"trace_id": f"t{i}", "user_id": "u1"})
    # Only report 2 out of 10 → 20% coverage → impaired
    outcome_store.record("u1", "t0", "accepted")
    outcome_store.record("u1", "t1", "accepted")
    coverage = outcome_store.assess_coverage("u1", trace_store)
    assert coverage["state"] == "impaired"
    assert "below 30%" in coverage["warning"]


# ---------------------------------------------------------------------------
# PendingAnchorStore
# ---------------------------------------------------------------------------

def test_pending_anchor_stage_and_list(conn):
    store = PendingAnchorStore(conn)
    now = datetime.now()
    a = make_anchor()
    a["pending_confirmation_at"] = now
    a["seeded_from"] = "seed_user_data"
    store.stage(a)
    pending = store.list_for_user("u1")
    assert len(pending) == 1
    assert pending[0]["seeded_from"] == "seed_user_data"


def test_pending_anchor_delete(conn):
    store = PendingAnchorStore(conn)
    now = datetime.now()
    a = make_anchor(anchor_id="pa-1")
    a["pending_confirmation_at"] = now
    store.stage(a)
    store.delete("pa-1")
    assert store.get("pa-1") is None
