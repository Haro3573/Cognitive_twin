"""
Tests for Step 8: async loops.

Sections:
  1. Schema / storage method tests (real in-memory SQLite)
  2. Mutation function tests (mocked LLM)
  3. Promotion engine tests (mocked stores)
  4. Memory decay tests (mocked stores)
  5. Anchor consolidation tests (mocked stores)
  6. Reversal reviewer tests (mocked stores)
  7. Outcome processor tests (mocked stores)
  8. Integration smoke tests (real SQLite, mocked LLM)
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

from src.storage.db import init_schema
from src.storage.outcome_store import OutcomeStore
from src.storage.governance_store import GovernanceStore
from src.storage.proposal_queue import ProposalQueue
from src.storage.shard_store import ShardStore
from src.storage.anchor_store import AnchorStore
from src.storage.review_store import ReviewStore

from src.storage.models import EditAnalysisModel, ProposedRuleModel, RejectionPatternModel

from src.async_loops.mutations import (
    analyze_edit,
    rule_might_be_responsible,
    generate_modified_rule,
    extract_rule_pattern_from_rejection,
    check_accumulators_for_promotion,
    DEPRECATION_TRIGGER,
    NEW_RULE_ACCUMULATOR_THRESHOLD,
)
from src.async_loops.promotion_engine import promote_proposals
from src.async_loops.memory_decay import decay_memory
from src.async_loops.anchor_consolidation import consolidate_anchors
from src.async_loops.reversal_reviewer import review_reversals
from src.async_loops.outcome_processor import process_outcomes


# ===========================================================================
# Helpers
# ===========================================================================

def _mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _setup_db():
    conn = _mem_conn()
    init_schema(conn)
    return conn


def _make_stores(conn):
    """Creates real storage objects on a shared in-memory DB."""
    from unittest.mock import MagicMock

    gov = GovernanceStore(conn)
    pq = ProposalQueue(conn, gov)
    outcomes = OutcomeStore(conn)
    review = ReviewStore(conn)

    # ShardStore and AnchorStore need Chroma; use mocks for in-memory tests
    shards = MagicMock()
    shards._conn = conn
    shards._embed_fn = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
    shards.add_from_trace = MagicMock(return_value=str(uuid.uuid4()))
    shards.list_for_decay.return_value = []
    shards.embed = lambda text: [0.1, 0.2, 0.3]

    anchors = MagicMock()

    from src.storage.trace_store import TraceStore
    from src.storage.pending_anchor_store import PendingAnchorStore
    traces = TraceStore(conn)
    pending_anchors = MagicMock()

    from src.deps import Stores
    return Stores(
        shards=shards,
        anchors=anchors,
        governance=gov,
        traces=traces,
        proposals=pq,
        pending_anchors=pending_anchors,
        outcomes=outcomes,
        reviews=review,
    )


def _make_rule(conn, user_id="u1", statement="Always be direct", confidence=0.7):
    gov = GovernanceStore(conn)
    return gov.add_rule({
        "rule_id": str(uuid.uuid4()),
        "user_id": user_id,
        "version": 1,
        "statement": statement,
        "confidence": confidence,
        "evidence_count": 1,
        "context_scope": [],
        "supporting_traces": [],
        "contradicting_traces": [],
        "activated_at": datetime.now(),
        "supersedes": None,
        "rule_class": "preference",
    })


def _make_trace(conn, user_id="u1", rule_basis=None, bootstrap=False, decision_content="Accept"):
    from src.storage.trace_store import TraceStore
    ts = TraceStore(conn)
    trace_id = uuid.uuid4().hex
    state = {
        "user_id": user_id,
        "rule_basis": rule_basis or [],
        "selected_hypothesis": {"id": "h1", "content": decision_content},
        "perceived_context": {"domain_tags": ["work"]},
        "output_payload": {
            "decision": {"id": "h1", "content": decision_content},
            "confidences": {"alignment": 0.8, "reproduction": 0.7, "divergence": 0.1},
            "annotations": {"bootstrap_mode": bootstrap},
            "rule_basis": rule_basis or [],
            "trace_id": trace_id,
            "alternatives": [],
        },
        "trace_id": trace_id,
    }
    ts.save(trace_id, state)
    return trace_id


def _make_outcome(conn, trace_id, user_id="u1", outcome_type="accepted", **kwargs):
    os = OutcomeStore(conn)
    return os.record(
        user_id=user_id,
        trace_id=trace_id,
        outcome_type=outcome_type,
        **kwargs,
    )


def _make_shard_store(conn):
    """Creates a real ShardStore with an in-memory Chroma collection."""
    import chromadb
    from unittest.mock import patch
    from src.storage.shard_store import ShardStore
    ephemeral = chromadb.EphemeralClient()
    with patch("src.storage.shard_store.chromadb.PersistentClient", return_value=ephemeral):
        store = ShardStore(conn, "/fake", embed_fn=lambda texts: [[0.1, 0.2, 0.3]] * len(texts))
    return store


def _make_anchor_store(conn):
    """Creates a real AnchorStore with an in-memory Chroma collection."""
    import chromadb
    from unittest.mock import patch
    from src.storage.anchor_store import AnchorStore
    ephemeral = chromadb.EphemeralClient()
    with patch("src.storage.anchor_store.chromadb.PersistentClient", return_value=ephemeral):
        store = AnchorStore(conn, "/fake", embed_fn=lambda texts: [[0.1, 0.2, 0.3]] * len(texts))
    return store


def _insert_anchor(store, conn, user_id="u1", days_old=0, supporting=None, contradicting=None):
    """Inserts an anchor with last_reinforced_at backdated by days_old."""
    anchor_id = str(uuid.uuid4())
    now = datetime.now() - timedelta(days=days_old)
    conn.execute(
        """
        INSERT INTO anchors
            (anchor_id, user_id, statement, structured_form, confidence,
             supporting_shard_ids, contradicting_shard_ids, context_scope,
             established_at, last_reinforced_at, last_user_confirmed_at, embedding)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            anchor_id, user_id, f"anchor-{anchor_id[:8]}", None, 0.8,
            json.dumps(supporting or []),
            json.dumps(contradicting or []),
            json.dumps([]),
            now.isoformat(), now.isoformat(), None, json.dumps([0.1, 0.2, 0.3]),
        ),
    )
    conn.commit()
    return anchor_id


# ===========================================================================
# 1. Schema / storage method tests
# ===========================================================================

class TestSchemaAndStorageMethods:

    def test_init_schema_creates_all_new_tables(self):
        conn = _setup_db()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "demoted_anchors" in tables
        assert "review_items" in tables

    def test_outcomes_has_processed_at_column(self):
        conn = _setup_db()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(outcomes)").fetchall()}
        assert "processed_at" in cols

    def test_governance_has_status_column(self):
        conn = _setup_db()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(governance_rules)").fetchall()}
        assert "status" in cols

    def test_outcome_unprocessed_returns_nulls(self):
        conn = _setup_db()
        trace_id = _make_trace(conn)
        oid = _make_outcome(conn, trace_id)

        os = OutcomeStore(conn)
        unprocessed = os.unprocessed("u1")
        assert len(unprocessed) == 1
        assert unprocessed[0]["outcome_id"] == oid

    def test_mark_processed_sets_timestamp(self):
        conn = _setup_db()
        trace_id = _make_trace(conn)
        oid = _make_outcome(conn, trace_id)

        os = OutcomeStore(conn)
        os.mark_processed(oid)

        row = conn.execute(
            "SELECT processed_at FROM outcomes WHERE outcome_id = ?", (oid,)
        ).fetchone()
        assert row["processed_at"] is not None

    def test_unprocessed_excludes_processed(self):
        conn = _setup_db()
        trace_id = _make_trace(conn)
        oid = _make_outcome(conn, trace_id)

        os = OutcomeStore(conn)
        os.mark_processed(oid)
        assert os.unprocessed("u1") == []

    def test_governance_modify_creates_new_version(self):
        conn = _setup_db()
        rule_id = _make_rule(conn)

        gov = GovernanceStore(conn)
        new_id = gov.modify(rule_id, {
            "statement": "Be direct but kind",
            "confidence_adjustment": -0.1,
            "rule_class": "preference",
        })

        new_rule = gov.get(new_id)
        assert new_rule is not None
        assert new_rule["statement"] == "Be direct but kind"
        assert new_rule["version"] == 2
        assert new_rule["supersedes"] == rule_id

    def test_governance_modify_marks_old_superseded(self):
        conn = _setup_db()
        rule_id = _make_rule(conn, confidence=0.7)

        gov = GovernanceStore(conn)
        gov.modify(rule_id, {"confidence_adjustment": -0.1})

        row = conn.execute(
            "SELECT superseded_by FROM governance_rules WHERE rule_id = ?",
            (rule_id,),
        ).fetchone()
        assert row["superseded_by"] is not None

    def test_governance_deprecate_excludes_from_active(self):
        conn = _setup_db()
        rule_id = _make_rule(conn)

        gov = GovernanceStore(conn)
        gov.deprecate(rule_id)

        active = gov.query_active_rules("u1")
        assert all(r["rule_id"] != rule_id for r in active)

    def test_governance_count_active_excludes_deprecated(self):
        conn = _setup_db()
        rule_id = _make_rule(conn)

        gov = GovernanceStore(conn)
        before = gov.count_active_rules("u1")
        gov.deprecate(rule_id)
        after = gov.count_active_rules("u1")
        assert after == before - 1

    def test_proposal_eligible_for_review_at_threshold(self):
        conn = _setup_db()
        gov = GovernanceStore(conn)
        pq = ProposalQueue(conn, gov)

        # Add 3 proposals with same merge key (upsert accumulates)
        for _ in range(3):
            pq.add("u1", {
                "type": "adjust_weight",
                "rationale": "weight test",
                "supporting_traces": ["t1"],
                "context": {},
                "weight": 1.0,
            })

        eligible = pq.eligible_for_review("u1")
        assert len(eligible) == 1  # threshold for adjust_weight = BASE_THRESHOLD = 3

    def test_proposal_not_eligible_below_threshold(self):
        conn = _setup_db()
        gov = GovernanceStore(conn)
        pq = ProposalQueue(conn, gov)

        pq.add("u1", {
            "type": "adjust_weight",
            "rationale": "weight test",
            "supporting_traces": ["t1"],
            "context": {},
            "weight": 1.0,
        })

        eligible = pq.eligible_for_review("u1")
        assert len(eligible) == 0  # evidence_count=1 < threshold=3

    def test_review_store_enqueue_and_list_pending(self):
        conn = _setup_db()
        rs = ReviewStore(conn)

        review_id = rs.enqueue("u1", "dormant_anchor", "anc-1", {"statement": "X"})
        pending = rs.list_pending("u1")

        assert len(pending) == 1
        assert pending[0]["review_id"] == review_id
        assert pending[0]["context"]["statement"] == "X"

    def test_review_store_record_response_removes_from_pending(self):
        conn = _setup_db()
        rs = ReviewStore(conn)

        rid = rs.enqueue("u1", "dormant_anchor", "anc-1", {})
        rs.record_response(rid, "confirm")

        assert rs.list_pending("u1") == []

    def test_review_store_count_pending(self):
        conn = _setup_db()
        rs = ReviewStore(conn)

        rs.enqueue("u1", "dormant_anchor", "a1", {})
        rs.enqueue("u1", "dormant_anchor", "a2", {})
        assert rs.count_pending("u1") == 2

    # --- ShardStore real-SQLite tests ---

    def test_shard_add_from_trace_creates_sqlite_row(self):
        conn = _setup_db()
        store = _make_shard_store(conn)
        state = {
            "user_id": "u1",
            "selected_hypothesis": {"content": "Be direct"},
            "perceived_context": {"domain_tags": ["work"]},
        }
        outcome = {"outcome_id": "o1", "outcome_type": "accepted"}
        shard_id = store.add_from_trace(state, outcome)

        row = conn.execute(
            "SELECT * FROM shards WHERE shard_id = ?", (shard_id,)
        ).fetchone()
        assert row is not None
        assert row["user_id"] == "u1"
        assert row["content"] == "Be direct"

    def test_shard_add_from_trace_content_override(self):
        conn = _setup_db()
        store = _make_shard_store(conn)
        state = {
            "user_id": "u1",
            "selected_hypothesis": {"content": "Original"},
            "perceived_context": {},
        }
        outcome = {"outcome_id": "o1", "outcome_type": "edited"}
        shard_id = store.add_from_trace(state, outcome, content_override="Edited version")

        row = conn.execute(
            "SELECT content FROM shards WHERE shard_id = ?", (shard_id,)
        ).fetchone()
        assert row["content"] == "Edited version"

    def test_shard_compress_updates_sqlite(self):
        conn = _setup_db()
        store = _make_shard_store(conn)
        state = {
            "user_id": "u1",
            "selected_hypothesis": {"content": "Detailed response about X"},
            "perceived_context": {},
        }
        shard_id = store.add_from_trace(state, {"outcome_id": "o1", "outcome_type": "accepted"})

        store.compress(shard_id, "summary of X", [0.5, 0.6, 0.7], new_level=1)

        row = conn.execute(
            "SELECT content, compression_level FROM shards WHERE shard_id = ?", (shard_id,)
        ).fetchone()
        assert row["content"] == "summary of X"
        assert row["compression_level"] == 1

    def test_shard_delete_removes_sqlite_row(self):
        conn = _setup_db()
        store = _make_shard_store(conn)
        state = {"user_id": "u1", "selected_hypothesis": {"content": "X"}, "perceived_context": {}}
        shard_id = store.add_from_trace(state, {"outcome_id": "o1", "outcome_type": "accepted"})

        store.delete(shard_id)

        row = conn.execute(
            "SELECT shard_id FROM shards WHERE shard_id = ?", (shard_id,)
        ).fetchone()
        assert row is None

    def test_shard_list_for_decay_returns_old_shards(self):
        conn = _setup_db()
        store = _make_shard_store(conn)

        # Insert one old shard directly (200 days ago)
        old_id = str(uuid.uuid4())
        old_time = (datetime.now() - timedelta(days=200)).isoformat()
        conn.execute(
            """INSERT INTO shards
               (shard_id, user_id, context, content, compression_level,
                created_at, last_activated_at, activation_count, decay_score,
                domain_tags, embedding)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (old_id, "u1", "{}", "old content", 0,
             old_time, old_time, 1, 0.2, "[]", "[0.1,0.2,0.3]"),
        )
        conn.commit()

        # Insert one recent shard
        store.add_from_trace(
            {"user_id": "u1", "selected_hypothesis": {"content": "recent"}, "perceived_context": {}},
            {"outcome_id": "o2", "outcome_type": "accepted"},
        )

        stale = store.list_for_decay("u1", min_age_days=90)
        assert len(stale) == 1
        assert stale[0]["shard_id"] == old_id

    # --- AnchorStore real-SQLite tests ---

    def test_anchor_list_dormant_returns_old(self):
        conn = _setup_db()
        store = _make_anchor_store(conn)

        old_id = _insert_anchor(store, conn, days_old=200)
        _insert_anchor(store, conn, days_old=10)  # recent

        dormant = store.list_dormant("u1", days=180)
        assert len(dormant) == 1
        assert dormant[0]["anchor_id"] == old_id

    def test_anchor_list_dormant_excludes_recent(self):
        conn = _setup_db()
        store = _make_anchor_store(conn)
        _insert_anchor(store, conn, days_old=5)

        dormant = store.list_dormant("u1", days=180)
        assert dormant == []

    def test_anchor_list_contradicted(self):
        conn = _setup_db()
        store = _make_anchor_store(conn)

        # More contradicting than supporting → should appear
        contr_id = _insert_anchor(
            store, conn, supporting=["s1"], contradicting=["c1", "c2", "c3"]
        )
        # More supporting → should not appear
        _insert_anchor(store, conn, supporting=["s1", "s2"], contradicting=["c1"])

        contradicted = store.list_contradicted("u1")
        assert len(contradicted) == 1
        assert contradicted[0]["anchor_id"] == contr_id

    def test_anchor_confirm_sets_confirmed_at(self):
        conn = _setup_db()
        store = _make_anchor_store(conn)
        anchor_id = _insert_anchor(store, conn)

        store.confirm_anchor(anchor_id)

        row = conn.execute(
            "SELECT last_user_confirmed_at FROM anchors WHERE anchor_id = ?",
            (anchor_id,),
        ).fetchone()
        assert row["last_user_confirmed_at"] is not None

    def test_anchor_demote_moves_to_demoted_table(self):
        conn = _setup_db()
        store = _make_anchor_store(conn)
        anchor_id = _insert_anchor(store, conn)

        store.demote_anchor(anchor_id, reason="test demotion")

        # Removed from active anchors
        active = conn.execute(
            "SELECT anchor_id FROM anchors WHERE anchor_id = ?", (anchor_id,)
        ).fetchone()
        assert active is None

        # Present in demoted_anchors
        demoted = conn.execute(
            "SELECT anchor_id, demotion_reason FROM demoted_anchors WHERE anchor_id = ?",
            (anchor_id,),
        ).fetchone()
        assert demoted is not None
        assert demoted["demotion_reason"] == "test demotion"

    def test_anchor_demote_noop_if_not_found(self):
        conn = _setup_db()
        store = _make_anchor_store(conn)
        # Should not raise
        store.demote_anchor("nonexistent-id")


# ===========================================================================
# 2. Mutation function tests
# ===========================================================================

class TestAnalyzeEdit:

    def test_trivial_edit_fast_path(self):
        result = analyze_edit("Hello world", "Hello world!", llm=None)
        assert result.substantive is False
        assert result.edit_type == "cosmetic"
        assert result.confidence == 1.0

    def test_substantive_edit_no_llm_fallback(self):
        result = analyze_edit("Accept the meeting", "Decline the meeting entirely", llm=None)
        assert result.substantive is True
        assert result.edit_type in ("tone_shift", "directional_change")
        assert result.confidence == 0.3

    def test_llm_path_called_for_substantive_edit(self):
        llm = MagicMock()
        expected = EditAnalysisModel(
            substantive=True,
            edit_type="directional_change",
            pattern="changed from accept to decline",
            confidence=0.9,
        )
        llm.with_structured_output.return_value.invoke.return_value = expected

        result = analyze_edit("Accept the meeting", "Decline the meeting entirely", llm=llm)
        llm.with_structured_output.assert_called_once()
        assert result.edit_type == "directional_change"

    def test_trivial_edit_skips_llm(self):
        llm = MagicMock()
        analyze_edit("Hi", "Hi!", llm=llm)
        llm.with_structured_output.assert_not_called()


class TestRuleMightBeResponsible:

    def _make_gov(self):
        gov = MagicMock()
        return gov

    def test_returns_false_if_rule_not_in_basis(self):
        gov = self._make_gov()
        analysis = EditAnalysisModel(
            substantive=True, edit_type="tone_shift",
            pattern="changed tone", confidence=0.8,
        )
        embed_fn = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
        assert not rule_might_be_responsible(
            "rule-999", analysis, ["rule-1", "rule-2"], gov, embed_fn
        )

    def test_returns_false_if_rule_has_no_embedding(self):
        gov = MagicMock()
        gov.get.return_value = {"rule_id": "r1", "embedding": []}
        analysis = EditAnalysisModel(
            substantive=True, edit_type="tone_shift",
            pattern="changed tone", confidence=0.8,
        )
        embed_fn = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
        assert not rule_might_be_responsible("r1", analysis, ["r1"], gov, embed_fn)

    def test_returns_true_when_cosine_above_threshold(self):
        gov = MagicMock()
        gov.get.return_value = {"rule_id": "r1", "embedding": [1.0, 0.0, 0.0]}
        analysis = EditAnalysisModel(
            substantive=True, edit_type="directional_change",
            pattern="changed tone", confidence=0.8,
        )
        # embed_fn returns identical vector → cosine = 1.0
        embed_fn = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
        assert rule_might_be_responsible("r1", analysis, ["r1"], gov, embed_fn)

    def test_returns_false_when_cosine_below_threshold(self):
        gov = MagicMock()
        gov.get.return_value = {"rule_id": "r1", "embedding": [1.0, 0.0, 0.0]}
        analysis = EditAnalysisModel(
            substantive=True, edit_type="directional_change",
            pattern="changed tone", confidence=0.8,
        )
        # embed_fn returns orthogonal vector → cosine = 0.0
        embed_fn = lambda texts: [[0.0, 1.0, 0.0] for _ in texts]
        assert not rule_might_be_responsible("r1", analysis, ["r1"], gov, embed_fn)


class TestGenerateModifiedRule:

    def test_returns_none_without_llm(self):
        result = generate_modified_rule(
            {"statement": "Be direct", "rule_class": "preference", "context_scope": []},
            EditAnalysisModel(
                substantive=True, edit_type="tone_shift",
                pattern="softer tone", confidence=0.8,
            ),
            llm=None,
        )
        assert result is None

    def test_llm_called_with_structured_output(self):
        llm = MagicMock()
        original = "Always respond directly and concisely to avoid confusion"
        modified = "Respond directly when the situation is clear"
        expected = ProposedRuleModel(
            statement=modified,
            context_scope=[],
            rule_class="preference",
            confidence_adjustment=-0.1,
            rationale="user softened tone",
        )
        llm.with_structured_output.return_value.invoke.return_value = expected
        analysis = EditAnalysisModel(
            substantive=True, edit_type="tone_shift",
            pattern="softer tone", confidence=0.8,
        )

        result = generate_modified_rule(
            {"statement": original, "rule_class": "preference", "context_scope": [], "confidence": 0.7},
            analysis,
            llm=llm,
        )
        assert result is not None
        assert result.statement == modified

    def test_rejects_out_of_scope_context(self):
        llm = MagicMock()
        original = "Always respond directly and concisely in all contexts"
        bad_proposal = ProposedRuleModel(
            statement="Be direct in health and work contexts",
            context_scope=["health", "work"],  # not a subset of original []
            rule_class="preference",
            confidence_adjustment=-0.1,
            rationale="test",
        )
        llm.with_structured_output.return_value.invoke.return_value = bad_proposal
        analysis = EditAnalysisModel(
            substantive=True, edit_type="tone_shift",
            pattern="softer tone", confidence=0.8,
        )

        result = generate_modified_rule(
            {"statement": original, "rule_class": "preference", "context_scope": [], "confidence": 0.7},
            analysis,
            llm=llm,
        )
        # Non-empty scope is not a subset of empty scope → fail validation → None
        assert result is None

    def test_clamps_positive_confidence_adjustment(self):
        llm = MagicMock()
        original = "Always respond directly and concisely to avoid confusion"
        modified = "Respond directly when the situation seems clear enough"
        proposal_with_positive_adj = ProposedRuleModel(
            statement=modified,
            context_scope=[],
            rule_class="preference",
            confidence_adjustment=0.3,  # positive — invalid for edit
            rationale="test",
        )
        llm.with_structured_output.return_value.invoke.return_value = proposal_with_positive_adj
        analysis = EditAnalysisModel(
            substantive=True, edit_type="tone_shift",
            pattern="test", confidence=0.8,
        )

        result = generate_modified_rule(
            {"statement": original, "rule_class": "preference", "context_scope": [], "confidence": 0.7},
            analysis,
            llm=llm,
        )
        # Clamped to 0.0
        assert result is not None
        assert result.confidence_adjustment == 0.0


class TestExtractRulePatternFromRejection:

    def test_returns_none_without_llm(self):
        result = extract_rule_pattern_from_rejection("too formal", {}, llm=None)
        assert result is None

    def test_returns_none_when_pattern_not_detected(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = RejectionPatternModel(
            pattern_detected=False,
            proposed_rule=None,
            confidence=0.4,
        )
        result = extract_rule_pattern_from_rejection("too formal", {}, llm=llm)
        assert result is None

    def test_returns_none_when_confidence_below_floor(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = RejectionPatternModel(
            pattern_detected=True,
            proposed_rule=ProposedRuleModel(
                statement="avoid formal language",
                context_scope=[],
                rule_class="preference",
                confidence_adjustment=0.3,
                rationale="test",
            ),
            confidence=0.5,  # below 0.6 floor
        )
        result = extract_rule_pattern_from_rejection("too formal", {}, llm=llm)
        assert result is None

    def test_returns_proposed_rule_when_confident(self):
        llm = MagicMock()
        proposed = ProposedRuleModel(
            statement="avoid formal language",
            context_scope=[],
            rule_class="preference",
            confidence_adjustment=0.3,
            rationale="user rejected formal tone",
        )
        llm.with_structured_output.return_value.invoke.return_value = RejectionPatternModel(
            pattern_detected=True,
            proposed_rule=proposed,
            confidence=0.85,
        )
        result = extract_rule_pattern_from_rejection("too formal", {}, llm=llm)
        assert result is not None
        assert result.statement == "avoid formal language"


class TestCheckAccumulatorsForPromotion:

    def test_does_nothing_without_accumulators(self):
        pq = MagicMock()
        pq.list.return_value = []
        check_accumulators_for_promotion("u1", pq, llm=None)
        pq.add.assert_not_called()

    def test_skips_synthesis_without_llm(self):
        pq = MagicMock()
        pq.list.return_value = [
            {"context_signature": ["work"], "rationale": f"obs {i}", "supporting_traces": []}
            for i in range(NEW_RULE_ACCUMULATOR_THRESHOLD)
        ]
        check_accumulators_for_promotion("u1", pq, llm=None)
        pq.add.assert_not_called()

    def test_synthesises_when_threshold_met(self):
        pq = MagicMock()
        siblings = [
            {
                "context_signature": ["work"],
                "rationale": f"obs {i}",
                "supporting_traces": [f"t{i}"],
                "status": "active",
                "proposal_id": str(uuid.uuid4()),
            }
            for i in range(NEW_RULE_ACCUMULATOR_THRESHOLD)
        ]
        pq.list.return_value = siblings

        llm = MagicMock()
        synthesised = ProposedRuleModel(
            statement="Never use jargon",
            context_scope=["work"],
            rule_class="preference",
            confidence_adjustment=0.3,
            rationale="pattern from 5 observations",
        )
        llm.with_structured_output.return_value.invoke.return_value = synthesised

        check_accumulators_for_promotion("u1", pq, llm=llm)

        pq.add.assert_called_once()
        call_args = pq.add.call_args[0]
        assert call_args[0] == "u1"
        assert call_args[1]["type"] == "add_rule"

    def test_marks_siblings_superseded(self):
        pq = MagicMock()
        siblings = [
            {
                "context_signature": ["work"],
                "rationale": f"obs {i}",
                "supporting_traces": [f"t{i}"],
                "status": "active",
                "proposal_id": str(uuid.uuid4()),
            }
            for i in range(NEW_RULE_ACCUMULATOR_THRESHOLD)
        ]
        pq.list.return_value = siblings

        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = ProposedRuleModel(
            statement="Never use jargon",
            context_scope=["work"],
            rule_class="preference",
            confidence_adjustment=0.3,
            rationale="synthesised",
        )

        check_accumulators_for_promotion("u1", pq, llm=llm)

        assert pq.update.call_count == NEW_RULE_ACCUMULATOR_THRESHOLD
        for upd_call in pq.update.call_args_list:
            assert upd_call[0][0]["status"] == "superseded_by_add_rule_proposal"


# ===========================================================================
# 3. Promotion engine tests
# ===========================================================================

class TestPromotionEngine:

    def _make_stores_with_proposals(self, proposal_types):
        stores = MagicMock()
        proposals = []
        for p_type, target_id in proposal_types:
            proposals.append({
                "proposal_id": str(uuid.uuid4()),
                "user_id": "u1",
                "type": p_type,
                "target_rule_id": target_id,
                "proposed_rule": {
                    "statement": "new rule",
                    "rule_class": "preference",
                    "context_scope": [],
                    "confidence_adjustment": 0.1,
                    "rationale": "test",
                },
                "evidence_count": 5,
                "promotion_threshold": 3,
                "supporting_traces": ["t1"],
                "status": "active",
            })
        stores.proposals.eligible_for_review.return_value = proposals
        stores.governance.get.return_value = {
            "rule_id": "target-rule",
            "user_id": "u1",
            "statement": "old rule",
            "confidence": 0.7,
            "rule_class": "preference",
            "context_scope": [],
            "version": 1,
        }
        return stores

    def test_returns_zero_when_no_eligible(self):
        stores = MagicMock()
        stores.proposals.eligible_for_review.return_value = []
        result = promote_proposals("u1", stores)
        assert result == {"promoted": 0, "skipped": 0}

    def test_add_rule_calls_governance_add(self):
        stores = self._make_stores_with_proposals([("add_rule", None)])
        promote_proposals("u1", stores)
        stores.governance.add_rule.assert_called_once()

    def test_modify_rule_calls_governance_modify(self):
        stores = self._make_stores_with_proposals([("modify_rule", "target-rule")])
        promote_proposals("u1", stores)
        stores.governance.modify.assert_called_once()

    def test_deprecate_rule_calls_governance_deprecate(self):
        stores = self._make_stores_with_proposals([("deprecate_rule", "target-rule")])
        promote_proposals("u1", stores)
        stores.governance.deprecate.assert_called_once()

    def test_missing_target_rule_counts_as_skipped(self):
        stores = self._make_stores_with_proposals([("modify_rule", "gone-rule")])
        stores.governance.get.return_value = None  # rule not found
        result = promote_proposals("u1", stores)
        assert result["skipped"] == 1
        assert result["promoted"] == 0

    def test_proposals_marked_promoted_or_discarded(self):
        stores = self._make_stores_with_proposals([("add_rule", None)])
        promote_proposals("u1", stores)
        assert stores.proposals.update.called
        update_call = stores.proposals.update.call_args[0][0]
        assert update_call["status"] == "promoted"


# ===========================================================================
# 4. Memory decay tests
# ===========================================================================

class TestMemoryDecay:

    def test_returns_zeros_when_no_stale_shards(self):
        stores = MagicMock()
        stores.shards.list_for_decay.return_value = []
        result = decay_memory("u1", stores)
        assert result == {"decayed": 0, "compressed": 0, "deleted": 0}

    def test_deletes_highly_decayed_compressed_shard(self):
        stores = MagicMock()
        old_time = datetime.now() - timedelta(days=400)
        stores.shards.list_for_decay.return_value = [{
            "shard_id": "s1",
            "decay_score": 0.9,  # >= DELETE_THRESHOLD = 0.85
            "compression_level": 1,
            "activation_count": 1,
            "last_activated_at": old_time,
            "content": "old memory",
        }]
        result = decay_memory("u1", stores)
        stores.shards.delete.assert_called_once_with("s1")
        assert result["deleted"] == 1

    def test_does_not_delete_uncompressed_shard_above_threshold(self):
        stores = MagicMock()
        old_time = datetime.now() - timedelta(days=400)
        conn = MagicMock()
        conn.execute.return_value = MagicMock()
        stores.shards._conn = conn
        stores.shards.list_for_decay.return_value = [{
            "shard_id": "s1",
            "decay_score": 0.9,
            "compression_level": 0,  # not yet compressed
            "activation_count": 1,
            "last_activated_at": old_time,
            "content": "old memory",
        }]
        result = decay_memory("u1", stores)
        stores.shards.delete.assert_not_called()
        assert result["decayed"] == 1

    def test_compresses_when_llm_available(self):
        stores = MagicMock()
        old_time = datetime.now() - timedelta(days=200)
        conn = MagicMock()
        conn.execute.return_value = MagicMock()
        stores.shards._conn = conn
        stores.shards.embed.return_value = [0.1, 0.2]
        stores.shards.list_for_decay.return_value = [{
            "shard_id": "s1",
            "decay_score": 0.55,  # >= COMPRESS_THRESHOLD = 0.5
            "compression_level": 0,
            "activation_count": 2,  # <= COMPRESS_MAX_ACTIVATIONS = 3
            "last_activated_at": old_time,
            "content": "A very long old memory entry about meetings",
        }]

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="User prefers short meetings.")

        result = decay_memory("u1", stores, llm=llm)
        stores.shards.compress.assert_called_once()
        assert result["compressed"] == 1


# ===========================================================================
# 5. Anchor consolidation tests
# ===========================================================================

class TestAnchorConsolidation:

    def _make_anchor(self, anchor_id="a1", days_dormant=200, contradicting=0, supporting=5):
        old_time = (datetime.now() - timedelta(days=days_dormant))
        return {
            "anchor_id": anchor_id,
            "statement": "Work-life balance matters",
            "confidence": 0.8,
            "last_reinforced_at": old_time,
            "supporting_shard_ids": ["s"] * supporting,
            "contradicting_shard_ids": ["c"] * contradicting,
        }

    def test_dormant_anchor_enqueued(self):
        stores = MagicMock()
        review_store = MagicMock()
        review_store.list_pending.return_value = []
        stores.anchors.list_dormant.return_value = [self._make_anchor()]
        stores.anchors.list_contradicted.return_value = []

        result = consolidate_anchors("u1", stores, review_store)
        assert result["dormant_queued"] == 1
        review_store.enqueue.assert_called_once()
        call_args = review_store.enqueue.call_args[1]
        assert call_args["item_type"] == "dormant_anchor"

    def test_contradicted_anchor_enqueued(self):
        stores = MagicMock()
        review_store = MagicMock()
        review_store.list_pending.return_value = []
        stores.anchors.list_dormant.return_value = []
        stores.anchors.list_contradicted.return_value = [
            self._make_anchor("a2", days_dormant=10, contradicting=4, supporting=2)
        ]

        result = consolidate_anchors("u1", stores, review_store)
        assert result["contradicted_queued"] == 1
        call_args = review_store.enqueue.call_args[1]
        assert call_args["item_type"] == "contradicted_anchor"

    def test_already_pending_not_re_enqueued(self):
        stores = MagicMock()
        review_store = MagicMock()
        review_store.list_pending.return_value = [
            {"item_id": "a1", "item_type": "dormant_anchor"}
        ]
        stores.anchors.list_dormant.return_value = [self._make_anchor("a1")]
        stores.anchors.list_contradicted.return_value = []

        result = consolidate_anchors("u1", stores, review_store)
        assert result["dormant_queued"] == 0
        review_store.enqueue.assert_not_called()


# ===========================================================================
# 6. Reversal reviewer tests
# ===========================================================================

class TestReversalReviewer:

    def test_returns_zero_when_no_rejected_outcomes(self):
        conn = _setup_db()
        stores = _make_stores(conn)
        rs = ReviewStore(conn)
        result = review_reversals("u1", stores, rs)
        assert result == {"queued": 0}

    def test_queues_reversal_when_rule_ratio_above_threshold(self):
        conn = _setup_db()
        rule_id = _make_rule(conn)
        trace_id = _make_trace(conn, rule_basis=[rule_id])
        _make_outcome(conn, trace_id, outcome_type="rejected", rejection_reason="wrong")

        # Add enough contradicting evidence to cross 0.3
        gov = GovernanceStore(conn)
        for _ in range(3):
            gov.add_contradicting_evidence(rule_id, "some-trace")
        # supporting evidence = 0, contradicting = 3 → ratio = 1.0 ≥ 0.3

        stores = _make_stores(conn)
        rs = ReviewStore(conn)
        result = review_reversals("u1", stores, rs)
        assert result["queued"] >= 1

    def test_does_not_requeue_already_pending(self):
        conn = _setup_db()
        rule_id = _make_rule(conn)
        trace_id = _make_trace(conn, rule_basis=[rule_id])
        _make_outcome(conn, trace_id, outcome_type="rejected")

        gov = GovernanceStore(conn)
        for _ in range(3):
            gov.add_contradicting_evidence(rule_id, "t")

        stores = _make_stores(conn)
        rs = ReviewStore(conn)

        # First run queues it
        review_reversals("u1", stores, rs)
        # Second run should not re-queue
        result = review_reversals("u1", stores, rs)
        assert result["queued"] == 0


# ===========================================================================
# 7. Outcome processor tests
# ===========================================================================

class TestOutcomeProcessor:

    def test_returns_zeros_when_no_unprocessed(self):
        stores = MagicMock()
        stores.outcomes.unprocessed.return_value = []
        result = process_outcomes("u1", stores)
        assert result == {"processed": 0, "shards_created": 0, "proposals_added": 0}

    def test_accepted_creates_shard(self):
        stores = MagicMock()
        trace_id = uuid.uuid4().hex
        stores.outcomes.unprocessed.return_value = [{
            "outcome_id": "o1",
            "trace_id": trace_id,
            "outcome_type": "accepted",
            "edited_content": None,
            "rejection_reason": None,
        }]
        stores.traces.get.return_value = {
            "user_id": "u1",
            "rule_basis": [],
            "selected_hypothesis": {"content": "Accept the meeting"},
            "perceived_context": {"domain_tags": ["work"]},
            "output_payload": {"annotations": {"bootstrap_mode": False}},
        }
        stores.shards.add_from_trace.return_value = "new-shard"
        stores.shards._embed_fn = lambda texts: [[0.0]] * len(texts)

        result = process_outcomes("u1", stores)
        stores.shards.add_from_trace.assert_called_once()
        assert result["shards_created"] == 1
        assert result["processed"] == 1

    def test_accepted_reinforces_rules_non_bootstrap(self):
        stores = MagicMock()
        trace_id = uuid.uuid4().hex
        stores.outcomes.unprocessed.return_value = [{
            "outcome_id": "o1",
            "trace_id": trace_id,
            "outcome_type": "accepted",
            "edited_content": None,
            "rejection_reason": None,
        }]
        stores.traces.get.return_value = {
            "user_id": "u1",
            "rule_basis": ["rule-1", "rule-2"],
            "selected_hypothesis": {"content": "Accept"},
            "perceived_context": {},
            "output_payload": {"annotations": {"bootstrap_mode": False}},
        }
        stores.shards._embed_fn = lambda texts: [[0.0]] * len(texts)

        process_outcomes("u1", stores)
        assert stores.governance.reinforce_rule.call_count == 2

    def test_accepted_does_not_reinforce_rules_in_bootstrap(self):
        stores = MagicMock()
        trace_id = uuid.uuid4().hex
        stores.outcomes.unprocessed.return_value = [{
            "outcome_id": "o1",
            "trace_id": trace_id,
            "outcome_type": "accepted",
            "edited_content": None,
            "rejection_reason": None,
        }]
        stores.traces.get.return_value = {
            "user_id": "u1",
            "rule_basis": ["rule-1"],
            "selected_hypothesis": {"content": "Accept"},
            "perceived_context": {},
            "output_payload": {"annotations": {"bootstrap_mode": True}},
        }
        stores.shards._embed_fn = lambda texts: [[0.0]] * len(texts)

        process_outcomes("u1", stores)
        stores.governance.reinforce_rule.assert_not_called()

    def test_mark_processed_always_called(self):
        stores = MagicMock()
        stores.outcomes.unprocessed.return_value = [{
            "outcome_id": "o1",
            "trace_id": "missing-trace",
            "outcome_type": "accepted",
            "edited_content": None,
            "rejection_reason": None,
        }]
        stores.traces.get.return_value = None  # orphan trace
        stores.shards._embed_fn = lambda texts: [[0.0]] * len(texts)

        process_outcomes("u1", stores)
        stores.outcomes.mark_processed.assert_called_once_with("o1")

    def test_orphan_trace_skips_shard_creation(self):
        stores = MagicMock()
        stores.outcomes.unprocessed.return_value = [{
            "outcome_id": "o1",
            "trace_id": "missing-trace",
            "outcome_type": "accepted",
            "edited_content": None,
            "rejection_reason": None,
        }]
        stores.traces.get.return_value = None
        stores.shards._embed_fn = lambda texts: [[0.0]] * len(texts)

        result = process_outcomes("u1", stores)
        stores.shards.add_from_trace.assert_not_called()
        assert result["shards_created"] == 0

    def test_edited_creates_shard_from_edited_content(self):
        stores = MagicMock()
        trace_id = uuid.uuid4().hex
        stores.outcomes.unprocessed.return_value = [{
            "outcome_id": "o1",
            "trace_id": trace_id,
            "outcome_type": "edited",
            "edited_content": "Accept the meeting but keep it short",
            "rejection_reason": None,
        }]
        stores.traces.get.return_value = {
            "user_id": "u1",
            "rule_basis": [],
            "selected_hypothesis": {"content": "Accept the meeting"},
            "perceived_context": {},
            "output_payload": {"annotations": {"bootstrap_mode": False}},
        }
        stores.shards._embed_fn = lambda texts: [[0.0]] * len(texts)

        process_outcomes("u1", stores)
        stores.shards.add_from_trace.assert_called_once()
        _, call_kwargs = stores.shards.add_from_trace.call_args
        assert call_kwargs.get("content_override") == "Accept the meeting but keep it short"

    def test_rejected_adds_contradicting_evidence(self):
        stores = MagicMock()
        trace_id = uuid.uuid4().hex
        stores.outcomes.unprocessed.return_value = [{
            "outcome_id": "o1",
            "trace_id": trace_id,
            "outcome_type": "rejected",
            "edited_content": None,
            "rejection_reason": "wrong tone",
        }]
        stores.traces.get.return_value = {
            "user_id": "u1",
            "rule_basis": ["rule-1"],
            "selected_hypothesis": {"content": "Accept"},
            "perceived_context": {},
            "output_payload": {"annotations": {"bootstrap_mode": False}},
        }
        stores.governance.rule_contradicting_evidence_ratio.return_value = 0.2
        stores.shards._embed_fn = lambda texts: [[0.0]] * len(texts)

        process_outcomes("u1", stores)
        # Called once with the rule_id and the trace_id (not outcome_id)
        stores.governance.add_contradicting_evidence.assert_called_once()
        call_args = stores.governance.add_contradicting_evidence.call_args[0]
        assert call_args[0] == "rule-1"

    def test_rejected_proposes_deprecation_above_trigger(self):
        stores = MagicMock()
        trace_id = uuid.uuid4().hex
        stores.outcomes.unprocessed.return_value = [{
            "outcome_id": "o1",
            "trace_id": trace_id,
            "outcome_type": "rejected",
            "edited_content": None,
            "rejection_reason": "wrong",
        }]
        stores.traces.get.return_value = {
            "user_id": "u1",
            "rule_basis": ["rule-1"],
            "selected_hypothesis": {"content": "Accept"},
            "perceived_context": {},
            "output_payload": {"annotations": {"bootstrap_mode": False}},
        }
        stores.governance.rule_contradicting_evidence_ratio.return_value = DEPRECATION_TRIGGER + 0.1
        stores.shards._embed_fn = lambda texts: [[0.0]] * len(texts)

        process_outcomes("u1", stores)
        stores.proposals.add.assert_called()
        calls = stores.proposals.add.call_args_list
        proposal_types = [c[0][1]["type"] for c in calls]
        assert "deprecate_rule" in proposal_types

    def test_bootstrap_rejection_skips_proposals(self):
        stores = MagicMock()
        trace_id = uuid.uuid4().hex
        stores.outcomes.unprocessed.return_value = [{
            "outcome_id": "o1",
            "trace_id": trace_id,
            "outcome_type": "rejected",
            "edited_content": None,
            "rejection_reason": "wrong",
        }]
        stores.traces.get.return_value = {
            "user_id": "u1",
            "rule_basis": ["rule-1"],
            "selected_hypothesis": {"content": "Accept"},
            "perceived_context": {},
            "output_payload": {"annotations": {"bootstrap_mode": True}},
        }
        stores.shards._embed_fn = lambda texts: [[0.0]] * len(texts)
        # Ratio below deprecation trigger so no deprecation proposal is generated
        stores.governance.rule_contradicting_evidence_ratio.return_value = 0.1

        process_outcomes("u1", stores)
        # Bootstrap rejections still record contradicting evidence (needed for deprecation trigger)
        stores.governance.add_contradicting_evidence.assert_called_once()
        # But bootstrap rejections do NOT generate rule proposals
        stores.proposals.add.assert_not_called()


# ===========================================================================
# 8. Integration smoke tests (real SQLite + mocked LLM)
# ===========================================================================

class TestIntegrationSmoke:

    def test_full_accepted_outcome_flow(self):
        conn = _setup_db()
        rule_id = _make_rule(conn)
        trace_id = _make_trace(conn, rule_basis=[rule_id])
        _make_outcome(conn, trace_id, outcome_type="accepted")

        stores = _make_stores(conn)
        result = process_outcomes("u1", stores, llm=None)

        assert result["processed"] == 1
        assert result["shards_created"] == 1
        # outcome marked processed
        os = OutcomeStore(conn)
        assert os.unprocessed("u1") == []

    def test_full_rejected_outcome_flow(self):
        conn = _setup_db()
        rule_id = _make_rule(conn)
        trace_id = _make_trace(conn, rule_basis=[rule_id])
        _make_outcome(conn, trace_id, outcome_type="rejected", rejection_reason="wrong tone")

        stores = _make_stores(conn)

        # Process outcome → adds contradicting evidence to real governance store
        result = process_outcomes("u1", stores, llm=None)
        assert result["processed"] == 1
        assert OutcomeStore(conn).unprocessed("u1") == []

    def test_full_promotion_flow(self):
        conn = _setup_db()
        gov = GovernanceStore(conn)
        pq = ProposalQueue(conn, gov)

        # Add 3 adjust_weight proposals (threshold = 3) to trigger eligibility
        for _ in range(3):
            pq.add("u1", {
                "type": "adjust_weight",
                "rationale": "test",
                "supporting_traces": ["t1"],
                "context": {},
                "weight": 1.0,
            })

        from src.deps import Stores
        from src.storage.trace_store import TraceStore
        stores = MagicMock()
        stores.proposals = pq
        stores.governance = gov

        result = promote_proposals("u1", stores)
        assert result["promoted"] == 1

        # Promoted proposal should no longer be eligible
        assert pq.eligible_for_review("u1") == []

    def test_anchor_consolidation_flow(self):
        conn = _setup_db()
        rs = ReviewStore(conn)

        stores = MagicMock()
        stores.anchors.list_dormant.return_value = [{
            "anchor_id": "a1",
            "statement": "Work-life balance",
            "confidence": 0.8,
            "last_reinforced_at": datetime.now() - timedelta(days=200),
            "supporting_shard_ids": ["s1"],
            "contradicting_shard_ids": [],
        }]
        stores.anchors.list_contradicted.return_value = []
        stores.reviews = rs

        result = consolidate_anchors("u1", stores, rs)
        assert result["dormant_queued"] == 1
        assert rs.count_pending("u1") == 1
