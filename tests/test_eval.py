"""
Tests for Step 9: eval harness (src/eval.py).

Uses real in-memory SQLite for all stores that only need SQLite, and
MagicMock for AnchorStore (whose Chroma constructor we avoid).
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.storage.db import init_schema
from src.storage.outcome_store import OutcomeStore
from src.storage.review_store import ReviewStore
from src.storage.trace_store import TraceStore
from src.storage.proposal_queue import ProposalQueue
from src.deps import Stores
from src.eval import (
    compute_dashboard,
    DIVERGENCE_ALERT_THRESHOLD,
    REVERSAL_RELIABILITY_FLOOR,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    return conn


def _stores(conn, *, anchor_mock=None):
    """Returns a Stores instance using real SQLite stores and MagicMock for Chroma-backed ones."""
    anchor = anchor_mock or MagicMock()
    if anchor_mock is None:
        anchor.count_for_user.return_value = 0
        anchor.count_demoted.return_value = 0
        anchor.count_demoted_in_period.return_value = 0
    return Stores(
        shards=MagicMock(),
        anchors=anchor,
        governance=MagicMock(),
        traces=TraceStore(conn),
        proposals=ProposalQueue(conn),
        pending_anchors=MagicMock(),
        outcomes=OutcomeStore(conn),
        reviews=ReviewStore(conn),
    )


def _record_outcome(conn, user_id, trace_id, outcome_type, reported_at=None):
    reported_at = reported_at or datetime.now().isoformat()
    conn.execute(
        """
        INSERT INTO outcomes
            (outcome_id, user_id, trace_id, outcome_type,
             original_content, edited_content, rejection_reason,
             alternative_id, reported_at, meta_weight)
        VALUES (?,?,?,?,NULL,NULL,NULL,NULL,?,1.0)
        """,
        (str(uuid.uuid4()), user_id, trace_id, outcome_type, reported_at),
    )
    conn.commit()


def _save_trace(conn, user_id, trace_id, *, alignment=None, reproduction=None,
                is_off_baseline=False):
    payload = None
    if alignment is not None or reproduction is not None:
        payload = {
            "confidences": {
                "alignment": alignment,
                "reproduction": reproduction,
                "divergence": abs((alignment or 0) - (reproduction or 0)),
            }
        }
    state = {
        "user_id": user_id,
        "trace_id": trace_id,
        "is_off_baseline": is_off_baseline,
        "output_payload": payload,
    }
    conn.execute(
        "INSERT INTO traces (trace_id, user_id, data, created_at) VALUES (?,?,?,?)",
        (trace_id, user_id, json.dumps(state), datetime.now().isoformat()),
    )
    conn.commit()


def _enqueue_reversal(conn, user_id, trace_id, *, response=None):
    review_id = str(uuid.uuid4())
    responded_at = datetime.now().isoformat() if response is not None else None
    conn.execute(
        """
        INSERT INTO review_items
            (review_id, user_id, item_type, item_id, context, surfaced_at,
             response, responded_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (review_id, user_id, "reversal_pattern", trace_id, "{}",
         datetime.now().isoformat(), response, responded_at),
    )
    conn.commit()
    return review_id


def _insert_proposal(conn, user_id, status):
    conn.execute(
        """
        INSERT INTO proposals
            (proposal_id, user_id, type, target_rule_id, proposed_rule,
             rationale, evidence_count, weight, promotion_threshold,
             supporting_traces, first_observed, last_reinforced,
             context_signature, merge_key, delta, status)
        VALUES (?,?,?,NULL,NULL,'',1,1.0,3,'[]',?,?,?,?,NULL,?)
        """,
        (str(uuid.uuid4()), user_id, "add_rule",
         datetime.now().isoformat(), datetime.now().isoformat(),
         "[]", f"add_rule:::{status}::{uuid.uuid4()}", status),
    )
    conn.commit()


# ------------------------------------------------------------------
# Dashboard structure
# ------------------------------------------------------------------

class TestDashboardStructure:
    def test_keys_present_on_empty_data(self):
        conn = _conn()
        stores = _stores(conn)
        result = compute_dashboard("u1", stores)
        assert set(result.keys()) == {
            "user_id", "computed_at", "period_days",
            "coverage", "acceptance", "reversal_resistance",
            "secondary", "latency",
        }

    def test_latency_not_tracked(self):
        conn = _conn()
        stores = _stores(conn)
        result = compute_dashboard("u1", stores)
        assert result["latency"] == {"status": "not_tracked"}

    def test_secondary_keys(self):
        conn = _conn()
        stores = _stores(conn)
        result = compute_dashboard("u1", stores)
        assert set(result["secondary"].keys()) == {
            "confidence_calibration",
            "divergence_rate",
            "off_baseline_precision",
            "anchor_stability",
            "promotion_rejection_rate",
            "bootstrap_exit_time",
        }

    def test_bootstrap_exit_time_not_tracked(self):
        conn = _conn()
        stores = _stores(conn)
        result = compute_dashboard("u1", stores)
        assert result["secondary"]["bootstrap_exit_time"]["status"] == "not_tracked"


# ------------------------------------------------------------------
# Acceptance metric
# ------------------------------------------------------------------

class TestAcceptance:
    def test_empty_returns_none_rate(self):
        conn = _conn()
        stores = _stores(conn)
        result = compute_dashboard("u1", stores)
        assert result["acceptance"]["rate"] is None
        assert result["acceptance"]["total_outcomes"] == 0

    def test_all_accepted(self):
        conn = _conn()
        uid = "u1"
        for _ in range(3):
            _record_outcome(conn, uid, str(uuid.uuid4()), "accepted")
        stores = _stores(conn)
        acc = compute_dashboard(uid, stores)["acceptance"]
        assert acc["rate"] == 1.0
        assert acc["accepted"] == 3
        assert acc["rejected"] == 0

    def test_mixed_outcomes(self):
        conn = _conn()
        uid = "u1"
        _record_outcome(conn, uid, str(uuid.uuid4()), "accepted")
        _record_outcome(conn, uid, str(uuid.uuid4()), "accepted")
        _record_outcome(conn, uid, str(uuid.uuid4()), "rejected")
        _record_outcome(conn, uid, str(uuid.uuid4()), "edited")
        stores = _stores(conn)
        acc = compute_dashboard(uid, stores)["acceptance"]
        assert acc["total_outcomes"] == 4
        assert acc["accepted"] == 2
        assert acc["rejected"] == 1
        assert acc["edited"] == 1
        assert acc["rate"] == pytest.approx(0.5)

    def test_excludes_other_users(self):
        conn = _conn()
        _record_outcome(conn, "other", str(uuid.uuid4()), "accepted")
        stores = _stores(conn)
        acc = compute_dashboard("u1", stores)["acceptance"]
        assert acc["total_outcomes"] == 0
        assert acc["rate"] is None

    def test_excludes_outcomes_outside_window(self):
        conn = _conn()
        uid = "u1"
        old = (datetime.now() - timedelta(days=40)).isoformat()
        _record_outcome(conn, uid, str(uuid.uuid4()), "accepted", reported_at=old)
        stores = _stores(conn)
        acc = compute_dashboard(uid, stores, days=30)["acceptance"]
        assert acc["total_outcomes"] == 0


# ------------------------------------------------------------------
# Coverage & learning_impaired propagation
# ------------------------------------------------------------------

class TestCoverage:
    def test_learning_impaired_propagates_to_acceptance(self):
        conn = _conn()
        uid = "u1"
        # Add many traces but zero outcomes → impaired
        for _ in range(5):
            _save_trace(conn, uid, str(uuid.uuid4()))
        stores = _stores(conn)
        result = compute_dashboard(uid, stores)
        assert result["coverage"]["learning_impaired"] is True
        assert result["acceptance"]["learning_impaired"] is True
        assert result["reversal_resistance"]["learning_impaired"] is True

    def test_learning_impaired_false_when_healthy(self):
        conn = _conn()
        uid = "u1"
        # One trace + one accepted outcome → 100% coverage → healthy
        tid = str(uuid.uuid4())
        _save_trace(conn, uid, tid)
        _record_outcome(conn, uid, tid, "accepted")
        stores = _stores(conn)
        result = compute_dashboard(uid, stores)
        assert result["coverage"]["learning_impaired"] is False
        assert result["acceptance"]["learning_impaired"] is False


# ------------------------------------------------------------------
# Reversal resistance
# ------------------------------------------------------------------

class TestReversalResistance:
    def test_empty_returns_none_rates(self):
        conn = _conn()
        stores = _stores(conn)
        rr = compute_dashboard("u1", stores)["reversal_resistance"]
        assert rr["engagement_rate"] is None
        assert rr["confirmation_rate"] is None
        assert rr["total_prompted"] == 0
        assert rr["reliable"] is False
        assert rr["flag"] == "UNRELIABLE"

    def test_all_confirmed(self):
        conn = _conn()
        uid = "u1"
        for _ in range(4):
            tid = str(uuid.uuid4())
            _save_trace(conn, uid, tid)
            _enqueue_reversal(conn, uid, tid, response="confirmed")
        stores = _stores(conn)
        rr = compute_dashboard(uid, stores)["reversal_resistance"]
        assert rr["total_prompted"] == 4
        assert rr["engaged"] == 4
        assert rr["confirmed"] == 4
        assert rr["engagement_rate"] == pytest.approx(1.0)
        assert rr["confirmation_rate"] == pytest.approx(1.0)
        assert rr["reliable"] is True
        assert rr["flag"] is None

    def test_engagement_below_floor_is_unreliable(self):
        conn = _conn()
        uid = "u1"
        # 10 total, only 2 engaged → engagement_rate = 0.2 < 0.30
        for i in range(10):
            tid = str(uuid.uuid4())
            _save_trace(conn, uid, tid)
            resp = "confirmed" if i < 2 else None
            _enqueue_reversal(conn, uid, tid, response=resp)
        stores = _stores(conn)
        rr = compute_dashboard(uid, stores)["reversal_resistance"]
        assert rr["engagement_rate"] == pytest.approx(0.2)
        assert rr["reliable"] is False
        assert rr["flag"] == "UNRELIABLE"

    def test_engagement_at_floor_is_reliable(self):
        conn = _conn()
        uid = "u1"
        # 10 total, 3 engaged → engagement_rate = 0.30 exactly
        for i in range(10):
            tid = str(uuid.uuid4())
            _save_trace(conn, uid, tid)
            resp = "confirmed" if i < 3 else None
            _enqueue_reversal(conn, uid, tid, response=resp)
        stores = _stores(conn)
        rr = compute_dashboard(uid, stores)["reversal_resistance"]
        assert rr["engagement_rate"] == pytest.approx(REVERSAL_RELIABILITY_FLOOR)
        assert rr["reliable"] is True

    def test_engaged_but_none_confirmed(self):
        conn = _conn()
        uid = "u1"
        for _ in range(4):
            tid = str(uuid.uuid4())
            _save_trace(conn, uid, tid)
            _enqueue_reversal(conn, uid, tid, response="dismissed")
        stores = _stores(conn)
        rr = compute_dashboard(uid, stores)["reversal_resistance"]
        assert rr["engaged"] == 4
        assert rr["confirmed"] == 0
        assert rr["confirmation_rate"] == pytest.approx(0.0)

    def test_no_engagement_none_confirmation_rate(self):
        conn = _conn()
        uid = "u1"
        # Items enqueued but no responses
        for _ in range(5):
            tid = str(uuid.uuid4())
            _save_trace(conn, uid, tid)
            _enqueue_reversal(conn, uid, tid, response=None)
        stores = _stores(conn)
        rr = compute_dashboard(uid, stores)["reversal_resistance"]
        assert rr["total_prompted"] == 5
        assert rr["engaged"] == 0
        assert rr["confirmation_rate"] is None


# ------------------------------------------------------------------
# Confidence calibration
# ------------------------------------------------------------------

class TestConfidenceCalibration:
    def test_four_bins_always_returned(self):
        conn = _conn()
        stores = _stores(conn)
        cal = compute_dashboard("u1", stores)["secondary"]["confidence_calibration"]
        assert len(cal) == 4
        labels = [b["bin_label"] for b in cal]
        assert labels == ["[0.0,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]

    def test_zero_count_bin_has_none_rate(self):
        conn = _conn()
        stores = _stores(conn)
        cal = compute_dashboard("u1", stores)["secondary"]["confidence_calibration"]
        for b in cal:
            assert b["count"] == 0
            assert b["acceptance_rate"] is None

    def test_trace_sorted_into_correct_bin(self):
        conn = _conn()
        uid = "u1"
        tid = str(uuid.uuid4())
        _save_trace(conn, uid, tid, alignment=0.75, reproduction=0.70)
        _record_outcome(conn, uid, tid, "accepted")
        stores = _stores(conn)
        cal = compute_dashboard(uid, stores)["secondary"]["confidence_calibration"]
        # alignment=0.75 → [0.6,0.8) bin
        high_bin = next(b for b in cal if b["bin_label"] == "[0.6,0.8)")
        assert high_bin["count"] == 1
        assert high_bin["acceptance_rate"] == pytest.approx(1.0)

    def test_rejected_outcome_lowers_bin_rate(self):
        conn = _conn()
        uid = "u1"
        tid1 = str(uuid.uuid4())
        tid2 = str(uuid.uuid4())
        _save_trace(conn, uid, tid1, alignment=0.9, reproduction=0.85)
        _save_trace(conn, uid, tid2, alignment=0.95, reproduction=0.90)
        _record_outcome(conn, uid, tid1, "accepted")
        _record_outcome(conn, uid, tid2, "rejected")
        stores = _stores(conn)
        cal = compute_dashboard(uid, stores)["secondary"]["confidence_calibration"]
        top_bin = next(b for b in cal if b["bin_label"] == "[0.8,1.0]")
        assert top_bin["count"] == 2
        assert top_bin["acceptance_rate"] == pytest.approx(0.5)

    def test_trace_without_outcome_excluded(self):
        conn = _conn()
        uid = "u1"
        _save_trace(conn, uid, str(uuid.uuid4()), alignment=0.85, reproduction=0.80)
        # No outcome recorded for this trace
        stores = _stores(conn)
        cal = compute_dashboard(uid, stores)["secondary"]["confidence_calibration"]
        assert all(b["count"] == 0 for b in cal)


# ------------------------------------------------------------------
# Divergence rate
# ------------------------------------------------------------------

class TestDivergenceRate:
    def test_empty_returns_none(self):
        conn = _conn()
        stores = _stores(conn)
        dr = compute_dashboard("u1", stores)["secondary"]["divergence_rate"]
        assert dr["rate"] is None
        assert dr["sample_size"] == 0
        assert dr["alert"] is False

    def test_high_divergence_triggers_no_alert(self):
        conn = _conn()
        uid = "u1"
        # |0.9 - 0.6| = 0.3 > 0.2 → diverged; rate = 1.0 > 0.15 → no alert
        _save_trace(conn, uid, str(uuid.uuid4()), alignment=0.9, reproduction=0.6)
        stores = _stores(conn)
        dr = compute_dashboard(uid, stores)["secondary"]["divergence_rate"]
        assert dr["rate"] == pytest.approx(1.0)
        assert dr["alert"] is False  # rate is NOT below target, no alert needed

    def test_low_divergence_triggers_alert(self):
        conn = _conn()
        uid = "u1"
        # All traces with tiny divergence → rate = 0.0 < 0.15 → alert
        for _ in range(5):
            _save_trace(conn, uid, str(uuid.uuid4()), alignment=0.8, reproduction=0.81)
        stores = _stores(conn)
        dr = compute_dashboard(uid, stores)["secondary"]["divergence_rate"]
        assert dr["rate"] == pytest.approx(0.0)
        assert dr["alert"] is True

    def test_traces_without_both_confidences_excluded(self):
        conn = _conn()
        uid = "u1"
        # Trace with only alignment, no reproduction
        state = {
            "user_id": uid,
            "trace_id": str(uuid.uuid4()),
            "is_off_baseline": False,
            "output_payload": {"confidences": {"alignment": 0.8}},
        }
        tid = state["trace_id"]
        conn.execute(
            "INSERT INTO traces (trace_id, user_id, data, created_at) VALUES (?,?,?,?)",
            (tid, uid, json.dumps(state), datetime.now().isoformat()),
        )
        conn.commit()
        stores = _stores(conn)
        dr = compute_dashboard(uid, stores)["secondary"]["divergence_rate"]
        assert dr["sample_size"] == 0


# ------------------------------------------------------------------
# Off-baseline precision
# ------------------------------------------------------------------

class TestOffBaselinePrecision:
    def test_empty_returns_none(self):
        conn = _conn()
        stores = _stores(conn)
        obp = compute_dashboard("u1", stores)["secondary"]["off_baseline_precision"]
        assert obp["rate"] is None
        assert obp["sample_size"] == 0

    def test_all_off_baseline(self):
        conn = _conn()
        uid = "u1"
        for _ in range(3):
            tid = str(uuid.uuid4())
            _save_trace(conn, uid, tid, is_off_baseline=True)
            _enqueue_reversal(conn, uid, tid)
        stores = _stores(conn)
        obp = compute_dashboard(uid, stores)["secondary"]["off_baseline_precision"]
        assert obp["sample_size"] == 3
        assert obp["rate"] == pytest.approx(1.0)

    def test_none_off_baseline(self):
        conn = _conn()
        uid = "u1"
        for _ in range(3):
            tid = str(uuid.uuid4())
            _save_trace(conn, uid, tid, is_off_baseline=False)
            _enqueue_reversal(conn, uid, tid)
        stores = _stores(conn)
        obp = compute_dashboard(uid, stores)["secondary"]["off_baseline_precision"]
        assert obp["rate"] == pytest.approx(0.0)

    def test_missing_trace_counts_as_not_off_baseline(self):
        conn = _conn()
        uid = "u1"
        # Enqueue reversal for a trace that was never saved
        _enqueue_reversal(conn, uid, "nonexistent-trace-id")
        stores = _stores(conn)
        obp = compute_dashboard(uid, stores)["secondary"]["off_baseline_precision"]
        assert obp["sample_size"] == 1
        assert obp["rate"] == pytest.approx(0.0)


# ------------------------------------------------------------------
# Anchor stability
# ------------------------------------------------------------------

class TestAnchorStability:
    def test_no_anchors_returns_none_revision_rate(self):
        conn = _conn()
        anchor = MagicMock()
        anchor.count_for_user.return_value = 0
        anchor.count_demoted.return_value = 0
        anchor.count_demoted_in_period.return_value = 0
        stores = _stores(conn, anchor_mock=anchor)
        stab = compute_dashboard("u1", stores)["secondary"]["anchor_stability"]
        assert stab["active_count"] == 0
        assert stab["demoted_count"] == 0
        assert stab["revision_rate_per_year"] is None

    def test_with_anchors_computes_annualized_rate(self):
        conn = _conn()
        anchor = MagicMock()
        anchor.count_for_user.return_value = 5
        anchor.count_demoted.return_value = 3
        # 2 demotions in 30 days → (2/30)*365 ≈ 24.33/year
        anchor.count_demoted_in_period.return_value = 2
        stores = _stores(conn, anchor_mock=anchor)
        stab = compute_dashboard("u1", stores, days=30)["secondary"]["anchor_stability"]
        assert stab["active_count"] == 5
        assert stab["demoted_count"] == 3
        expected = (2 / 30) * 365.0
        assert stab["revision_rate_per_year"] == pytest.approx(expected)


# ------------------------------------------------------------------
# Promotion rejection rate
# ------------------------------------------------------------------

class TestPromotionRejectionRate:
    def test_no_proposals_returns_none(self):
        conn = _conn()
        stores = _stores(conn)
        prr = compute_dashboard("u1", stores)["secondary"]["promotion_rejection_rate"]
        assert prr["rate"] is None
        assert prr["promoted"] == 0
        assert prr["discarded"] == 0

    def test_all_promoted(self):
        conn = _conn()
        uid = "u1"
        for _ in range(3):
            _insert_proposal(conn, uid, "promoted")
        stores = _stores(conn)
        prr = compute_dashboard(uid, stores)["secondary"]["promotion_rejection_rate"]
        assert prr["promoted"] == 3
        assert prr["discarded"] == 0
        assert prr["rate"] == pytest.approx(0.0)

    def test_all_discarded(self):
        conn = _conn()
        uid = "u1"
        for _ in range(2):
            _insert_proposal(conn, uid, "discarded")
        stores = _stores(conn)
        prr = compute_dashboard(uid, stores)["secondary"]["promotion_rejection_rate"]
        assert prr["discarded"] == 2
        assert prr["rate"] == pytest.approx(1.0)

    def test_mixed_resolved(self):
        conn = _conn()
        uid = "u1"
        _insert_proposal(conn, uid, "promoted")
        _insert_proposal(conn, uid, "promoted")
        _insert_proposal(conn, uid, "discarded")
        stores = _stores(conn)
        prr = compute_dashboard(uid, stores)["secondary"]["promotion_rejection_rate"]
        assert prr["rate"] == pytest.approx(1 / 3)

    def test_active_proposals_excluded_from_denominator(self):
        conn = _conn()
        uid = "u1"
        _insert_proposal(conn, uid, "active")
        _insert_proposal(conn, uid, "promoted")
        stores = _stores(conn)
        prr = compute_dashboard(uid, stores)["secondary"]["promotion_rejection_rate"]
        # Only promoted + discarded → denominator = 1
        assert prr["promoted"] == 1
        assert prr["discarded"] == 0
        assert prr["rate"] == pytest.approx(0.0)
