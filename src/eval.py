"""
Eval harness: computes a metrics dashboard per spec §13.

compute_dashboard(user_id, stores, days=30) -> dict

All reads are from existing stores — no new tables are created. The result is
a point-in-time snapshot; callers may log or render it but this module does
not persist it.
"""

from datetime import datetime

from src.deps import Stores

DIVERGENCE_THRESHOLD = 0.2
DIVERGENCE_ALERT_THRESHOLD = 0.15
REVERSAL_RELIABILITY_FLOOR = 0.30
CONFIDENCE_BINS = [0.0, 0.4, 0.6, 0.8, 1.01]
_BIN_LABELS = ["[0.0,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]


def compute_dashboard(user_id: str, stores: Stores, days: int = 30) -> dict:
    """
    Returns a metrics snapshot for user_id over the last `days` days.

    All numeric rates are float | None (None when denominator is zero).
    """
    coverage = stores.outcomes.assess_coverage(user_id, stores.traces)
    learning_impaired = coverage["state"] == "impaired"

    return {
        "user_id": user_id,
        "computed_at": datetime.now().isoformat(),
        "period_days": days,
        "coverage": {**coverage, "learning_impaired": learning_impaired},
        "acceptance": _acceptance(user_id, stores, days, learning_impaired),
        "reversal_resistance": _reversal_resistance(user_id, stores, learning_impaired),
        "secondary": _secondary(user_id, stores, days),
        "latency": {"status": "not_tracked"},
    }


# ------------------------------------------------------------------
# Primary metrics
# ------------------------------------------------------------------

def _acceptance(user_id: str, stores: Stores, days: int, learning_impaired: bool) -> dict:
    outcomes = stores.outcomes.list_recent(user_id, days)
    total = len(outcomes)
    accepted = sum(1 for o in outcomes if o["outcome_type"] == "accepted")
    edited = sum(1 for o in outcomes if o["outcome_type"] == "edited")
    rejected = sum(1 for o in outcomes if o["outcome_type"] == "rejected")
    return {
        "rate": accepted / total if total > 0 else None,
        "total_outcomes": total,
        "accepted": accepted,
        "edited": edited,
        "rejected": rejected,
        "learning_impaired": learning_impaired,
    }


def _reversal_resistance(user_id: str, stores: Stores, learning_impaired: bool) -> dict:
    items = stores.reviews.list_by_type(user_id, "reversal_pattern")
    total_prompted = len(items)
    engaged = sum(1 for i in items if i.get("responded_at") is not None)
    confirmed = sum(1 for i in items if i.get("response") == "confirmed")

    engagement_rate = engaged / total_prompted if total_prompted > 0 else None
    confirmation_rate = confirmed / engaged if engaged > 0 else None

    reliable = engagement_rate is not None and engagement_rate >= REVERSAL_RELIABILITY_FLOOR

    return {
        "engagement_rate": engagement_rate,
        "confirmation_rate": confirmation_rate,
        "total_prompted": total_prompted,
        "engaged": engaged,
        "confirmed": confirmed,
        "reliable": reliable,
        "flag": None if reliable else "UNRELIABLE",
        "learning_impaired": learning_impaired,
    }


# ------------------------------------------------------------------
# Secondary metrics
# ------------------------------------------------------------------

def _secondary(user_id: str, stores: Stores, days: int) -> dict:
    return {
        "confidence_calibration": _confidence_calibration(user_id, stores, days),
        "divergence_rate": _divergence_rate(user_id, stores, days),
        "off_baseline_precision": _off_baseline_precision(user_id, stores),
        "anchor_stability": _anchor_stability(user_id, stores, days),
        "promotion_rejection_rate": _promotion_rejection_rate(user_id, stores),
        "bootstrap_exit_time": {
            "status": "not_tracked",
            "reason": "no user creation timestamp",
        },
    }


def _confidence_calibration(user_id: str, stores: Stores, days: int) -> list[dict]:
    traces = stores.traces.list_recent(user_id, days)
    outcomes_by_trace = {
        o["trace_id"]: o["outcome_type"]
        for o in stores.outcomes.list_recent(user_id, days)
    }

    bins = [
        {"low": CONFIDENCE_BINS[i], "high": CONFIDENCE_BINS[i + 1], "accepted": 0, "total": 0}
        for i in range(len(CONFIDENCE_BINS) - 1)
    ]

    for trace in traces:
        output_payload = trace.get("output_payload") or {}
        confidences = output_payload.get("confidences") or {}
        alignment = confidences.get("alignment")
        if alignment is None:
            continue

        outcome_type = outcomes_by_trace.get(trace.get("trace_id"))
        if outcome_type is None:
            continue

        for b in bins:
            if b["low"] <= alignment < b["high"]:
                b["total"] += 1
                if outcome_type == "accepted":
                    b["accepted"] += 1
                break

    return [
        {
            "bin_label": _BIN_LABELS[i],
            "low": b["low"],
            "high": b["high"],
            "acceptance_rate": b["accepted"] / b["total"] if b["total"] > 0 else None,
            "count": b["total"],
        }
        for i, b in enumerate(bins)
    ]


def _divergence_rate(user_id: str, stores: Stores, days: int) -> dict:
    traces = stores.traces.list_recent(user_id, days)
    sample_size = 0
    diverged = 0

    for trace in traces:
        output_payload = trace.get("output_payload") or {}
        confidences = output_payload.get("confidences") or {}
        alignment = confidences.get("alignment")
        reproduction = confidences.get("reproduction")
        if alignment is None or reproduction is None:
            continue
        sample_size += 1
        if abs(alignment - reproduction) > DIVERGENCE_THRESHOLD:
            diverged += 1

    rate = diverged / sample_size if sample_size > 0 else None
    return {
        "rate": rate,
        "sample_size": sample_size,
        "alert": rate is not None and rate < DIVERGENCE_ALERT_THRESHOLD,
        "target": DIVERGENCE_ALERT_THRESHOLD,
    }


def _off_baseline_precision(user_id: str, stores: Stores) -> dict:
    items = stores.reviews.list_by_type(user_id, "reversal_pattern")
    sample_size = len(items)
    if sample_size == 0:
        return {"rate": None, "sample_size": 0}

    off_baseline = sum(
        1
        for item in items
        if _trace_is_off_baseline(item["item_id"], stores)
    )
    return {"rate": off_baseline / sample_size, "sample_size": sample_size}


def _trace_is_off_baseline(trace_id: str, stores: Stores) -> bool:
    trace = stores.traces.get(trace_id)
    return bool(trace and trace.get("is_off_baseline"))


def _anchor_stability(user_id: str, stores: Stores, days: int) -> dict:
    active_count = stores.anchors.count_for_user(user_id)
    demoted_count = stores.anchors.count_demoted(user_id)
    demoted_in_period = stores.anchors.count_demoted_in_period(user_id, days)

    if active_count + demoted_count == 0:
        revision_rate = None
    else:
        revision_rate = (demoted_in_period / days) * 365.0 if days > 0 else None

    return {
        "active_count": active_count,
        "demoted_count": demoted_count,
        "revision_rate_per_year": revision_rate,
    }


def _promotion_rejection_rate(user_id: str, stores: Stores) -> dict:
    counts = stores.proposals.count_by_status(user_id)
    promoted = counts.get("promoted", 0)
    discarded = counts.get("discarded", 0)
    total_resolved = promoted + discarded
    return {
        "rate": discarded / total_resolved if total_resolved > 0 else None,
        "promoted": promoted,
        "discarded": discarded,
    }
