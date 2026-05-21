"""
ProposalQueue: staging area between meta_learn_node and the promotion engine.

All governance mutations are staged here. The promotion engine (async, not yet built)
reviews proposals and promotes/discards. The request path never writes to
GovernanceStore directly.

Key invariant (patch §B1, load-bearing):
  Merge key = type::target_rule_id::context_signature
  Same key → upsert (accumulate evidence).
  Different key → new proposal.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Literal, Optional

from .models import ProposalModel

BASE_THRESHOLD = 3


def context_signature(context: dict) -> list[str]:
    """Deterministic signature for a context dict. Used as merge key dimension."""
    sig = sorted(context.get("domain_tags", []))
    if context.get("situation_type"):
        sig.append(f"situation:{context['situation_type']}")
    return sig


def proposal_merge_key(proposal_type: str, target_rule_id: Optional[str], ctx_sig: list[str]) -> str:
    """Merge key: type::target_rule_id::context_signature (pipe-joined, sorted)."""
    ctx_str = "|".join(sorted(ctx_sig))
    return f"{proposal_type}::{target_rule_id or ''}::{ctx_str}"


def compute_promotion_threshold(
    proposal_type: str,
    target_rule_confidence: Optional[float],
) -> int:
    """
    Adaptive threshold from §6.3. Computed once at first insert, never on upsert.

    modify_rule:  BASE * (1 + 2 * rule_confidence) — harder to modify high-confidence rules
    add_rule:     BASE * 1.5 (~4-5)
    deprecate_rule: BASE * 2 * rule_confidence + 1 — deprecation needs strong evidence
    adjust_weight: BASE (3)
    investigate_*: 1 (accumulators, not gated)
    """
    conf = target_rule_confidence or 0.0

    if proposal_type == "modify_rule":
        return int(BASE_THRESHOLD * (1 + 2 * conf))
    elif proposal_type == "add_rule":
        return int(BASE_THRESHOLD * 1.5)
    elif proposal_type == "deprecate_rule":
        return int(BASE_THRESHOLD * 2 * conf) + 1
    elif proposal_type == "adjust_weight":
        return BASE_THRESHOLD
    else:  # investigate_divergence, investigate_rule, investigate_new_rule
        return 1


def _row_to_proposal(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["proposed_rule"] = json.loads(d["proposed_rule"]) if d["proposed_rule"] else None
    d["supporting_traces"] = json.loads(d["supporting_traces"])
    d["context_signature"] = json.loads(d["context_signature"])
    d["first_observed"] = datetime.fromisoformat(d["first_observed"])
    d["last_reinforced"] = datetime.fromisoformat(d["last_reinforced"])
    return d


class ProposalQueue:
    def __init__(self, conn: sqlite3.Connection, governance_store=None) -> None:
        self._conn = conn
        # governance_store is optional; used to look up rule confidence for threshold.
        # Injected to avoid circular imports (governance_store does not import this module).
        self._gov = governance_store

    def _rule_confidence(self, rule_id: Optional[str]) -> Optional[float]:
        if rule_id is None or self._gov is None:
            return None
        rule = self._gov.get(rule_id)
        return rule["confidence"] if rule else None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, user_id: str, proposal: dict) -> dict:
        """
        Upserts a proposal by merge key.

        Callers pass raw `context` dict; this method computes context_signature
        and merge_key internally. Existing proposals are updated in-place;
        new proposals get a fresh entry with computed promotion_threshold.
        """
        ctx_sig = context_signature(proposal.get("context", {}))
        p_type = proposal["type"]
        target_rule_id = proposal.get("target_rule_id")
        key = proposal_merge_key(p_type, target_rule_id, ctx_sig)
        now = datetime.now()

        existing = self._find_by_key(user_id, key)
        if existing:
            # Upsert: accumulate evidence
            traces = existing["supporting_traces"]
            traces.extend(proposal.get("supporting_traces", []))
            new_weight = max(existing["weight"], proposal.get("weight", 1.0))
            new_proposed_rule = proposal.get("proposed_rule") or existing.get("proposed_rule")

            self._conn.execute(
                """
                UPDATE proposals
                SET evidence_count = evidence_count + 1,
                    supporting_traces = ?,
                    last_reinforced = ?,
                    weight = ?,
                    proposed_rule = ?
                WHERE proposal_id = ?
                """,
                (
                    json.dumps(traces),
                    now.isoformat(),
                    new_weight,
                    json.dumps(new_proposed_rule) if new_proposed_rule else None,
                    existing["proposal_id"],
                ),
            )
            self._conn.commit()
            return self.get(existing["proposal_id"])

        # New proposal
        proposal_id = str(uuid.uuid4())
        threshold = compute_promotion_threshold(
            p_type, self._rule_confidence(target_rule_id)
        )

        self._conn.execute(
            """
            INSERT INTO proposals
                (proposal_id, user_id, type, target_rule_id, proposed_rule,
                 rationale, evidence_count, weight, promotion_threshold,
                 supporting_traces, first_observed, last_reinforced,
                 context_signature, merge_key, delta, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                proposal_id,
                user_id,
                p_type,
                target_rule_id,
                json.dumps(proposal.get("proposed_rule")) if proposal.get("proposed_rule") else None,
                proposal.get("rationale", ""),
                1,
                proposal.get("weight", 1.0),
                threshold,
                json.dumps(proposal.get("supporting_traces", [])),
                now.isoformat(),
                now.isoformat(),
                json.dumps(ctx_sig),
                key,
                proposal.get("delta"),
                "active",
            ),
        )
        self._conn.commit()
        return self.get(proposal_id)

    def update(self, proposal: dict) -> None:
        """Updates status and mutable fields of an existing proposal."""
        self._conn.execute(
            """
            UPDATE proposals
            SET status = ?, proposed_rule = ?, last_reinforced = ?
            WHERE proposal_id = ?
            """,
            (
                proposal["status"],
                json.dumps(proposal.get("proposed_rule")) if proposal.get("proposed_rule") else None,
                datetime.now().isoformat(),
                proposal["proposal_id"],
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, proposal_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
        return _row_to_proposal(row) if row else None

    def _find_by_key(self, user_id: str, merge_key: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM proposals WHERE user_id = ? AND merge_key = ? AND status = 'active'",
            (user_id, merge_key),
        ).fetchone()
        return _row_to_proposal(row) if row else None

    def list(
        self,
        user_id: str,
        p_type: Optional[str] = None,
        status: str = "active",
    ) -> list[dict]:
        if p_type:
            rows = self._conn.execute(
                "SELECT * FROM proposals WHERE user_id = ? AND type = ? AND status = ?",
                (user_id, p_type, status),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM proposals WHERE user_id = ? AND status = ?",
                (user_id, status),
            ).fetchall()
        return [_row_to_proposal(r) for r in rows]

    def count_active(self, user_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM proposals WHERE user_id = ? AND status = 'active'",
            (user_id,),
        ).fetchone()
        return row[0] if row else 0

    def count_by_status(self, user_id: str) -> dict:
        """Returns {status: count} for all known proposal statuses."""
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM proposals WHERE user_id = ? GROUP BY status",
            (user_id,),
        ).fetchall()
        result: dict = {"active": 0, "promoted": 0, "discarded": 0,
                        "superseded_by_add_rule_proposal": 0}
        for row in rows:
            status = row["status"]
            if status in result:
                result[status] = row["n"]
        return result

    def eligible_for_review(self, user_id: str) -> list[dict]:
        """Returns active proposals that have reached their promotion threshold."""
        rows = self._conn.execute(
            """
            SELECT * FROM proposals
            WHERE user_id = ? AND status = 'active'
              AND evidence_count >= promotion_threshold
            ORDER BY last_reinforced DESC
            """,
            (user_id,),
        ).fetchall()
        return [_row_to_proposal(r) for r in rows]

    def delete_stale(self, user_id: str, older_than_days: int = 30, min_evidence: int = 2) -> int:
        """Discards proposals older than N days with insufficient evidence."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
        result = self._conn.execute(
            """
            UPDATE proposals SET status = 'discarded'
            WHERE user_id = ? AND status = 'active'
              AND first_observed < ? AND evidence_count < ?
            """,
            (user_id, cutoff, min_evidence),
        )
        self._conn.commit()
        return result.rowcount
