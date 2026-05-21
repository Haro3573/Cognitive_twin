"""
GovernanceStore: append-only rule store with supersession.

Active rules: rows where superseded_by IS NULL.
Governance mutations go through ProposalQueue → promotion engine;
the request path only reads from here.

query_active_rules applies context filtering:
  - Empty context_scope = universal (applies everywhere)
  - Non-empty context_scope = rule applies if context_scope ∩ context.domain_tags ≠ ∅
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Optional

from .models import GovernanceRuleModel

DEPRECATION_TRIGGER = 0.6  # patch §A4: >60% contradicting traces → propose deprecation


def _row_to_rule(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["structured_form"] = json.loads(d["structured_form"]) if d["structured_form"] else None
    d["context_scope"] = json.loads(d["context_scope"])
    d["supporting_traces"] = json.loads(d["supporting_traces"])
    d["contradicting_traces"] = json.loads(d["contradicting_traces"])
    d["activated_at"] = datetime.fromisoformat(d["activated_at"])
    # superseded_by is an internal column; strip it from the returned dict
    d.pop("superseded_by", None)
    return d


class GovernanceStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # Write (only called by promotion engine, never by request path)
    # ------------------------------------------------------------------

    def add_rule(self, rule: dict) -> str:
        """Inserts a new rule. Returns rule_id."""
        model = GovernanceRuleModel.model_validate(rule)
        rule_id = model.rule_id or str(uuid.uuid4())
        model.rule_id = rule_id

        self._conn.execute(
            """
            INSERT INTO governance_rules
                (rule_id, user_id, version, statement, structured_form, confidence,
                 evidence_count, context_scope, supporting_traces, contradicting_traces,
                 activated_at, supersedes, superseded_by, rule_class)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)
            """,
            (
                model.rule_id,
                model.user_id,
                model.version,
                model.statement,
                json.dumps(model.structured_form) if model.structured_form else None,
                model.confidence,
                model.evidence_count,
                json.dumps(model.context_scope),
                json.dumps(model.supporting_traces),
                json.dumps(model.contradicting_traces),
                model.activated_at.isoformat(),
                model.supersedes,
                model.rule_class,
            ),
        )
        # If this rule supersedes another, mark that rule as superseded
        if model.supersedes:
            self._conn.execute(
                "UPDATE governance_rules SET superseded_by = ? WHERE rule_id = ?",
                (rule_id, model.supersedes),
            )
        self._conn.commit()
        return rule_id

    def reinforce_rule(self, rule_id: str, trace_id: str, weight: float = 1.0) -> None:
        """Append supporting trace, increment evidence_count, nudge confidence up."""
        row = self._conn.execute(
            "SELECT supporting_traces, confidence FROM governance_rules WHERE rule_id = ?",
            (rule_id,),
        ).fetchone()
        if not row:
            return
        traces = json.loads(row["supporting_traces"])
        traces.append(trace_id)
        new_conf = min(1.0, row["confidence"] + weight * 0.05)
        self._conn.execute(
            """
            UPDATE governance_rules
            SET supporting_traces = ?, evidence_count = evidence_count + 1, confidence = ?
            WHERE rule_id = ?
            """,
            (json.dumps(traces), new_conf, rule_id),
        )
        self._conn.commit()

    def add_contradicting_evidence(self, rule_id: str, trace_id: str, weight: float = 1.0) -> None:
        """Append contradicting trace, nudge confidence down."""
        row = self._conn.execute(
            "SELECT contradicting_traces, confidence FROM governance_rules WHERE rule_id = ?",
            (rule_id,),
        ).fetchone()
        if not row:
            return
        traces = json.loads(row["contradicting_traces"])
        traces.append(trace_id)
        new_conf = max(0.0, row["confidence"] - weight * 0.05)
        self._conn.execute(
            """
            UPDATE governance_rules
            SET contradicting_traces = ?, confidence = ?
            WHERE rule_id = ?
            """,
            (json.dumps(traces), new_conf, rule_id),
        )
        self._conn.commit()

    def modify(self, rule_id: str, proposed_rule: dict) -> str:
        """
        Creates a new rule version superseding rule_id with changes from proposed_rule.

        Returns new_rule_id. The old rule gets superseded_by = new_rule_id.
        confidence_adjustment in proposed_rule is applied as a delta.
        """
        old = self.get(rule_id)
        if not old:
            raise ValueError(f"Rule {rule_id!r} not found")

        delta = proposed_rule.get("confidence_adjustment", 0.0)
        new_conf = max(0.0, min(1.0, old["confidence"] + delta))

        new_rule = {
            "rule_id": str(uuid.uuid4()),
            "user_id": old["user_id"],
            "version": old.get("version", 1) + 1,
            "statement": proposed_rule.get("statement") or old["statement"],
            "confidence": new_conf,
            "evidence_count": 0,
            "context_scope": proposed_rule.get("context_scope") or old["context_scope"],
            "supporting_traces": [],
            "contradicting_traces": [],
            "activated_at": datetime.now(),
            "supersedes": rule_id,
            "rule_class": proposed_rule.get("rule_class") or old["rule_class"],
        }
        return self.add_rule(new_rule)

    def deprecate(self, rule_id: str) -> None:
        """Marks rule as deprecated. Deprecated rules are excluded from active queries."""
        self._conn.execute(
            "UPDATE governance_rules SET status = 'deprecated' WHERE rule_id = ?",
            (rule_id,),
        )
        self._conn.commit()

    def rule_contradicting_evidence_ratio(self, rule_id: str) -> float:
        row = self._conn.execute(
            "SELECT supporting_traces, contradicting_traces FROM governance_rules WHERE rule_id = ?",
            (rule_id,),
        ).fetchone()
        if not row:
            return 0.0
        sup = len(json.loads(row["supporting_traces"]))
        con = len(json.loads(row["contradicting_traces"]))
        denom = sup + con
        return con / denom if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # Read (request-path safe — no lock needed)
    # ------------------------------------------------------------------

    def get(self, rule_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM governance_rules WHERE rule_id = ?", (rule_id,)
        ).fetchone()
        return _row_to_rule(row) if row else None

    def query_active_rules(
        self,
        user_id: str,
        context: Optional[dict] = None,
        min_confidence: float = 0.0,
    ) -> list[dict]:
        """
        Returns active rules (not superseded) for user_id.

        Context filtering:
          - Rule with empty context_scope matches any context (universal).
          - Rule with non-empty context_scope matches if it overlaps with
            context.domain_tags (domain overlap).
        """
        rows = self._conn.execute(
            """
            SELECT * FROM governance_rules
            WHERE user_id = ? AND superseded_by IS NULL
              AND confidence >= ?
              AND (status IS NULL OR status = 'active')
            """,
            (user_id, min_confidence),
        ).fetchall()

        rules = [_row_to_rule(r) for r in rows]

        if context is None:
            return rules

        domain_tags = set(context.get("domain_tags", []))
        return [
            r for r in rules
            if not r["context_scope"] or (set(r["context_scope"]) & domain_tags)
        ]

    def current_version(self, user_id: str) -> int:
        """Returns the highest rule version number for user (used as a monotonic stamp)."""
        row = self._conn.execute(
            "SELECT MAX(version) FROM governance_rules WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row[0] or 0

    def count_active_rules(self, user_id: str) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(*) FROM governance_rules
            WHERE user_id = ? AND superseded_by IS NULL
              AND (status IS NULL OR status = 'active')
            """,
            (user_id,),
        ).fetchone()
        return row[0] if row else 0

    def search_all_versions(self, user_id: str) -> list[dict]:
        """Returns all rule versions including superseded (for reasoner history tool)."""
        rows = self._conn.execute(
            "SELECT * FROM governance_rules WHERE user_id = ? ORDER BY version ASC",
            (user_id,),
        ).fetchall()
        return [_row_to_rule(r) for r in rows]
