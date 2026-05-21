"""
Async background loops for the Cognitive Twin (spec §12 / patch §B3).

Each loop is a standalone function — no scheduler is bundled here.
The caller (cron, script, or test) invokes them on demand.

Execution order (loosely):
  1. outcome_processor  — digest raw outcomes into evidence
  2. promotion_engine   — promote eligible proposals into governance rules
  3. memory_decay       — age and compress stale shards
  4. anchor_consolidation — surface dormant/contradicted anchors for review
  5. reversal_reviewer  — surface rejection patterns for user confirmation
"""
