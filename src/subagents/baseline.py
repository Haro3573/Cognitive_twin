"""
compute_baseline_deviation: cosine similarity between recent shards and anchors.

Algorithm (spec §6.3):
  1. If len(anchors) < 5 OR len(recent_shards) < 10 → return 0.0
  2. For each anchor: find K=5 most similar shards by cosine similarity
  3. anchor_consistency = mean of those similarities
  4. overall_consistency = mean of per-anchor consistencies
  5. deviation = 1.0 − overall_consistency, clamped to [0, 1]
"""

import math
from typing import Optional

_K = 5


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def compute_baseline_deviation(
    shards: list[dict],
    anchors: list[dict],
    min_anchors: int = 5,
    min_shards: int = 10,
) -> float:
    """Returns deviation score in [0, 1]. Returns 0.0 when data is insufficient."""
    shard_embeddings = [s["embedding"] for s in shards if s.get("embedding")]
    anchor_embeddings = [a["embedding"] for a in anchors if a.get("embedding")]

    if len(anchor_embeddings) < min_anchors or len(shard_embeddings) < min_shards:
        return 0.0

    per_anchor: list[float] = []
    for anc_emb in anchor_embeddings:
        sims = sorted(
            (_cosine(anc_emb, s_emb) for s_emb in shard_embeddings),
            reverse=True,
        )[:_K]
        per_anchor.append(sum(sims) / len(sims))

    overall_consistency = sum(per_anchor) / len(per_anchor)
    return max(0.0, min(1.0, 1.0 - overall_consistency))
