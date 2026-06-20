"""
TaskPilot AI — Semantic Deduplication Engine

Two-stage waterfall pipeline that collapses duplicate tasks:

1. **Exact Hash Filter**: Normalizes text and groups by xxHash-64 digest.
   Identical (after normalisation) tasks are merged immediately.
2. **Semantic Filter**: Embeds remaining tasks with a sentence-transformer,
   builds a cosine-similarity matrix, clusters via Union-Find, and merges
   any cluster whose pairwise similarity exceeds the threshold.

The merge strategy is *conservative* — it keeps the most information from
every constituent task so nothing is lost during consolidation.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional

import faiss
import numpy as np
import xxhash
from sentence_transformers import SentenceTransformer

from src.schemas import Task, SeverityLevel

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Module-level model cache
# ────────────────────────────────────────────────────────────
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


# ────────────────────────────────────────────────────────────
# Severity ranking helper
# ────────────────────────────────────────────────────────────

_SEVERITY_RANK: dict[SeverityLevel, int] = {
    SeverityLevel.LOW: 1,
    SeverityLevel.MEDIUM: 2,
    SeverityLevel.HIGH: 3,
    SeverityLevel.CRITICAL: 4,
}


def _severity_rank(severity: SeverityLevel) -> int:
    """Return a numeric rank for *severity* (higher = more severe).

    CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1
    """
    return _SEVERITY_RANK.get(severity, 0)


# ────────────────────────────────────────────────────────────
# Union-Find (Disjoint Set Union)
# ────────────────────────────────────────────────────────────

class UnionFind:
    """Weighted-rank Union-Find with path compression for clustering."""

    def __init__(self, n: int) -> None:
        self.parent: list[int] = list(range(n))
        self.rank: list[int] = [0] * n

    def find(self, x: int) -> int:
        """Find the root representative of *x* with path compression."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Merge the sets containing *x* and *y* by rank."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def get_clusters(self) -> dict[int, list[int]]:
        """Return a mapping of root → list of member indices."""
        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            clusters[self.find(i)].append(i)
        return dict(clusters)


# ────────────────────────────────────────────────────────────
# Text normalisation & hashing
# ────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Lowercase, strip, remove punctuation, and collapse whitespace."""
    text = text.lower().strip()
    text = _PUNCTUATION_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _hash_text(text: str) -> str:
    """Return the xxHash-64 hex digest of *text*."""
    return xxhash.xxh64(text).hexdigest()


# ────────────────────────────────────────────────────────────
# Stage 1 — Exact Hash Filter
# ────────────────────────────────────────────────────────────

def _exact_dedup(tasks: list[Task]) -> list[Task]:
    """Remove exact duplicates by grouping on normalised-text hash.

    Tasks whose ``semantic_text()`` normalises to the same string are
    merged into a single canonical task via :func:`_merge_tasks`.
    """
    if len(tasks) <= 1:
        return list(tasks)

    hash_groups: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        digest = _hash_text(_normalize_text(task.semantic_text()))
        hash_groups[digest].append(task)

    result: list[Task] = []
    exact_dup_count = 0
    for digest, group in hash_groups.items():
        if len(group) > 1:
            merged = _merge_tasks(group)
            result.append(merged)
            exact_dup_count += len(group) - 1
            logger.debug(
                "Exact-hash merge: %d tasks → 1 (hash=%s, titles=%s)",
                len(group),
                digest[:12],
                [t.title for t in group],
            )
        else:
            result.append(group[0])

    logger.info(
        "Stage 1 (Exact Hash): removed %d exact duplicate(s) — %d → %d tasks",
        exact_dup_count,
        len(tasks),
        len(result),
    )
    return result


# ────────────────────────────────────────────────────────────
# Stage 2 — Semantic Filter
# ────────────────────────────────────────────────────────────

def _get_embedding_model() -> SentenceTransformer:
    """Load (and cache) the sentence-transformer model.

    Uses ``all-MiniLM-L6-v2`` which produces 384-dimensional normalised
    embeddings — small, fast, and good enough for task deduplication.
    """
    global _EMBEDDING_MODEL  # noqa: PLW0603
    if _EMBEDDING_MODEL is None:
        logger.info("Loading sentence-transformer model 'all-MiniLM-L6-v2' …")
        _EMBEDDING_MODEL = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Sentence-transformer model loaded successfully.")
    return _EMBEDDING_MODEL


def _compute_embeddings(tasks: list[Task]) -> np.ndarray:
    """Encode each task's ``semantic_text()`` into a normalised 384-d vector.

    Returns
    -------
    np.ndarray
        Shape ``(len(tasks), 384)`` with L2-normalised rows.
    """
    model = _get_embedding_model()
    texts = [task.semantic_text() for task in tasks]
    embeddings: np.ndarray = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings


def _semantic_dedup(
    tasks: list[Task],
    threshold: float = 0.85,
) -> list[Task]:
    """Cluster semantically similar tasks and merge each cluster.

    1. Compute normalised embeddings for all tasks.
    2. Build a cosine-similarity matrix via dot product.
    3. Union-Find any pair whose similarity ≥ *threshold*.
    4. Merge each multi-member cluster with :func:`_merge_tasks`.
    """
    if len(tasks) <= 1:
        return list(tasks)

    # --- Embed ---
    embeddings = _compute_embeddings(tasks)

    # --- FAISS Vector Indexing (Cosine similarity via Inner Product of L2-normalised vectors) ---
    n = len(tasks)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # Perform range search to find all pairs with similarity >= threshold
    lims, D, I = index.range_search(embeddings, threshold)

    # --- Union-Find clustering ---
    uf = UnionFind(n)

    for i in range(n):
        start, end = lims[i], lims[i + 1]
        for j_idx in range(start, end):
            j = I[j_idx]
            sim = D[j_idx]
            if i < j:  # Only log and union once per pair
                logger.debug(
                    "Semantic match (sim=%.4f ≥ %.2f): '%s' ↔ '%s'",
                    sim,
                    threshold,
                    tasks[i].title,
                    tasks[j].title,
                )
                uf.union(i, j)

    clusters = uf.get_clusters()

    # --- Merge clusters ---
    result: list[Task] = []
    semantic_dup_count = 0
    for root, members in clusters.items():
        cluster_tasks = [tasks[idx] for idx in members]
        if len(cluster_tasks) > 1:
            merged = _merge_tasks(cluster_tasks)
            result.append(merged)
            semantic_dup_count += len(cluster_tasks) - 1
            logger.info(
                "Semantic merge: %d tasks → 1 (titles=%s)",
                len(cluster_tasks),
                [t.title for t in cluster_tasks],
            )
        else:
            result.append(cluster_tasks[0])

    logger.info(
        "Stage 2 (Semantic): removed %d semantic duplicate(s) — %d → %d tasks",
        semantic_dup_count,
        len(tasks),
        len(result),
    )
    return result


# ────────────────────────────────────────────────────────────
# Merge logic
# ────────────────────────────────────────────────────────────

def _merge_tasks(tasks: list[Task]) -> Task:
    """Merge a group of duplicate tasks into a single canonical task.

    Merge strategy (conservative — keeps the most information):

    * **title** / **description**: longest string (most descriptive).
    * **severity**: highest rank (CRITICAL > HIGH > MEDIUM > LOW).
    * **deadline**: earliest non-``None`` value.
    * **source_lineage**: union of all lineage lists.
    * **is_blocker**: ``True`` if *any* constituent is a blocker.
    * **impact_score** / **urgency_score**: maximum value.
    * **dependencies** / **labels**: union (deduplicated).
    * **id**: kept from the first task in the list.
    * **raw_text**: concatenation of all raw texts separated by
      ``'\\n---\\n'``.
    """
    if len(tasks) == 1:
        return tasks[0]

    # Title & description — longest wins
    best_title = max(tasks, key=lambda t: len(t.title)).title
    best_description = max(tasks, key=lambda t: len(t.description)).description

    # Severity — highest rank wins
    best_severity = max(tasks, key=lambda t: _severity_rank(t.severity)).severity

    # Deadline — earliest non-None
    deadlines = [t.deadline for t in tasks if t.deadline is not None]
    best_deadline = min(deadlines) if deadlines else None

    # Source lineage — union (preserve order, deduplicate)
    seen_lineage: set[str] = set()
    merged_lineage: list[str] = []
    for task in tasks:
        for ref in task.source_lineage:
            if ref not in seen_lineage:
                seen_lineage.add(ref)
                merged_lineage.append(ref)

    # Boolean OR
    best_is_blocker = any(t.is_blocker for t in tasks)

    # Numeric — take maximums
    best_impact = max(t.impact_score for t in tasks)
    best_urgency = max(t.urgency_score for t in tasks)

    # Dependencies & labels — union (preserve order, deduplicate)
    seen_deps: set[str] = set()
    merged_deps: list[str] = []
    for task in tasks:
        for dep in task.dependencies:
            if dep not in seen_deps:
                seen_deps.add(dep)
                merged_deps.append(dep)

    seen_labels: set[str] = set()
    merged_labels: list[str] = []
    for task in tasks:
        for label in task.labels:
            if label not in seen_labels:
                seen_labels.add(label)
                merged_labels.append(label)

    # Raw text — concatenation
    merged_raw = "\n---\n".join(t.raw_text for t in tasks if t.raw_text)

    # Build merged task — keep the first task's identity fields
    first = tasks[0]
    merged = first.model_copy(
        update={
            "title": best_title,
            "description": best_description,
            "severity": best_severity,
            "deadline": best_deadline,
            "source_lineage": merged_lineage,
            "is_blocker": best_is_blocker,
            "impact_score": best_impact,
            "urgency_score": best_urgency,
            "dependencies": merged_deps,
            "labels": merged_labels,
            "raw_text": merged_raw,
        }
    )
    return merged


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def deduplicate_tasks(
    tasks: list[Task],
    similarity_threshold: float = 0.85,
) -> list[Task]:
    """Run the full two-stage deduplication waterfall.

    Parameters
    ----------
    tasks:
        Raw list of :class:`Task` objects, potentially containing duplicates.
    similarity_threshold:
        Cosine-similarity cutoff for the semantic stage (default 0.85).
        Pairs scoring at or above this value are considered duplicates.

    Returns
    -------
    list[Task]
        Deduplicated list with merged tasks retaining the richest metadata.
    """
    original_count = len(tasks)

    if original_count == 0:
        logger.info("Deduplication skipped: empty task list.")
        return []

    if original_count == 1:
        logger.info("Deduplication skipped: only one task provided.")
        return list(tasks)

    # Stage 1 — Exact hash deduplication
    stage1_result = _exact_dedup(tasks)

    # Stage 2 — Semantic deduplication
    final_result = _semantic_dedup(stage1_result, threshold=similarity_threshold)

    logger.info(
        "Deduplication complete: %d tasks → %d unique tasks",
        original_count,
        len(final_result),
    )
    return final_result
