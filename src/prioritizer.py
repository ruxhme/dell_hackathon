"""
TaskPilot AI — Deterministic Prioritization Module

Implements a RICE-inspired weighted scoring formula to rank tasks:

    P(t) = w_s · S(t)  +  w_u · U(t)  +  w_d · D(t)  +  w_i · I(t)

where each component captures one dimension of priority:
    S  – severity (categorical → numeric mapping)
    U  – urgency  (deadline-aware exponential decay or pre-computed score)
    D  – dependency (binary blocker flag)
    I  – impact   (pre-computed business-impact score)

After deterministic ranking, an LLM generates concise natural-language
rationales for the top-N tasks (at most 10) to explain *why* each task
landed at its position.  Remaining tasks receive formulaic rationales
derived directly from the numeric breakdown.
"""

import logging
import math
import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.schemas import (
    Task,
    PrioritizedTask,
    PriorityWeights,
    SeverityLevel,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Scoring Component Functions
# ────────────────────────────────────────────────────────────

# Mapping from categorical severity levels to numeric scores.
_SEVERITY_MAP: dict[SeverityLevel, float] = {
    SeverityLevel.CRITICAL: 1.0,
    SeverityLevel.HIGH: 0.75,
    SeverityLevel.MEDIUM: 0.5,
    SeverityLevel.LOW: 0.25,
}


def _severity_score(severity: SeverityLevel) -> float:
    """Convert a categorical severity level to a normalized numeric score.

    Args:
        severity: The severity level of the task.

    Returns:
        A float in [0.25, 1.0] representing the severity score.
    """
    return _SEVERITY_MAP.get(severity, 0.5)


def _urgency_score(task: Task) -> float:
    """Compute time-aware urgency using exponential decay against a 72-hour window.

    When a task has an explicit deadline the score ramps from 0 → 1 as the
    deadline approaches, with an exponential boost kicking in below 12 hours.
    Tasks without a deadline fall back to their pre-computed ``urgency_score``.

    Args:
        task: The task to evaluate.

    Returns:
        A float in [0.0, 1.0] representing how urgent the task is right now.
    """
    if task.deadline is not None:
        hours_remaining = max(
            0.0,
            (task.deadline - datetime.now(task.deadline.tzinfo)).total_seconds() / 3600.0,
        )

        # Linear ramp over a 72-hour planning window
        score = max(0.0, min(1.0, 1.0 - (hours_remaining / 72.0)))

        # Exponential boost for tasks inside the 12-hour danger zone
        score = min(1.0, score * (1.0 + math.exp(-hours_remaining / 6.0 + 2.0) * 0.3))

        return round(score, 6)

    # No deadline — rely on the value already set by the source loader.
    return task.urgency_score


def _dependency_score(task: Task) -> float:
    """Return a binary dependency score based on the blocker flag.

    Args:
        task: The task to evaluate.

    Returns:
        1.0 if the task is a blocker, 0.0 otherwise.
    """
    return 1.0 if task.is_blocker else 0.0


def _impact_score(task: Task) -> float:
    """Return the pre-computed business-impact score.

    Args:
        task: The task to evaluate.

    Returns:
        The task's impact_score field, a float in [0.0, 1.0].
    """
    return task.impact_score


# ────────────────────────────────────────────────────────────
# Composite Scoring
# ────────────────────────────────────────────────────────────


def compute_priority_score(
    task: Task,
    weights: PriorityWeights,
) -> tuple[float, dict[str, float]]:
    """Apply the weighted scoring formula and return both the total and the breakdown.

    The formula is:
        P(t) = w_s · S(t)  +  w_u · U(t)  +  w_d · D(t)  +  w_i · I(t)

    Each component is multiplied by its corresponding weight and the weighted
    values are returned alongside the total for downstream transparency.

    Args:
        task: The task to score.
        weights: The weight configuration to apply.

    Returns:
        A tuple of (total_score, breakdown_dict) where breakdown_dict contains
        the individually weighted component values keyed as ``severity``,
        ``urgency``, ``dependency``, and ``impact``.
    """
    raw_s = _severity_score(task.severity)
    raw_u = _urgency_score(task)
    raw_d = _dependency_score(task)
    raw_i = _impact_score(task)

    weighted_s = weights.severity * raw_s
    weighted_u = weights.urgency * raw_u
    weighted_d = weights.dependencies * raw_d
    weighted_i = weights.impact * raw_i

    total = weighted_s + weighted_u + weighted_d + weighted_i

    # Clamp to [0, 1] to satisfy the PrioritizedTask schema constraint.
    total = max(0.0, min(1.0, round(total, 6)))

    breakdown = {
        "severity": round(weighted_s, 6),
        "urgency": round(weighted_u, 6),
        "dependency": round(weighted_d, 6),
        "impact": round(weighted_i, 6),
    }

    return total, breakdown


# ────────────────────────────────────────────────────────────
# Ranking
# ────────────────────────────────────────────────────────────


def rank_tasks(
    tasks: list[Task],
    weights: Optional[PriorityWeights] = None,
) -> list[PrioritizedTask]:
    """Score, sort, and rank a list of tasks without generating rationales.

    This is the pure-deterministic step of the pipeline.  Every task receives
    a numeric priority score and an ordinal rank (1 = highest priority).

    Args:
        tasks: The tasks to rank.
        weights: Optional custom weight configuration. Falls back to
            ``PriorityWeights()`` defaults (severity=0.35, urgency=0.30,
            dependencies=0.20, impact=0.15) when omitted.

    Returns:
        A list of ``PrioritizedTask`` objects sorted by descending
        priority_score, with ranks assigned starting at 1.
        Rationale fields are left empty at this stage.
    """
    if not tasks:
        logger.warning("rank_tasks called with an empty task list.")
        return []

    effective_weights = weights or PriorityWeights()
    logger.info(
        "Ranking %d tasks with weights: severity=%.2f, urgency=%.2f, "
        "dependencies=%.2f, impact=%.2f",
        len(tasks),
        effective_weights.severity,
        effective_weights.urgency,
        effective_weights.dependencies,
        effective_weights.impact,
    )

    scored: list[tuple[Task, float, dict[str, float]]] = []
    for task in tasks:
        total, breakdown = compute_priority_score(task, effective_weights)
        scored.append((task, total, breakdown))

    # Stable sort descending by total score.
    scored.sort(key=lambda item: item[1], reverse=True)

    prioritized: list[PrioritizedTask] = []
    for rank, (task, total, breakdown) in enumerate(scored, start=1):
        prioritized.append(
            PrioritizedTask(
                task=task,
                priority_score=total,
                rank=rank,
                rationale="",
                severity_component=breakdown["severity"],
                urgency_component=breakdown["urgency"],
                dependency_component=breakdown["dependency"],
                impact_component=breakdown["impact"],
            )
        )

    logger.info("Ranking complete. Top score: %.4f, Bottom score: %.4f",
                prioritized[0].priority_score,
                prioritized[-1].priority_score)

    return prioritized


# ────────────────────────────────────────────────────────────
# LLM Rationale Generation
# ────────────────────────────────────────────────────────────

_LLM_RATIONALE_LIMIT = 10  # Only call the LLM for the top N tasks.


def _get_llm() -> ChatGroq:
    """Instantiate the Groq LLM client used for rationale generation.

    Returns:
        A ``ChatGroq`` instance configured with low temperature
        for deterministic-leaning outputs.

    Raises:
        ValueError: If the ``GROQ_API_KEY`` environment variable is missing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Cannot initialise LLM for rationale generation."
        )

    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0.2,
        api_key=api_key,
    )


def _build_rationale_prompt() -> ChatPromptTemplate:
    """Build the prompt template used to generate ranking rationales.

    Returns:
        A ``ChatPromptTemplate`` with placeholders for task metadata and
        scoring breakdown.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a task prioritization analyst. Given a task and its "
                "scoring breakdown, generate a concise 1-2 sentence rationale "
                "explaining WHY this task is ranked at this position. Reference "
                "the specific scoring variables (severity, urgency, dependencies, "
                "impact) and their values. Be precise and data-driven.",
            ),
            (
                "human",
                "Task: {task_title}\n"
                "Description: {task_description}\n"
                "Rank: {rank} of {total_tasks}\n"
                "Priority Score: {priority_score:.4f}\n"
                "Scoring Breakdown:\n"
                "  - Severity component:   {severity_component:.4f}\n"
                "  - Urgency component:    {urgency_component:.4f}\n"
                "  - Dependency component: {dependency_component:.4f}\n"
                "  - Impact component:     {impact_component:.4f}\n\n"
                "Generate a concise rationale for this ranking.",
            ),
        ]
    )


def _formulaic_rationale(pt: PrioritizedTask, total_tasks: int) -> str:
    """Generate a deterministic, formulaic rationale from the scoring breakdown.

    Used for tasks outside the top-N LLM window, or as a fallback when the
    LLM is unavailable.

    Args:
        pt: The prioritized task to describe.
        total_tasks: Total number of tasks in the ranked list (for context).

    Returns:
        A human-readable rationale string.
    """
    parts: list[str] = []

    # Identify the dominant scoring driver for a more informative rationale.
    components = {
        "severity": pt.severity_component,
        "urgency": pt.urgency_component,
        "dependency": pt.dependency_component,
        "impact": pt.impact_component,
    }
    dominant = max(components, key=components.get)  # type: ignore[arg-type]

    parts.append(
        f"Ranked #{pt.rank} of {total_tasks} with a priority score of "
        f"{pt.priority_score:.4f}."
    )
    parts.append(
        f"Primary driver: {dominant} ({components[dominant]:.4f}). "
        f"Breakdown — severity: {pt.severity_component:.4f}, "
        f"urgency: {pt.urgency_component:.4f}, "
        f"dependency: {pt.dependency_component:.4f}, "
        f"impact: {pt.impact_component:.4f}."
    )

    return " ".join(parts)


def generate_rationales(
    prioritized_tasks: list[PrioritizedTask],
) -> list[PrioritizedTask]:
    """Enrich prioritized tasks with natural-language rationales.

    For the top-N tasks (capped at ``_LLM_RATIONALE_LIMIT``), the LLM is
    invoked to produce an insightful, human-readable explanation.  All
    remaining tasks — and *all* tasks when the API key is absent — receive
    a formulaic, deterministic rationale derived from the numeric breakdown.

    Args:
        prioritized_tasks: Ranked tasks (must already be sorted by rank).

    Returns:
        The same list with ``.rationale`` fields populated.
    """
    if not prioritized_tasks:
        return prioritized_tasks

    total_tasks = len(prioritized_tasks)

    # Attempt LLM initialisation; fall back gracefully on failure.
    llm = None
    prompt = None
    try:
        llm = _get_llm()
        prompt = _build_rationale_prompt()
        logger.info(
            "LLM available — generating rationales for top %d tasks.",
            min(_LLM_RATIONALE_LIMIT, total_tasks),
        )
    except (ValueError, Exception) as exc:
        logger.warning(
            "LLM unavailable (%s). All tasks will receive formulaic rationales.",
            exc,
        )

    for pt in prioritized_tasks:
        # Top-N tasks get LLM rationales if the model is available.
        if llm is not None and prompt is not None and pt.rank <= _LLM_RATIONALE_LIMIT:
            try:
                chain = prompt | llm
                response = chain.invoke(
                    {
                        "task_title": pt.task.title,
                        "task_description": pt.task.description or "(no description)",
                        "rank": pt.rank,
                        "total_tasks": total_tasks,
                        "priority_score": pt.priority_score,
                        "severity_component": pt.severity_component,
                        "urgency_component": pt.urgency_component,
                        "dependency_component": pt.dependency_component,
                        "impact_component": pt.impact_component,
                    }
                )
                pt.rationale = response.content.strip()
                logger.debug(
                    "LLM rationale for rank #%d (%s): %s",
                    pt.rank,
                    pt.task.title,
                    pt.rationale[:80],
                )
            except Exception as exc:
                logger.error(
                    "LLM rationale generation failed for '%s' (rank #%d): %s. "
                    "Falling back to formulaic rationale.",
                    pt.task.title,
                    pt.rank,
                    exc,
                )
                pt.rationale = _formulaic_rationale(pt, total_tasks)
        else:
            # Outside top-N window or no LLM — use the formulaic fallback.
            pt.rationale = _formulaic_rationale(pt, total_tasks)

    logger.info("Rationale generation complete for %d tasks.", total_tasks)
    return prioritized_tasks


# ────────────────────────────────────────────────────────────
# Public API — Full Pipeline
# ────────────────────────────────────────────────────────────


def prioritize_tasks(
    tasks: list[Task],
    weights: Optional[PriorityWeights] = None,
) -> list[PrioritizedTask]:
    """End-to-end prioritization pipeline: score → rank → explain.

    This is the primary public entry point for the prioritization module.
    It chains the deterministic ranking step with the LLM-augmented rationale
    generation to produce a fully annotated, rank-ordered task list.

    Args:
        tasks: Raw (unranked) tasks from any loader.
        weights: Optional custom weight configuration.  Uses sensible defaults
            when omitted.

    Returns:
        A list of ``PrioritizedTask`` objects sorted by descending priority,
        each annotated with a human-readable rationale.
    """
    if not tasks:
        logger.warning("prioritize_tasks called with no tasks. Returning empty list.")
        return []

    logger.info("Starting prioritization pipeline for %d tasks.", len(tasks))

    # Step 1: Deterministic scoring and ranking.
    ranked = rank_tasks(tasks, weights)

    # Step 2: Augment with natural-language rationales.
    enriched = generate_rationales(ranked)

    # Log the top 3 tasks for operational visibility.
    top_n = min(3, len(enriched))
    for pt in enriched[:top_n]:
        logger.info(
            "  #%d  [%.4f]  %s  (sev=%.3f  urg=%.3f  dep=%.3f  imp=%.3f)",
            pt.rank,
            pt.priority_score,
            pt.task.title[:60],
            pt.severity_component,
            pt.urgency_component,
            pt.dependency_component,
            pt.impact_component,
        )

    logger.info("Prioritization pipeline complete. %d tasks ranked.", len(enriched))
    return enriched
