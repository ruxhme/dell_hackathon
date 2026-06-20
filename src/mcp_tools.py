"""
TaskPilot AI — FastMCP Tool Server

Exposes the core pipeline operations as MCP-compliant tools that the
LangGraph agent can invoke autonomously.  Each @mcp.tool decorated
function is automatically schema-inspected and made available to the
language model via the Model Context Protocol.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastmcp import FastMCP

from src.schemas import (
    Task,
    PrioritizedTask,
    PriorityWeights,
    TaskSource,
)
from src.loaders import (
    load_all_structured_tasks,
    load_emails_raw,
    load_meeting_transcripts_raw,
    load_chaos_defect,
)
from src.extractor import extract_all_unstructured
from src.deduplicator import deduplicate_tasks
from src.prioritizer import prioritize_tasks

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# MCP Server Instance
# ────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="TaskPilot Tools",
    instructions=(
        "Pipeline tools for the TaskPilot AI agentic task prioritization "
        "system.  Provides ingestion, deduplication, prioritization, and "
        "chaos-injection capabilities."
    ),
)


# ────────────────────────────────────────────────────────────
# In-Memory State (shared across tool invocations)
# ────────────────────────────────────────────────────────────

_state: dict = {
    "raw_tasks": [],
    "deduplicated_tasks": [],
    "prioritized_tasks": [],
    "chaos_injected": False,
}


def get_state() -> dict:
    """Return the current pipeline state (for agent/UI access)."""
    return _state


def reset_state() -> None:
    """Reset all pipeline state to initial values."""
    _state["raw_tasks"] = []
    _state["deduplicated_tasks"] = []
    _state["prioritized_tasks"] = []
    _state["chaos_injected"] = False


# ────────────────────────────────────────────────────────────
# MCP Tools
# ────────────────────────────────────────────────────────────

@mcp.tool()
def ingest_all_tasks() -> str:
    """
    Ingest tasks from ALL data sources.

    Loads structured tasks from Jira and ServiceNow, then extracts hidden
    tasks from emails and meeting transcripts using LLM-powered parsing.
    Returns a summary of ingested items.
    """
    logger.info("🔄 Starting full task ingestion pipeline...")

    # Phase 1: Structured sources
    structured_tasks = load_all_structured_tasks()
    logger.info(f"  ✅ Loaded {len(structured_tasks)} structured tasks (Jira + ServiceNow)")

    # Phase 2: Unstructured sources (LLM extraction)
    emails = load_emails_raw()
    meetings = load_meeting_transcripts_raw()
    extracted_tasks = extract_all_unstructured(emails, meetings)
    logger.info(f"  ✅ Extracted {len(extracted_tasks)} tasks from unstructured sources")

    # Combine
    all_tasks = structured_tasks + extracted_tasks
    _state["raw_tasks"] = all_tasks

    summary = (
        f"Ingestion complete. Discovered {len(all_tasks)} total tasks:\n"
        f"  • {len(structured_tasks)} from structured sources "
        f"(Jira + ServiceNow)\n"
        f"  • {len(extracted_tasks)} extracted from unstructured text "
        f"(emails + meetings)\n\n"
        f"Sources breakdown:\n"
    )

    source_counts: dict[str, int] = {}
    for t in all_tasks:
        source_counts[t.source.value] = source_counts.get(t.source.value, 0) + 1
    for src, count in sorted(source_counts.items()):
        summary += f"  • {src}: {count} tasks\n"

    logger.info(f"📊 {summary}")
    return summary


@mcp.tool()
def deduplicate_current_tasks(similarity_threshold: float = 0.85) -> str:
    """
    Run the two-stage deduplication engine on the current task list.

    Stage 1: Exact hash matching (xxhash) for identical text.
    Stage 2: Semantic similarity (all-MiniLM-L6-v2 + FAISS) for
             conceptually overlapping tasks.

    Args:
        similarity_threshold: Cosine similarity threshold for semantic
                              matching. Default 0.85 (conservative).
    """
    if not _state["raw_tasks"]:
        return "No tasks to deduplicate. Run ingest_all_tasks first."

    original_count = len(_state["raw_tasks"])
    logger.info(
        f"🔄 Running deduplication on {original_count} tasks "
        f"(threshold={similarity_threshold})..."
    )

    deduped = deduplicate_tasks(
        _state["raw_tasks"],
        similarity_threshold=similarity_threshold,
    )
    _state["deduplicated_tasks"] = deduped

    removed = original_count - len(deduped)
    summary = (
        f"Deduplication complete:\n"
        f"  • Input: {original_count} tasks\n"
        f"  • Output: {len(deduped)} unique tasks\n"
        f"  • Removed: {removed} duplicates "
        f"({removed / original_count * 100:.1f}% reduction)\n"
        f"  • Similarity threshold: {similarity_threshold}\n\n"
        f"Merged tasks (with multi-source lineage):\n"
    )

    for t in deduped:
        if len(t.source_lineage) > 1:
            summary += (
                f"  • \"{t.title}\" — merged from "
                f"{len(t.source_lineage)} sources: "
                f"{', '.join(t.source_lineage)}\n"
            )

    logger.info(f"📊 {summary}")
    return summary


@mcp.tool()
def prioritize_current_tasks(
    weight_severity: float = 0.35,
    weight_urgency: float = 0.30,
    weight_dependencies: float = 0.20,
    weight_impact: float = 0.15,
) -> str:
    """
    Run deterministic prioritization on the current deduplicated task list.

    Uses a weighted RICE-inspired formula:
    P(t) = w_s·S(t) + w_u·U(t) + w_d·D(t) + w_i·I(t)

    Args:
        weight_severity:     Weight for severity dimension (default 0.35).
        weight_urgency:      Weight for urgency / deadline (default 0.30).
        weight_dependencies: Weight for blocker status (default 0.20).
        weight_impact:       Weight for business impact (default 0.15).
    """
    tasks_to_prioritize = (
        _state["deduplicated_tasks"]
        if _state["deduplicated_tasks"]
        else _state["raw_tasks"]
    )

    if not tasks_to_prioritize:
        return "No tasks to prioritize. Run ingest_all_tasks first."

    weights = PriorityWeights(
        severity=weight_severity,
        urgency=weight_urgency,
        dependencies=weight_dependencies,
        impact=weight_impact,
    )

    logger.info(
        f"🔄 Prioritizing {len(tasks_to_prioritize)} tasks with weights: "
        f"S={weight_severity}, U={weight_urgency}, "
        f"D={weight_dependencies}, I={weight_impact}..."
    )

    prioritized = prioritize_tasks(tasks_to_prioritize, weights=weights)
    _state["prioritized_tasks"] = prioritized

    summary = f"📋 **Prioritized Daily Plan** ({len(prioritized)} tasks):\n\n"
    for pt in prioritized:
        severity_icon = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }.get(pt.task.severity.value, "⚪")

        deadline_str = (
            pt.task.deadline.strftime("%b %d, %H:%M")
            if pt.task.deadline
            else "No deadline"
        )

        summary += (
            f"**#{pt.rank}** {severity_icon} **{pt.task.title}**\n"
            f"  Score: {pt.priority_score:.3f} | "
            f"Severity: {pt.task.severity.value.upper()} | "
            f"Deadline: {deadline_str} | "
            f"Blocker: {'Yes' if pt.task.is_blocker else 'No'}\n"
            f"  📝 {pt.rationale}\n"
            f"  🔗 Sources: {', '.join(pt.task.source_lineage)}\n\n"
        )

    logger.info(f"✅ Prioritization complete. Top task: {prioritized[0].task.title}")
    return summary


@mcp.tool()
def inject_chaos_defect() -> str:
    """
    Simulate a critical production incident injection mid-session.

    Loads a pre-configured P0 defect (production database failure) and
    injects it into the current pipeline.  Triggers immediate
    re-deduplication and re-prioritization of all tasks.

    This demonstrates the system's dynamic adaptability.
    """
    logger.info("🔥 CHAOS INJECTION: Loading critical production defect...")

    chaos_task = load_chaos_defect()
    _state["raw_tasks"].append(chaos_task)
    _state["chaos_injected"] = True

    # Re-run deduplication on the augmented dataset
    deduped = deduplicate_tasks(_state["raw_tasks"])
    _state["deduplicated_tasks"] = deduped

    # Re-run prioritization
    prioritized = prioritize_tasks(deduped)
    _state["prioritized_tasks"] = prioritized

    # Build alert
    alert = (
        "⚠️ **CRITICAL ALERT: New P0 Defect Detected!**\n\n"
        f"**{chaos_task.title}**\n"
        f"{chaos_task.description}\n\n"
        f"• Severity: 🔴 CRITICAL\n"
        f"• SLA Breach: {chaos_task.deadline.strftime('%b %d, %H:%M') if chaos_task.deadline else 'IMMINENT'}\n"
        f"• Impact: System-wide outage\n\n"
        "The daily plan has been **completely reprioritized** to accommodate "
        "this impending SLA breach.\n\n"
        "---\n\n"
    )

    # Append updated plan
    alert += f"📋 **Updated Prioritized Plan** ({len(prioritized)} tasks):\n\n"
    for pt in prioritized[:5]:  # Show top 5
        severity_icon = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }.get(pt.task.severity.value, "⚪")

        alert += (
            f"**#{pt.rank}** {severity_icon} **{pt.task.title}** "
            f"(Score: {pt.priority_score:.3f})\n"
            f"  {pt.rationale}\n\n"
        )

    if len(prioritized) > 5:
        alert += f"... and {len(prioritized) - 5} more tasks\n"

    logger.info("🔥 Chaos injection complete. Plan reprioritized.")
    return alert


@mcp.tool()
def get_task_details(task_title: str) -> str:
    """
    Retrieve full details for a specific task by title (partial match).

    Args:
        task_title: Full or partial title of the task to look up.
    """
    # Search in prioritized tasks first, then deduplicated, then raw
    search_pools = [
        _state["prioritized_tasks"],
        _state["deduplicated_tasks"],
        _state["raw_tasks"],
    ]

    for pool in search_pools:
        for item in pool:
            task = item.task if isinstance(item, PrioritizedTask) else item
            if task_title.lower() in task.title.lower():
                detail = (
                    f"📋 **Task Details: {task.title}**\n\n"
                    f"• **ID:** {task.id}\n"
                    f"• **Source:** {task.source.value}\n"
                    f"• **Severity:** {task.severity.value.upper()}\n"
                    f"• **Status:** {task.status.value}\n"
                    f"• **Assignee:** {task.assignee or 'Unassigned'}\n"
                    f"• **Deadline:** {task.deadline.strftime('%Y-%m-%d %H:%M') if task.deadline else 'None'}\n"
                    f"• **Is Blocker:** {'Yes' if task.is_blocker else 'No'}\n"
                    f"• **Impact Score:** {task.impact_score:.2f}\n"
                    f"• **Urgency Score:** {task.urgency_score:.2f}\n"
                    f"• **Story Points:** {task.story_points or 'N/A'}\n"
                    f"• **Labels:** {', '.join(task.labels) if task.labels else 'None'}\n"
                    f"• **Dependencies:** {', '.join(task.dependencies) if task.dependencies else 'None'}\n"
                    f"• **Source Lineage:** {', '.join(task.source_lineage)}\n\n"
                    f"**Description:**\n{task.description}\n"
                )

                if isinstance(item, PrioritizedTask):
                    detail += (
                        f"\n**Priority Analysis:**\n"
                        f"• Rank: #{item.rank}\n"
                        f"• Score: {item.priority_score:.3f}\n"
                        f"• Severity Component: {item.severity_component:.3f}\n"
                        f"• Urgency Component: {item.urgency_component:.3f}\n"
                        f"• Dependency Component: {item.dependency_component:.3f}\n"
                        f"• Impact Component: {item.impact_component:.3f}\n"
                        f"• Rationale: {item.rationale}\n"
                    )

                return detail

    return f"No task found matching '{task_title}'."


@mcp.tool()
def get_daily_plan() -> str:
    """
    Execute the complete end-to-end pipeline and return the optimized
    daily execution plan.

    Pipeline: Ingest → Deduplicate → Prioritize → Format Plan

    This is the primary entry point for generating a fresh daily plan
    from scratch.
    """
    logger.info("🚀 Executing full end-to-end pipeline for daily plan...")

    # Step 1: Ingest
    ingest_summary = ingest_all_tasks()

    # Step 2: Deduplicate
    dedup_summary = deduplicate_current_tasks()

    # Step 3: Prioritize
    priority_summary = prioritize_current_tasks()

    # Combine into comprehensive report
    report = (
        "# 🎯 TaskPilot AI — Your Optimized Daily Plan\n\n"
        "---\n\n"
        "## 📥 Data Ingestion\n"
        f"{ingest_summary}\n\n"
        "---\n\n"
        "## 🔍 Deduplication Results\n"
        f"{dedup_summary}\n\n"
        "---\n\n"
        "## 📋 Prioritized Execution Order\n"
        f"{priority_summary}\n"
    )

    logger.info("✅ Full pipeline execution complete.")
    return report


@mcp.tool()
def get_pipeline_status() -> str:
    """
    Return the current state of the pipeline for diagnostic purposes.
    Shows counts of tasks at each stage and whether chaos has been injected.
    """
    status = (
        "📊 **Pipeline Status**\n\n"
        f"• Raw tasks ingested: {len(_state['raw_tasks'])}\n"
        f"• After deduplication: {len(_state['deduplicated_tasks'])}\n"
        f"• Prioritized tasks: {len(_state['prioritized_tasks'])}\n"
        f"• Chaos injected: {'🔥 Yes' if _state['chaos_injected'] else '❌ No'}\n"
    )
    return status
