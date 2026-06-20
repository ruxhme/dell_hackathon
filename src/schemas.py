"""
TaskPilot AI — Pydantic Schema Definitions

Canonical data models enforcing strict validation for the entire pipeline.
Every task, regardless of origin, is normalized into these schemas before
any downstream processing (deduplication, prioritization, or display).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ────────────────────────────────────────────────────────────
# Enumerations
# ────────────────────────────────────────────────────────────

class TaskSource(str, Enum):
    """Origin system from which a task was ingested."""
    JIRA = "jira"
    GITHUB = "github"
    SERVICENOW = "servicenow"
    EMAIL = "email"
    MEETING = "meeting"
    CHAOS_INJECTION = "chaos_injection"


class SeverityLevel(str, Enum):
    """Normalized severity ranking used across all sources."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(str, Enum):
    """Current lifecycle status of a task."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"


# ────────────────────────────────────────────────────────────
# Core Task Model
# ────────────────────────────────────────────────────────────

class Task(BaseModel):
    """
    Canonical task representation.

    Every task entering the pipeline — whether from a Jira board, ServiceNow
    incident, email thread, or meeting transcript — is normalized into this
    schema.  Downstream modules (deduplication, prioritization) depend on
    the structural guarantees enforced here.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique internal identifier for this task.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Concise summary of the task.",
    )
    description: str = Field(
        default="",
        max_length=5000,
        description="Detailed description of what the task entails.",
    )
    source: TaskSource = Field(
        ...,
        description="Origin system that produced this task.",
    )
    source_id: str = Field(
        default="",
        description="Identifier in the origin system (e.g. JIRA-1234, INC00012).",
    )
    source_lineage: list[str] = Field(
        default_factory=list,
        description=(
            "Audit trail of all origin references. Preserved across merges "
            "so every consolidated task can be traced back to its roots."
        ),
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.MEDIUM,
        description="Normalized severity / priority level.",
    )
    urgency_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Normalized urgency score (0 = no rush, 1 = imminent breach).",
    )
    deadline: Optional[datetime] = Field(
        default=None,
        description="Hard deadline or SLA expiry, if applicable.",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs or labels of tasks that this item blocks.",
    )
    is_blocker: bool = Field(
        default=False,
        description="True if this task currently blocks other team members.",
    )
    impact_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Business impact assessment (0 = negligible, 1 = critical).",
    )
    status: TaskStatus = Field(
        default=TaskStatus.OPEN,
        description="Current lifecycle status.",
    )
    assignee: Optional[str] = Field(
        default=None,
        description="Person or team assigned to this task.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the task was first ingested.",
    )
    raw_text: str = Field(
        default="",
        description="Original unprocessed text, preserved for audit.",
    )
    story_points: Optional[int] = Field(
        default=None,
        ge=0,
        description="Estimated effort in story points (agile sources only).",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Tags or categories from the source system.",
    )

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Task title cannot be blank or whitespace-only.")
        return v.strip()

    @field_validator("source_lineage", mode="before")
    @classmethod
    def ensure_lineage_is_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    def semantic_text(self) -> str:
        """Combined text used for embedding and deduplication."""
        return f"{self.title} {self.description}".strip()


# ────────────────────────────────────────────────────────────
# LLM Extraction Output Schemas
# ────────────────────────────────────────────────────────────

class ExtractedTask(BaseModel):
    """
    Schema for a single task extracted from unstructured text by the LLM.

    This is intentionally a subset of the full Task model — the LLM only
    needs to identify the core fields.  The loader will enrich the result
    into a full Task object afterwards.
    """

    title: str = Field(
        ...,
        description="A concise, actionable title for the extracted task.",
    )
    description: str = Field(
        default="",
        description="Detailed description of the task, if available.",
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.MEDIUM,
        description="Assessed severity based on context clues in the text.",
    )
    deadline: Optional[str] = Field(
        default=None,
        description=(
            "Deadline mentioned in the text, as an ISO-8601 string "
            "(e.g. '2026-06-22T17:00:00'). None if no deadline is stated."
        ),
    )
    is_blocker: bool = Field(
        default=False,
        description="True if the text implies this blocks other people or work.",
    )
    impact_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated business impact from 0 (negligible) to 1 (critical).",
    )


class ExtractedTaskList(BaseModel):
    """
    Wrapper for structured LLM extraction output.

    Accommodates the reality that a single email or meeting transcript
    may contain zero, one, or many hidden tasks.
    """

    tasks: list[ExtractedTask] = Field(
        default_factory=list,
        description="List of tasks extracted from the text. Empty if none found.",
    )
    source_summary: str = Field(
        default="",
        description="Brief summary of what the source text was about.",
    )


# ────────────────────────────────────────────────────────────
# Prioritization Output Schema
# ────────────────────────────────────────────────────────────

class PrioritizedTask(BaseModel):
    """
    A task annotated with its computed priority score, rank, and a
    human-readable rationale explaining why it was ranked at this position.
    """

    task: Task = Field(
        ...,
        description="The underlying task object.",
    )
    priority_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Computed priority score (higher = more urgent).",
    )
    rank: int = Field(
        ...,
        ge=1,
        description="Ordinal position in the prioritized plan (1 = top priority).",
    )
    rationale: str = Field(
        default="",
        description=(
            "Natural-language feature-attribution statement explaining "
            "the scoring variables that drove this ranking."
        ),
    )

    # Breakdown of individual scoring components for transparency
    severity_component: float = Field(default=0.0, description="Weighted severity score.")
    urgency_component: float = Field(default=0.0, description="Weighted urgency score.")
    dependency_component: float = Field(default=0.0, description="Weighted dependency score.")
    impact_component: float = Field(default=0.0, description="Weighted impact score.")


# ────────────────────────────────────────────────────────────
# Configuration Schema
# ────────────────────────────────────────────────────────────

class PriorityWeights(BaseModel):
    """Configurable weights for the prioritization formula."""

    severity: float = Field(default=0.35, ge=0.0, le=1.0)
    urgency: float = Field(default=0.30, ge=0.0, le=1.0)
    dependencies: float = Field(default=0.20, ge=0.0, le=1.0)
    impact: float = Field(default=0.15, ge=0.0, le=1.0)

    @field_validator("impact")
    @classmethod
    def weights_must_sum_to_one(cls, v, info):
        total = (
            info.data.get("severity", 0.35)
            + info.data.get("urgency", 0.30)
            + info.data.get("dependencies", 0.20)
            + v
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Priority weights must sum to 1.0, got {total:.2f}. "
                "Adjust weights so they total exactly 1.0."
            )
        return v