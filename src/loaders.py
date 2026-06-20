"""
TaskPilot AI — Data Loaders

Ingests tasks from heterogeneous source systems (JIRA, ServiceNow, email,
meeting transcripts, chaos-injection) and normalises each record into the
canonical ``Task`` schema.

Design principles
-----------------
* **Never crash the pipeline.**  Every file-read and record-parse is
  wrapped in try / except.  Malformed records are logged and skipped.
* **Deterministic mapping.**  Priority / urgency / status values are
  converted through explicit lookup tables — no implicit fall-throughs.
* **Pure functions.**  Loaders have no side-effects beyond reading
  the filesystem and emitting log messages.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.schemas import (
    SeverityLevel,
    Task,
    TaskSource,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"

# Mapping tables — kept module-level for clarity and testability.
_JIRA_PRIORITY_MAP: dict[str, SeverityLevel] = {
    "Highest": SeverityLevel.CRITICAL,
    "High": SeverityLevel.HIGH,
    "Medium": SeverityLevel.MEDIUM,
    "Low": SeverityLevel.LOW,
}

_SERVICENOW_URGENCY_MAP: dict[int, SeverityLevel] = {
    1: SeverityLevel.CRITICAL,
    2: SeverityLevel.HIGH,
    3: SeverityLevel.MEDIUM,
    4: SeverityLevel.LOW,
}

_JIRA_STATUS_MAP: dict[str, TaskStatus] = {
    "To Do": TaskStatus.OPEN,
    "Open": TaskStatus.OPEN,
    "Backlog": TaskStatus.OPEN,
    "In Progress": TaskStatus.IN_PROGRESS,
    "In Review": TaskStatus.IN_PROGRESS,
    "In Development": TaskStatus.IN_PROGRESS,
    "Blocked": TaskStatus.BLOCKED,
    "Done": TaskStatus.DONE,
    "Closed": TaskStatus.DONE,
    "Resolved": TaskStatus.DONE,
}

_SEVERITY_IMPACT_MAP: dict[SeverityLevel, float] = {
    SeverityLevel.CRITICAL: 0.95,
    SeverityLevel.HIGH: 0.80,
    SeverityLevel.MEDIUM: 0.50,
    SeverityLevel.LOW: 0.30,
}

_SERVICENOW_IMPACT_MAP: dict[int, float] = {
    1: 0.95,
    2: 0.75,
    3: 0.50,
    4: 0.25,
}


# ────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────


def _read_json(filename: str) -> list | dict:
    """Read and parse a JSON file from ``DATA_DIR``.

    Parameters
    ----------
    filename:
        Name of the JSON file relative to *DATA_DIR*.

    Returns
    -------
    list | dict
        Parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.
    """
    filepath: Path = DATA_DIR / filename
    logger.debug("Loading data file: %s", filepath)
    with filepath.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _map_jira_priority(priority: str) -> SeverityLevel:
    """Map a JIRA priority string to the canonical ``SeverityLevel``.

    Parameters
    ----------
    priority:
        Raw priority label from the JIRA payload
        (e.g. ``"Highest"``, ``"High"``, ``"Medium"``, ``"Low"``).

    Returns
    -------
    SeverityLevel
        Corresponding severity enum member, defaulting to ``MEDIUM``
        for unrecognised values.
    """
    severity = _JIRA_PRIORITY_MAP.get(priority)
    if severity is None:
        logger.warning(
            "Unknown JIRA priority '%s' — defaulting to MEDIUM.", priority
        )
        return SeverityLevel.MEDIUM
    return severity


def _map_servicenow_urgency(urgency: int) -> SeverityLevel:
    """Map a ServiceNow integer urgency value to ``SeverityLevel``.

    Parameters
    ----------
    urgency:
        Numeric urgency code (1 = most urgent, 4 = least).

    Returns
    -------
    SeverityLevel
        Corresponding severity enum member, defaulting to ``MEDIUM``.
    """
    severity = _SERVICENOW_URGENCY_MAP.get(urgency)
    if severity is None:
        logger.warning(
            "Unknown ServiceNow urgency %r — defaulting to MEDIUM.", urgency
        )
        return SeverityLevel.MEDIUM
    return severity


def _map_jira_status(status: str) -> TaskStatus:
    """Map a JIRA status string to the canonical ``TaskStatus``.

    Parameters
    ----------
    status:
        Raw status label from the JIRA payload
        (e.g. ``"To Do"``, ``"In Progress"``, ``"Done"``).

    Returns
    -------
    TaskStatus
        Corresponding status enum member, defaulting to ``OPEN``.
    """
    mapped = _JIRA_STATUS_MAP.get(status)
    if mapped is None:
        logger.warning(
            "Unknown JIRA status '%s' — defaulting to OPEN.", status
        )
        return TaskStatus.OPEN
    return mapped


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Best-effort ISO-8601 datetime parse.

    Parameters
    ----------
    value:
        A datetime string (ISO-8601) or ``None``.

    Returns
    -------
    Optional[datetime]
        The parsed datetime, or ``None`` if parsing fails or *value* is falsy.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        logger.warning("Could not parse datetime '%s': %s", value, exc)
        return None


def _compute_urgency_from_deadline(deadline: Optional[datetime]) -> float:
    """Compute a normalised urgency score based on deadline proximity.

    The formula is::

        urgency = max(0, 1 - (hours_remaining / 72))

    …clamped to the ``[0, 1]`` range.  If no deadline is provided the
    function returns a neutral ``0.5``.

    Parameters
    ----------
    deadline:
        The task deadline (timezone-aware or naive).

    Returns
    -------
    float
        Urgency score in ``[0.0, 1.0]``.
    """
    if deadline is None:
        return 0.5

    now = datetime.now(tz=deadline.tzinfo if deadline.tzinfo else None)
    delta = deadline - now
    hours_remaining = delta.total_seconds() / 3600.0
    urgency = 1.0 - (hours_remaining / 72.0)
    return max(0.0, min(1.0, urgency))


# ────────────────────────────────────────────────────────────
# Public loaders
# ────────────────────────────────────────────────────────────


def load_jira_tasks() -> list[Task]:
    """Load and normalise JIRA tasks from ``data/jira_tasks.json``.

    Each JSON object is mapped to a canonical :class:`Task` with:

    * ``source`` = :attr:`TaskSource.JIRA`
    * ``source_id`` = the JIRA issue key
    * ``source_lineage`` = ``["jira://<key>"]``
    * ``severity`` derived from the JIRA *priority* field
    * ``deadline`` parsed from *due_date*
    * ``is_blocker`` set when the *dependencies* list is non-empty
    * ``urgency_score`` computed from deadline proximity
    * ``impact_score`` inferred from severity

    Returns
    -------
    list[Task]
        Successfully parsed tasks.  Malformed records are logged and
        skipped — the pipeline is never interrupted.
    """
    try:
        raw_tasks: list[dict] = _read_json("jira_tasks.json")  # type: ignore[assignment]
    except FileNotFoundError:
        logger.error("JIRA data file not found at %s/jira_tasks.json", DATA_DIR)
        return []
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in jira_tasks.json: %s", exc)
        return []

    tasks: list[Task] = []

    for idx, item in enumerate(raw_tasks):
        try:
            key: str = item.get("key", f"UNKNOWN-{idx}")
            severity = _map_jira_priority(item.get("priority", "Medium"))
            deadline = _parse_datetime(item.get("due_date"))
            dependencies: list[str] = item.get("dependencies", [])
            status = _map_jira_status(item.get("status", "Open"))

            task = Task(
                title=item.get("summary", item.get("title", f"JIRA Task {key}")),
                description=item.get("description", ""),
                source=TaskSource.JIRA,
                source_id=key,
                source_lineage=[f"jira://{key}"],
                severity=severity,
                urgency_score=_compute_urgency_from_deadline(deadline),
                deadline=deadline,
                dependencies=dependencies,
                is_blocker=len(dependencies) > 0,
                impact_score=_SEVERITY_IMPACT_MAP.get(severity, 0.5),
                status=status,
                assignee=item.get("assignee"),
                story_points=item.get("story_points"),
                labels=item.get("labels", []),
                raw_text=json.dumps(item, default=str),
            )
            tasks.append(task)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping JIRA record at index %d: %s", idx, exc
            )

    logger.info("Loaded %d JIRA task(s) from jira_tasks.json.", len(tasks))
    return tasks


def load_servicenow_incidents() -> list[Task]:
    """Load and normalise ServiceNow incidents from ``data/service_now_incidents.json``.

    Each JSON object is mapped to a canonical :class:`Task` with:

    * ``source`` = :attr:`TaskSource.SERVICENOW`
    * ``source_id`` = the incident number
    * ``source_lineage`` = ``["servicenow://<incident_number>"]``
    * ``severity`` derived from the ServiceNow *urgency* field
    * ``deadline`` parsed from *sla_deadline*
    * ``impact_score`` derived from the numeric *impact* field

    Returns
    -------
    list[Task]
        Successfully parsed tasks.
    """
    try:
        raw_incidents: list[dict] = _read_json("service_now_incidents.json")  # type: ignore[assignment]
    except FileNotFoundError:
        logger.error(
            "ServiceNow data file not found at %s/service_now_incidents.json",
            DATA_DIR,
        )
        return []
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in service_now_incidents.json: %s", exc)
        return []

    tasks: list[Task] = []

    for idx, item in enumerate(raw_incidents):
        try:
            incident_number: str = item.get(
                "incident_number", item.get("number", f"INC-UNKNOWN-{idx}")
            )
            urgency_raw = item.get("urgency", 3)
            severity = _map_servicenow_urgency(int(urgency_raw))
            deadline = _parse_datetime(item.get("sla_deadline"))
            impact_raw = item.get("impact", 3)
            impact_score = _SERVICENOW_IMPACT_MAP.get(
                int(impact_raw), 0.50
            )
            dependencies: list[str] = item.get("dependencies", [])

            task = Task(
                title=item.get(
                    "short_description",
                    item.get("title", f"ServiceNow Incident {incident_number}"),
                ),
                description=item.get("description", ""),
                source=TaskSource.SERVICENOW,
                source_id=incident_number,
                source_lineage=[f"servicenow://{incident_number}"],
                severity=severity,
                urgency_score=_compute_urgency_from_deadline(deadline),
                deadline=deadline,
                dependencies=dependencies,
                is_blocker=len(dependencies) > 0,
                impact_score=impact_score,
                status=TaskStatus.OPEN,
                assignee=item.get("assigned_to", item.get("assignee")),
                labels=item.get("labels", []),
                raw_text=json.dumps(item, default=str),
            )
            tasks.append(task)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping ServiceNow record at index %d: %s", idx, exc
            )

    logger.info(
        "Loaded %d ServiceNow incident(s) from service_now_incidents.json.",
        len(tasks),
    )
    return tasks


def load_emails_raw() -> list[dict]:
    """Load raw email payloads from ``data/emails.json``.

    The records are returned as plain dicts — they are meant to be fed
    to an LLM for extraction rather than parsed directly into :class:`Task`
    objects.

    Returns
    -------
    list[dict]
        Raw email dicts.  An empty list is returned if the file is
        missing or malformed.
    """
    try:
        data = _read_json("emails.json")
        if not isinstance(data, list):
            logger.warning("emails.json: expected a JSON array, got %s.", type(data).__name__)
            return []
        logger.info("Loaded %d raw email record(s) from emails.json.", len(data))
        return data  # type: ignore[return-value]
    except FileNotFoundError:
        logger.error("Email data file not found at %s/emails.json", DATA_DIR)
        return []
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in emails.json: %s", exc)
        return []


def load_meeting_transcripts_raw() -> list[dict]:
    """Load raw meeting-transcript payloads from ``data/meeting_transcripts.json``.

    Like :func:`load_emails_raw`, results are returned as plain dicts
    intended for LLM-based extraction.

    Returns
    -------
    list[dict]
        Raw transcript dicts.  An empty list on file/parse errors.
    """
    try:
        data = _read_json("meeting_transcripts.json")
        if not isinstance(data, list):
            logger.warning(
                "meeting_transcripts.json: expected a JSON array, got %s.",
                type(data).__name__,
            )
            return []
        logger.info(
            "Loaded %d raw meeting transcript(s) from meeting_transcripts.json.",
            len(data),
        )
        return data  # type: ignore[return-value]
    except FileNotFoundError:
        logger.error(
            "Meeting transcript file not found at %s/meeting_transcripts.json",
            DATA_DIR,
        )
        return []
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in meeting_transcripts.json: %s", exc)
        return []


def load_chaos_defect() -> Task:
    """Load the synthetic chaos-injection defect from ``data/chaos_defect.json``.

    This always produces a **single** :class:`Task` with maximum severity
    and urgency, representing a surprise production defect injected to
    stress-test the prioritisation engine.

    Returns
    -------
    Task
        A Task with ``source`` = :attr:`TaskSource.CHAOS_INJECTION`,
        ``severity`` = CRITICAL, ``urgency_score`` = 1.0, and
        ``impact_score`` = 1.0.

    Raises
    ------
    FileNotFoundError
        Propagated if the chaos defect file does not exist (intentional —
        chaos injection is mandatory when configured).
    """
    try:
        raw: dict = _read_json("chaos_defect.json")  # type: ignore[assignment]
    except FileNotFoundError:
        logger.error(
            "Chaos defect file not found at %s/chaos_defect.json", DATA_DIR
        )
        raise
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in chaos_defect.json: %s", exc)
        raise

    defect_id: str = raw.get("id", raw.get("defect_id", "CHAOS-001"))
    deadline = _parse_datetime(raw.get("deadline", raw.get("sla_deadline")))

    return Task(
        title=raw.get("title", raw.get("summary", "Chaos Injection Defect")),
        description=raw.get("description", ""),
        source=TaskSource.CHAOS_INJECTION,
        source_id=defect_id,
        source_lineage=[f"chaos://{defect_id}"],
        severity=SeverityLevel.CRITICAL,
        urgency_score=1.0,
        deadline=deadline,
        dependencies=raw.get("dependencies", []),
        is_blocker=True,
        impact_score=1.0,
        status=TaskStatus.OPEN,
        assignee=raw.get("assignee"),
        labels=raw.get("labels", ["chaos-injection"]),
        raw_text=json.dumps(raw, default=str),
    )


def load_all_structured_tasks() -> list[Task]:
    """Load and combine all structured task sources.

    Currently merges:

    * JIRA tasks  (via :func:`load_jira_tasks`)
    * ServiceNow incidents  (via :func:`load_servicenow_incidents`)

    Unstructured sources (emails, meeting transcripts) require LLM
    extraction and are *not* included here.

    Returns
    -------
    list[Task]
        Combined list of tasks from all structured sources.
    """
    jira_tasks = load_jira_tasks()
    servicenow_tasks = load_servicenow_incidents()

    combined: list[Task] = jira_tasks + servicenow_tasks
    logger.info(
        "Combined %d structured task(s) (%d JIRA + %d ServiceNow).",
        len(combined),
        len(jira_tasks),
        len(servicenow_tasks),
    )
    return combined
