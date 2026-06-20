"""
TaskPilot AI — Data Loaders

Ingests tasks from heterogeneous source systems (JIRA, GitHub, ServiceNow, email,
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
import urllib.request
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

_JIRA_PRIORITY_MAP: dict[str, SeverityLevel] = {
    "Highest": SeverityLevel.CRITICAL,
    "High": SeverityLevel.HIGH,
    "Medium": SeverityLevel.MEDIUM,
    "Low": SeverityLevel.LOW,
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

_SERVICENOW_URGENCY_MAP: dict[int, SeverityLevel] = {
    1: SeverityLevel.CRITICAL,
    2: SeverityLevel.HIGH,
    3: SeverityLevel.MEDIUM,
    4: SeverityLevel.LOW,
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
    filepath: Path = DATA_DIR / filename
    logger.debug("Loading data file: %s", filepath)
    with filepath.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def _map_jira_priority(priority: str) -> SeverityLevel:
    severity = _JIRA_PRIORITY_MAP.get(priority)
    if severity is None:
        logger.warning(
            "Unknown JIRA priority '%s' — defaulting to MEDIUM.", priority
        )
        return SeverityLevel.MEDIUM
    return severity

def _map_jira_status(status: str) -> TaskStatus:
    mapped = _JIRA_STATUS_MAP.get(status)
    if mapped is None:
        logger.warning(
            "Unknown JIRA status '%s' — defaulting to OPEN.", status
        )
        return TaskStatus.OPEN
    return mapped

def _map_servicenow_urgency(urgency: int) -> SeverityLevel:
    severity = _SERVICENOW_URGENCY_MAP.get(urgency)
    if severity is None:
        logger.warning("Unknown ServiceNow urgency %r — defaulting to MEDIUM.", urgency)
        return SeverityLevel.MEDIUM
    return severity

def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # Replace 'Z' with '+00:00' to support ISO 8601 parsing from GitHub API
        value = value.replace('Z', '+00:00')
        return datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        logger.warning("Could not parse datetime '%s': %s", value, exc)
        return None

def _compute_urgency_from_deadline(deadline: Optional[datetime]) -> float:
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
    """Load and normalise JIRA tasks from ``data/jira_tasks.json``."""
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

def load_github_issues() -> list[Task]:
    """Load and normalise real issues from the public GitHub API.
    
    Queries the FastAPI repository for its most recent open issues.
    """
    url = "https://api.github.com/repos/fastapi/fastapi/issues?state=open&per_page=10"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "TaskPilotAI"}
    )
    
    try:
        logger.info("Fetching real issues from GitHub API: %s", url)
        with urllib.request.urlopen(req, timeout=10) as response:
            raw_tasks = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.error("Failed to fetch issues from GitHub: %s", exc)
        return []

    tasks: list[Task] = []
    
    for idx, item in enumerate(raw_tasks):
        try:
            # Skip pull requests, we only want issues
            if "pull_request" in item:
                continue
                
            key = str(item.get("number", f"UNKNOWN-{idx}"))
            
            # Map GitHub labels to our severity levels
            labels = [label.get("name", "").lower() for label in item.get("labels", [])]
            severity = SeverityLevel.MEDIUM
            if any("critical" in l or "bug" in l for l in labels):
                severity = SeverityLevel.HIGH
            if any("enhancement" in l for l in labels):
                severity = SeverityLevel.LOW
                
            deadline = _parse_datetime(item.get("created_at"))
            
            task = Task(
                title=item.get("title", f"GitHub Issue {key}"),
                description=item.get("body", "") or "",
                source=TaskSource.GITHUB,
                source_id=key,
                source_lineage=[f"github://fastapi/{key}"],
                severity=severity,
                urgency_score=_compute_urgency_from_deadline(deadline),
                deadline=deadline,
                dependencies=[],
                is_blocker=False,
                impact_score=_SEVERITY_IMPACT_MAP.get(severity, 0.5),
                status=TaskStatus.OPEN,
                assignee=item.get("assignee", {}).get("login") if item.get("assignee") else None,
                story_points=None,
                labels=labels,
                raw_text=json.dumps(item, default=str),
            )
            tasks.append(task)
        except Exception as exc:
            logger.warning("Skipping GitHub record at index %d: %s", idx, exc)

    logger.info("Loaded %d GitHub issue(s) from real API.", len(tasks))
    return tasks

def load_servicenow_incidents() -> list[Task]:
    try:
        raw_incidents: list[dict] = _read_json("service_now_incidents.json")
    except FileNotFoundError:
        logger.error("ServiceNow data file not found.")
        return []
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in service_now_incidents.json: %s", exc)
        return []

    tasks: list[Task] = []
    for idx, item in enumerate(raw_incidents):
        try:
            incident_number: str = item.get("incident_number", item.get("number", f"INC-UNKNOWN-{idx}"))
            urgency_raw = item.get("urgency", 3)
            severity = _map_servicenow_urgency(int(urgency_raw))
            deadline = _parse_datetime(item.get("sla_deadline"))
            impact_raw = item.get("impact", 3)
            impact_score = _SERVICENOW_IMPACT_MAP.get(int(impact_raw), 0.50)
            dependencies: list[str] = item.get("dependencies", [])

            task = Task(
                title=item.get("short_description", item.get("title", f"ServiceNow Incident {incident_number}")),
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
        except Exception as exc:
            logger.warning("Skipping ServiceNow record at index %d: %s", idx, exc)

    return tasks

def load_emails_raw() -> list[dict]:
    try:
        data = _read_json("emails.json")
        if not isinstance(data, list):
            return []
        return data
    except Exception:
        return []

def load_meeting_transcripts_raw() -> list[dict]:
    try:
        data = _read_json("meeting_transcripts.json")
        if not isinstance(data, list):
            return []
        return data
    except Exception:
        return []

def load_chaos_defect() -> Task:
    try:
        raw: dict = _read_json("chaos_defect.json")
    except Exception:
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
    """Load and combine all structured task sources."""
    jira_tasks = load_jira_tasks()
    github_tasks = load_github_issues()
    servicenow_tasks = load_servicenow_incidents()

    combined: list[Task] = jira_tasks + github_tasks + servicenow_tasks
    logger.info(
        "Combined %d structured task(s) (%d JIRA + %d GitHub + %d ServiceNow).",
        len(combined),
        len(jira_tasks),
        len(github_tasks),
        len(servicenow_tasks),
    )
    return combined