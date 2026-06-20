"""
TaskPilot AI — LLM Extraction Engine

Uses Google Gemini (via LangChain) to extract structured, actionable tasks
from unstructured text sources such as emails and meeting transcripts.

Each piece of raw text is sent through a few-shot prompted LLM call that
returns an ``ExtractedTaskList``.  The extracted items are then enriched into
full ``Task`` objects with unique IDs, source lineage, and audit metadata.

Retry logic (up to 3 attempts) handles transient validation errors by feeding
the error message back into the LLM for self-correction.
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.schemas import (
    ExtractedTask,
    ExtractedTaskList,
    Task,
    TaskSource,
    SeverityLevel,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────

_MAX_RETRIES: int = 3


def _get_llm() -> ChatGoogleGenerativeAI:
    """Return a configured Gemini 1.5 Flash instance.

    The ``temperature`` is kept low (0.1) to favour deterministic,
    schema-compliant outputs over creative variation.

    Returns:
        ChatGoogleGenerativeAI: Ready-to-use LLM client.

    Raises:
        RuntimeError: If ``GOOGLE_API_KEY`` is not set in the environment.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please add it to your .env file."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1,
    )


def _build_extraction_prompt() -> ChatPromptTemplate:
    """Build a few-shot ChatPromptTemplate for task extraction.

    The prompt contains:
    * A system message defining the agent's role and output format.
    * Two few-shot examples:
        1. An email about a production deployment failure → one HIGH task.
        2. A meeting transcript with no actionable items → empty list.
    * A human message template accepting ``{text}`` and ``{source_type}``.

    Returns:
        ChatPromptTemplate: Compiled prompt template.
    """
    return ChatPromptTemplate.from_messages(
        [
            # ── System role ──────────────────────────────────────────
            (
                "system",
                (
                    "You are an expert task extraction agent. Your job is to "
                    "read unstructured text (emails, meeting transcripts, chat "
                    "logs) and extract every actionable task hidden within it.\n\n"
                    "Rules:\n"
                    "1. Only extract ACTIONABLE items — things someone needs to do.\n"
                    "2. Ignore status updates, FYIs, and social pleasantries.\n"
                    "3. For each task, provide a concise title, a clear description, "
                    "severity (critical / high / medium / low), whether it is a "
                    "blocker, estimated impact (0.0–1.0), and a deadline if "
                    "mentioned.\n"
                    "4. If the text contains NO actionable tasks, return an empty "
                    "tasks list.\n"
                    "5. Provide a brief source_summary describing the overall "
                    "content of the source text."
                ),
            ),
            # ── Few-shot example 1: email with a task ────────────────
            (
                "human",
                (
                    "Source type: email\n\n"
                    "Text:\n"
                    "Subject: URGENT – Production deployment failed\n"
                    "From: devops@acme.com\n\n"
                    "Hi team,\n\n"
                    "Last night's v2.8 deployment to production rolled back "
                    "automatically after health-checks failed. The root cause "
                    "appears to be a misconfigured database connection pool. "
                    "We need someone to investigate the connection-pool settings "
                    "and redeploy by end of day Friday. This is blocking the "
                    "QA team from running regression tests.\n\n"
                    "Thanks,\nDevOps"
                ),
            ),
            (
                "ai",
                (
                    '{{"tasks": ['
                    '{{"title": "Investigate and fix production database connection-pool misconfiguration", '
                    '"description": "The v2.8 production deployment rolled back due to health-check failures caused by a misconfigured database connection pool. '
                    'Investigate the connection-pool settings, apply the fix, and redeploy. This is blocking the QA team from running regression tests.", '
                    '"severity": "high", '
                    '"deadline": null, '
                    '"is_blocker": true, '
                    '"impact_score": 0.85}}'
                    '], "source_summary": "DevOps email reporting a failed v2.8 production deployment due to database connection-pool misconfiguration."}}'
                ),
            ),
            # ── Few-shot example 2: meeting with no tasks ────────────
            (
                "human",
                (
                    "Source type: meeting\n\n"
                    "Text:\n"
                    "Meeting: Weekly Team Standup – 2026-06-16\n\n"
                    "Alice: I finished the login page redesign yesterday.\n"
                    "Bob: Nice work, that looks great.\n"
                    "Carol: I'll be out on PTO next Monday.\n"
                    "Dave: All good on my end, nothing new to report."
                ),
            ),
            (
                "ai",
                (
                    '{{"tasks": [], '
                    '"source_summary": "Routine weekly standup with status updates and a PTO notice. No actionable tasks identified."}}'
                ),
            ),
            # ── Actual request ───────────────────────────────────────
            (
                "human",
                "Source type: {source_type}\n\nText:\n{text}",
            ),
        ]
    )


# ────────────────────────────────────────────────────────────
# Core extraction
# ────────────────────────────────────────────────────────────


def extract_tasks_from_text(
    raw_text: str,
    source_type: TaskSource,
    source_ref: str = "",
) -> list[Task]:
    """Extract actionable tasks from arbitrary unstructured text.

    The function calls Gemini with structured-output binding so the
    response is automatically parsed into an ``ExtractedTaskList``.
    Each ``ExtractedTask`` is then enriched into a full ``Task``.

    Up to ``_MAX_RETRIES`` attempts are made when a ``ValidationError``
    occurs — the error is fed back to the LLM so it can self-correct.

    Args:
        raw_text: The unstructured source text to analyse.
        source_type: Origin type (e.g. ``TaskSource.EMAIL``).
        source_ref: Freeform reference string for audit lineage
            (e.g. ``email://from=...&subject=...``).

    Returns:
        list[Task]: Zero or more fully-formed Task objects.
    """
    if not raw_text or not raw_text.strip():
        logger.warning("extract_tasks_from_text called with empty text — skipping.")
        return []

    # ── Guard: API key ───────────────────────────────────────
    try:
        llm = _get_llm()
    except RuntimeError:
        logger.warning(
            "GOOGLE_API_KEY is not configured. Returning empty task list."
        )
        return []

    structured_llm = llm.with_structured_output(ExtractedTaskList)
    prompt = _build_extraction_prompt()
    chain = prompt | structured_llm

    invoke_kwargs = {
        "text": raw_text,
        "source_type": source_type.value,
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(
                "Extraction attempt %d/%d for source_type=%s source_ref=%s",
                attempt,
                _MAX_RETRIES,
                source_type.value,
                source_ref or "(none)",
            )
            result: ExtractedTaskList = chain.invoke(invoke_kwargs)
            logger.info(
                "Extraction succeeded on attempt %d — %d task(s) found.",
                attempt,
                len(result.tasks),
            )
            break  # success
        except ValidationError as exc:
            last_error = exc
            logger.warning(
                "ValidationError on attempt %d/%d: %s",
                attempt,
                _MAX_RETRIES,
                exc,
            )
            # Feed the error back so the LLM can self-correct on next try.
            invoke_kwargs["text"] = (
                f"{raw_text}\n\n"
                f"[SYSTEM: Your previous response failed validation with the "
                f"following error. Please fix and try again.]\n"
                f"Validation error: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unexpected error during extraction (attempt %d/%d): %s",
                attempt,
                _MAX_RETRIES,
                exc,
                exc_info=True,
            )
            last_error = exc
            return []
    else:
        # All retry attempts exhausted.
        logger.error(
            "All %d extraction attempts failed for source_ref=%s. "
            "Last error: %s",
            _MAX_RETRIES,
            source_ref,
            last_error,
        )
        return []

    # ── Convert ExtractedTask → Task ─────────────────────────
    tasks: list[Task] = []
    for extracted in result.tasks:
        deadline_dt: Optional[datetime] = None
        if extracted.deadline:
            try:
                deadline_dt = datetime.fromisoformat(extracted.deadline)
            except (ValueError, TypeError):
                logger.warning(
                    "Could not parse deadline '%s' — setting to None.",
                    extracted.deadline,
                )

        task = Task(
            id=str(uuid.uuid4()),
            title=extracted.title,
            description=extracted.description,
            source=source_type,
            source_id="",
            source_lineage=[source_ref] if source_ref else [],
            severity=extracted.severity,
            deadline=deadline_dt,
            is_blocker=extracted.is_blocker,
            impact_score=extracted.impact_score,
            created_at=datetime.now(),
            raw_text=raw_text,
        )
        tasks.append(task)
        logger.debug("Created Task id=%s title=%r", task.id, task.title)

    return tasks


# ────────────────────────────────────────────────────────────
# Source-specific convenience wrappers
# ────────────────────────────────────────────────────────────


def extract_from_emails(email_data: list[dict]) -> list[Task]:
    """Extract tasks from a batch of email dictionaries.

    Each dict is expected to contain at least ``subject``, ``body``, and
    ``from`` keys.  The subject and body are concatenated into a single
    text block for extraction.

    Args:
        email_data: List of email dicts, each with ``from``, ``subject``,
            and ``body`` keys.

    Returns:
        list[Task]: Aggregated tasks from all emails.
    """
    if not email_data:
        logger.info("No email data provided — nothing to extract.")
        return []

    all_tasks: list[Task] = []
    for idx, email in enumerate(email_data, start=1):
        subject = email.get("subject", "(no subject)")
        body = email.get("body", "")
        sender = email.get("from", "unknown")

        combined_text = f"Subject: {subject}\nFrom: {sender}\n\n{body}"
        source_ref = f"email://from={sender}&subject={subject}"

        logger.info(
            "Processing email %d/%d from=%s subject=%r",
            idx,
            len(email_data),
            sender,
            subject,
        )

        tasks = extract_tasks_from_text(
            raw_text=combined_text,
            source_type=TaskSource.EMAIL,
            source_ref=source_ref,
        )
        all_tasks.extend(tasks)

    logger.info(
        "Email extraction complete: %d email(s) → %d task(s).",
        len(email_data),
        len(all_tasks),
    )
    return all_tasks


def extract_from_meetings(meeting_data: list[dict]) -> list[Task]:
    """Extract tasks from a batch of meeting transcript dictionaries.

    Each dict is expected to contain at least ``meeting_title`` and
    ``transcript`` keys.

    Args:
        meeting_data: List of meeting dicts, each with ``meeting_title``
            and ``transcript`` keys.

    Returns:
        list[Task]: Aggregated tasks from all meetings.
    """
    if not meeting_data:
        logger.info("No meeting data provided — nothing to extract.")
        return []

    all_tasks: list[Task] = []
    for idx, meeting in enumerate(meeting_data, start=1):
        title = meeting.get("meeting_title", "(untitled meeting)")
        transcript = meeting.get("transcript", "")

        combined_text = f"Meeting: {title}\n\n{transcript}"
        source_ref = f"meeting://{title}"

        logger.info(
            "Processing meeting %d/%d title=%r",
            idx,
            len(meeting_data),
            title,
        )

        tasks = extract_tasks_from_text(
            raw_text=combined_text,
            source_type=TaskSource.MEETING,
            source_ref=source_ref,
        )
        all_tasks.extend(tasks)

    logger.info(
        "Meeting extraction complete: %d meeting(s) → %d task(s).",
        len(meeting_data),
        len(all_tasks),
    )
    return all_tasks


# ────────────────────────────────────────────────────────────
# Unified entry point
# ────────────────────────────────────────────────────────────


def extract_all_unstructured(
    email_data: list[dict],
    meeting_data: list[dict],
) -> list[Task]:
    """Extract tasks from all unstructured sources in a single call.

    This is the primary entry point used by the pipeline orchestrator.
    It delegates to :func:`extract_from_emails` and
    :func:`extract_from_meetings`, then returns the combined results.

    Args:
        email_data: List of email dicts (``from``, ``subject``, ``body``).
        meeting_data: List of meeting dicts (``meeting_title``, ``transcript``).

    Returns:
        list[Task]: Combined list of tasks from all unstructured sources.
    """
    logger.info(
        "Starting unified unstructured extraction — "
        "%d email(s), %d meeting(s).",
        len(email_data),
        len(meeting_data),
    )

    email_tasks = extract_from_emails(email_data)
    meeting_tasks = extract_from_meetings(meeting_data)

    combined = email_tasks + meeting_tasks
    logger.info(
        "Unified extraction complete: %d total task(s) "
        "(%d from email, %d from meetings).",
        len(combined),
        len(email_tasks),
        len(meeting_tasks),
    )
    return combined
