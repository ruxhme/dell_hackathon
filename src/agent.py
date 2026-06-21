"""
TaskPilot AI — LangGraph Agent Orchestration

Implements the core agentic workflow as a state machine using LangGraph.
The agent operates in a ReAct (Reason → Act → Observe) loop, autonomously
selecting and invoking pipeline tools to fulfill user requests.

Key capabilities:
  - Autonomous tool selection and execution
  - Persistent conversational state management
  - Dynamic re-prioritization on state changes (chaos injection)
  - Human-in-the-loop intervention points
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Annotated, Any, Literal, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.mcp_tools import (
    get_state as get_pipeline_state,
    reset_state as reset_pipeline_state,
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
from src.schemas import PriorityWeights, PrioritizedTask, Task

load_dotenv()
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Agent State Schema
# ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Persistent state dictionary flowing through the LangGraph state machine.

    This state is continuously updated and passed along graph edges,
    ensuring total contextual continuity throughout the application lifecycle.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tasks: list[dict]
    prioritized_plan: list[dict]
    chaos_injected: bool
    processing_stage: str
    tool_outputs: list[dict]


# ────────────────────────────────────────────────────────────
# Tool Definitions (LangChain-compatible wrappers)
# ────────────────────────────────────────────────────────────

from langchain_core.tools import tool


@tool
def ingest_all_tasks_tool() -> str:
    """
    Ingest tasks from ALL data sources — Jira, ServiceNow, emails, and
    meeting transcripts.  Structured sources are parsed directly; 
    unstructured sources use LLM-powered extraction.
    Returns a summary of all discovered tasks.
    """
    logger.info("🔄 Tool: Ingesting all tasks...")

    # Structured sources
    structured = load_all_structured_tasks()

    # Unstructured sources
    emails = load_emails_raw()
    meetings = load_meeting_transcripts_raw()
    extracted = extract_all_unstructured(emails, meetings)

    all_tasks = structured + extracted

    # Update pipeline state
    pipeline_state = get_pipeline_state()
    pipeline_state["raw_tasks"] = all_tasks

    source_counts: dict[str, int] = {}
    for t in all_tasks:
        src = t.source.value
        source_counts[src] = source_counts.get(src, 0) + 1

    summary = (
        f"✅ Ingestion complete. Discovered {len(all_tasks)} total tasks:\n"
    )
    for src, count in sorted(source_counts.items()):
        summary += f"  • {src}: {count} tasks\n"
    summary += (
        f"\nBreakdown: {len(structured)} from structured sources, "
        f"{len(extracted)} extracted from unstructured text."
    )
    return summary


@tool
def deduplicate_tasks_tool(similarity_threshold: float = 0.85) -> str:
    """
    Run the two-stage deduplication engine on ingested tasks.
    Stage 1: Exact hash matching (xxhash).
    Stage 2: Semantic similarity (all-MiniLM-L6-v2 embeddings).

    Args:
        similarity_threshold: Cosine similarity threshold (0.0 to 1.0).
                              Default 0.85 for conservative dedup.
    """
    pipeline_state = get_pipeline_state()
    raw_tasks = pipeline_state.get("raw_tasks", [])

    if not raw_tasks:
        return "❌ No tasks available. Please run ingestion first."

    original_count = len(raw_tasks)
    deduped = deduplicate_tasks(raw_tasks, similarity_threshold=similarity_threshold)
    pipeline_state["deduplicated_tasks"] = deduped

    removed = original_count - len(deduped)

    summary = (
        f"✅ Deduplication complete:\n"
        f"  • Input: {original_count} tasks\n"
        f"  • Output: {len(deduped)} unique tasks\n"
        f"  • Removed: {removed} duplicates "
        f"({removed / max(original_count, 1) * 100:.1f}% reduction)\n\n"
        "Merged tasks with multiple sources:\n"
    )

    for t in deduped:
        if len(t.source_lineage) > 1:
            summary += (
                f"  • \"{t.title}\" — {len(t.source_lineage)} sources: "
                f"{', '.join(t.source_lineage)}\n"
            )

    return summary


@tool
def prioritize_tasks_tool(
    weight_severity: float = 0.35,
    weight_urgency: float = 0.30,
    weight_dependencies: float = 0.20,
    weight_impact: float = 0.15,
) -> str:
    """
    Run deterministic prioritization using weighted scoring formula:
    P(t) = w_s·S(t) + w_u·U(t) + w_d·D(t) + w_i·I(t)

    Args:
        weight_severity: Weight for severity dimension (0.0-1.0).
        weight_urgency: Weight for deadline urgency (0.0-1.0).
        weight_dependencies: Weight for blocker status (0.0-1.0).
        weight_impact: Weight for business impact (0.0-1.0).
    """
    pipeline_state = get_pipeline_state()
    tasks = (
        pipeline_state.get("deduplicated_tasks")
        or pipeline_state.get("raw_tasks", [])
    )

    if not tasks:
        return "❌ No tasks available. Please run ingestion first."

    weights = PriorityWeights(
        severity=weight_severity,
        urgency=weight_urgency,
        dependencies=weight_dependencies,
        impact=weight_impact,
    )

    prioritized = prioritize_tasks(tasks, weights=weights)
    pipeline_state["prioritized_tasks"] = prioritized

    summary = f"📋 **Prioritized Daily Plan** ({len(prioritized)} tasks):\n\n"
    for pt in prioritized:
        icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
            pt.task.severity.value, "⚪"
        )
        deadline = (
            pt.task.deadline.strftime("%b %d, %H:%M")
            if pt.task.deadline
            else "No deadline"
        )
        summary += (
            f"**#{pt.rank}** {icon} **{pt.task.title}**\n"
            f"  Score: {pt.priority_score:.3f} | "
            f"Severity: {pt.task.severity.value.upper()} | "
            f"Deadline: {deadline} | "
            f"Blocker: {'Yes' if pt.task.is_blocker else 'No'}\n"
            f"  📝 {pt.rationale}\n"
            f"  🔗 Sources: {', '.join(pt.task.source_lineage)}\n\n"
        )

    return summary


@tool
def inject_chaos_defect_tool() -> str:
    """
    Simulate a critical production incident injection mid-session.
    Loads a P0 defect and triggers immediate re-deduplication and
    re-prioritization of all tasks.  Demonstrates dynamic adaptability.
    """
    pipeline_state = get_pipeline_state()

    chaos_task = load_chaos_defect()
    pipeline_state["raw_tasks"].append(chaos_task)
    pipeline_state["chaos_injected"] = True

    # Re-run full pipeline
    deduped = deduplicate_tasks(pipeline_state["raw_tasks"])
    pipeline_state["deduplicated_tasks"] = deduped

    prioritized = prioritize_tasks(deduped)
    pipeline_state["prioritized_tasks"] = prioritized

    alert = (
        "⚠️ **CRITICAL ALERT: New P0 Defect Injected!**\n\n"
        f"**{chaos_task.title}**\n"
        f"{chaos_task.description}\n\n"
        "The daily plan has been **completely reprioritized**.\n\n"
        f"📋 **Updated Top 5 Priorities:**\n\n"
    )

    for pt in prioritized[:5]:
        icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
            pt.task.severity.value, "⚪"
        )
        alert += (
            f"**#{pt.rank}** {icon} **{pt.task.title}** "
            f"(Score: {pt.priority_score:.3f})\n"
            f"  {pt.rationale}\n\n"
        )

    return alert


@tool
def get_task_details_tool(task_title: str) -> str:
    """
    Retrieve full details for a specific task by title (partial match).

    Args:
        task_title: Full or partial title of the task to look up.
    """
    pipeline_state = get_pipeline_state()

    for pool_key in ["prioritized_tasks", "deduplicated_tasks", "raw_tasks"]:
        for item in pipeline_state.get(pool_key, []):
            task = item.task if isinstance(item, PrioritizedTask) else item
            if task_title.lower() in task.title.lower():
                detail = (
                    f"📋 **{task.title}**\n\n"
                    f"• Source: {task.source.value} | ID: {task.source_id}\n"
                    f"• Severity: {task.severity.value.upper()}\n"
                    f"• Deadline: {task.deadline.strftime('%Y-%m-%d %H:%M') if task.deadline else 'None'}\n"
                    f"• Blocker: {'Yes' if task.is_blocker else 'No'}\n"
                    f"• Impact: {task.impact_score:.2f}\n"
                    f"• Lineage: {', '.join(task.source_lineage)}\n\n"
                    f"**Description:** {task.description}\n"
                )
                if isinstance(item, PrioritizedTask):
                    detail += (
                        f"\n**Priority:** Rank #{item.rank}, "
                        f"Score {item.priority_score:.3f}\n"
                        f"**Rationale:** {item.rationale}\n"
                    )
                return detail

    return f"❌ No task found matching '{task_title}'."


@tool
def get_pipeline_status_tool() -> str:
    """Return current pipeline state: task counts and chaos injection status."""
    state = get_pipeline_state()
    return (
        f"📊 Pipeline Status:\n"
        f"  • Raw tasks: {len(state.get('raw_tasks', []))}\n"
        f"  • Deduplicated: {len(state.get('deduplicated_tasks', []))}\n"
        f"  • Prioritized: {len(state.get('prioritized_tasks', []))}\n"
        f"  • Chaos injected: {'🔥 Yes' if state.get('chaos_injected') else '❌ No'}"
    )


# ────────────────────────────────────────────────────────────
# All available tools
# ────────────────────────────────────────────────────────────

ALL_TOOLS = [
    ingest_all_tasks_tool,
    deduplicate_tasks_tool,
    prioritize_tasks_tool,
    inject_chaos_defect_tool,
    get_task_details_tool,
    get_pipeline_status_tool,
]


# ────────────────────────────────────────────────────────────
# System Prompt
# ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are **TaskPilot AI**, an intelligent task prioritization assistant designed to act as a personal digital chief of staff for software engineers.

## Your Capabilities
You have access to a suite of tools that allow you to:
1. **Ingest tasks** from multiple sources (Jira, ServiceNow, emails, meeting transcripts)
2. **Deduplicate tasks** using semantic similarity analysis
3. **Prioritize tasks** using a deterministic weighted scoring formula
4. **Handle critical incidents** via chaos defect injection
5. **Provide detailed task information** on demand
6. **Report pipeline status**

## Your Behavior
- When asked to generate a daily plan or prioritize tasks, you MUST call the tools in order: ingest → deduplicate → prioritize
- Always explain what you're doing at each step
- After prioritization, present the results clearly with rankings, scores, and rationales
- When chaos injection occurs, proactively alert the user about the reprioritized plan
- Be concise but thorough in your explanations
- Reference specific scoring variables (severity, urgency, dependencies, impact) when discussing priorities
- Always maintain traceability — reference source systems when discussing tasks

## Important Rules
- NEVER fabricate task data — only present information from the actual tool outputs
- ALWAYS use the tools rather than guessing at task information
- If asked about a specific task, use the get_task_details tool
- Keep your responses professional, data-driven, and actionable
- When presenting the prioritized plan, use the exact scores and rationales from the tools

Today's date is: {today}
"""


# ────────────────────────────────────────────────────────────
# LangGraph Node Functions
# ────────────────────────────────────────────────────────────

def _get_llm():
    """Initialize the language model with tool bindings."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Set it in your .env file."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.3,
        max_retries=2,
    )
    return llm.bind_tools(ALL_TOOLS)


def chat_node(state: AgentState) -> dict:
    """
    The primary reasoning node.  Receives the conversation history and
    decides whether to call a tool or produce a final response.

    Implements the 'Reason' step of the ReAct loop.
    """
    logger.info("🧠 Chat node: reasoning...")

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT.format(today=datetime.now().strftime("%Y-%m-%d"))
    )

    llm = _get_llm()
    messages = [system_msg] + list(state["messages"])

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        logger.error("LLM invoke failed (quota or network issue): %s", exc)
        from langchain_core.messages import AIMessage
        response = AIMessage(content="[System] The LLM API quota has been exhausted or an error occurred. The pipeline operations were attempted, but I cannot generate a detailed response right now. Please check your Groq API quota and verify your key is correct.")
    logger.info(
        f"🧠 Chat node response: "
        f"{'[tool_calls]' if response.tool_calls else '[text]'}"
    )

    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Conditional edge function.  Routes the graph to the tool executor
    if the LLM requested tool calls, otherwise ends the conversation turn.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(
            f"🔧 Routing to tools: "
            f"{[tc['name'] for tc in last_message.tool_calls]}"
        )
        return "tools"
    return "end"


# ────────────────────────────────────────────────────────────
# Graph Construction
# ────────────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """
    Construct the LangGraph state machine implementing the ReAct loop.

    Graph topology:
        chat_node → [tools | END]
        tools → chat_node

    The agent continues looping until it produces a final text response
    without requesting any tool calls.
    """
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)

    # Set entry point
    graph.set_entry_point("chat")

    # Add conditional edge from chat
    graph.add_conditional_edges(
        "chat",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # After tools, always go back to chat for next reasoning step
    graph.add_edge("tools", "chat")

    return graph


def compile_agent():
    """Build and compile the agent graph, ready for invocation."""
    graph = build_agent_graph()
    return graph.compile()


# ────────────────────────────────────────────────────────────
# Convenience: Run Agent
# ────────────────────────────────────────────────────────────

def run_agent(user_message: str, state: Optional[AgentState] = None) -> tuple[str, AgentState]:
    """
    Execute the agent with a user message and return the response.

    Args:
        user_message: The user's natural language input.
        state: Optional existing state for conversation continuity.

    Returns:
        Tuple of (agent_response_text, updated_state).
    """
    agent = compile_agent()

    if state is None:
        state = AgentState(
            messages=[],
            tasks=[],
            prioritized_plan=[],
            chaos_injected=False,
            processing_stage="idle",
            tool_outputs=[],
        )

    # Add the user message
    state["messages"] = list(state["messages"]) + [HumanMessage(content=user_message)]

    # Execute the graph
    result = agent.invoke(state)

    # Extract the final AI response
    ai_messages = [
        m for m in result["messages"]
        if isinstance(m, AIMessage) and m.content and not m.tool_calls
    ]

    response_text = ai_messages[-1].content if ai_messages else "I processed your request."

    return response_text, result


def stream_agent(user_message: str, state: Optional[AgentState] = None):
    """
    Stream the agent execution, yielding events as they occur.

    Yields dicts with keys:
        - "type": "tool_call" | "tool_result" | "response" | "token"
        - "content": the payload
    """
    agent = compile_agent()

    if state is None:
        state = AgentState(
            messages=[],
            tasks=[],
            prioritized_plan=[],
            chaos_injected=False,
            processing_stage="idle",
            tool_outputs=[],
        )

    state["messages"] = list(state["messages"]) + [HumanMessage(content=user_message)]

    for event in agent.stream(state, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "chat":
                messages = node_output.get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield {
                                    "type": "tool_call",
                                    "content": {
                                        "name": tc["name"],
                                        "args": tc["args"],
                                    },
                                }
                        elif msg.content:
                            yield {
                                "type": "response",
                                "content": msg.content,
                            }
            elif node_name == "tools":
                messages = node_output.get("messages", [])
                for msg in messages:
                    if isinstance(msg, ToolMessage):
                        yield {
                            "type": "tool_result",
                            "content": {
                                "name": msg.name,
                                "result": msg.content,
                            },
                        }
