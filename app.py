"""
TaskPilot AI — Streamlit Chat Interface

The presentation layer that consolidates backend processing into a
seamless, interactive conversational experience.  Features:

  - Chat-based interaction with the LangGraph agent
  - Real-time tool execution visibility via expander widgets
  - Dynamic task list with color-coded severity
  - "Simulate Chaos" sidebar button for mid-session defect injection
  - Persistent session state for conversation history
  - Configurable priority weight sliders
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# ── Ensure project root is importable ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent import stream_agent, AgentState, compile_agent
from src.mcp_tools import get_state as get_pipeline_state, reset_state as reset_pipeline_state
from src.schemas import PrioritizedTask, SeverityLevel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Page Configuration
# ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TaskPilot AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global Theme ──────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header Banner ─────────────────────────────────── */
    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .header-banner h1 {
        color: #e94560;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .header-banner p {
        color: #a8b2d1;
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
        font-weight: 300;
    }

    /* ── Task Cards ────────────────────────────────────── */
    .task-card {
        background: linear-gradient(135deg, #1e1e30 0%, #252540 100%);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .task-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }

    .task-card.critical { border-left-color: #e94560; }
    .task-card.high { border-left-color: #f59e0b; }
    .task-card.medium { border-left-color: #eab308; }
    .task-card.low { border-left-color: #22c55e; }

    .task-rank {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e94560;
        margin-right: 0.75rem;
    }

    .task-title {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    .task-meta {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    .task-score {
        font-size: 0.85rem;
        font-weight: 600;
        color: #38bdf8;
    }

    .task-rationale {
        font-size: 0.85rem;
        color: #a8b2d1;
        font-style: italic;
        margin-top: 0.4rem;
        line-height: 1.4;
    }

    /* ── Chaos Button ──────────────────────────────────── */
    .chaos-button {
        background: linear-gradient(135deg, #e94560 0%, #c62828 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        border: none;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
    }

    .chaos-button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.6);
    }

    /* ── Sidebar Styling ───────────────────────────────── */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .sidebar-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    /* ── Status Badges ─────────────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-badge.active { background: #064e3b; color: #34d399; }
    .status-badge.idle { background: #1e293b; color: #94a3b8; }
    .status-badge.chaos { background: #7f1d1d; color: #f87171; }

    /* ── Tool Execution Display ────────────────────────── */
    .tool-call-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }

    /* ── Alert Banner ──────────────────────────────────── */
    .alert-banner {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #f87171;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# Session State Initialization
# ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = None

if "chaos_triggered" not in st.session_state:
    st.session_state.chaos_triggered = False

if "pipeline_initialized" not in st.session_state:
    st.session_state.pipeline_initialized = False

if "weights" not in st.session_state:
    st.session_state.weights = {
        "severity": 0.35,
        "urgency": 0.30,
        "dependencies": 0.20,
        "impact": 0.15,
    }


# ────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────

def get_severity_icon(severity: str) -> str:
    """Return an emoji icon for the severity level."""
    return {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
    }.get(severity.lower(), "⚪")


def render_task_card(pt: PrioritizedTask) -> None:
    """Render a single prioritized task as a styled card."""
    task = pt.task
    severity_class = task.severity.value.lower()
    icon = get_severity_icon(task.severity.value)
    deadline_str = (
        task.deadline.strftime("%b %d, %H:%M")
        if task.deadline
        else "No deadline"
    )
    sources = ", ".join(task.source_lineage) if task.source_lineage else task.source.value

    st.markdown(f"""
    <div class="task-card {severity_class}">
        <div style="display: flex; align-items: center;">
            <span class="task-rank">#{pt.rank}</span>
            <div>
                <span class="task-title">{icon} {task.title}</span>
                <div class="task-meta">
                    {task.severity.value.upper()} • {deadline_str} •
                    {'🔗 Blocker' if task.is_blocker else '—'} •
                    <span class="task-score">Score: {pt.priority_score:.3f}</span>
                </div>
            </div>
        </div>
        <div class="task-rationale">📝 {pt.rationale}</div>
        <div class="task-meta" style="margin-top: 0.3rem;">🔗 {sources}</div>
    </div>
    """, unsafe_allow_html=True)


def render_task_list_sidebar() -> None:
    """Render the current prioritized task list in the sidebar."""
    pipeline_state = get_pipeline_state()
    prioritized = pipeline_state.get("prioritized_tasks", [])

    if not prioritized:
        st.sidebar.markdown("*No tasks prioritized yet. Ask me to generate your daily plan!*")
        return

    st.sidebar.markdown(f"### 📋 Current Plan ({len(prioritized)} tasks)")

    for pt in prioritized:
        task = pt.task
        icon = get_severity_icon(task.severity.value)
        st.sidebar.markdown(
            f"**#{pt.rank}** {icon} {task.title}  \n"
            f"<small style='color: #94a3b8;'>Score: {pt.priority_score:.3f}</small>",
            unsafe_allow_html=True,
        )


# ────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo / Title
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #e94560; margin: 0;">🎯 TaskPilot AI</h2>
        <p style="color: #94a3b8; font-size: 0.85rem;">Your Digital Chief of Staff</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Pipeline Status
    st.markdown('<div class="sidebar-title">📊 PIPELINE STATUS</div>', unsafe_allow_html=True)
    pipeline_state = get_pipeline_state()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Raw", len(pipeline_state.get("raw_tasks", [])))
    with col2:
        st.metric("Deduped", len(pipeline_state.get("deduplicated_tasks", [])))
    with col3:
        st.metric("Ranked", len(pipeline_state.get("prioritized_tasks", [])))

    if pipeline_state.get("chaos_injected"):
        st.markdown(
            '<span class="status-badge chaos">🔥 CHAOS ACTIVE</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Priority Weight Sliders
    st.markdown('<div class="sidebar-title">⚖️ PRIORITY WEIGHTS</div>', unsafe_allow_html=True)

    w_s = st.slider("Severity", 0.0, 1.0, st.session_state.weights["severity"], 0.05, key="w_severity")
    w_u = st.slider("Urgency", 0.0, 1.0, st.session_state.weights["urgency"], 0.05, key="w_urgency")
    w_d = st.slider("Dependencies", 0.0, 1.0, st.session_state.weights["dependencies"], 0.05, key="w_deps")
    w_i = st.slider("Impact", 0.0, 1.0, st.session_state.weights["impact"], 0.05, key="w_impact")

    total_w = w_s + w_u + w_d + w_i
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {total_w:.2f} — should be 1.0")
    else:
        st.session_state.weights = {
            "severity": w_s,
            "urgency": w_u,
            "dependencies": w_d,
            "impact": w_i,
        }

    st.divider()

    # Chaos Injector
    st.markdown('<div class="sidebar-title">💥 DYNAMIC ADAPTABILITY</div>', unsafe_allow_html=True)

    if st.button(
        "🔥 Simulate Chaos" if not st.session_state.chaos_triggered else "🔥 Chaos Already Injected",
        disabled=st.session_state.chaos_triggered,
        type="primary",
        use_container_width=True,
        key="chaos_btn",
    ):
        st.session_state.chaos_triggered = True

        # Inject via agent
        chaos_prompt = (
            "A critical production defect has just been reported! "
            "Inject the chaos defect and show me the reprioritized plan immediately."
        )
        st.session_state.messages.append({"role": "user", "content": chaos_prompt})
        st.rerun()

    if st.session_state.chaos_triggered:
        st.markdown("""
        <div class="alert-banner">
            <strong>⚠️ CHAOS MODE ACTIVE</strong><br>
            <small>A critical P0 defect has been injected. The plan has been reprioritized.</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Task List
    render_task_list_sidebar()

    st.divider()

    # Reset Button
    if st.button("🔄 Reset Pipeline", use_container_width=True, key="reset_btn"):
        reset_pipeline_state()
        st.session_state.messages = []
        st.session_state.agent_state = None
        st.session_state.chaos_triggered = False
        st.session_state.pipeline_initialized = False
        st.rerun()


# ────────────────────────────────────────────────────────────
# Main Chat Interface
# ────────────────────────────────────────────────────────────

# Header Banner
st.markdown("""
<div class="header-banner">
    <h1>🎯 TaskPilot AI</h1>
    <p>Your intelligent task prioritization assistant — aggregating, deduplicating, and ranking your work across all platforms.</p>
</div>
""", unsafe_allow_html=True)

# API Key Check
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error(
        "⚠️ **GOOGLE_API_KEY not found!** "
        "Create a `.env` file in the project root with your API key. "
        "See `.env.example` for the template."
    )
    st.info("Get your API key at: https://aistudio.google.com/apikey")
    st.stop()


# Welcome message
if not st.session_state.messages:
    welcome_msg = (
        "👋 Welcome to **TaskPilot AI** — your personal digital chief of staff!\n\n"
        "I can help you:\n"
        "- 📥 **Ingest tasks** from GitHub, ServiceNow, emails, and meetings\n"
        "- 🔍 **Deduplicate** overlapping items using semantic analysis\n"
        "- 📊 **Prioritize** your workload with a deterministic scoring formula\n"
        "- 🔥 **Handle chaos** when critical defects strike mid-day\n\n"
        "**Try asking me:**\n"
        '- *"Generate my daily plan"*\n'
        '- *"What are my highest priority tasks?"*\n'
        '- *"Tell me about the payment gateway issue"*\n'
        '- *"What\'s the current pipeline status?"*\n\n"What tasks were loaded from GitHub?"\n\n'
        "Use the **🔥 Simulate Chaos** button in the sidebar to test dynamic adaptability!"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})


# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "tool_calls" in msg:
            # Render tool call details in expanders
            for tc in msg["tool_calls"]:
                with st.expander(f"🔧 Tool: {tc['name']}", expanded=False):
                    st.code(f"Arguments: {tc.get('args', {})}", language="json")
                    if "result" in tc:
                        st.markdown(tc["result"])

        st.markdown(msg["content"])


# Chat Input
if user_input := st.chat_input("Ask TaskPilot AI anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Process with agent
    with st.chat_message("assistant"):
        tool_calls_display = []

        with st.spinner("🧠 TaskPilot AI is thinking..."):
            try:
                # Stream agent execution
                full_response = ""
                response_placeholder = st.empty()

                for event in stream_agent(
                    user_input,
                    state=st.session_state.agent_state,
                ):
                    if event["type"] == "tool_call":
                        tc = event["content"]
                        tool_calls_display.append(tc)
                        with st.expander(f"🔧 Calling: {tc['name']}", expanded=True):
                            st.code(
                                f"Tool: {tc['name']}\nArguments: {tc.get('args', {})}",
                                language="json",
                            )
                            st.info("⏳ Executing...")

                    elif event["type"] == "tool_result":
                        tr = event["content"]
                        # Update the last expander with result
                        with st.expander(f"✅ Result: {tr['name']}", expanded=False):
                            st.markdown(tr["result"])

                    elif event["type"] == "response":
                        full_response = event["content"]
                        response_placeholder.markdown(full_response)

                if not full_response:
                    full_response = "I've completed the requested operations. Check the sidebar for the updated task list."
                    response_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"❌ An error occurred: {str(e)}\n\nPlease check that your GOOGLE_API_KEY is valid and try again."
                st.error(full_response)
                logger.error(f"Agent error: {e}", exc_info=True)

        # Save to session state
        msg_data = {"role": "assistant", "content": full_response}
        if tool_calls_display:
            msg_data["tool_calls"] = tool_calls_display
        st.session_state.messages.append(msg_data)

    # Rerun to refresh sidebar metrics
    st.rerun()


# ────────────────────────────────────────────────────────────
# Footer — Task Cards (if prioritized)
# ────────────────────────────────────────────────────────────

pipeline_state = get_pipeline_state()
prioritized = pipeline_state.get("prioritized_tasks", [])

if prioritized:
    st.divider()
    st.markdown("## 📋 Current Prioritized Task List")

    for pt in prioritized:
        render_task_card(pt)