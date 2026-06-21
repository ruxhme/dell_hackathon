# TaskPilot AI

**Your Intelligent Digital Chief of Staff**

TaskPilot AI is an advanced agentic coding solution designed to aggregate, deduplicate, and prioritize tasks across disparate systems (Jira, ServiceNow, Emails, Meetings) into a single, cohesive, prioritized action plan.

## Features

1.  **Multi-Source Ingestion**: Automatically loads data from structured (Jira, ServiceNow) and unstructured (Emails, Meetings) sources.
2.  **LLM-Powered Extraction**: Uses Google Gemini to extract actionable tasks and implicit action items from natural language text.
3.  **Semantic Deduplication**: A two-stage waterfall engine using exact hashing and `sentence-transformers` to consolidate overlapping tasks (e.g., a P1 incident in ServiceNow and an urgent email from the VP).
4.  **Deterministic Prioritization**: A transparent, math-based scoring formula (RICE-inspired) combining severity, urgency, dependencies, and impact.
5.  **Agentic Orchestration**: Powered by LangGraph and FastMCP, allowing the AI to logically route data through the pipeline and respond dynamically to chat.
6.  **Chaos Injection**: A simulated real-world feature where a critical defect is introduced mid-stream to demonstrate the system's ability to instantly reprioritize.
7.  **Interactive Frontend**: A beautiful Streamlit chat UI that exposes the LangGraph agent's thoughts and allows users to tweak priority weights on the fly.

## Project Structure

```text
.
├── app.py                      # Main Streamlit frontend and entry point
├── data/                       # Simulated data sources
│   ├── chaos_defect.json       # P0 injection scenario
│   ├── emails.json             # Unstructured emails with implicit tasks
│   ├── jira_tasks.json         # Structured sprint data
│   ├── meeting_transcripts.json# Unstructured meeting logs
│   └── service_now_incidents.json # Structured incident data
├── requirements.txt            # Python dependencies
└── src/                        # Core application modules
    ├── agent.py                # LangGraph orchestration logic
    ├── deduplicator.py         # 2-stage semantic deduplication engine
    ├── evaluate.py             # Evaluation harness for quantitative metrics
    ├── extractor.py            # LLM unstructured text extraction
    ├── loaders.py              # Data source ingestion
    ├── mcp_tools.py            # FastMCP tool server bridge
    ├── prioritizer.py          # Deterministic ranking and LLM rationales
    └── schemas.py              # Pydantic data models
```

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    # (Assuming you are in the project root)
    ```

2.  **Set up the environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory based on `.env.example`:
    ```bash
    cp .env.example .env
    ```
    Add your Google Gemini API key to the `.env` file:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

1. Open your browser to the URL provided by Streamlit (usually `http://localhost:8501`).
2. Ask the agent to "Generate my daily plan" or "Prioritize my tasks".
3. Watch the pipeline extract, deduplicate, and rank the work.
4. Click the "🔥 Simulate Chaos" button in the sidebar to inject a P0 defect and see the system instantly reprioritize!

## Architecture Details

-   **Model**: Google Gemini 1.5 Flash (via `langchain-google-genai`)
-   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
-   **Orchestration**: `langgraph`
-   **Tools**: `fastmcp`
-   **UI**: `streamlit`
