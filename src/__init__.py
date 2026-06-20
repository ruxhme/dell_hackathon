# TaskPilot AI — Source Package
"""
TaskPilot AI: An enterprise-grade agentic task prioritization system.

Modules:
    schemas      — Pydantic models for canonical task representation
    loaders      — Data source ingestion adapters
    extractor    — LLM-powered unstructured text → structured task extraction
    deduplicator — Two-stage semantic deduplication engine
    prioritizer  — Deterministic multi-variable scoring with explainability
    mcp_tools    — FastMCP tool server exposing pipeline operations
    agent        — LangGraph state-machine orchestration
"""
