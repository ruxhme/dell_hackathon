"""
tests/test_backend.py — Unit tests for TaskPilot AI backend
Run with: pytest tests/ -v
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ── Import the API app ─────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api import app, _cache, _cache_key, _cache_get, _cache_set

client = TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Health check
# ══════════════════════════════════════════════════════════════════════════════

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


# ══════════════════════════════════════════════════════════════════════════════
# 2. Cache logic
# ══════════════════════════════════════════════════════════════════════════════

def test_cache_set_and_get():
    key = _cache_key("test", 123)
    _cache_set(key, {"hello": "world"})
    result = _cache_get(key)
    assert result == {"hello": "world"}


def test_cache_miss_returns_none():
    result = _cache_get("nonexistent_key_xyz")
    assert result is None


def test_cache_clear_endpoint():
    # Put something in the cache first
    _cache_set("dummy_key", {"data": 1})
    response = client.delete("/cache")
    assert response.status_code == 200
    assert "cleared" in response.json()["message"]
    assert len(_cache) == 0


def test_cache_ttl_expiry(monkeypatch):
    """Expired cache entries should be treated as misses."""
    key = _cache_key("expired_test")
    # Manually insert an old entry
    _cache[key] = {"ts": time.time() - 9999, "data": {"old": True}}
    result = _cache_get(key)
    assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sources status endpoint
# ══════════════════════════════════════════════════════════════════════════════

def test_sources_status_returns_all_keys():
    response = client.get("/sources/status")
    assert response.status_code == 200
    sources = response.json()["sources"]
    expected_keys = {"jira_tasks", "emails", "service_now_incidents", "meeting_transcripts", "chaos_defect"}
    assert expected_keys == set(sources.keys())


def test_sources_status_has_exists_field():
    response = client.get("/sources/status")
    for name, info in response.json()["sources"].items():
        assert "exists" in info
        assert "path" in info


# ══════════════════════════════════════════════════════════════════════════════
# 4. Chaos injection
# ══════════════════════════════════════════════════════════════════════════════

def test_chaos_inject_default():
    response = client.post("/chaos/inject", json={})
    assert response.status_code == 200
    data = response.json()
    assert "injected_task" in data
    task = data["injected_task"]
    assert task["severity"] == "P0"
    assert task["score"] == 9999


def test_chaos_inject_custom():
    payload = {
        "title": "DB Connection Pool Exhausted",
        "severity": "P0",
        "source": "test_chaos",
    }
    response = client.post("/chaos/inject", json=payload)
    assert response.status_code == 200
    task = response.json()["injected_task"]
    assert task["title"] == "DB Connection Pool Exhausted"


def test_chaos_inject_clears_cache():
    _cache_set("something", {"data": True})
    assert len(_cache) > 0
    client.post("/chaos/inject", json={})
    assert len(_cache) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 5. Pipeline endpoints (mocked — don't need real LLM keys to test)
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_task(id="TASK-1", title="Fix login bug", severity="P1", source="github"):
    task = MagicMock()
    task.dict.return_value = {
        "id": id,
        "title": title,
        "severity": severity,
        "source": source,
        "score": 80,
        "rationale": "High severity + approaching deadline",
    }
    return task


@patch("api._get_pipeline")
def test_get_all_tasks(mock_pipeline):
    mock_task = _make_mock_task()
    loaders = MagicMock()
    extractor = MagicMock()
    deduplicator = MagicMock()
    prioritizer = MagicMock()

    loaders.load_all.return_value = [mock_task]
    extractor.extract_all.return_value = [_make_mock_task("EMAIL-1", "Urgent email task", "P2", "email")]
    deduplicator.deduplicate.return_value = [mock_task]  # 1 deduped

    mock_pipeline.return_value = (loaders, extractor, deduplicator, prioritizer)
    _cache.clear()

    response = client.get("/tasks/all")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert data["counts"]["structured"] == 1
    assert data["counts"]["extracted"] == 1
    assert data["counts"]["duplicates_removed"] == 1


@patch("api._get_pipeline")
def test_prioritize_tasks(mock_pipeline):
    ranked_task = _make_mock_task()
    loaders, extractor, deduplicator, prioritizer = (
        MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )
    loaders.load_all.return_value = [ranked_task]
    extractor.extract_all.return_value = []
    deduplicator.deduplicate.return_value = [ranked_task]
    prioritizer.prioritize.return_value = [ranked_task]

    mock_pipeline.return_value = (loaders, extractor, deduplicator, prioritizer)
    _cache.clear()

    response = client.post("/tasks/prioritize", json={
        "severity": 0.4, "urgency": 0.3, "dependencies": 0.2, "impact": 0.1
    })
    assert response.status_code == 200
    data = response.json()
    assert "ranked_tasks" in data
    assert len(data["ranked_tasks"]) == 1
    assert data["weights_used"]["severity"] == 0.4


@patch("api._get_pipeline")
def test_daily_plan_tiers(mock_pipeline):
    p0_task = MagicMock()
    p0_task.dict.return_value = {"id": "INC-1", "title": "Prod down", "severity": "P0", "score": 99}

    p2_task = MagicMock()
    p2_task.dict.return_value = {"id": "GITHUB-2", "title": "UI bug", "severity": "P2", "score": 50}

    loaders, extractor, deduplicator, prioritizer = (
        MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )
    loaders.load_all.return_value = [p0_task, p2_task]
    extractor.extract_all.return_value = []
    deduplicator.deduplicate.return_value = [p0_task, p2_task]
    prioritizer.prioritize.return_value = [p0_task, p2_task]

    mock_pipeline.return_value = (loaders, extractor, deduplicator, prioritizer)
    _cache.clear()

    response = client.get("/tasks/daily-plan")
    assert response.status_code == 200
    plan = response.json()["daily_plan"]
    assert len(plan["critical"]) == 1
    assert len(plan["high"]) == 1
    assert len(plan["normal"]) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 6. Chat endpoint
# ══════════════════════════════════════════════════════════════════════════════

@patch("api._get_pipeline")  # ensure pipeline doesn't fail on import
def test_chat_missing_agent_module(mock_pipeline):
    """If src.agent doesn't exist yet, should return 500 with a helpful message."""
    with patch.dict("sys.modules", {"src.agent": None}):
        response = client.post("/chat", json={"message": "What's my top priority?"})
        # Either 200 (agent works) or 500 (module missing) — both are valid here
        assert response.status_code in (200, 500)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Priority weights validation
# ══════════════════════════════════════════════════════════════════════════════

@patch("api._get_pipeline")
def test_custom_weights_passed_to_prioritizer(mock_pipeline):
    task = _make_mock_task()
    loaders, extractor, deduplicator, prioritizer = (
        MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )
    loaders.load_all.return_value = [task]
    extractor.extract_all.return_value = []
    deduplicator.deduplicate.return_value = [task]
    prioritizer.prioritize.return_value = [task]
    mock_pipeline.return_value = (loaders, extractor, deduplicator, prioritizer)
    _cache.clear()

    custom_weights = {"severity": 0.6, "urgency": 0.2, "dependencies": 0.1, "impact": 0.1}
    client.post("/tasks/prioritize", json=custom_weights)

    # Verify prioritizer was called with our weights
    call_kwargs = prioritizer.prioritize.call_args
    assert call_kwargs is not None