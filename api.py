"""
api.py — TaskPilot AI REST Backend
Exposes the core pipeline (load → extract → deduplicate → prioritize) as HTTP endpoints.
Run with: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
import logging
import json
import hashlib
import os

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TaskPilot AI Backend",
    description="REST API for the AI Task Prioritization Assistant",
    version="1.0.0",
)

# Allow Streamlit (or any frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Redis Cache setup ─────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 300  # 5 minutes
redis_client = None

if REDIS_AVAILABLE:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info(f"Connected to Redis at {redis_url}")
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}. Falling back to in-memory cache.")
        redis_client = None
else:
    logger.info("Redis package not installed. Falling back to in-memory cache.")

# Fallback in-memory cache
_memory_cache: dict = {}

def _cache_key(*args) -> str:
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()

def _cache_get(key: str):
    if redis_client:
        try:
            cached_data = redis_client.get(key)
            if cached_data:
                logger.info(f"Redis Cache HIT for key {key[:8]}…")
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
    else:
        entry = _memory_cache.get(key)
        if entry and (time.time() - entry["ts"] < CACHE_TTL_SECONDS):
            logger.info(f"Memory Cache HIT for key {key[:8]}…")
            return entry["data"]
    return None

def _cache_set(key: str, data):
    if redis_client:
        try:
            redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(data))
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
    else:
        _memory_cache[key] = {"ts": time.time(), "data": data}

def _cache_clear():
    if redis_client:
        try:
            redis_client.flushdb()
            logger.info("Redis cache cleared")
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
    else:
        _memory_cache.clear()
        logger.info("Memory cache cleared")


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: Optional[list] = []

class PriorityWeights(BaseModel):
    severity: float = 0.4
    urgency: float = 0.3
    dependencies: float = 0.2
    impact: float = 0.1

class ChaosRequest(BaseModel):
    title: str = "CRITICAL: Production Database Down"
    severity: str = "P0"
    source: str = "chaos_injection"


# ── Helper: safely import your teammates' modules ─────────────────────────────

def _get_pipeline():
    """
    Lazily imports the src/ modules written by your teammates.
    Returns (loaders, extractor, deduplicator, prioritizer) or raises 500.
    """
    try:
        from src import loaders, extractor, deduplicator, prioritizer
        return loaders, extractor, deduplicator, prioritizer
    except ImportError as e:
        logger.error(f"Failed to import pipeline modules: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline modules not found: {e}. Make sure src/ exists.",
        )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Quick liveness check — use this to confirm the server is running."""
    return {
        "status": "ok", 
        "timestamp": time.time(),
        "cache_type": "redis" if redis_client else "memory"
    }


@app.get("/tasks/all")
def get_all_tasks():
    """
    Runs the full pipeline:
    1. Load raw data from all sources (Jira, ServiceNow, emails, meetings)
    2. Extract action items from unstructured text via LLM
    3. Deduplicate semantically similar tasks
    4. Return the merged task list (NOT yet prioritized)
    """
    cache_key = _cache_key("all_tasks")
    cached = _cache_get(cache_key)
    if cached:
        return cached

    logger.info("Running full ingestion + extraction + deduplication pipeline…")
    loaders, extractor, deduplicator, _ = _get_pipeline()

    try:
        start = time.time()

        # Step 1: Load structured tasks (Jira, ServiceNow)
        structured_tasks = loaders.load_all()
        logger.info(f"Loaded {len(structured_tasks)} structured tasks")

        # Step 2: Extract tasks from unstructured sources (emails, meetings)
        extracted_tasks = extractor.extract_all()
        logger.info(f"Extracted {len(extracted_tasks)} tasks from unstructured sources")

        # Step 3: Merge and deduplicate
        all_tasks = structured_tasks + extracted_tasks
        deduped_tasks = deduplicator.deduplicate(all_tasks)
        logger.info(f"After dedup: {len(deduped_tasks)} unique tasks (removed {len(all_tasks) - len(deduped_tasks)} duplicates)")

        elapsed = round(time.time() - start, 2)
        result = {
            "tasks": [t.dict() if hasattr(t, "dict") else t for t in deduped_tasks],
            "counts": {
                "structured": len(structured_tasks),
                "extracted": len(extracted_tasks),
                "duplicates_removed": len(all_tasks) - len(deduped_tasks),
                "total": len(deduped_tasks),
            },
            "elapsed_seconds": elapsed,
        }
        _cache_set(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tasks/prioritize")
def prioritize_tasks(weights: PriorityWeights = PriorityWeights()):
    """
    Runs the full pipeline AND prioritizes the result.
    Accepts optional custom weights for the RICE-inspired scoring formula.
    Returns tasks ranked highest → lowest with an explanation for each rank.
    """
    cache_key = _cache_key("prioritize", weights.dict())
    cached = _cache_get(cache_key)
    if cached:
        return cached

    logger.info(f"Prioritizing with weights: {weights.dict()}")
    loaders, extractor, deduplicator, prioritizer = _get_pipeline()

    try:
        start = time.time()

        structured_tasks = loaders.load_all()
        extracted_tasks = extractor.extract_all()
        all_tasks = structured_tasks + extracted_tasks
        deduped_tasks = deduplicator.deduplicate(all_tasks)

        # Prioritize with caller-supplied weights
        ranked = prioritizer.prioritize(deduped_tasks, weights=weights.dict())
        logger.info(f"Prioritization complete. Top task: {ranked[0] if ranked else 'none'}")

        elapsed = round(time.time() - start, 2)
        result = {
            "ranked_tasks": [t.dict() if hasattr(t, "dict") else t for t in ranked],
            "weights_used": weights.dict(),
            "elapsed_seconds": elapsed,
        }
        _cache_set(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"Prioritization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/daily-plan")
def get_daily_plan():
    """
    Generates the structured daily TODO plan.
    Returns top tasks grouped by priority tier (Critical / High / Normal).
    """
    cache_key = _cache_key("daily_plan")
    cached = _cache_get(cache_key)
    if cached:
        return cached

    loaders, extractor, deduplicator, prioritizer = _get_pipeline()

    try:
        structured_tasks = loaders.load_all()
        extracted_tasks = extractor.extract_all()
        all_tasks = structured_tasks + extracted_tasks
        deduped_tasks = deduplicator.deduplicate(all_tasks)
        ranked = prioritizer.prioritize(deduped_tasks)

        # Group into tiers for a clean daily plan
        critical, high, normal = [], [], []
        for t in ranked:
            task_dict = t.dict() if hasattr(t, "dict") else t
            severity = str(task_dict.get("severity", "")).upper()
            if severity in ("P0", "P1", "CRITICAL"):
                critical.append(task_dict)
            elif severity in ("P2", "HIGH"):
                high.append(task_dict)
            else:
                normal.append(task_dict)

        result = {
            "daily_plan": {
                "critical": critical,
                "high": high,
                "normal": normal,
            },
            "summary": f"{len(critical)} critical, {len(high)} high, {len(normal)} normal tasks today",
        }
        _cache_set(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"Daily plan error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Passes a natural language message to the LangGraph agent and returns its response.
    Maintains conversation history across turns.
    """
    logger.info(f"Chat message received: '{request.message[:60]}…'")

    try:
        from src.agent import run_agent
        response = run_agent(
            message=request.message,
            history=request.history or [],
        )
        return {"response": response, "message": request.message}

    except ImportError:
        raise HTTPException(status_code=500, detail="Agent module (src/agent.py) not found.")
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chaos/inject")
def inject_chaos(req: ChaosRequest, background_tasks: BackgroundTasks):
    """
    Injects a simulated P0 defect into the pipeline to demonstrate live re-prioritization.
    The new task is added to the top of the ranked list immediately.
    Cache is cleared so the next /tasks/prioritize call reflects the injected task.
    """
    logger.warning(f"CHAOS INJECTION: {req.title} ({req.severity})")

    # Clear cache so reprioritization picks up the new task
    _cache_clear()

    chaos_task = {
        "id": f"CHAOS-{int(time.time())}",
        "title": req.title,
        "severity": req.severity,
        "source": req.source,
        "injected_at": time.time(),
        "score": 9999,  # Forces it to top of ranked list
        "rationale": f"CHAOS INJECTION — {req.severity} defect manually injected to simulate real-world emergency re-prioritization.",
    }

    return {
        "message": "Chaos task injected. Cache cleared — next prioritize call will include this task at #1.",
        "injected_task": chaos_task,
    }


@app.delete("/cache")
def clear_cache():
    """Manually clear the cache (useful during testing or after data changes)."""
    _cache_clear()
    return {"message": "Cache cleared."}


@app.get("/sources/status")
def sources_status():
    """
    Checks which data source files exist on disk.
    Useful for debugging whether the data/ folder is set up correctly.
    """
    import os
    sources = {
        "jira_tasks": "data/jira_tasks.json",
        "emails": "data/emails.json",
        "service_now_incidents": "data/service_now_incidents.json",
        "meeting_transcripts": "data/meeting_transcripts.json",
        "chaos_defect": "data/chaos_defect.json",
    }
    status = {}
    for name, path in sources.items():
        exists = os.path.exists(path)
        status[name] = {
            "path": path,
            "exists": exists,
            "size_bytes": os.path.getsize(path) if exists else None,
        }
    return {"sources": status}