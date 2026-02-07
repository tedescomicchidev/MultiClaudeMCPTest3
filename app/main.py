"""
Parallel Agent Scheduler — FastAPI Web Application.

A demo web app implementing the production blueprint from
"Running Claude Code in Kubernetes" for scheduling parallel agent swarms.

Architecture (from the PDF):
- Meta-Agent Orchestrator pattern with uber-agent + worker swarm
- Task queue for distributing work (in-memory; Redis in production)
- SSE streaming for real-time progress monitoring
- Parallel agent spawning via asyncio (simulates SDK session forking)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.models import SwarmRequest
from app.swarm_manager import SwarmManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global swarm manager (created at startup) ───────────────────────────
manager: SwarmManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set. Requests will fail unless the "
            "anthropic library finds credentials elsewhere."
        )
    manager = SwarmManager(api_key=api_key, model=model)
    logger.info("SwarmManager initialized (model=%s)", model)
    yield


app = FastAPI(
    title="Parallel Agent Scheduler",
    description=(
        "Schedule and monitor swarms of parallel Claude agents. "
        "Based on the Meta-Agent Orchestrator pattern from the "
        "Kubernetes production blueprint."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files (frontend)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── Routes ──────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page frontend."""
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path) as f:
        return HTMLResponse(content=f.read())


@app.post("/api/swarms")
async def create_swarm(req: SwarmRequest):
    """
    Create a new agent swarm.

    The uber-agent will decompose the idea into subtasks and dispatch
    them to parallel worker agents.
    """
    swarm = await manager.create_swarm(idea=req.idea, num_agents=req.num_agents)
    return {
        "swarm_id": swarm.id,
        "status": swarm.status.value,
        "num_agents": swarm.num_agents,
        "idea": swarm.idea,
    }


@app.get("/api/swarms")
async def list_swarms():
    """List all swarms."""
    swarms = await manager.list_swarms()
    return [
        {
            "id": s.id,
            "idea": s.idea[:100],
            "status": s.status.value,
            "num_agents": s.num_agents,
            "progress": s.progress,
            "created_at": s.created_at.isoformat(),
        }
        for s in swarms
    ]


@app.get("/api/swarms/{swarm_id}")
async def get_swarm(swarm_id: str):
    """Get the full state of a swarm."""
    swarm = await manager.get_swarm(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    return {
        "id": swarm.id,
        "idea": swarm.idea,
        "status": swarm.status.value,
        "num_agents": swarm.num_agents,
        "progress": swarm.progress,
        "subtasks": [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "status": t.status.value,
                "agent_id": t.agent_id,
                "result": t.result,
                "error": t.error,
            }
            for t in swarm.subtasks
        ],
        "synthesis": swarm.synthesis,
        "error": swarm.error,
        "created_at": swarm.created_at.isoformat(),
        "completed_at": swarm.completed_at.isoformat() if swarm.completed_at else None,
    }


@app.get("/api/swarms/{swarm_id}/stream")
async def stream_swarm(swarm_id: str, request: Request):
    """
    SSE endpoint for real-time swarm progress.

    From the PDF (p.5):
      "SSE is often used for real-time streaming of agent responses."

    The client connects and receives events as:
      event: <event_type>
      data: <json>
    """
    swarm = await manager.get_swarm(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")

    async def event_generator():
        async for event in manager.subscribe(swarm_id):
            if await request.is_disconnected():
                break
            yield (
                f"event: {event.event_type}\n"
                f"data: {json.dumps(event.data)}\n\n"
            )
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
