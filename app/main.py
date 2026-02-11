"""
Parallel Agent Scheduler — FastAPI Web Application.

Implements the production blueprint from "Running Claude Code in Kubernetes"
for scheduling parallel agent swarms and agent teams on Kubernetes.

Architecture:
- Meta-Agent Orchestrator pattern with uber-agent + worker swarm
- Agent Team mode with delegate lead + role-specialized members
- Kubernetes pod-based agent execution (replaces asyncio.gather)
- Task queue for distributing work (in-memory; Redis in production)
- SSE streaming for real-time progress monitoring
- Worker callback endpoint for pod result reporting

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

from app.agent_team import AgentTeam
from app.k8s_manager import K8sPodManager
from app.models import TeamRequest, SwarmRequest, WorkerResult
from app.swarm_manager import SwarmManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global managers (created at startup) ──────────────────────────────
manager: SwarmManager
team_manager: AgentTeam


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager, team_manager
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    namespace = os.environ.get("K8S_NAMESPACE", "default")
    worker_image = os.environ.get("WORKER_IMAGE", "parallel-agent-worker:latest")
    orchestrator_service = os.environ.get("ORCHESTRATOR_SERVICE", "agent-orchestrator")

    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set. Requests will fail unless the "
            "anthropic library finds credentials elsewhere."
        )

    # Shared K8s pod manager for both swarm and team modes
    k8s_mgr = K8sPodManager(
        namespace=namespace,
        worker_image=worker_image,
        orchestrator_service=orchestrator_service,
    )

    manager = SwarmManager(
        api_key=api_key, model=model, k8s_manager=k8s_mgr, namespace=namespace
    )
    team_manager = AgentTeam(
        api_key=api_key, model=model, k8s_manager=k8s_mgr, namespace=namespace
    )

    logger.info(
        "Initialized managers (model=%s, namespace=%s, worker_image=%s)",
        model, namespace, worker_image,
    )
    yield


app = FastAPI(
    title="Parallel Agent Scheduler",
    description=(
        "Schedule and monitor swarms of parallel Claude agents running "
        "as Kubernetes pods. Supports both anonymous swarm mode and "
        "role-based agent team mode with delegate lead. "
        "Based on the Meta-Agent Orchestrator pattern from the "
        "Kubernetes production blueprint."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Serve static files (frontend)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── Frontend Route ───────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page frontend."""
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path) as f:
        return HTMLResponse(content=f.read())


# ── Swarm Routes (original, now Kubernetes-backed) ───────────────────────

@app.post("/api/swarms")
async def create_swarm(req: SwarmRequest):
    """
    Create a new agent swarm.

    The uber-agent decomposes the idea into subtasks and dispatches
    them to parallel worker pods on Kubernetes.
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
    """SSE endpoint for real-time swarm progress."""
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


# ── Worker Callback Route ────────────────────────────────────────────────

@app.post("/api/callback")
async def worker_callback(result: WorkerResult):
    """
    Callback endpoint for worker pods to report results.

    Each worker pod POSTs its result here after completing (or failing)
    its assigned subtask. The result is routed to the appropriate
    swarm or team manager.
    """
    # Try swarm manager first, then team manager
    await manager.handle_worker_result(result)
    await team_manager.handle_worker_result(
        swarm_id=result.swarm_id,
        task_id=result.task_id,
        worker_id=result.worker_id,
        status=result.status.value,
        result=result.result,
        error=result.error,
    )
    return {"status": "ok"}


# ── Agent Team Routes ────────────────────────────────────────────────────

@app.post("/api/teams")
async def create_team(req: TeamRequest):
    """
    Create a new agent team with delegate mode.

    Spawns role-specialized agents (planner, architect, devil's advocate,
    k8s expert) as Kubernetes pods. The team lead operates in delegate
    mode, restricted to coordination-only actions.
    """
    team = await team_manager.create_team(
        idea=req.idea, delegate_mode=req.delegate_mode
    )
    return {
        "team_id": team.id,
        "status": team.status.value,
        "delegate_mode": team.delegate_mode,
        "idea": team.idea,
    }


@app.get("/api/teams")
async def list_teams():
    """List all agent teams."""
    teams = await team_manager.list_teams()
    return [
        {
            "id": t.id,
            "idea": t.idea[:100],
            "status": t.status.value,
            "delegate_mode": t.delegate_mode,
            "agents": {
                aid: {
                    "role": a.role.value,
                    "status": a.status.value,
                    "pod_name": a.pod_name,
                }
                for aid, a in t.agents.items()
            },
            "progress": t.progress,
            "created_at": t.created_at.isoformat(),
        }
        for t in teams
    ]


@app.get("/api/teams/{team_id}")
async def get_team(team_id: str):
    """Get the full state of an agent team."""
    team = await team_manager.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return {
        "id": team.id,
        "idea": team.idea,
        "status": team.status.value,
        "delegate_mode": team.delegate_mode,
        "plan": {
            "id": team.plan.id,
            "title": team.plan.title,
            "description": team.plan.description,
            "phases": team.plan.phases,
            "status": team.plan.status.value,
            "approved_by": team.plan.approved_by,
        } if team.plan else None,
        "agents": {
            aid: {
                "id": a.id,
                "role": a.role.value,
                "status": a.status.value,
                "pod_name": a.pod_name,
                "result": a.result,
                "error": a.error,
            }
            for aid, a in team.agents.items()
        },
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
            for t in team.subtasks
        ],
        "synthesis": team.synthesis,
        "error": team.error,
        "progress": team.progress,
        "event_log": team.event_log[-50:],  # Last 50 events
        "created_at": team.created_at.isoformat(),
        "completed_at": team.completed_at.isoformat() if team.completed_at else None,
    }


@app.post("/api/teams/{team_id}/approve-plan")
async def approve_team_plan(team_id: str):
    """Approve the team's implementation plan (delegate mode action)."""
    team = await team_manager.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    if not team.plan:
        raise HTTPException(status_code=400, detail="No plan to approve")
    await team_manager.approve_plan(team_id, team.plan.id)
    return {"status": "approved", "plan_id": team.plan.id}


@app.post("/api/teams/{team_id}/reject-plan")
async def reject_team_plan(team_id: str, feedback: str = "Needs revision"):
    """Reject the team's plan with feedback (delegate mode action)."""
    team = await team_manager.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    if not team.plan:
        raise HTTPException(status_code=400, detail="No plan to reject")
    await team_manager.reject_plan(team_id, team.plan.id, feedback)
    return {"status": "rejected", "plan_id": team.plan.id, "feedback": feedback}


@app.get("/api/teams/{team_id}/stream")
async def stream_team(team_id: str, request: Request):
    """SSE endpoint for real-time agent team progress."""
    team = await team_manager.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    async def event_generator():
        async for event in team_manager.subscribe(team_id):
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
