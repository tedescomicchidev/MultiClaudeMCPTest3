"""
Swarm Manager — orchestrates the full lifecycle of an agent swarm.

Implements the production blueprint architecture:
1. Uber-agent decomposes the idea into subtasks
2. Subtasks are enqueued in the task queue
3. Worker agents are spawned in parallel (asyncio.gather)
4. File-locking ensures data integrity on shared state
5. Uber-agent synthesizes final results
6. SSE events are emitted at every state transition

From the PDF (p.7):
  "Using the Claude Agent SDK, developers can utilize 'session forking' to spawn
   parallel agents from a single active session. This bypasses the overhead of a
   fresh initialization."

We simulate session forking via asyncio.gather with a shared Anthropic client,
which reuses the underlying HTTP connection pool — analogous to the SDK's
in-process approach.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import anthropic

from app.models import (
    SubTask,
    SwarmEvent,
    SwarmState,
    SwarmStatus,
    TaskStatus,
)
from app.task_queue import TaskQueue
from app.uber_agent import decompose_idea, synthesize_results
from app.worker_agent import execute_subtask

logger = logging.getLogger(__name__)


class SwarmManager:
    """
    Manages all active swarms and coordinates their lifecycle.

    In production on Kubernetes, this would be backed by a StatefulSet with
    dedicated PVCs per agent pod and a Redis task queue. The SwarmManager
    would run as the Meta-Agent Orchestrator pod.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514") -> None:
        self._swarms: dict[str, SwarmState] = {}
        self._queues: dict[str, TaskQueue] = {}
        self._event_queues: dict[str, list[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
        self._model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()

    async def create_swarm(self, idea: str, num_agents: int) -> SwarmState:
        """Create a new swarm and start it in the background."""
        swarm = SwarmState(idea=idea, num_agents=num_agents)
        queue = TaskQueue()

        async with self._lock:
            self._swarms[swarm.id] = swarm
            self._queues[swarm.id] = queue
            self._event_queues[swarm.id] = []

        # Fire and forget — the swarm runs in the background
        asyncio.create_task(self._run_swarm(swarm.id))
        logger.info("Created swarm %s with %d agents", swarm.id, num_agents)
        return swarm

    async def get_swarm(self, swarm_id: str) -> Optional[SwarmState]:
        async with self._lock:
            return self._swarms.get(swarm_id)

    async def list_swarms(self) -> list[SwarmState]:
        async with self._lock:
            return list(self._swarms.values())

    async def subscribe(self, swarm_id: str) -> AsyncGenerator[SwarmEvent, None]:
        """Subscribe to real-time SSE events for a swarm."""
        q: asyncio.Queue[Optional[SwarmEvent]] = asyncio.Queue()
        async with self._lock:
            if swarm_id not in self._event_queues:
                return
            self._event_queues[swarm_id].append(q)

        try:
            while True:
                event = await q.get()
                if event is None:
                    break
                yield event
        finally:
            async with self._lock:
                if swarm_id in self._event_queues:
                    try:
                        self._event_queues[swarm_id].remove(q)
                    except ValueError:
                        pass

    async def _emit(self, swarm_id: str, event_type: str, data: dict) -> None:
        """Broadcast an SSE event to all subscribers of this swarm."""
        event = SwarmEvent(event_type=event_type, swarm_id=swarm_id, data=data)
        async with self._lock:
            for q in self._event_queues.get(swarm_id, []):
                await q.put(event)

    async def _close_subscribers(self, swarm_id: str) -> None:
        """Signal all subscribers that the stream is done."""
        async with self._lock:
            for q in self._event_queues.get(swarm_id, []):
                await q.put(None)

    async def _run_swarm(self, swarm_id: str) -> None:
        """Full swarm lifecycle: decompose -> parallel execute -> synthesize."""
        swarm = self._swarms[swarm_id]
        queue = self._queues[swarm_id]

        try:
            # ── Phase 1: Decomposition ──────────────────────────────────
            swarm.status = SwarmStatus.DECOMPOSING
            await self._emit(swarm_id, "status", {
                "status": swarm.status.value,
                "message": "Uber-agent is decomposing the idea into subtasks...",
            })

            subtasks = await decompose_idea(
                idea=swarm.idea,
                num_agents=swarm.num_agents,
                client=self._client,
                model=self._model,
            )
            swarm.subtasks = subtasks

            await self._emit(swarm_id, "decomposition_complete", {
                "subtasks": [
                    {"id": t.id, "title": t.title, "description": t.description}
                    for t in subtasks
                ],
            })

            # Enqueue all subtasks
            for task in subtasks:
                await queue.enqueue(task)

            # ── Phase 2: Parallel Execution ─────────────────────────────
            swarm.status = SwarmStatus.RUNNING
            await self._emit(swarm_id, "status", {
                "status": swarm.status.value,
                "message": f"Spawning {len(subtasks)} parallel worker agents...",
            })

            # Spawn all workers in parallel — analogous to SDK parallel forking
            # (50-75ms per agent as per the blueprint benchmarks)
            await asyncio.gather(
                *[
                    self._run_worker(swarm_id, task, f"worker-{i}")
                    for i, task in enumerate(subtasks)
                ]
            )

            # ── Phase 3: Synthesis ──────────────────────────────────────
            await self._emit(swarm_id, "status", {
                "status": "synthesizing",
                "message": "Uber-agent is synthesizing worker results...",
            })

            synthesis = await synthesize_results(
                idea=swarm.idea,
                subtasks=swarm.subtasks,
                client=self._client,
                model=self._model,
            )
            swarm.synthesis = synthesis
            swarm.status = SwarmStatus.COMPLETED
            swarm.completed_at = datetime.now(timezone.utc)

            await self._emit(swarm_id, "completed", {
                "status": swarm.status.value,
                "synthesis": synthesis,
                "progress": swarm.progress,
            })

        except Exception as exc:
            logger.exception("Swarm %s failed", swarm_id)
            swarm.status = SwarmStatus.FAILED
            swarm.error = str(exc)
            await self._emit(swarm_id, "error", {
                "status": swarm.status.value,
                "error": str(exc),
            })

        finally:
            await self._close_subscribers(swarm_id)

    async def _run_worker(self, swarm_id: str, task: SubTask, worker_id: str) -> None:
        """
        Run a single worker agent for one subtask.

        In the Kubernetes production blueprint, each worker would be an isolated
        pod with its own PVC and context window. Here we simulate that with
        concurrent async tasks sharing the HTTP client pool.
        """
        try:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now(timezone.utc)
            task.agent_id = worker_id

            await self._emit(swarm_id, "worker_started", {
                "task_id": task.id,
                "worker_id": worker_id,
                "title": task.title,
            })

            result = await execute_subtask(
                subtask=task,
                client=self._client,
                model=self._model,
                worker_id=worker_id,
            )

            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now(timezone.utc)

            # Update the queue state as well
            queue = self._queues[swarm_id]
            await queue.complete(task.id, result)

            await self._emit(swarm_id, "worker_completed", {
                "task_id": task.id,
                "worker_id": worker_id,
                "title": task.title,
                "result": result,
                "progress": self._swarms[swarm_id].progress,
            })

        except Exception as exc:
            logger.exception("Worker %s failed on task %s", worker_id, task.id)
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            task.completed_at = datetime.now(timezone.utc)

            queue = self._queues[swarm_id]
            await queue.fail(task.id, str(exc))

            await self._emit(swarm_id, "worker_failed", {
                "task_id": task.id,
                "worker_id": worker_id,
                "title": task.title,
                "error": str(exc),
            })
