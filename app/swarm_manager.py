"""
Swarm Manager — orchestrates the full lifecycle of an agent swarm.

Refactored from asyncio.gather to Kubernetes pod-based execution:
1. Uber-agent decomposes the idea into subtasks (runs in-process)
2. Subtasks are enqueued in the task queue
3. Worker agents are spawned as Kubernetes pods via K8sPodManager
4. Workers POST results back to the orchestrator callback endpoint
5. Uber-agent synthesizes final results (runs in-process)
6. SSE events are emitted at every state transition

From the PDF (p.7-8):
  "In a production setting, each agent runs in its own container with
   isolated context windows. Using the Claude Agent SDK, developers can
   utilize 'session forking' to spawn parallel agents."

Each worker pod now replaces what was previously an asyncio coroutine,
providing true container isolation per the production blueprint.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import anthropic

from app.k8s_manager import K8sPodManager
from app.models import (
    AgentRole,
    SubTask,
    SwarmEvent,
    SwarmState,
    SwarmStatus,
    TaskStatus,
    WorkerResult,
)
from app.task_queue import TaskQueue
from app.uber_agent import decompose_idea, synthesize_results

logger = logging.getLogger(__name__)


class SwarmManager:
    """
    Manages all active swarms and coordinates their lifecycle.

    Now backed by Kubernetes pods instead of asyncio coroutines. Each
    worker agent runs in its own pod with isolated context, matching
    the production blueprint architecture.

    The SwarmManager runs as the Meta-Agent Orchestrator pod. Worker
    pods are created via K8sPodManager and report results back via
    HTTP callbacks.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        k8s_manager: Optional[K8sPodManager] = None,
        namespace: str = "default",
    ) -> None:
        self._swarms: dict[str, SwarmState] = {}
        self._queues: dict[str, TaskQueue] = {}
        self._event_queues: dict[str, list[asyncio.Queue]] = {}
        self._pending_results: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()
        self._k8s = k8s_manager or K8sPodManager(namespace=namespace)

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

    # ── Worker Result Callback ───────────────────────────────────────────

    async def handle_worker_result(self, result: WorkerResult) -> None:
        """
        Process a result callback from a worker pod.

        Called by the /api/callback endpoint when a worker pod POSTs
        its result back. Updates task state and resolves the pending
        future so the swarm can continue.
        """
        swarm = self._swarms.get(result.swarm_id)
        if not swarm:
            logger.warning("Received callback for unknown swarm %s", result.swarm_id)
            return

        # Update the subtask
        for task in swarm.subtasks:
            if task.id == result.task_id:
                if result.status == TaskStatus.COMPLETED:
                    task.status = TaskStatus.COMPLETED
                    task.result = result.result
                else:
                    task.status = TaskStatus.FAILED
                    task.error = result.error
                task.completed_at = datetime.now(timezone.utc)
                break

        # Update the queue
        queue = self._queues.get(result.swarm_id)
        if queue:
            if result.status == TaskStatus.COMPLETED:
                await queue.complete(result.task_id, result.result or "")
            else:
                await queue.fail(result.task_id, result.error or "unknown error")

        # Emit SSE event
        if result.status == TaskStatus.COMPLETED:
            await self._emit(result.swarm_id, "worker_completed", {
                "task_id": result.task_id,
                "worker_id": result.worker_id,
                "title": next(
                    (t.title for t in swarm.subtasks if t.id == result.task_id), ""
                ),
                "result": result.result,
                "progress": swarm.progress,
            })
        else:
            await self._emit(result.swarm_id, "worker_failed", {
                "task_id": result.task_id,
                "worker_id": result.worker_id,
                "title": next(
                    (t.title for t in swarm.subtasks if t.id == result.task_id), ""
                ),
                "error": result.error,
            })

        # Resolve the pending future
        future_key = f"{result.swarm_id}:{result.task_id}"
        if future_key in self._pending_results:
            future = self._pending_results.pop(future_key)
            if not future.done():
                future.set_result(result)

        logger.info(
            "Worker %s callback for task %s: %s",
            result.worker_id, result.task_id, result.status.value,
        )

    # ── Swarm Lifecycle ──────────────────────────────────────────────────

    async def _run_swarm(self, swarm_id: str) -> None:
        """Full swarm lifecycle: decompose -> spawn pods -> wait -> synthesize."""
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

            # ── Phase 2: Spawn Worker Pods ──────────────────────────────
            swarm.status = SwarmStatus.RUNNING
            await self._emit(swarm_id, "status", {
                "status": swarm.status.value,
                "message": f"Spawning {len(subtasks)} worker pods on Kubernetes...",
            })

            # Create a pod for each subtask and wait for all to complete
            await self._spawn_and_wait_workers(swarm_id, subtasks)

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
            # Clean up all worker pods
            await self._k8s.cleanup_swarm_pods(swarm_id)

    async def _spawn_and_wait_workers(
        self, swarm_id: str, subtasks: list[SubTask]
    ) -> None:
        """
        Spawn a Kubernetes pod for each subtask and wait for all to complete.

        Replaces the previous asyncio.gather() approach with actual pod
        creation. Each worker runs in isolation and POSTs results back
        via the callback endpoint.
        """
        futures = []

        for i, task in enumerate(subtasks):
            worker_id = f"worker-{i}"
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now(timezone.utc)
            task.agent_id = worker_id

            await self._emit(swarm_id, "worker_started", {
                "task_id": task.id,
                "worker_id": worker_id,
                "title": task.title,
            })

            # Create the K8s pod
            try:
                pod_name = await self._k8s.create_worker_pod(
                    swarm_id=swarm_id,
                    task=task,
                    role=AgentRole.WORKER,
                    worker_id=worker_id,
                )
                logger.info(
                    "Spawned pod %s for task %s (%s)",
                    pod_name, task.id, task.title,
                )
            except Exception as exc:
                logger.error("Failed to create pod for task %s: %s", task.id, exc)
                task.status = TaskStatus.FAILED
                task.error = f"Pod creation failed: {exc}"
                task.completed_at = datetime.now(timezone.utc)
                await self._emit(swarm_id, "worker_failed", {
                    "task_id": task.id,
                    "worker_id": worker_id,
                    "title": task.title,
                    "error": str(exc),
                })
                continue

            # Create a future to wait for the callback
            future_key = f"{swarm_id}:{task.id}"
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            self._pending_results[future_key] = future
            futures.append((task, future))

        # Wait for all workers to report back (with timeout)
        for task, future in futures:
            try:
                await asyncio.wait_for(future, timeout=300.0)
            except asyncio.TimeoutError:
                logger.warning("Worker for task %s timed out", task.id)
                if task.status == TaskStatus.IN_PROGRESS:
                    task.status = TaskStatus.FAILED
                    task.error = "Worker pod timed out"
                    task.completed_at = datetime.now(timezone.utc)
                    await self._emit(swarm_id, "worker_failed", {
                        "task_id": task.id,
                        "worker_id": task.agent_id,
                        "title": task.title,
                        "error": "Worker pod timed out",
                    })
