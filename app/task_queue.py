"""
Task Queue for the Meta-Agent Orchestrator pattern.

Production blueprint recommends Redis for task distribution. This module
provides an async in-memory queue for the demo, with the interface designed
to be swapped out for Redis (or any broker) in production.

From the PDF (p.6-7):
  "The Meta-Agent analyzes the requirements, breaks them into independent,
   parallelizable tasks, and queues them in a system like Redis. Worker agents
   pull these tasks from the queue and execute them simultaneously."
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from app.models import SubTask, TaskStatus

logger = logging.getLogger(__name__)


class TaskQueue:
    """
    Async in-memory task queue.

    In production, replace with a Redis-backed implementation using the same
    interface (enqueue / dequeue / complete / fail / get_task).
    """

    def __init__(self) -> None:
        self._tasks: dict[str, SubTask] = {}
        self._pending: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()

    async def enqueue(self, task: SubTask) -> None:
        async with self._lock:
            self._tasks[task.id] = task
            await self._pending.put(task.id)
            logger.info("Enqueued task %s: %s", task.id, task.title)

    async def dequeue(self, timeout: float = 30.0) -> Optional[SubTask]:
        """Pull the next pending task. Returns None on timeout."""
        try:
            task_id = await asyncio.wait_for(self._pending.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now(timezone.utc)
            return task

    async def complete(self, task_id: str, result: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = datetime.now(timezone.utc)
                logger.info("Task %s completed", task_id)

    async def fail(self, task_id: str, error: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.FAILED
                task.error = error
                task.completed_at = datetime.now(timezone.utc)
                logger.warning("Task %s failed: %s", task_id, error)

    async def get_task(self, task_id: str) -> Optional[SubTask]:
        async with self._lock:
            return self._tasks.get(task_id)

    async def all_tasks(self) -> list[SubTask]:
        async with self._lock:
            return list(self._tasks.values())

    @property
    def pending_count(self) -> int:
        return self._pending.qsize()
