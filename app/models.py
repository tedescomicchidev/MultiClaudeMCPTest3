"""
Pydantic models for the Parallel Agent Scheduler.

Follows the Meta-Agent Orchestrator pattern from the production blueprint:
- A SwarmRequest captures the user's idea and desired parallelism.
- The UberAgent decomposes it into SubTasks queued for worker agents.
- Each SubTask tracks its own lifecycle independently.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SwarmStatus(str, Enum):
    INITIALIZING = "initializing"
    DECOMPOSING = "decomposing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SubTask(BaseModel):
    """A single unit of work assigned to one worker agent."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    agent_id: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class SwarmRequest(BaseModel):
    """User-submitted request to start an agent swarm."""

    idea: str
    num_agents: int = Field(default=3, ge=1, le=20)


class SwarmState(BaseModel):
    """Full state of a running swarm, managed by the uber-agent."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    idea: str
    num_agents: int
    status: SwarmStatus = SwarmStatus.INITIALIZING
    subtasks: list[SubTask] = Field(default_factory=list)
    synthesis: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    @property
    def progress(self) -> dict:
        total = len(self.subtasks)
        if total == 0:
            return {"total": 0, "completed": 0, "failed": 0, "running": 0, "pending": 0}
        return {
            "total": total,
            "completed": sum(1 for t in self.subtasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.subtasks if t.status == TaskStatus.FAILED),
            "running": sum(1 for t in self.subtasks if t.status == TaskStatus.IN_PROGRESS),
            "pending": sum(1 for t in self.subtasks if t.status == TaskStatus.PENDING),
        }


class SwarmEvent(BaseModel):
    """Server-Sent Event payload for real-time UI updates."""

    event_type: str
    swarm_id: str
    data: dict
