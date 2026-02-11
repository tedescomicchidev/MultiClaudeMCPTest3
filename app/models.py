"""
Pydantic models for the Parallel Agent Scheduler.

Follows the Meta-Agent Orchestrator pattern from the production blueprint:
- A SwarmRequest captures the user's idea and desired parallelism.
- The UberAgent decomposes it into SubTasks queued for worker agents.
- Each SubTask tracks its own lifecycle independently.

Extended with Agent Team models for Kubernetes-based orchestration:
- AgentRole defines team member specializations.
- AgentState tracks individual agent pods.
- TeamState manages the full team lifecycle with delegate mode.
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


# ── Agent Team Models ────────────────────────────────────────────────────


class AgentRole(str, Enum):
    """Roles within an agent team. Each maps to a Kubernetes pod."""

    LEAD = "lead"
    ARCHITECT = "architect"
    K8S_EXPERT = "k8s_expert"
    DEVILS_ADVOCATE = "devils_advocate"
    PLANNER = "planner"
    WORKER = "worker"


class PlanStatus(str, Enum):
    """Lifecycle of a team plan requiring lead approval."""

    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class AgentStatus(str, Enum):
    """Pod-backed agent lifecycle states."""

    PENDING = "pending"          # Pod not yet created
    CREATING = "creating"        # Pod spec submitted to K8s API
    RUNNING = "running"          # Pod is Running phase
    COMPLETED = "completed"      # Pod succeeded
    FAILED = "failed"            # Pod failed
    TERMINATED = "terminated"    # Pod deleted by lead


class DelegateAction(str, Enum):
    """
    Actions available in delegate mode.

    When the team lead operates in delegate mode, it is restricted to
    these coordination-only actions. It cannot execute subtasks or call
    Claude for implementation work.
    """

    SPAWN_AGENT = "spawn_agent"
    MESSAGE_AGENT = "message_agent"
    SHUTDOWN_AGENT = "shutdown_agent"
    MANAGE_TASKS = "manage_tasks"
    APPROVE_PLAN = "approve_plan"
    REJECT_PLAN = "reject_plan"


class AgentState(BaseModel):
    """State of a single agent running as a Kubernetes pod."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: AgentRole
    pod_name: Optional[str] = None
    namespace: str = "default"
    status: AgentStatus = AgentStatus.PENDING
    assigned_task: Optional[SubTask] = None
    messages: list[dict] = Field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TeamPlan(BaseModel):
    """
    A plan created by the planner agent and requiring lead approval.

    The plan breaks down the idea into phases, assigns roles, and must
    be approved by the lead before any implementation begins. This
    enforces the delegate mode constraint.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str
    description: str
    phases: list[dict] = Field(default_factory=list)
    role_assignments: dict[str, str] = Field(default_factory=dict)
    status: PlanStatus = PlanStatus.DRAFT
    created_by: Optional[str] = None
    approved_by: Optional[str] = None
    feedback: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: Optional[datetime] = None


class TeamState(BaseModel):
    """
    Full state of an agent team running on Kubernetes.

    Extends the swarm concept with:
    - Named agent roles instead of anonymous workers
    - Delegate mode for the lead (coordination-only)
    - Plan approval workflow before implementation
    - Per-agent pod tracking
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    idea: str
    status: SwarmStatus = SwarmStatus.INITIALIZING
    delegate_mode: bool = True
    agents: dict[str, AgentState] = Field(default_factory=dict)
    plan: Optional[TeamPlan] = None
    subtasks: list[SubTask] = Field(default_factory=list)
    synthesis: Optional[str] = None
    error: Optional[str] = None
    event_log: list[dict] = Field(default_factory=list)
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


class TeamRequest(BaseModel):
    """User-submitted request to create an agent team."""

    idea: str
    delegate_mode: bool = Field(
        default=True,
        description="Restrict the lead to coordination-only actions",
    )
    namespace: str = Field(default="default", description="Kubernetes namespace")


class TeamEvent(BaseModel):
    """Server-Sent Event payload for team progress updates."""

    event_type: str
    team_id: str
    data: dict


class WorkerResult(BaseModel):
    """Result payload posted back by a worker pod to the orchestrator."""

    task_id: str
    worker_id: str
    swarm_id: str
    status: TaskStatus
    result: Optional[str] = None
    error: Optional[str] = None
