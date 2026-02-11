"""
Agent Team — orchestrates a team of role-specialized agents on Kubernetes.

Implements the delegate mode pattern where the team lead is restricted to
coordination-only actions (spawning, messaging, shutting down agents, and
managing tasks). The lead CANNOT execute implementation work directly.

Team workflow:
1. Lead receives the idea and spawns the planner agent (pod)
2. Planner creates a structured plan
3. Architect reviews and refines the plan
4. Devil's Advocate challenges the plan
5. Lead approves (or rejects) the plan
6. K8s Expert validates deployment aspects
7. Worker agents are spawned to implement approved tasks
8. Lead synthesizes all results

Each agent runs as an isolated Kubernetes pod via K8sPodManager.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

import anthropic

from app.k8s_manager import K8sPodManager
from app.models import (
    AgentRole,
    AgentState,
    AgentStatus,
    DelegateAction,
    PlanStatus,
    SubTask,
    SwarmStatus,
    TaskStatus,
    TeamEvent,
    TeamPlan,
    TeamState,
)
from app.task_queue import TaskQueue

logger = logging.getLogger(__name__)

# ── Lead system prompt — coordination only, no implementation ────────────

LEAD_SYSTEM_PROMPT = """\
You are the Team Lead operating in DELEGATE MODE. You coordinate the agent \
team but NEVER implement tasks yourself.

Your ONLY allowed actions are:
1. SPAWN_AGENT — Create a new agent pod with a specific role
2. MESSAGE_AGENT — Send instructions or feedback to an agent
3. SHUTDOWN_AGENT — Terminate an agent pod
4. MANAGE_TASKS — Assign, reassign, or update task status
5. APPROVE_PLAN — Approve the team's implementation plan
6. REJECT_PLAN — Reject a plan with feedback for revision

You MUST NOT:
- Write code or implementation details
- Execute subtasks directly
- Call Claude for implementation work
- Bypass the plan approval workflow

Your role is to coordinate, delegate, and synthesize results from your team.
"""


class DelegateModeViolation(Exception):
    """Raised when the lead attempts an action outside delegate mode."""
    pass


class AgentTeam:
    """
    Manages a team of agents running as Kubernetes pods.

    The team lead operates in delegate mode, restricted to coordination
    actions defined in DelegateAction. All implementation work is
    performed by team member agents running in their own pods.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        k8s_manager: Optional[K8sPodManager] = None,
        namespace: str = "default",
    ) -> None:
        self._teams: dict[str, TeamState] = {}
        self._queues: dict[str, TaskQueue] = {}
        self._event_queues: dict[str, list[asyncio.Queue]] = {}
        self._pending_results: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._model = model
        self._client = (
            anthropic.AsyncAnthropic(api_key=api_key)
            if api_key
            else anthropic.AsyncAnthropic()
        )
        self._k8s = k8s_manager or K8sPodManager(namespace=namespace)
        self._namespace = namespace

    # ── Team Lifecycle ───────────────────────────────────────────────────

    async def create_team(self, idea: str, delegate_mode: bool = True) -> TeamState:
        """
        Create a new agent team and start the orchestration workflow.

        Spawns the core team (planner, architect, devil's advocate, k8s expert)
        as Kubernetes pods. The lead runs in-process in delegate mode.
        """
        team = TeamState(idea=idea, delegate_mode=delegate_mode)
        queue = TaskQueue()

        async with self._lock:
            self._teams[team.id] = team
            self._queues[team.id] = queue
            self._event_queues[team.id] = []

        # Fire the orchestration in the background
        asyncio.create_task(self._run_team(team.id))
        logger.info("Created team %s (delegate_mode=%s)", team.id, delegate_mode)
        return team

    async def get_team(self, team_id: str) -> Optional[TeamState]:
        async with self._lock:
            return self._teams.get(team_id)

    async def list_teams(self) -> list[TeamState]:
        async with self._lock:
            return list(self._teams.values())

    # ── Delegate Mode Actions ────────────────────────────────────────────

    def _enforce_delegate_mode(self, team: TeamState, action: DelegateAction) -> None:
        """Verify the action is allowed under delegate mode."""
        if not team.delegate_mode:
            return
        allowed = {a.value for a in DelegateAction}
        if action.value not in allowed:
            raise DelegateModeViolation(
                f"Action '{action.value}' is not permitted in delegate mode. "
                f"Allowed actions: {', '.join(sorted(allowed))}"
            )

    async def spawn_agent(
        self,
        team_id: str,
        role: AgentRole,
        task: Optional[SubTask] = None,
    ) -> AgentState:
        """
        Spawn a new agent as a Kubernetes pod.

        Delegate mode action: SPAWN_AGENT.
        """
        team = self._teams[team_id]
        self._enforce_delegate_mode(team, DelegateAction.SPAWN_AGENT)

        agent = AgentState(role=role)
        if task:
            agent.assigned_task = task

        # Create K8s pod
        subtask = task or SubTask(
            title=f"{role.value} agent task",
            description=f"Perform {role.value} duties for: {team.idea}",
        )

        try:
            agent.status = AgentStatus.CREATING
            pod_name = await self._k8s.create_worker_pod(
                swarm_id=team_id,
                task=subtask,
                role=role,
                worker_id=f"{role.value}-{agent.id[:6]}",
            )
            agent.pod_name = pod_name
            agent.status = AgentStatus.RUNNING
            agent.started_at = datetime.now(timezone.utc)
        except Exception as exc:
            agent.status = AgentStatus.FAILED
            agent.error = str(exc)
            logger.error("Failed to spawn %s agent: %s", role.value, exc)

        async with self._lock:
            team.agents[agent.id] = agent

        await self._emit(team_id, "agent_spawned", {
            "agent_id": agent.id,
            "role": role.value,
            "pod_name": agent.pod_name,
            "status": agent.status.value,
        })

        logger.info(
            "Spawned %s agent %s (pod=%s) for team %s",
            role.value, agent.id, agent.pod_name, team_id,
        )
        return agent

    async def message_agent(
        self,
        team_id: str,
        agent_id: str,
        message: str,
        sender: str = "lead",
    ) -> None:
        """
        Send a message to an agent.

        Delegate mode action: MESSAGE_AGENT.
        Messages are stored in the agent's message log and can be used
        for inter-agent communication.
        """
        team = self._teams[team_id]
        self._enforce_delegate_mode(team, DelegateAction.MESSAGE_AGENT)

        agent = team.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found in team {team_id}")

        msg = {
            "from": sender,
            "to": agent_id,
            "content": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        agent.messages.append(msg)

        await self._emit(team_id, "agent_message", {
            "agent_id": agent_id,
            "role": agent.role.value,
            "message": message,
            "sender": sender,
        })

    async def shutdown_agent(self, team_id: str, agent_id: str) -> None:
        """
        Shut down an agent by deleting its Kubernetes pod.

        Delegate mode action: SHUTDOWN_AGENT.
        """
        team = self._teams[team_id]
        self._enforce_delegate_mode(team, DelegateAction.SHUTDOWN_AGENT)

        agent = team.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found in team {team_id}")

        if agent.pod_name:
            await self._k8s.delete_pod(agent.pod_name)

        agent.status = AgentStatus.TERMINATED
        agent.completed_at = datetime.now(timezone.utc)

        await self._emit(team_id, "agent_shutdown", {
            "agent_id": agent_id,
            "role": agent.role.value,
            "pod_name": agent.pod_name,
        })

        logger.info("Shutdown agent %s (pod=%s)", agent_id, agent.pod_name)

    async def approve_plan(self, team_id: str, plan_id: str) -> None:
        """
        Approve the team plan, allowing implementation to begin.

        Delegate mode action: APPROVE_PLAN.
        """
        team = self._teams[team_id]
        self._enforce_delegate_mode(team, DelegateAction.APPROVE_PLAN)

        if not team.plan or team.plan.id != plan_id:
            raise ValueError(f"Plan {plan_id} not found in team {team_id}")

        team.plan.status = PlanStatus.APPROVED
        team.plan.approved_by = "lead"
        team.plan.approved_at = datetime.now(timezone.utc)

        await self._emit(team_id, "plan_approved", {
            "plan_id": plan_id,
            "title": team.plan.title,
        })

        logger.info("Plan %s approved for team %s", plan_id, team_id)

    async def reject_plan(
        self, team_id: str, plan_id: str, feedback: str
    ) -> None:
        """
        Reject the plan with feedback for revision.

        Delegate mode action: REJECT_PLAN.
        """
        team = self._teams[team_id]
        self._enforce_delegate_mode(team, DelegateAction.REJECT_PLAN)

        if not team.plan or team.plan.id != plan_id:
            raise ValueError(f"Plan {plan_id} not found in team {team_id}")

        team.plan.status = PlanStatus.REJECTED
        team.plan.feedback = feedback

        await self._emit(team_id, "plan_rejected", {
            "plan_id": plan_id,
            "feedback": feedback,
        })

        logger.info("Plan %s rejected for team %s: %s", plan_id, team_id, feedback)

    # ── Worker Result Callback ───────────────────────────────────────────

    async def handle_worker_result(
        self,
        swarm_id: str,
        task_id: str,
        worker_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Process a result callback from a worker pod.

        Called by the /api/callback endpoint when a worker pod POSTs
        its result back to the orchestrator.
        """
        team = self._teams.get(swarm_id)
        if not team:
            logger.warning("Received callback for unknown team %s", swarm_id)
            return

        # Update the agent state
        for agent in team.agents.values():
            if agent.assigned_task and agent.assigned_task.id == task_id:
                if status == "completed":
                    agent.status = AgentStatus.COMPLETED
                    agent.result = result
                    agent.assigned_task.status = TaskStatus.COMPLETED
                    agent.assigned_task.result = result
                else:
                    agent.status = AgentStatus.FAILED
                    agent.error = error
                    agent.assigned_task.status = TaskStatus.FAILED
                    agent.assigned_task.error = error
                agent.completed_at = datetime.now(timezone.utc)
                break

        # Update the subtask in the team state
        for task in team.subtasks:
            if task.id == task_id:
                if status == "completed":
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                else:
                    task.status = TaskStatus.FAILED
                    task.error = error
                task.completed_at = datetime.now(timezone.utc)
                break

        # Update the queue
        queue = self._queues.get(swarm_id)
        if queue:
            if status == "completed":
                await queue.complete(task_id, result or "")
            else:
                await queue.fail(task_id, error or "unknown error")

        event_type = "worker_completed" if status == "completed" else "worker_failed"
        await self._emit(swarm_id, event_type, {
            "task_id": task_id,
            "worker_id": worker_id,
            "status": status,
            "result": result,
            "error": error,
            "progress": team.progress,
        })

        # Resolve the pending future if one exists
        future_key = f"{swarm_id}:{task_id}"
        if future_key in self._pending_results:
            future = self._pending_results.pop(future_key)
            if not future.done():
                future.set_result({"status": status, "result": result, "error": error})

        logger.info(
            "Worker %s result for task %s: %s", worker_id, task_id, status
        )

    # ── Team Orchestration ───────────────────────────────────────────────

    async def _run_team(self, team_id: str) -> None:
        """
        Full team lifecycle with delegate mode enforcement.

        Phase 1: Spawn planner → generate plan
        Phase 2: Spawn architect + devil's advocate → review plan
        Phase 3: Lead approves plan (delegate mode gate)
        Phase 4: Spawn K8s expert → validate deployment aspects
        Phase 5: Spawn workers → execute approved tasks
        Phase 6: Lead synthesizes results (via Claude, coordination only)
        """
        team = self._teams[team_id]

        try:
            # ── Phase 1: Planning ────────────────────────────────────
            team.status = SwarmStatus.DECOMPOSING
            await self._emit(team_id, "status", {
                "status": team.status.value,
                "message": "Lead is spawning the planner agent...",
            })

            plan = await self._phase_planning(team_id)
            team.plan = plan

            await self._emit(team_id, "plan_created", {
                "plan_id": plan.id,
                "title": plan.title,
                "description": plan.description,
                "phases": plan.phases,
                "status": plan.status.value,
            })

            # ── Phase 2: Review ──────────────────────────────────────
            await self._emit(team_id, "status", {
                "status": "reviewing",
                "message": "Architect and Devil's Advocate are reviewing the plan...",
            })

            await self._phase_review(team_id)

            # ── Phase 3: Plan Approval Gate ──────────────────────────
            # In delegate mode, the lead must explicitly approve
            plan.status = PlanStatus.PENDING_APPROVAL
            await self._emit(team_id, "plan_pending_approval", {
                "plan_id": plan.id,
                "title": plan.title,
                "message": "Plan requires lead approval before implementation begins.",
            })

            # Auto-approve for the orchestration flow (in production,
            # this would wait for an explicit API call from the lead)
            await self.approve_plan(team_id, plan.id)

            # ── Phase 4: K8s Validation ──────────────────────────────
            await self._emit(team_id, "status", {
                "status": "validating",
                "message": "K8s Expert is validating deployment configuration...",
            })

            await self._phase_k8s_validation(team_id)

            # ── Phase 5: Implementation ──────────────────────────────
            team.status = SwarmStatus.RUNNING
            await self._emit(team_id, "status", {
                "status": team.status.value,
                "message": f"Spawning {len(team.subtasks)} worker pods...",
            })

            await self._phase_implementation(team_id)

            # ── Phase 6: Synthesis ───────────────────────────────────
            await self._emit(team_id, "status", {
                "status": "synthesizing",
                "message": "Lead is synthesizing team results...",
            })

            synthesis = await self._phase_synthesis(team_id)
            team.synthesis = synthesis
            team.status = SwarmStatus.COMPLETED
            team.completed_at = datetime.now(timezone.utc)

            await self._emit(team_id, "completed", {
                "status": team.status.value,
                "synthesis": synthesis,
                "progress": team.progress,
            })

        except Exception as exc:
            logger.exception("Team %s failed", team_id)
            team.status = SwarmStatus.FAILED
            team.error = str(exc)
            await self._emit(team_id, "error", {
                "status": team.status.value,
                "error": str(exc),
            })

        finally:
            await self._close_subscribers(team_id)
            # Clean up pods
            await self._k8s.cleanup_swarm_pods(team_id)

    async def _phase_planning(self, team_id: str) -> TeamPlan:
        """Spawn planner agent to create the implementation plan."""
        team = self._teams[team_id]

        planning_task = SubTask(
            title="Create implementation plan",
            description=(
                f"Create a detailed implementation plan for the following idea. "
                f"Break it into phases with clear task assignments for each team "
                f"role (architect, k8s_expert, worker). Output as structured JSON "
                f"with 'title', 'description', 'phases' (array of phase objects "
                f"with 'name', 'tasks' array), and 'role_assignments' dict.\n\n"
                f"Idea: {team.idea}"
            ),
        )

        planner = await self.spawn_agent(team_id, AgentRole.PLANNER, planning_task)

        # Wait for planner pod to complete
        result = await self._wait_for_agent_result(team_id, planning_task.id)

        # Parse the plan from the planner's output
        plan_text = result.get("result", "")
        plan = self._parse_plan(plan_text, team.idea)
        plan.created_by = planner.id

        # Shutdown planner pod
        await self.shutdown_agent(team_id, planner.id)

        return plan

    async def _phase_review(self, team_id: str) -> None:
        """Spawn architect and devil's advocate to review the plan."""
        team = self._teams[team_id]
        plan = team.plan

        plan_summary = (
            f"Plan: {plan.title}\n"
            f"Description: {plan.description}\n"
            f"Phases: {json.dumps(plan.phases, indent=2)}"
        )

        # Spawn both reviewers in parallel (as separate pods)
        architect_task = SubTask(
            title="Review architecture",
            description=(
                f"Review the following implementation plan from an architectural "
                f"perspective. Identify strengths, weaknesses, and suggest "
                f"improvements.\n\n{plan_summary}"
            ),
        )

        advocate_task = SubTask(
            title="Challenge assumptions",
            description=(
                f"Critically review the following implementation plan. Identify "
                f"potential flaws, edge cases, security concerns, and scalability "
                f"issues. Challenge every assumption.\n\n{plan_summary}"
            ),
        )

        architect = await self.spawn_agent(team_id, AgentRole.ARCHITECT, architect_task)
        advocate = await self.spawn_agent(team_id, AgentRole.DEVILS_ADVOCATE, advocate_task)

        # Wait for both reviews
        arch_result, adv_result = await asyncio.gather(
            self._wait_for_agent_result(team_id, architect_task.id),
            self._wait_for_agent_result(team_id, advocate_task.id),
        )

        # Store reviews as messages to the lead
        await self.message_agent(
            team_id, list(team.agents.keys())[0] if team.agents else "",
            f"Architecture Review:\n{arch_result.get('result', 'No result')}",
            sender="architect",
        )
        await self.message_agent(
            team_id, list(team.agents.keys())[0] if team.agents else "",
            f"Critical Review:\n{adv_result.get('result', 'No result')}",
            sender="devils_advocate",
        )

        # Shutdown reviewer pods
        await self.shutdown_agent(team_id, architect.id)
        await self.shutdown_agent(team_id, advocate.id)

    async def _phase_k8s_validation(self, team_id: str) -> None:
        """Spawn K8s expert to validate deployment aspects."""
        team = self._teams[team_id]
        plan = team.plan

        k8s_task = SubTask(
            title="Validate Kubernetes deployment",
            description=(
                f"Review the following plan and validate the Kubernetes deployment "
                f"aspects. Check pod specs, resource limits, RBAC, networking, "
                f"and security. Suggest improvements.\n\n"
                f"Plan: {plan.title}\n"
                f"Phases: {json.dumps(plan.phases, indent=2)}"
            ),
        )

        k8s_agent = await self.spawn_agent(team_id, AgentRole.K8S_EXPERT, k8s_task)
        result = await self._wait_for_agent_result(team_id, k8s_task.id)

        await self.message_agent(
            team_id, list(team.agents.keys())[0] if team.agents else "",
            f"K8s Validation:\n{result.get('result', 'No result')}",
            sender="k8s_expert",
        )

        await self.shutdown_agent(team_id, k8s_agent.id)

    async def _phase_implementation(self, team_id: str) -> None:
        """Spawn worker pods to implement the approved plan."""
        team = self._teams[team_id]
        plan = team.plan
        queue = self._queues[team_id]

        # Create subtasks from the plan phases
        subtasks = []
        for i, phase in enumerate(plan.phases):
            for j, task_desc in enumerate(phase.get("tasks", [phase])):
                if isinstance(task_desc, dict):
                    title = task_desc.get("name", task_desc.get("title", f"Task {i+1}.{j+1}"))
                    description = task_desc.get("description", str(task_desc))
                else:
                    title = f"Phase {i+1} Task {j+1}"
                    description = str(task_desc)

                subtask = SubTask(title=title, description=description)
                subtasks.append(subtask)

        # If no structured tasks were parsed, create a single implementation task
        if not subtasks:
            subtasks.append(SubTask(
                title="Implement plan",
                description=f"Implement the following plan:\n{plan.description}",
            ))

        team.subtasks = subtasks

        await self._emit(team_id, "decomposition_complete", {
            "subtasks": [
                {"id": t.id, "title": t.title, "description": t.description}
                for t in subtasks
            ],
        })

        # Enqueue and spawn worker pods
        for task in subtasks:
            await queue.enqueue(task)

        # Spawn all workers as separate pods
        futures = []
        for i, task in enumerate(subtasks):
            worker_id = f"worker-{i}"
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now(timezone.utc)
            task.agent_id = worker_id

            agent = await self.spawn_agent(team_id, AgentRole.WORKER, task)

            await self._emit(team_id, "worker_started", {
                "task_id": task.id,
                "worker_id": worker_id,
                "title": task.title,
            })

            futures.append(self._wait_for_agent_result(team_id, task.id))

        # Wait for all workers to complete
        await asyncio.gather(*futures, return_exceptions=True)

    async def _phase_synthesis(self, team_id: str) -> str:
        """Lead synthesizes results from all team members."""
        team = self._teams[team_id]

        worker_summaries = []
        for i, task in enumerate(team.subtasks, 1):
            status = "COMPLETED" if task.result else "FAILED"
            output = task.result or task.error or "(no output)"
            worker_summaries.append(
                f"### Worker {i}: {task.title}\n"
                f"**Status:** {status}\n"
                f"**Output:**\n{output}\n"
            )

        # Lead uses Claude for synthesis (this is coordination, not implementation)
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=LEAD_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"## Original Idea\n{team.idea}\n\n"
                        f"## Team Plan\n{team.plan.title}: {team.plan.description}\n\n"
                        f"## Worker Results\n\n{''.join(worker_summaries)}\n\n"
                        "Synthesize these results into a coherent final summary. "
                        "As the team lead, summarize what was accomplished and "
                        "identify any remaining gaps."
                    ),
                }
            ],
        )

        return response.content[0].text

    # ── Agent Result Waiting ─────────────────────────────────────────────

    async def _wait_for_agent_result(
        self,
        team_id: str,
        task_id: str,
        timeout: float = 300.0,
    ) -> dict:
        """
        Wait for a worker pod to report its result via callback.

        Creates a Future that is resolved by handle_worker_result()
        when the pod POSTs back.
        """
        future_key = f"{team_id}:{task_id}"
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending_results[future_key] = future

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self._pending_results.pop(future_key, None)
            logger.warning("Timeout waiting for task %s in team %s", task_id, team_id)
            return {"status": "failed", "error": "Pod timed out", "result": None}

    # ── Plan Parsing ─────────────────────────────────────────────────────

    def _parse_plan(self, plan_text: str, idea: str) -> TeamPlan:
        """Parse a plan from the planner agent's output."""
        # Try to extract JSON from the response
        try:
            # Strip markdown fences if present
            raw = plan_text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            data = json.loads(raw)
            return TeamPlan(
                title=data.get("title", "Implementation Plan"),
                description=data.get("description", idea),
                phases=data.get("phases", []),
                role_assignments=data.get("role_assignments", {}),
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback: create a simple plan from the text
            return TeamPlan(
                title="Implementation Plan",
                description=plan_text[:500] if plan_text else idea,
                phases=[{"name": "Implementation", "tasks": [{"name": "Execute plan", "description": plan_text or idea}]}],
                role_assignments={},
            )

    # ── SSE Event Broadcasting ───────────────────────────────────────────

    async def subscribe(self, team_id: str) -> AsyncGenerator[TeamEvent, None]:
        """Subscribe to real-time SSE events for a team."""
        q: asyncio.Queue[Optional[TeamEvent]] = asyncio.Queue()
        async with self._lock:
            if team_id not in self._event_queues:
                return
            self._event_queues[team_id].append(q)

        try:
            while True:
                event = await q.get()
                if event is None:
                    break
                yield event
        finally:
            async with self._lock:
                if team_id in self._event_queues:
                    try:
                        self._event_queues[team_id].remove(q)
                    except ValueError:
                        pass

    async def _emit(self, team_id: str, event_type: str, data: dict) -> None:
        """Broadcast an SSE event to all subscribers."""
        event = TeamEvent(event_type=event_type, team_id=team_id, data=data)

        # Also log to team event log
        team = self._teams.get(team_id)
        if team:
            team.event_log.append({
                "type": event_type,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        async with self._lock:
            for q in self._event_queues.get(team_id, []):
                await q.put(event)

    async def _close_subscribers(self, team_id: str) -> None:
        """Signal all subscribers that the stream is done."""
        async with self._lock:
            for q in self._event_queues.get(team_id, []):
                await q.put(None)
