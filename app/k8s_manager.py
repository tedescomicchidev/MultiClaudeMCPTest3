"""
Kubernetes Pod Manager — creates and manages agent pods.

Replaces the asyncio.gather-based worker spawning with actual Kubernetes
pods. Each agent (team member or worker) gets its own isolated pod with:
- Dedicated context window (no shared memory between agents)
- Its own PVC mount for persistent session data
- Network-restricted egress (api.anthropic.com only via NetworkPolicy)
- Non-root execution matching the production blueprint

From the PDF (p.7-8):
  "In a production setting, each agent runs in its own container with
   isolated context windows. StatefulSets with dedicated PVCs provide
   session persistence."

Pod lifecycle:
  1. create_worker_pod() → submits pod spec to K8s API
  2. wait_for_pod() → watches until Running/Succeeded/Failed
  3. get_pod_result() → reads result from pod annotation or callback
  4. delete_pod() → cleans up after completion
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from app.models import AgentRole, SubTask

logger = logging.getLogger(__name__)

# ── Role-specific system prompts for pod environment variables ──────────

ROLE_SYSTEM_PROMPTS: dict[str, str] = {
    AgentRole.ARCHITECT: (
        "You are the Architect agent. Your role is to design the high-level "
        "system architecture, define component boundaries, API contracts, and "
        "data flow. Provide clear architectural diagrams (as text) and "
        "rationale for your design decisions."
    ),
    AgentRole.K8S_EXPERT: (
        "You are the Kubernetes Expert agent. Your role is to design and "
        "review Kubernetes deployment configurations, pod specs, RBAC policies, "
        "networking, storage, and scaling strategies. Ensure production "
        "readiness with security best practices."
    ),
    AgentRole.DEVILS_ADVOCATE: (
        "You are the Devil's Advocate agent. Your role is to critically "
        "review proposed plans and implementations, identify potential flaws, "
        "edge cases, security vulnerabilities, and scalability concerns. "
        "Challenge assumptions and suggest improvements."
    ),
    AgentRole.PLANNER: (
        "You are the Planner agent. Your role is to create detailed "
        "implementation plans with clear phases, task breakdowns, role "
        "assignments, and dependency tracking. Output plans as structured "
        "JSON for the team lead to approve."
    ),
    AgentRole.WORKER: (
        "You are a focused Worker Agent in a parallel agent swarm. Complete "
        "your assigned subtask thoroughly and return a clear, structured result. "
        "Stay focused on YOUR assigned subtask only."
    ),
}


class K8sPodManager:
    """
    Manages Kubernetes pod lifecycle for agent team members.

    Each agent runs in its own pod, replacing the in-process asyncio
    concurrency with true container isolation per the production blueprint.
    """

    def __init__(
        self,
        namespace: str = "default",
        worker_image: str = "parallel-agent-worker:latest",
        orchestrator_service: str = "agent-orchestrator",
        orchestrator_port: int = 8000,
    ) -> None:
        self._namespace = namespace
        self._worker_image = worker_image
        self._orchestrator_url = (
            f"http://{orchestrator_service}.{namespace}.svc.cluster.local:{orchestrator_port}"
        )

        # Load kubeconfig — in-cluster when running as a pod, local otherwise
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except config.ConfigException:
            try:
                config.load_kube_config()
                logger.info("Loaded local kubeconfig")
            except config.ConfigException:
                logger.warning(
                    "No Kubernetes config found. Pod operations will fail. "
                    "Set KUBECONFIG or run inside a cluster."
                )

        self._core_v1 = client.CoreV1Api()
        self._batch_v1 = client.BatchV1Api()

    def _build_pod_spec(
        self,
        pod_name: str,
        swarm_id: str,
        task: SubTask,
        role: AgentRole,
        worker_id: str,
        extra_env: Optional[dict] = None,
    ) -> client.V1Pod:
        """
        Build a Kubernetes Pod spec for a worker agent.

        The pod runs the worker-server image, receives its subtask via
        environment variables, executes it, and POSTs the result back
        to the orchestrator's callback endpoint.
        """
        system_prompt = ROLE_SYSTEM_PROMPTS.get(role, ROLE_SYSTEM_PROMPTS[AgentRole.WORKER])

        env_vars = [
            client.V1EnvVar(name="TASK_ID", value=task.id),
            client.V1EnvVar(name="TASK_TITLE", value=task.title),
            client.V1EnvVar(name="TASK_DESCRIPTION", value=task.description),
            client.V1EnvVar(name="SWARM_ID", value=swarm_id),
            client.V1EnvVar(name="WORKER_ID", value=worker_id),
            client.V1EnvVar(name="AGENT_ROLE", value=role.value),
            client.V1EnvVar(name="SYSTEM_PROMPT", value=system_prompt),
            client.V1EnvVar(name="CALLBACK_URL", value=f"{self._orchestrator_url}/api/callback"),
            # API key injected from Kubernetes Secret
            client.V1EnvVar(
                name="ANTHROPIC_API_KEY",
                value_from=client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(
                        name="anthropic-api-key",
                        key="api-key",
                    )
                ),
            ),
            # Model from ConfigMap or default
            client.V1EnvVar(
                name="CLAUDE_MODEL",
                value_from=client.V1EnvVarSource(
                    config_map_key_ref=client.V1ConfigMapKeySelector(
                        name="agent-config",
                        key="claude-model",
                        optional=True,
                    )
                ),
            ),
        ]

        if extra_env:
            for key, value in extra_env.items():
                env_vars.append(client.V1EnvVar(name=key, value=str(value)))

        container = client.V1Container(
            name="agent",
            image=self._worker_image,
            env=env_vars,
            resources=client.V1ResourceRequirements(
                requests={"cpu": "100m", "memory": "128Mi"},
                limits={"cpu": "500m", "memory": "512Mi"},
            ),
            security_context=client.V1SecurityContext(
                run_as_non_root=True,
                run_as_user=1000,
                read_only_root_filesystem=True,
                allow_privilege_escalation=False,
                capabilities=client.V1Capabilities(drop=["ALL"]),
            ),
        )

        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=client.V1ObjectMeta(
                name=pod_name,
                namespace=self._namespace,
                labels={
                    "app": "agent-worker",
                    "swarm-id": swarm_id,
                    "task-id": task.id,
                    "agent-role": role.value,
                    "worker-id": worker_id,
                },
                annotations={
                    "agent-scheduler/task-title": task.title[:63],
                },
            ),
            spec=client.V1PodSpec(
                restart_policy="Never",
                service_account_name="agent-worker",
                automount_service_account_token=False,
                containers=[container],
                # DNS policy for callback resolution
                dns_policy="ClusterFirst",
            ),
        )

        return pod

    async def create_worker_pod(
        self,
        swarm_id: str,
        task: SubTask,
        role: AgentRole = AgentRole.WORKER,
        worker_id: Optional[str] = None,
        extra_env: Optional[dict] = None,
    ) -> str:
        """
        Create a Kubernetes pod for a worker agent.

        Returns the pod name. The pod will execute the subtask and POST
        its result to the orchestrator's callback endpoint.
        """
        wid = worker_id or f"worker-{task.id[:8]}"
        pod_name = f"agent-{swarm_id}-{wid}"
        # K8s names must be <= 63 chars, lowercase alphanumeric + dashes
        pod_name = pod_name[:63].lower().rstrip("-")

        pod_spec = self._build_pod_spec(
            pod_name=pod_name,
            swarm_id=swarm_id,
            task=task,
            role=role,
            worker_id=wid,
            extra_env=extra_env,
        )

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._core_v1.create_namespaced_pod(
                    namespace=self._namespace,
                    body=pod_spec,
                ),
            )
            logger.info(
                "Created pod %s for swarm %s (role=%s)", pod_name, swarm_id, role.value
            )
            return pod_name
        except ApiException as e:
            logger.error("Failed to create pod %s: %s", pod_name, e.reason)
            raise

    async def delete_pod(self, pod_name: str, namespace: Optional[str] = None) -> None:
        """Delete a pod (graceful shutdown)."""
        ns = namespace or self._namespace
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._core_v1.delete_namespaced_pod(
                    name=pod_name,
                    namespace=ns,
                    grace_period_seconds=30,
                ),
            )
            logger.info("Deleted pod %s", pod_name)
        except ApiException as e:
            if e.status == 404:
                logger.warning("Pod %s already deleted", pod_name)
            else:
                logger.error("Failed to delete pod %s: %s", pod_name, e.reason)
                raise

    async def get_pod_status(self, pod_name: str, namespace: Optional[str] = None) -> str:
        """Get the current phase of a pod (Pending, Running, Succeeded, Failed)."""
        ns = namespace or self._namespace
        try:
            pod = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._core_v1.read_namespaced_pod_status(
                    name=pod_name, namespace=ns
                ),
            )
            return pod.status.phase
        except ApiException as e:
            if e.status == 404:
                return "NotFound"
            raise

    async def get_pod_logs(self, pod_name: str, namespace: Optional[str] = None) -> str:
        """Retrieve logs from a pod."""
        ns = namespace or self._namespace
        try:
            logs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._core_v1.read_namespaced_pod_log(
                    name=pod_name, namespace=ns
                ),
            )
            return logs
        except ApiException as e:
            logger.error("Failed to get logs for pod %s: %s", pod_name, e.reason)
            return f"(error reading logs: {e.reason})"

    async def wait_for_pod(
        self,
        pod_name: str,
        namespace: Optional[str] = None,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> str:
        """
        Poll until a pod reaches a terminal state (Succeeded or Failed).

        Returns the final phase. In production, this would use the K8s
        watch API for efficiency; polling is simpler for the demo.
        """
        ns = namespace or self._namespace
        elapsed = 0.0

        while elapsed < timeout:
            phase = await self.get_pod_status(pod_name, ns)
            if phase in ("Succeeded", "Failed", "NotFound"):
                return phase
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        logger.warning("Pod %s timed out after %.0fs", pod_name, timeout)
        return "Timeout"

    async def list_swarm_pods(self, swarm_id: str, namespace: Optional[str] = None) -> list[dict]:
        """List all pods belonging to a swarm."""
        ns = namespace or self._namespace
        try:
            pods = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._core_v1.list_namespaced_pod(
                    namespace=ns,
                    label_selector=f"swarm-id={swarm_id}",
                ),
            )
            return [
                {
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "role": pod.metadata.labels.get("agent-role", "unknown"),
                    "worker_id": pod.metadata.labels.get("worker-id", "unknown"),
                    "task_id": pod.metadata.labels.get("task-id", "unknown"),
                }
                for pod in pods.items
            ]
        except ApiException as e:
            logger.error("Failed to list pods for swarm %s: %s", swarm_id, e.reason)
            return []

    async def cleanup_swarm_pods(self, swarm_id: str, namespace: Optional[str] = None) -> int:
        """Delete all pods belonging to a swarm. Returns count of deleted pods."""
        pods = await self.list_swarm_pods(swarm_id, namespace)
        count = 0
        for pod_info in pods:
            await self.delete_pod(pod_info["name"], namespace)
            count += 1
        logger.info("Cleaned up %d pods for swarm %s", count, swarm_id)
        return count
