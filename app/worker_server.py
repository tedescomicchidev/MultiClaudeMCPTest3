"""
Worker Server — standalone process that runs inside each agent pod.

Each Kubernetes pod runs this server as its main process. On startup it:
1. Reads the subtask from environment variables
2. Calls the Claude API with the role-specific system prompt
3. POSTs the result back to the orchestrator's callback endpoint
4. Exits (pod terminates with success or failure)

This replaces the in-process execute_subtask() coroutine from the
asyncio-based architecture. Each worker now runs in complete isolation
with its own container, network namespace, and context window.

Usage (inside a pod):
    python -m app.worker_server

Environment variables (set by K8sPodManager):
    TASK_ID, TASK_TITLE, TASK_DESCRIPTION, SWARM_ID, WORKER_ID,
    AGENT_ROLE, SYSTEM_PROMPT, CALLBACK_URL, ANTHROPIC_API_KEY,
    CLAUDE_MODEL
"""

from __future__ import annotations

import json
import logging
import os
import sys

import anthropic
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] worker: %(message)s",
)
logger = logging.getLogger(__name__)


def get_required_env(name: str) -> str:
    """Read a required environment variable or exit."""
    value = os.environ.get(name)
    if not value:
        logger.error("Missing required environment variable: %s", name)
        sys.exit(1)
    return value


def run_worker() -> None:
    """
    Main entry point for the worker pod.

    Reads task from environment, calls Claude, and reports back.
    Runs synchronously since each pod handles exactly one task.
    """
    task_id = get_required_env("TASK_ID")
    task_title = get_required_env("TASK_TITLE")
    task_description = get_required_env("TASK_DESCRIPTION")
    swarm_id = get_required_env("SWARM_ID")
    worker_id = get_required_env("WORKER_ID")
    agent_role = os.environ.get("AGENT_ROLE", "worker")
    system_prompt = get_required_env("SYSTEM_PROMPT")
    callback_url = get_required_env("CALLBACK_URL")
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    logger.info(
        "Worker %s starting (role=%s, task=%s: %s)",
        worker_id, agent_role, task_id, task_title,
    )

    result = None
    error = None
    status = "completed"

    try:
        # ── Call Claude API ──────────────────────────────────────────
        api_client = anthropic.Anthropic()

        response = api_client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"## Your Assigned Subtask\n"
                        f"**Title:** {task_title}\n\n"
                        f"**Instructions:**\n{task_description}"
                    ),
                }
            ],
        )

        result = response.content[0].text
        logger.info("Worker %s completed task %s", worker_id, task_id)

    except Exception as exc:
        logger.exception("Worker %s failed on task %s", worker_id, task_id)
        error = str(exc)
        status = "failed"

    # ── Report result back to orchestrator ────────────────────────────
    payload = {
        "task_id": task_id,
        "worker_id": worker_id,
        "swarm_id": swarm_id,
        "status": status,
        "result": result,
        "error": error,
    }

    try:
        with httpx.Client(timeout=30.0) as http:
            resp = http.post(callback_url, json=payload)
            resp.raise_for_status()
            logger.info(
                "Worker %s reported result to orchestrator (status=%d)",
                worker_id, resp.status_code,
            )
    except Exception as exc:
        # If callback fails, log the result so it can be retrieved from pod logs
        logger.error("Failed to report result to orchestrator: %s", exc)
        logger.info("RESULT_PAYLOAD: %s", json.dumps(payload))

    # Exit with appropriate code
    if status == "failed":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    run_worker()
