"""
Worker Agent — executes a single subtask using Claude.

From the PDF (p.3, 7):
  "The SDK is particularly suited for building 'swarms' of agents that coordinate
   through a shared state model."

  "Parallel spawning using the SDK has been clocked at 50-75ms per agent, a
   10-20x improvement over the 750ms required for sequential spawning."

Each worker agent:
1. Receives a SubTask from the queue
2. Calls Claude with the subtask description as a focused prompt
3. Returns the result (or error) to the swarm manager
"""

from __future__ import annotations

import logging
from typing import Optional

import anthropic

from app.models import SubTask

logger = logging.getLogger(__name__)

WORKER_SYSTEM_PROMPT = """\
You are a focused Worker Agent in a parallel agent swarm. You have been assigned \
one specific subtask. Complete it thoroughly and return a clear, structured result.

Guidelines:
- Stay focused on YOUR assigned subtask only.
- Be thorough but concise in your output.
- If the task asks for code, provide working code with brief explanations.
- If the task asks for analysis, provide structured insights.
- Do not reference or depend on other workers' outputs.
"""


async def execute_subtask(
    subtask: SubTask,
    client: anthropic.AsyncAnthropic,
    model: str = "claude-sonnet-4-20250514",
    worker_id: Optional[str] = None,
) -> str:
    """
    Execute a single subtask by calling Claude.

    This simulates what the Claude Agent SDK does in-process with sub-millisecond
    tool calls. In a full production deployment, each worker would run inside its
    own container/pod with isolated context windows per the blueprint.
    """
    agent_label = worker_id or subtask.agent_id or subtask.id
    logger.info("Worker %s starting subtask: %s", agent_label, subtask.title)

    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=WORKER_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"## Your Assigned Subtask\n"
                    f"**Title:** {subtask.title}\n\n"
                    f"**Instructions:**\n{subtask.description}"
                ),
            }
        ],
    )

    result = response.content[0].text
    logger.info("Worker %s completed subtask: %s", agent_label, subtask.title)
    return result
