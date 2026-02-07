"""
Uber-Agent (Meta-Agent Orchestrator).

From the PDF (p.6):
  "In a swarm architecture, a 'Meta-Agent' runs in a specialized mode where
   it focuses on task decomposition rather than code writing. The Meta-Agent
   analyzes the requirements, breaks them into independent, parallelizable
   tasks, and queues them in a system like Redis."

This module implements the uber-agent that:
1. Receives the user's idea
2. Calls Claude to decompose it into N independent subtasks
3. Enqueues each subtask for parallel worker execution
4. Synthesizes final results once all workers complete
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import anthropic

from app.models import SubTask

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DECOMPOSITION_SYSTEM_PROMPT = """\
You are a Meta-Agent Orchestrator. Your sole job is to decompose a user's idea \
into exactly {num_agents} independent, parallelizable subtasks that can each be \
assigned to a separate worker agent.

Rules:
- Each subtask MUST be independent (no dependencies between subtasks).
- Each subtask should be roughly equal in scope.
- Output ONLY valid JSON — an array of objects with "title" and "description" keys.
- "title" should be short (< 80 chars).
- "description" should be a clear, self-contained instruction a worker agent can \
  execute without seeing the other subtasks.

Example output:
[
  {{"title": "Design the database schema", "description": "Create a PostgreSQL schema ..."}},
  {{"title": "Build the REST API", "description": "Implement FastAPI endpoints ..."}}
]
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are a Meta-Agent Orchestrator reviewing the completed results from a swarm \
of parallel worker agents. Synthesize their outputs into a single coherent \
summary that addresses the original idea.

Be concise but thorough. Highlight key contributions from each worker and note \
any gaps or conflicts between their outputs.
"""


async def decompose_idea(
    idea: str,
    num_agents: int,
    client: anthropic.AsyncAnthropic,
    model: str = "claude-sonnet-4-20250514",
) -> list[SubTask]:
    """Use Claude to break the user's idea into N independent subtasks."""
    logger.info("Decomposing idea into %d subtasks", num_agents)

    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=DECOMPOSITION_SYSTEM_PROMPT.format(num_agents=num_agents),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Decompose this idea into exactly {num_agents} independent, "
                    f"parallelizable subtasks:\n\n{idea}"
                ),
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    tasks_data = json.loads(raw)

    subtasks = []
    for item in tasks_data:
        subtasks.append(
            SubTask(
                title=item["title"],
                description=item["description"],
            )
        )
    logger.info("Created %d subtasks", len(subtasks))
    return subtasks


async def synthesize_results(
    idea: str,
    subtasks: list[SubTask],
    client: anthropic.AsyncAnthropic,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Combine all worker results into a final synthesis."""
    logger.info("Synthesizing results from %d subtasks", len(subtasks))

    worker_summaries = []
    for i, task in enumerate(subtasks, 1):
        status = "COMPLETED" if task.result else "FAILED"
        output = task.result or task.error or "(no output)"
        worker_summaries.append(
            f"### Worker {i}: {task.title}\n"
            f"**Status:** {status}\n"
            f"**Output:**\n{output}\n"
        )

    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"## Original Idea\n{idea}\n\n"
                    f"## Worker Results\n\n{''.join(worker_summaries)}\n\n"
                    "Please synthesize these results into a coherent final summary."
                ),
            }
        ],
    )

    return response.content[0].text
