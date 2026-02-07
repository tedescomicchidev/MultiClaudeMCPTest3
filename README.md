# Parallel Agent Scheduler

A Python web app that schedules user-defined numbers of parallel Claude agents
working as a swarm. Built following the **Meta-Agent Orchestrator pattern** from
the "Running Claude Code in Kubernetes" production blueprint.

## Architecture

```
User Request
    |
    v
[FastAPI Web App] --SSE--> [Browser UI]
    |
    v
[Uber-Agent / Meta-Agent Orchestrator]
    |  Decomposes idea into N independent subtasks
    |
    +---> [Task Queue] (in-memory; Redis in production)
    |        |
    |        +---> [Worker Agent 0] --|
    |        +---> [Worker Agent 1] --|--> Parallel Execution
    |        +---> [Worker Agent N] --|       (asyncio.gather)
    |
    v
[Synthesis] -- Uber-agent combines all worker results
```

**Key patterns from the production blueprint:**

- **Meta-Agent Orchestrator** (PDF p.6): Uber-agent decomposes tasks, workers
  execute in parallel, results are synthesized
- **Parallel spawning** (PDF p.7): Workers launched via `asyncio.gather`,
  simulating the SDK's 50-75ms session forking
- **Task queue** (PDF p.6): In-memory async queue (swap for Redis in production)
- **SSE streaming** (PDF p.5): Real-time progress via Server-Sent Events
- **Non-root container** (PDF p.8): Dockerfile runs as unprivileged `agent` user

## Quick Start

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## Docker

```bash
docker build -t parallel-agent-scheduler .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... parallel-agent-scheduler
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Model to use for agents |

## Project Structure

```
app/
├── main.py            # FastAPI entry point, routes, SSE endpoint
├── models.py          # Pydantic models (SwarmState, SubTask, etc.)
├── uber_agent.py      # Meta-Agent: idea decomposition + synthesis
├── worker_agent.py    # Worker agent: executes a single subtask
├── swarm_manager.py   # Orchestrates full swarm lifecycle
├── task_queue.py      # Async task queue (in-memory / Redis)
└── static/
    └── index.html     # Single-page frontend with SSE client
```

## Production Notes

For Kubernetes deployment (per the blueprint):

- Use **StatefulSets** with dedicated PVCs for session-persistent agents
- Deploy a **Redis** instance to replace the in-memory task queue
- Apply **NetworkPolicies** restricting egress to `api.anthropic.com` only
- Use the **Proxy Pattern** (Envoy sidecar) to avoid API keys in agent containers
- Consider **gVisor** or **Kata Containers** for kernel-level isolation
- Scale workers via **HPA** driven by Redis queue depth
