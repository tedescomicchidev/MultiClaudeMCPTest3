# Parallel Agent Scheduler

A Python web app that orchestrates parallel Claude agents running as
**Kubernetes pods**. Supports two modes: anonymous **Swarm Mode** and
role-based **Agent Team Mode** with delegate lead.

Built following the **Meta-Agent Orchestrator pattern** from the
"Running Claude Code in Kubernetes" production blueprint.

## Architecture

```
User Request
    |
    v
[FastAPI Orchestrator Pod] --SSE--> [Browser UI]
    |                          <--callback-- [Worker Pods]
    v
[Team Lead / Uber-Agent] (delegate mode: coordination only)
    |
    +---> [Planner Pod]      -- creates implementation plan
    +---> [Architect Pod]    -- reviews system design
    +---> [Devil's Advocate Pod] -- challenges assumptions
    +---> [K8s Expert Pod]   -- validates deployment
    |
    +---> Plan Approval Gate (lead must approve before implementation)
    |
    +---> [Worker Pod 0] --|
    +---> [Worker Pod 1] --|--> Parallel Execution (K8s pods)
    +---> [Worker Pod N] --|
    |
    v
[Synthesis] -- Lead combines all results
```

## Two Operating Modes

### Swarm Mode
Anonymous workers execute subtasks in parallel. The uber-agent decomposes
the idea, spawns worker pods, and synthesizes results.

### Team Mode (with Delegate Lead)
Role-specialized agents coordinate through a lead that operates in
**delegate mode** — restricted to coordination-only actions:
- `SPAWN_AGENT` — Create a new agent pod
- `MESSAGE_AGENT` — Send instructions to an agent
- `SHUTDOWN_AGENT` — Terminate an agent pod
- `MANAGE_TASKS` — Assign and track tasks
- `APPROVE_PLAN` / `REJECT_PLAN` — Gate implementation

The lead **cannot** execute subtasks or write code directly.

## Key Changes from tmux-based Architecture

| Before (tmux/asyncio) | After (Kubernetes) |
|---|---|
| `asyncio.gather()` spawns coroutines | `K8sPodManager` creates K8s pods |
| In-process shared memory | Isolated containers with own context |
| tmux sessions for agents | Kubernetes pods per agent |
| No role enforcement | Delegate mode restricts lead to coordination |
| No plan approval | Plan must be approved before implementation |
| Single Dockerfile | Orchestrator + Worker Dockerfiles |

## Quick Start (Local Development)

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
# Build orchestrator image
docker build -t parallel-agent-scheduler .

# Build worker image
docker build -t parallel-agent-worker:latest -f Dockerfile.worker .

# Run orchestrator (local dev — no K8s)
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... parallel-agent-scheduler
```

## Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl create secret generic anthropic-api-key \
  --from-literal=api-key=sk-ant-... \
  -n agent-scheduler
kubectl apply -f k8s/

# Verify
kubectl get pods -n agent-scheduler
kubectl port-forward svc/agent-orchestrator 8000:8000 -n agent-scheduler
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Model for all agents |
| `K8S_NAMESPACE` | `default` | Kubernetes namespace |
| `WORKER_IMAGE` | `parallel-agent-worker:latest` | Worker pod container image |
| `ORCHESTRATOR_SERVICE` | `agent-orchestrator` | K8s service name for callbacks |

## Project Structure

```
app/
├── main.py            # FastAPI entry point, routes, SSE, callback endpoint
├── models.py          # Pydantic models (SwarmState, TeamState, AgentRole, etc.)
├── uber_agent.py      # Meta-Agent: idea decomposition + synthesis
├── worker_agent.py    # Worker agent: executes a single subtask (legacy)
├── worker_server.py   # Standalone worker process for K8s pods
├── swarm_manager.py   # Swarm lifecycle (K8s pod-backed)
├── agent_team.py      # Agent team with delegate mode + plan approval
├── k8s_manager.py     # Kubernetes pod lifecycle management
├── task_queue.py      # Async task queue (in-memory / Redis)
└── static/
    └── index.html     # Frontend with Swarm + Team mode tabs

k8s/
├── namespace.yaml     # agent-scheduler namespace
├── secret.yaml        # Anthropic API key secret
├── configmap.yaml     # Agent configuration
├── rbac.yaml          # ServiceAccounts + Roles for orchestrator & workers
├── deployment.yaml    # Orchestrator deployment
├── service.yaml       # Orchestrator ClusterIP service
└── networkpolicy.yaml # Egress restrictions per blueprint

Dockerfile             # Orchestrator container image
Dockerfile.worker      # Worker pod container image
```

## API Endpoints

### Swarm Mode
| Method | Route | Description |
|---|---|---|
| POST | `/api/swarms` | Create a new swarm |
| GET | `/api/swarms` | List all swarms |
| GET | `/api/swarms/{id}` | Get swarm state |
| GET | `/api/swarms/{id}/stream` | SSE event stream |

### Team Mode
| Method | Route | Description |
|---|---|---|
| POST | `/api/teams` | Create a team (delegate mode) |
| GET | `/api/teams` | List all teams |
| GET | `/api/teams/{id}` | Get team state |
| GET | `/api/teams/{id}/stream` | SSE event stream |
| POST | `/api/teams/{id}/approve-plan` | Approve implementation plan |
| POST | `/api/teams/{id}/reject-plan` | Reject plan with feedback |

### Worker Callback
| Method | Route | Description |
|---|---|---|
| POST | `/api/callback` | Worker pods report results here |

## Production Notes

For Kubernetes deployment (per the blueprint):

- Use **StatefulSets** with dedicated PVCs for session-persistent agents
- Deploy a **Redis** instance to replace the in-memory task queue
- Apply **NetworkPolicies** restricting egress to `api.anthropic.com` only
- Use the **Proxy Pattern** (Envoy sidecar) to avoid API keys in agent containers
- Consider **gVisor** or **Kata Containers** for kernel-level isolation
- Scale workers via **HPA** driven by Redis queue depth
