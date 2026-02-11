# Orchestrator — Production Container Image
#
# Runs the Meta-Agent Orchestrator which manages agent teams and swarms
# on Kubernetes. Creates worker pods via the K8s API.
#
# Based on the production blueprint from "Running Claude Code in Kubernetes":
# - Minimal base image (Alpine) to reduce attack surface
# - Non-root user for least-privilege execution
# - Only outbound HTTPS to api.anthropic.com required
#
# Build:
#   docker build -t parallel-agent-scheduler .
#
# Run (local dev):
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... parallel-agent-scheduler

FROM python:3.12-alpine

# Runtime dependencies
RUN apk add --no-cache tini

# Non-root user (as per the blueprint's security requirements)
RUN adduser -D -h /app agent
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Switch to non-root
USER agent

# Expose the web port
EXPOSE 8000

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["tini", "--"]

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
