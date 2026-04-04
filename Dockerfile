FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

# Mount your config at /app/agent.json
CMD ["colony-agent", "run", "--config", "agent.json"]
