# GLM API Monitor Proxy

A transparent reverse proxy for Zhipu GLM API that tracks token usage, tool calls, request latency, and error rates. Data is stored in TimescaleDB and visualized through Grafana.

[中文文档](README.zh-CN.md)

## How It Works

```
Client → Proxy (:8000) → open.bigmodel.cn
               ↓
        Stats Extraction
               ↓
        TimescaleDB ← Grafana
```

Clients only need to change the API host to point at the proxy. Everything else (API key, request paths, parameters) stays the same. Request and response bodies are **never** stored.

## What It Tracks

- **Token usage**: prompt_tokens, completion_tokens, total_tokens
- **Tool calls**: function names and call counts
- **Request latency**: per-request duration (ms)
- **Error tracking**: HTTP error codes, timeout records
- **API key grouping**: stats grouped by masked API key (first6...last6)

## Quick Start

### Option A: Use Pre-built Image (Recommended)

Docker Hub: [realjustinwu/glm-api-monitor](https://hub.docker.com/r/realjustinwu/glm-api-monitor)

Create a `docker-compose.yml`:

```yaml
services:
  proxy:
    image: realjustinwu/glm-api-monitor:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://glm:glm@timescaledb:5432/glm_monitor
    depends_on:
      timescaledb:
        condition: service_healthy
    restart: unless-stopped

  timescaledb:
    image: timescale/timescaledb:latest-pg16
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=glm
      - POSTGRES_PASSWORD=glm
      - POSTGRES_DB=glm_monitor
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U glm -d glm_monitor"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  timescaledb_data:
```

Or run with Docker only (you need a separate TimescaleDB instance):

```bash
docker run -d --name glm-monitor \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://glm:glm@your-db-host:5432/glm_monitor \
  realjustinwu/glm-api-monitor:latest
```

### Option B: Build from Source

```bash
git clone https://github.com/realjustinwu/glm-api-monitor.git
cd glm-api-monitor
docker compose up -d
```

### Configure Client

Change the GLM API base URL to the proxy address:

```python
# Before
client = ZhipuAI(api_key="your-api-key")

# After (assuming proxy runs on localhost:8000)
client = ZhipuAI(api_key="your-api-key", base_url="http://localhost:8000/api/paas/v4")
```

Or with OpenAI-compatible interface:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-glm-api-key",
    base_url="http://localhost:8000/api/paas/v4"
)
```

### 3. Configure Grafana

1. Add a PostgreSQL datasource in Grafana pointing to TimescaleDB:
   - Host: `localhost:5432`
   - Database: `glm_monitor`
   - User: `glm`
   - Password: `glm`

2. Import Dashboard: `grafana/glm-monitor.json`

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_UPSTREAM_URL` | `https://open.bigmodel.cn` | Upstream API base URL |
| `DATABASE_URL` | `postgresql://glm:glm@timescaledb:5432/glm_monitor` | Database connection string |
| `PROXY_PORT` | `8000` | Proxy listen port |
| `REQUEST_TIMEOUT` | `120` | Upstream request timeout (seconds) |

## Database Schema

### api_requests (partitioned by day)

| Column | Type | Description |
|--------|------|-------------|
| time | timestamptz | Request timestamp |
| request_id | text | GLM request ID |
| api_key | text | Masked API key (first6...last6) |
| model | text | Model name |
| is_streaming | boolean | Streaming or not |
| prompt_tokens | integer | Input token count |
| completion_tokens | integer | Output token count |
| total_tokens | integer | Total token count |
| finish_reason | text | Finish reason |
| latency_ms | float | Request duration in ms |
| error_message | text | Error message |

### tool_calls (partitioned by day)

| Column | Type | Description |
|--------|------|-------------|
| time | timestamptz | Request timestamp |
| request_id | text | Associated request ID |
| tool_name | text | Function name |

### Continuous Aggregates

- `api_requests_hourly`: hourly aggregation (tokens, request count, avg latency, errors)
- `api_requests_daily`: daily aggregation
- `tool_calls_hourly`: hourly tool call counts

Raw data retained for 90 days, aggregates kept indefinitely.

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the proxy locally
uvicorn app.main:app --port 8000
```

If you don't have TimescaleDB running locally, the proxy will still work — requests are forwarded normally and stats are logged (writes to DB will fail gracefully).

### Run Unit Tests

```bash
pytest tests/ -v
```

### Run Integration Tests

Integration tests start a local proxy and make real API calls to Zhipu:

```bash
GLM_API_KEY=your-key pytest -m integration -v -s
```

## Project Structure

```
app/
├── main.py      # FastAPI proxy entry point
├── config.py    # Environment configuration
├── db.py        # Database connection pool and schema
├── stats.py     # Stats extraction logic
└── writer.py    # Async database writer
grafana/
└── glm-monitor.json  # Grafana dashboard
tests/                 # 16 unit tests
```

## License

MIT
