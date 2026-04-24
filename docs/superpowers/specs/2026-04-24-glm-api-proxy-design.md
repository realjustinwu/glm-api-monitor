# GLM API Monitor Proxy - Design Spec

## Overview

A transparent reverse proxy for Zhipu GLM API that intercepts requests, forwards them upstream, extracts usage statistics from responses, and stores them in TimescaleDB. Clients only need to change the API host to point at this proxy.

## Architecture

```
Client → FastAPI Proxy (:8000) → open.bigmodel.cn
                ↓
         Stats Extractor
                ↓
         TimescaleDB (:5432) ← Grafana (:3000)
```

All deployed via Docker Compose.

## Tech Stack

- **Proxy**: Python 3.12 + FastAPI + httpx (async HTTP client)
- **Database**: TimescaleDB (PostgreSQL extension)
- **Visualization**: Grafana with PostgreSQL datasource
- **Deployment**: Docker Compose

## Components

### 1. FastAPI Proxy Service

Single service handling all GLM API paths under `/api/paas/v4/{path}`:

- Extract API key from `Authorization: Bearer <key>` header
- Extract model name from request body
- Record request start time
- Forward request to `https://open.bigmodel.cn/api/paas/v4/{path}` with original headers
- Parse response to extract statistics
- Return response to client unchanged

**Streaming handling**: SSE responses are streamed through to client in real-time. The proxy buffers the last chunk containing `usage` and `finish_reason` to extract stats after stream ends.

### 2. Statistics Extractor

Extracts from each response:

- **Token usage**: `prompt_tokens`, `completion_tokens`, `total_tokens`
- **Tool calls**: each `tool_calls[].function.name` (one row per tool call)
- **Metadata**: request_id, model, finish_reason, latency_ms
- **Error info**: HTTP status code, error message on failure
- **API key**: first 6 chars + "..." + last 6 chars (e.g. `abc123...xyz789`)

### 3. Database Schema

**Table `api_requests`** (hypertable, partitioned by `time`):

| Column | Type | Description |
|--------|------|-------------|
| time | timestamptz | Request timestamp |
| request_id | text | GLM request ID |
| api_key | text | Masked key (first6...last6) |
| model | text | Model name |
| is_streaming | boolean | Whether streaming |
| prompt_tokens | integer | Input tokens |
| completion_tokens | integer | Output tokens |
| total_tokens | integer | Total tokens |
| finish_reason | text | stop/tool_calls/length/sensitive/error |
| latency_ms | float | Request duration in ms |
| error_message | text | Error message (nullable) |

**Table `tool_calls`** (hypertable, partitioned by `time`):

| Column | Type | Description |
|--------|------|-------------|
| time | timestamptz | Request timestamp |
| request_id | text | Associated request ID |
| tool_name | text | Function name called |

**Continuous Aggregates**:

- `api_requests_hourly`: sum of tokens, count of requests, avg latency — grouped by (time_bucket(1h), api_key, model)
- `tool_calls_hourly`: count of calls — grouped by (time_bucket(1h), api_key, tool_name)
- `api_requests_daily`: same as hourly but daily bucket

**Retention policy**: raw data kept 90 days, daily aggregates kept indefinitely.

### 4. Grafana Dashboard

Pre-provisioned dashboards via provisioning:

- Token usage over time (by model, by API key)
- Request count and latency distribution
- Error rate trend
- Tool call frequency (top tools, by API key)
- Daily/weekly/monthly cost estimation

### 5. Docker Compose

Three services:

- `proxy`: FastAPI app on port 8000
- `timescaledb`: TimescaleDB on port 5432
- `grafana`: Grafana on port 3000

## Request Flow

### Non-streaming

1. Client sends `POST /api/paas/v4/chat/completions`
2. Proxy records start time, extracts API key + model from request
3. Forwards to `https://open.bigmodel.cn/api/paas/v4/chat/completions`
4. Receives full response, extracts `usage`, `finish_reason`, `tool_calls`
5. Writes stats to TimescaleDB
6. Returns response to client unchanged

### Streaming

1. Same as above for request handling
2. Proxy opens SSE stream to upstream
3. Each chunk is forwarded to client immediately (no buffering delay)
4. Proxy tracks the last chunk with `usage`/`finish_reason`
5. After stream ends (`data: [DONE]`), writes stats to TimescaleDB

### Error Cases

- Upstream returns HTTP error: record status code + error message, return error to client
- Upstream timeout: record timeout error, return 504 to client
- Proxy internal error: return 500 to client, log error

## Security

- Full API keys are never stored — only masked version (first6...last6)
- Request/response bodies are NOT stored
- Database credentials managed via Docker environment variables
- Proxy does not add or modify any authentication

## Configuration

Environment variables:

- `GLM_UPSTREAM_URL`: upstream base URL (default: `https://open.bigmodel.cn`)
- `DATABASE_URL`: TimescaleDB connection string
- `PROXY_PORT`: proxy listen port (default: 8000)
- `REQUEST_TIMEOUT`: upstream request timeout in seconds (default: 120)
