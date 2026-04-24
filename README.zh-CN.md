# GLM API Monitor Proxy

智谱 GLM API 的透明反向代理，用于统计 token 用量、工具调用、请求延迟和错误率。数据存储在 TimescaleDB 中，通过 Grafana 可视化。

[English](README.md)

## 工作原理

```
Client → Proxy (:8000) → open.bigmodel.cn
               ↓
        统计数据提取
               ↓
        TimescaleDB ← Grafana
```

客户端只需将 API host 从 `open.bigmodel.cn` 改为代理地址，其余（API key、请求路径、参数）完全不变。请求和响应 body 不会被存储。

## 统计内容

- **Token 用量**：prompt_tokens、completion_tokens、total_tokens
- **工具调用**：被调用的函数名称及次数
- **请求延迟**：每次请求的耗时（ms）
- **错误追踪**：HTTP 错误码、超时记录
- **API Key 分组**：按脱敏后的 API key（前6位...后6位）分类统计

## 快速开始

### 方式 A：使用预构建镜像（推荐）

Docker Hub: [realjustinwu/glm-api-monitor](https://hub.docker.com/r/realjustinwu/glm-api-monitor)

创建 `docker-compose.yml`：

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

或仅使用 Docker 运行（需要单独的 TimescaleDB 实例）：

```bash
docker run -d --name glm-monitor \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://glm:glm@your-db-host:5432/glm_monitor \
  realjustinwu/glm-api-monitor:latest
```

### 方式 B：从源码构建

```bash
git clone https://github.com/realjustinwu/glm-api-monitor.git
cd glm-api-monitor
docker compose up -d
```

### 修改客户端配置

将 GLM API 的 base URL 改为代理地址：

```python
# 原来
client = ZhipuAI(api_key="your-api-key")

# 改为（假设代理运行在本机 8000 端口）
client = ZhipuAI(api_key="your-api-key", base_url="http://localhost:8000/api/paas/v4")
```

或使用 OpenAI 兼容接口：

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-glm-api-key",
    base_url="http://localhost:8000/api/paas/v4"
)
```

### 3. 配置 Grafana

1. 在 Grafana 中添加 PostgreSQL 数据源，指向 TimescaleDB：
   - Host: `localhost:5432`
   - Database: `glm_monitor`
   - User: `glm`
   - Password: `glm`

2. 导入 Dashboard：`grafana/glm-monitor.json`

## 配置项

通过环境变量或 `.env` 文件配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GLM_UPSTREAM_URL` | `https://open.bigmodel.cn` | 上游 API 地址 |
| `DATABASE_URL` | `postgresql://glm:glm@timescaledb:5432/glm_monitor` | 数据库连接字符串 |
| `PROXY_PORT` | `8000` | 代理监听端口 |
| `REQUEST_TIMEOUT` | `120` | 上游请求超时（秒） |

## 数据库表结构

### api_requests（按天分区）

| 列 | 类型 | 说明 |
|----|------|------|
| time | timestamptz | 请求时间 |
| request_id | text | GLM 请求 ID |
| api_key | text | 脱敏 API key（前6...后6） |
| model | text | 模型名称 |
| is_streaming | boolean | 是否流式 |
| prompt_tokens | integer | 输入 token 数 |
| completion_tokens | integer | 输出 token 数 |
| total_tokens | integer | 总 token 数 |
| finish_reason | text | 结束原因 |
| latency_ms | float | 请求耗时 ms |
| error_message | text | 错误信息 |

### tool_calls（按天分区）

| 列 | 类型 | 说明 |
|----|------|------|
| time | timestamptz | 请求时间 |
| request_id | text | 关联请求 ID |
| tool_name | text | 工具/函数名称 |

### 连续聚合视图

- `api_requests_hourly`：按小时聚合（token、请求数、平均延迟、错误数）
- `api_requests_daily`：按天聚合
- `tool_calls_hourly`：按小时聚合工具调用次数

原始数据保留 90 天，聚合数据永久保留。

## 本地开发

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

## 项目结构

```
app/
├── main.py      # FastAPI 代理入口
├── config.py    # 环境变量配置
├── db.py        # 数据库连接池和建表
├── stats.py     # 统计数据提取
└── writer.py    # 异步写入数据库
grafana/
└── glm-monitor.json  # Grafana Dashboard
tests/                 # 16 个单元测试
```
