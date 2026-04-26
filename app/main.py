from __future__ import annotations

import json
import logging
import time

import httpx
from fastapi import FastAPI, Request, Response

from app.config import settings
from app.db import init_db, close_db
from app.stats import mask_api_key, extract_stats, StreamingStatsCollector, extract_anthropic_stats, AnthropicStreamingStatsCollector
from app.writer import write_stats

logger = logging.getLogger(__name__)

app = FastAPI(title="GLM API Monitor Proxy")


@app.on_event("startup")
async def startup():
    try:
        await init_db(settings.database_url)
    except Exception as exc:
        logger.warning("Database unavailable (%s), proxy will run without stats persistence", exc)
    logger.info("Proxy started, upstream=%s", settings.glm_upstream_url)


@app.on_event("shutdown")
async def shutdown():
    try:
        await close_db()
    except Exception:
        pass


async def _proxy_non_streaming(
    client: httpx.AsyncClient, method: str, path: str, request: Request,
    *, api_path: str, extract_fn=extract_stats,
) -> Response:
    start = time.monotonic()
    api_key_raw = ""
    model = ""
    request_body = {}

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key_raw = auth_header[7:]

    body = await request.body()
    try:
        request_body = json.loads(body)
        model = request_body.get("model", "")
    except Exception:
        pass

    is_streaming = request_body.get("stream", False)
    api_key_masked = mask_api_key(api_key_raw) if api_key_raw else ""

    upstream_url = f"{settings.glm_upstream_url}{path}"

    upstream_headers = dict(request.headers)
    upstream_headers.pop("host", None)
    upstream_headers.pop("content-length", None)

    try:
        upstream_resp = await client.request(
            method=method,
            url=upstream_url,
            content=body,
            headers=upstream_headers,
        )
    except httpx.TimeoutException as exc:
        latency = (time.monotonic() - start) * 1000
        await write_stats(stats={
            "request_id": "",
            "api_key": api_key_masked,
            "model": model,
            "is_streaming": is_streaming,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "finish_reason": "error",
            "latency_ms": latency,
            "error_message": f"Timeout: {exc}",
            "tool_calls": [],
            "api_path": api_path,
        })
        return Response(content=str(exc), status_code=504)

    latency = (time.monotonic() - start) * 1000

    # Extract stats
    try:
        resp_json = upstream_resp.json()
    except Exception:
        resp_json = {}

    stats = extract_fn(resp_json if upstream_resp.status_code == 200 else {})
    stats["api_key"] = api_key_masked
    stats["is_streaming"] = False
    stats["latency_ms"] = latency
    stats["api_path"] = api_path

    if upstream_resp.status_code >= 400:
        stats["finish_reason"] = "error"
        stats["error_message"] = f"HTTP {upstream_resp.status_code}"

    await write_stats(stats=stats)

    resp_headers = dict(upstream_resp.headers)
    resp_headers.pop("content-encoding", None)
    resp_headers.pop("transfer-encoding", None)

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=resp_headers,
    )


async def _proxy_streaming(
    client: httpx.AsyncClient, request: Request, path: str,
    *, api_path: str, collector_cls=StreamingStatsCollector,
) -> Response:
    from starlette.responses import StreamingResponse

    start = time.monotonic()
    api_key_raw = ""
    model = ""

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key_raw = auth_header[7:]

    body = await request.body()
    try:
        request_body = json.loads(body)
        model = request_body.get("model", "")
    except Exception:
        pass

    api_key_masked = mask_api_key(api_key_raw) if api_key_raw else ""
    upstream_url = f"{settings.glm_upstream_url}{path}"

    upstream_headers = dict(request.headers)
    upstream_headers.pop("host", None)
    upstream_headers.pop("content-length", None)

    req = client.build_request(
        method="POST",
        url=upstream_url,
        content=body,
        headers=upstream_headers,
    )
    upstream_resp = await client.send(req, stream=True)

    collector = collector_cls()
    api_key_final = api_key_masked
    model_final = model
    api_path_final = api_path

    async def stream_and_collect():
        buffer = ""
        try:
            async for chunk in upstream_resp.aiter_bytes():
                text = chunk.decode("utf-8", errors="replace")
                buffer += text
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    collector.process_chunk(line)
                yield chunk
        finally:
            latency = (time.monotonic() - start) * 1000
            stats = collector.to_stats_dict()
            stats["api_key"] = api_key_final
            stats["model"] = stats["model"] or model_final
            stats["is_streaming"] = True
            stats["latency_ms"] = latency
            stats["api_path"] = api_path_final

            if upstream_resp.status_code >= 400:
                stats["finish_reason"] = "error"
                stats["error_message"] = f"HTTP {upstream_resp.status_code}"

            await write_stats(stats=stats)
            await upstream_resp.aclose()
            await client.aclose()

    stream_headers = dict(upstream_resp.headers)
    stream_headers.pop("content-encoding", None)
    stream_headers.pop("transfer-encoding", None)

    return StreamingResponse(
        stream_and_collect(),
        status_code=upstream_resp.status_code,
        headers=stream_headers,
        media_type="text/event-stream",
    )


@app.api_route("/api/paas/v4/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_route(request: Request, path: str):
    body = await request.body()
    try:
        request_body = json.loads(body)
    except Exception:
        request_body = {}

    is_streaming = request_body.get("stream", False)
    api_path = f"/api/paas/v4/{path}"

    if is_streaming:
        client = httpx.AsyncClient(timeout=settings.request_timeout)
        return await _proxy_streaming(client, request, api_path, api_path=api_path)

    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        method = request.method
        return await _proxy_non_streaming(client, method, api_path, request, api_path=api_path)


@app.api_route("/api/anthropic/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def anthropic_proxy_route(request: Request, path: str):
    body = await request.body()
    try:
        request_body = json.loads(body)
    except Exception:
        request_body = {}

    is_streaming = request_body.get("stream", False)
    api_path = f"/api/anthropic/{path}"

    if is_streaming:
        client = httpx.AsyncClient(timeout=settings.request_timeout)
        return await _proxy_streaming(
            client, request, api_path, api_path=api_path,
            collector_cls=AnthropicStreamingStatsCollector,
        )

    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        method = request.method
        return await _proxy_non_streaming(
            client, method, api_path, request,
            api_path=api_path, extract_fn=extract_anthropic_stats,
        )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def fallback_proxy(request: Request, path: str):
    """Proxy all unmatched requests to upstream and log them for future API support."""
    method = request.method
    full_path = f"/{path}" if path else "/"
    logger.info(
        "Unmatched API request: method=%s path=%s content_length=%s",
        method, full_path, request.headers.get("content-length", "0"),
    )

    body = await request.body()
    upstream_url = f"{settings.glm_upstream_url}{full_path}"

    upstream_headers = dict(request.headers)
    upstream_headers.pop("host", None)
    upstream_headers.pop("content-length", None)

    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        try:
            upstream_resp = await client.request(
                method=method,
                url=upstream_url,
                content=body,
                headers=upstream_headers,
            )
        except httpx.TimeoutException as exc:
            logger.error("Fallback proxy timeout: %s %s -> %s", method, full_path, exc)
            return Response(content=str(exc), status_code=504)

        logger.info(
            "Fallback proxy response: method=%s path=%s status=%d",
            method, full_path, upstream_resp.status_code,
        )

        resp_headers = dict(upstream_resp.headers)
        resp_headers.pop("content-encoding", None)
        resp_headers.pop("transfer-encoding", None)

        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=resp_headers,
        )
