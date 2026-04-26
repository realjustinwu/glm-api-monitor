"""Integration tests — real Zhipu API calls through the proxy.

Run with:
    GLM_API_KEY=xxx.yyy pytest -m integration -v -s
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import httpx
import pytest

PROXY_PORT = int(os.environ.get("PROXY_PORT", "18765"))
PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://glm:glm@localhost:5432/glm_monitor"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def proxy_server():
    """Start uvicorn in a subprocess for the whole test session."""
    env = os.environ.copy()
    env["PROXY_PORT"] = str(PROXY_PORT)
    env["DATABASE_URL"] = DATABASE_URL

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", str(PROXY_PORT),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for proxy to be ready
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{PROXY_URL}/docs", timeout=1)
            if resp.status_code == 200:
                break
        except httpx.ConnectError:
            pass
        time.sleep(0.3)
    else:
        proc.terminate()
        out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        pytest.fail(f"Proxy did not start in 10s:\n{out}")

    yield proc

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture()
def api_key():
    key = os.environ.get("GLM_API_KEY")
    if not key:
        pytest.skip("GLM_API_KEY not set")
    return key


@pytest.fixture()
async def client(proxy_server):
    async with httpx.AsyncClient(base_url=PROXY_URL, timeout=120) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOLS_PAYLOAD = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
]


def collect_streaming_usage(lines: list[str]) -> dict:
    """Find the last valid JSON with usage from SSE data lines."""
    last_usage = {}
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = data.get("usage")
        if usage:
            last_usage = usage
            break
    return last_usage


def collect_anthropic_streaming_usage(lines: list[str]) -> dict:
    """Find usage from Anthropic SSE events."""
    result = {}
    event = ""
    for line in lines:
        line = line.strip()
        if line.startswith("event: "):
            event = line[7:]
            continue
        if not line.startswith("data: "):
            continue
        try:
            data = json.loads(line[6:])
        except json.JSONDecodeError:
            continue

        if event == "message_start":
            msg = data.get("message", data)
            u = msg.get("usage", {})
            result["input_tokens"] = u.get("input_tokens", 0)
            result["output_tokens"] = u.get("output_tokens", 0)
        elif event == "message_delta":
            u = data.get("usage", {})
            result["output_tokens"] = u.get("output_tokens", result.get("output_tokens", 0))
            result["input_tokens"] = u.get("input_tokens", result.get("input_tokens", 0))

    return result


def print_sse(lines: list[str], label: str) -> None:
    """Print all SSE lines for debugging."""
    print(f"\n--- {label} SSE ---")
    for line in lines:
        if line.strip():
            print(line)
    print(f"--- end ---")


# ---------------------------------------------------------------------------
# Test 1: GLM non-streaming chat
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_glm_chat_non_streaming(client: httpx.AsyncClient, api_key: str):
    resp = await client.post(
        "/api/paas/v4/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "glm-4-flash",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    print(f"\n--- GLM non-streaming response ---")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    usage = data.get("usage", {})
    assert usage.get("prompt_tokens", 0) > 0, f"prompt_tokens should be > 0, got {usage}"
    assert usage.get("completion_tokens", 0) > 0
    assert usage.get("total_tokens", 0) > 0
    assert data.get("model", ""), "model should not be empty"


# ---------------------------------------------------------------------------
# Test 2: GLM streaming chat
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_glm_chat_streaming(client: httpx.AsyncClient, api_key: str):
    async with client.stream(
        "POST",
        "/api/paas/v4/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "glm-4-flash",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        lines: list[str] = []
        async for line in resp.aiter_lines():
            lines.append(line)

    assert lines, "Should receive SSE lines"
    print_sse(lines, "GLM streaming chat")
    usage = collect_streaming_usage(lines)
    print(f"Parsed usage: {usage}")
    assert usage.get("total_tokens", 0) > 0, f"Streaming should report usage, got {usage}"


# ---------------------------------------------------------------------------
# Test 3: GLM tool call non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_glm_tool_call_non_streaming(client: httpx.AsyncClient, api_key: str):
    resp = await client.post(
        "/api/paas/v4/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "glm-4-flash",
            "messages": [{"role": "user", "content": "What's the weather in Beijing?"}],
            "tools": TOOLS_PAYLOAD,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    print(f"\n--- GLM tool call non-streaming response ---")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    choice = data.get("choices", [{}])[0]
    tool_calls = choice.get("message", {}).get("tool_calls", [])
    assert len(tool_calls) > 0, f"Expected tool_calls, got {json.dumps(data, indent=2)}"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert choice.get("finish_reason") == "tool_calls"


# ---------------------------------------------------------------------------
# Test 4: GLM tool call streaming
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_glm_tool_call_streaming(client: httpx.AsyncClient, api_key: str):
    async with client.stream(
        "POST",
        "/api/paas/v4/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "glm-4-flash",
            "messages": [{"role": "user", "content": "What's the weather in Shanghai?"}],
            "tools": TOOLS_PAYLOAD,
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        lines: list[str] = []
        async for line in resp.aiter_lines():
            lines.append(line)

    assert lines, "Should receive SSE lines"
    print_sse(lines, "GLM tool call streaming")

    # Check that tool_calls appear in streaming output
    found_tool_call = False
    for line in lines:
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = data.get("choices", [{}])[0].get("delta", {})
        if "tool_calls" in delta:
            found_tool_call = True
            break

    assert found_tool_call, f"Expected tool_calls in streaming output, got {len(lines)} lines"


# ---------------------------------------------------------------------------
# Test 5: Anthropic non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_anthropic_non_streaming(client: httpx.AsyncClient, api_key: str):
    resp = await client.post(
        "/api/anthropic/v1/messages",
        headers={
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": "glm-4-flash",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say hello in one word"}],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    print(f"\n--- Anthropic non-streaming response ---")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    usage = data.get("usage", {})
    assert usage.get("input_tokens", 0) > 0, f"input_tokens should be > 0, got {usage}"
    assert usage.get("output_tokens", 0) > 0, f"output_tokens should be > 0, got {usage}"


# ---------------------------------------------------------------------------
# Test 6: Anthropic streaming — verify prompt_tokens > 0
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_anthropic_streaming_prompt_tokens(client: httpx.AsyncClient, api_key: str):
    async with client.stream(
        "POST",
        "/api/anthropic/v1/messages",
        headers={
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": "glm-4-flash",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        lines: list[str] = []
        async for line in resp.aiter_lines():
            lines.append(line)

    assert lines, "Should receive SSE lines"
    print_sse(lines, "Anthropic streaming")
    usage = collect_anthropic_streaming_usage(lines)
    print(f"Parsed usage: {usage}")

    assert usage.get("input_tokens", 0) > 0, (
        f"prompt_tokens (input_tokens) should be > 0, got {usage}"
    )
    assert usage.get("output_tokens", 0) > 0, (
        f"output_tokens should be > 0, got {usage}"
    )


# ---------------------------------------------------------------------------
# Test 7: Coding API — fallback proxy
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_coding_api_fallback(client: httpx.AsyncClient, api_key: str):
    resp = await client.post(
        "/api/coding/paas/v4/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "glm-4-flash",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    print(f"\n--- Coding API response ---")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    usage = data.get("usage", {})
    assert usage.get("total_tokens", 0) > 0, f"Coding API should return usage, got {usage}"
    assert data.get("model", ""), "model should not be empty"
