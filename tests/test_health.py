import asyncio

import httpx
import pytest

from app import health
from app.config import settings


@pytest.fixture(autouse=True)
def clean():
    health._health.clear()
    object.__setattr__(settings, "openrouter_api_key", "k")
    yield
    health._health.clear()
    object.__setattr__(settings, "openrouter_api_key", "")


REAL_CLIENT = httpx.AsyncClient  # captured once: patching twice must not wrap the first patch


def patch_transport(monkeypatch, handler):
    real = REAL_CLIENT

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)

    monkeypatch.setattr(health.httpx, "AsyncClient", factory)


def tool_ok(request):
    """Healthy model: a tool call for the tool probe, non-empty text for the plain-generation probe."""
    if b'"tools"' in request.content:
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{"function": {"name": "search"}}]}}]})
    return httpx.Response(200, json={"choices": [{"message": {"content": '["a","b","c"]'}}]})


def test_unverified_models_are_not_ok():
    assert health.view("m")["status"] == "unknown"
    assert health.is_blocked("m") == ""  # unknown is not blocked, but the UI will not paint it as available


def test_probe_ok_requires_a_real_tool_call(monkeypatch):
    patch_transport(monkeypatch, tool_ok)
    asyncio.run(health.probe("m-ok"))
    assert health.view("m-ok")["status"] == "ok" and health.view("m-ok")["latency_ms"] is not None

    patch_transport(monkeypatch, lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]}))
    asyncio.run(health.probe("m-notools"))
    v = health.view("m-notools")
    assert v["status"] == "error" and "herramientas" in v["detail"]


@pytest.mark.parametrize(
    "response,status",
    [
        (httpx.Response(429, headers={"retry-after": "120"}), "limited"),
        (httpx.Response(403), "error"),
        (httpx.Response(404), "error"),
        (httpx.Response(500), "error"),
        (httpx.Response(200, json={"error": "overloaded"}), "error"),
    ],
)
def test_probe_classifies_failures(monkeypatch, response, status):
    patch_transport(monkeypatch, lambda r: response)
    asyncio.run(health.probe("m"))
    assert health.view("m")["status"] == status


def test_limited_reports_retry_time_and_expires(monkeypatch):
    patch_transport(monkeypatch, lambda r: httpx.Response(429, headers={"retry-after": "120"}))
    asyncio.run(health.probe("m"))
    v = health.view("m")
    assert 100 <= v["retry_in"] <= 120
    assert health.is_blocked("m")
    assert health.view("m", now=v["checked_at"] + 200)["status"] == "unknown"  # needs re-verification


def test_timeout_is_an_error(monkeypatch):
    def boom(request):
        raise httpx.ReadTimeout("slow")

    patch_transport(monkeypatch, boom)
    asyncio.run(health.probe("m"))
    assert health.view("m")["detail"] == "no responde"


def test_probe_stale_skips_fresh_verdicts(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return tool_ok(request)

    patch_transport(monkeypatch, handler)
    asyncio.run(health.probe_stale(["a", "b"]))
    asyncio.run(health.probe_stale(["a", "b"]))  # both fresh now
    assert len(calls) == 4  # 2 requests (tool call + plain text) per model, none on the second pass


def test_real_call_outcomes_update_the_registry():
    health.mark("m", "limited", "sin cuota", retry_in=30)
    assert health.is_blocked("m")
    health.mark("m", "ok")
    assert health.is_blocked("m") == "" and health.view("m")["status"] == "ok"


def test_models_endpoint_verifies_before_reporting(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    object.__setattr__(settings, "model_pool", ("m-a", "m-b"))
    patch_transport(monkeypatch, lambda r: tool_ok(r) if b'"m-a"' in r.content else httpx.Response(429, headers={"retry-after": "90"}))
    client = TestClient(main.app)
    before = {m["id"]: m["status"] for m in client.get("/api/models").json()["models"]}
    assert before == {"m-a": "unknown", "m-b": "unknown"}  # nothing is painted as available unverified
    after = {m["id"]: m for m in client.get("/api/models?check=1").json()["models"]}
    assert after["m-a"]["status"] == "ok" and after["m-b"]["status"] == "limited" and after["m-b"]["retry_in"] > 0
    # a pinned model that is known to be down is rejected by the API
    r = client.post("/api/run", json={"goal": "hola mundo", "model": "m-b"})
    assert r.status_code == 422 and "no está disponible" in r.json()["detail"]


def test_model_that_passes_tools_but_answers_empty_is_not_available(monkeypatch):
    def handler(request):
        if b'"tools"' in request.content:
            return tool_ok(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})

    patch_transport(monkeypatch, handler)
    asyncio.run(health.probe("reasoner"))
    v = health.view("reasoner")
    assert v["status"] == "error" and v["detail"] == "respuestas vacías"


def test_openrouter_requests_ask_for_low_reasoning_and_leave_headroom():
    from app.llm import LLM

    body = LLM._body([{"role": "user", "content": "x"}], "openrouter::v/m:free", max_tokens=512)
    assert body["model"] == "v/m:free" and body["reasoning"] == {"effort": "low"} and body["max_tokens"] == 512 + 1024
    plain = LLM._body([{"role": "user", "content": "x"}], "m-a", max_tokens=512)
    assert "reasoning" not in plain and plain["max_tokens"] == 512


DAILY_429 = httpx.Response(
    429,
    headers={"x-ratelimit-reset": str(int((__import__("time").time() + 7200) * 1000))},
    json={"error": {"message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day"}},
)


def test_daily_free_limit_takes_down_every_model_of_the_provider(monkeypatch):
    object.__setattr__(settings, "model_pool", ("m-a", "openrouter::v/x:free", "openrouter::v/y:free"))
    patch_transport(monkeypatch, lambda r: DAILY_429)
    asyncio.run(health.probe("openrouter::v/x:free"))
    for t in ("openrouter::v/x:free", "openrouter::v/y:free"):
        v = health.view(t)
        assert v["status"] == "limited" and v["detail"] == "límite diario gratuito" and 7000 < v["retry_in"] <= 7200
    assert health.view("m-a")["status"] == "unknown"  # other providers are unaffected


def test_automatic_checks_skip_metered_providers_but_manual_ones_do_not(monkeypatch):
    calls = []

    def handler(request):
        calls.append(json_model(request))
        return tool_ok(request)

    def json_model(request):
        import json

        return json.loads(request.content)["model"]

    patch_transport(monkeypatch, handler)
    targets = ["m-a", "openrouter::v/x:free"]
    asyncio.run(health.probe_stale(targets))
    assert set(calls) == {"m-a"}  # OpenRouter untouched: its free quota is tiny and shared
    assert health.view("openrouter::v/x:free")["status"] == "unknown"
    asyncio.run(health.probe_stale(targets, manual=True))
    assert "v/x:free" in calls and health.view("openrouter::v/x:free")["status"] == "ok"


def test_openrouter_ok_verdicts_last_six_hours():
    health.mark("openrouter::v/x:free", "ok")
    h = health._health["openrouter::v/x:free"]
    assert not health._stale("openrouter::v/x:free", h.checked_at + 3600)
    assert health._stale("openrouter::v/x:free", h.checked_at + 7 * 3600)
    health.mark("m-a", "ok")
    assert health._stale("m-a", health._health["m-a"].checked_at + 400)


def test_desktop_token_is_enforced_only_when_configured(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    client = TestClient(main.app)
    monkeypatch.delenv("APP_TOKEN", raising=False)
    assert client.get("/api/health").status_code == 200  # browser mode: open, as before
    monkeypatch.setenv("APP_TOKEN", "s3cret")
    assert client.get("/api/health").status_code == 403
    assert client.get("/", headers={"cookie": "agent_token=wrong"}).status_code == 403
    assert client.get("/api/health", headers={"cookie": "agent_token=s3cret"}).status_code == 200
    assert client.get("/vendor/marked.min.js", headers={"cookie": "agent_token=s3cret"}).status_code == 200
