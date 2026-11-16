"""Which models actually work right now.

A model is only shown as available after a real (tiny) tool-calling request succeeded. Real runs keep the
registry fresh as a side effect: a success marks the model ok, a 429 marks it limited until it should recover.
"""

import asyncio
import time
from dataclasses import dataclass

import httpx

from .config import available_targets, settings, split_target

TTL_OK = 300.0  # seconds before an ok verdict is re-verified
# OpenRouter's free tier is ~50 requests/day for the WHOLE account (1000 with $10 credit): every probe spends it.
# Those providers are only probed on an explicit user request, and their ok verdicts last longer.
METERED = {"openrouter"}
TTL_OK_METERED = 6 * 3600.0
TTL_BAD = 60.0  # seconds before an error verdict is re-verified (limited ones expire with their retry time)
PROBE_TIMEOUT = 40.0
PLAIN_PROBE_TOKENS = 512  # the smallest max_tokens the agent uses for free-form answers
_PROBE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "search", "description": "Search the web",
        "parameters": {"type": "object", "required": ["arg"], "properties": {"arg": {"type": "string"}}},
    },
}]


@dataclass
class ModelHealth:
    status: str = "unknown"  # unknown | ok | limited | error
    detail: str = ""
    checked_at: float = 0.0
    until: float = 0.0  # limited: epoch time at which the quota should be back
    latency_ms: int | None = None


_health: dict[str, ModelHealth] = {}
_lock = asyncio.Lock()


def mark(target: str, status: str, detail: str = "", retry_in: float = 0.0, latency_ms: int | None = None) -> None:
    now = time.time()
    prev = _health.get(target)
    _health[target] = ModelHealth(
        status, detail, now, now + retry_in if status == "limited" else 0.0,
        latency_ms if latency_ms is not None else (prev.latency_ms if prev and status == "ok" else None),
    )


def view(target: str, now: float | None = None) -> dict:
    """Current verdict; limited/error ones that expired go back to 'unknown' (they need a new check)."""
    now = now or time.time()
    h = _health.get(target) or ModelHealth()
    status = h.status
    if status == "limited" and now >= h.until:
        status = "unknown"
    elif status == "error" and now - h.checked_at >= TTL_BAD:
        status = "unknown"
    return {
        "status": status, "detail": h.detail if status != "unknown" else "", "latency_ms": h.latency_ms,
        "retry_in": max(0, round(h.until - now)) if status == "limited" else 0, "checked_at": h.checked_at,
    }


def is_blocked(target: str) -> str:
    """Reason the model is known to be unusable right now, or ''."""
    v = view(target)
    return (v["detail"] or v["status"]) if v["status"] in ("limited", "error") else ""


def _stale(target: str, now: float) -> bool:
    h = _health.get(target)
    if not h or h.status == "unknown":
        return True
    if h.status == "ok":
        return now - h.checked_at >= (TTL_OK_METERED if split_target(target)[0] in METERED else TTL_OK)
    return view(target, now)["status"] == "unknown"


def reset_seconds(headers) -> float | None:
    """Seconds until the quota resets, from X-RateLimit-Reset (epoch milliseconds, OpenRouter style)."""
    try:
        return max(0.0, float(headers["x-ratelimit-reset"]) / 1000.0 - time.time())
    except (KeyError, ValueError):
        return None


def mark_daily_limit(provider: str, retry_in: float) -> None:
    """The daily free quota is per account, so when one model hits it every model of that provider is out."""
    for t in available_targets():
        if split_target(t)[0] == provider:
            mark(t, "limited", "límite diario gratuito", retry_in=retry_in)


def _endpoint(provider: str) -> tuple[str, str]:
    if provider == "openrouter":
        return settings.openrouter_base_url, settings.openrouter_api_key
    return settings.llm_base_url, settings.llm_api_key


def _extra(body: dict) -> dict:
    return {k: v for k, v in body.items() if k not in ("model", "messages")}


def _tool_verdict(r: httpx.Response) -> str:
    """'' if the response holds a tool call, else why not."""
    try:
        msg = r.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError):
        return "respuesta inválida"
    return "" if msg.get("tool_calls") else "ignora las herramientas"


async def probe(target: str) -> None:
    """One forced tool call, no retries: this must report the state of the model, not wait it out."""
    from .llm import LLM  # local import: llm.py imports this module, so a top-level import would be circular

    provider, name = split_target(target)
    base, key = _endpoint(provider)
    body = {
        "model": name, "max_tokens": 200, "tools": _PROBE_TOOLS, "tool_choice": "required",
        "messages": [{"role": "user", "content": "Use the search tool with the argument: ping"}],
    }
    if "gpt-oss" in name:
        body["reasoning_effort"] = "low"
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as c:
            url, headers = base.rstrip("/") + "/chat/completions", {"Authorization": f"Bearer {key}"}
            r = await c.post(url, json=LLM._body(body["messages"], target, **_extra(body)), headers=headers)
            ms = int((time.monotonic() - t0) * 1000)
            if r.status_code == 200:
                verdict = _tool_verdict(r)
                if verdict:
                    return mark(target, "error", verdict)
                # Second check: plain text generation within the agent's smallest budget. A model can pass the tool
                # call and still answer everything else with empty content (all tokens spent thinking).
                plain = await c.post(url, headers=headers, json=LLM._body(
                    [{"role": "user", "content": 'Devuelve SOLO una lista JSON con 3 consultas de búsqueda sobre buscadores, p. ej. ["a","b","c"].'}],
                    target, max_tokens=PLAIN_PROBE_TOKENS))
                if plain.status_code == 200:
                    try:
                        content = plain.json()["choices"][0]["message"].get("content") or ""
                    except (ValueError, KeyError, IndexError, TypeError):
                        content = ""
                    if not content.strip():
                        return mark(target, "error", "respuestas vacías")
                    return mark(target, "ok", latency_ms=ms)
                r = plain  # the second request failed: classify that failure below
    except httpx.TimeoutException:
        return mark(target, "error", "no responde")
    except httpx.HTTPError as e:
        return mark(target, "error", f"sin conexión ({type(e).__name__})")
    if r.status_code == 429:
        try:
            wait = float(r.headers.get("retry-after", ""))
        except ValueError:
            wait = reset_seconds(r.headers) or 60.0
        wait = min(max(wait, 5.0), 86400.0)
        if "free-models-per-day" in r.text:
            return mark_daily_limit(provider, wait)
        return mark(target, "limited", "sin cuota", retry_in=wait)
    if r.status_code in (401, 403):
        return mark(target, "error", f"sin acceso (HTTP {r.status_code})")
    if r.status_code == 404:
        return mark(target, "error", "no existe")
    return mark(target, "error", f"HTTP {r.status_code}")


async def probe_stale(targets: list[str], manual: bool = False) -> None:
    """Verify every target without a fresh verdict, concurrently. Only one batch runs at a time.

    Automatic checks skip metered providers; `manual=True` (the user pressed the button) includes them."""
    async with _lock:
        now = time.time()
        todo = [t for t in targets if _stale(t, now) and (manual or split_target(t)[0] not in METERED)]
        if todo:
            await asyncio.gather(*(probe(t) for t in todo))
