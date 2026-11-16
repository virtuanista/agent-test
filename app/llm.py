import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Optional

import httpx
from loguru import logger

from . import health
from .config import available_targets, settings, split_target  # noqa: F401  (re-exported)

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
# Seconds a single call may spend waiting for quota before giving up. Rate limits are temporary, so this is
# generous: the run looks "infinite" and only fails when the quota really does not come back.
MAX_WAIT_TOTAL = float(os.getenv("LLM_MAX_WAIT", "1800"))
MAX_ATTEMPTS = 500  # safety net only; the wait budget above is what normally ends a hopeless call
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


class LLMError(Exception):
    pass


def parse_duration(value: str | None) -> float | None:
    """'697ms' / '1m26.4s' / '4h27m50.4s' / '12' -> seconds."""
    if not value:
        return None
    parts = _DURATION.findall(value)
    if parts:
        return sum(float(n) * _UNIT[u] for n, u in parts)
    try:
        return float(value)
    except ValueError:
        return None


_LIMIT_BODY = re.compile(r"Limit (\d+), Used (\d+), Requested (\d+)")
_TRY_AGAIN = re.compile(r"try again in ((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)")


@dataclass
class _ModelState:
    blocked_until: float = 0.0  # monotonic time: do not use this model before
    remaining_tokens: Optional[int] = None  # from x-ratelimit-remaining-tokens
    window_resets_at: float = 0.0  # monotonic time at which remaining_tokens refills
    # Smallest estimated request size that this model rejected as larger than its whole per-minute limit.
    # Waiting never helps for those, so such requests skip the model instead of retrying it.
    unfit_est: float = float("inf")
    fails: int = 0  # consecutive retryable failures; reset on success. Drives exponential backoff.


class LLM:
    """Client for an OpenAI-compatible /chat/completions endpoint with a pool of models.

    Free tiers limit tokens-per-minute *per model* (Groq: 8000 TPM each). Instead of sleeping on a 429,
    the call moves to the next model of the pool; it only waits when every model is exhausted.
    """

    # Process-wide counters, read by eval/run_eval.py
    stats: dict[str, int] = {"calls": 0, "retries": 0, "switches": 0}

    def __init__(self) -> None:
        # Called as on_wait(seconds, status_code) before sleeping; lets the UI show why it paused.
        self.on_wait: Optional[Callable[[float, int], None]] = None
        self.pin: Optional[str] = None  # model chosen by the user: tried first for every call, the pool backs it up
        self._state: dict[str, _ModelState] = {}
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=httpx.Timeout(90, connect=10),
        )
        self._extra: dict[str, httpx.AsyncClient] = {}
        if settings.openrouter_api_key:
            self._extra["openrouter"] = httpx.AsyncClient(
                base_url=settings.openrouter_base_url,
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}", "X-Title": "agent-test"},
                timeout=httpx.Timeout(90, connect=10),
            )

    def _client_for(self, target: str) -> httpx.AsyncClient:
        provider, _ = split_target(target)
        return self._client if provider == "default" else self._extra[provider]

    # --- request building -------------------------------------------------

    @staticmethod
    def _body(messages: list[dict], model: str, **kw: Any) -> dict:
        provider, name = split_target(model)
        body = {"model": name, "messages": messages, "max_tokens": 2048, **kw}
        if "gpt-oss" in name:
            body.setdefault("reasoning_effort", "low")
        if provider == "openrouter":
            # The free OpenRouter models are reasoning models: thinking tokens count against max_tokens, so a
            # small budget comes back with empty content (finish_reason=length). Ask for little thinking and
            # leave headroom for it.
            body.setdefault("reasoning", {"effort": "low"})
            body["max_tokens"] += 1024
        return body

    @staticmethod
    def _estimate(messages: list[dict], kw: dict) -> int:
        chars = len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(kw.get("tools", [])))
        return int(chars / 3.3) + 150  # prompt side only; mixed Spanish/English text

    # --- model pool -------------------------------------------------------

    def _order(self, preferred: Optional[str]) -> list[str]:
        first = self.pin or preferred or settings.model
        usable = [m for m in settings.model_pool if split_target(m)[0] in ("default", *self._extra)]
        return [first] + [m for m in usable if m != first]

    def _state_of(self, model: str) -> _ModelState:
        return self._state.setdefault(model, _ModelState())

    def _ready_at(self, model: str, est: int, now: float) -> float:
        """Monotonic time when `model` can take a request of ~est tokens (<= now means ready)."""
        st = self._state_of(model)
        if est >= st.unfit_est:
            return float("inf")  # this request is bigger than the model's per-minute limit
        at = st.blocked_until
        if st.remaining_tokens is not None and st.remaining_tokens < est and st.window_resets_at > now:
            at = max(at, st.window_resets_at)
        return at

    async def _acquire(self, preferred: Optional[str], est: int, waited: list[float]) -> str:
        order = self._order(preferred)
        while True:
            now = time.monotonic()
            ready = {m: self._ready_at(m, est, now) for m in order}
            for m in order:
                if ready[m] <= now:
                    return m
            usable = [t for t in ready.values() if t != float("inf")]
            if not usable:
                raise LLMError(
                    f"request of ~{est} tokens exceeds the per-minute limit of every model in the pool; "
                    "it has to be made smaller"
                )
            wait = max(0.25, min(min(usable) - now, 30.0))
            waited[0] += wait
            if waited[0] > MAX_WAIT_TOTAL:
                raise LLMError("every model of the pool is rate limited; gave up waiting")
            logger.warning(f"all models rate limited; waiting {wait:.1f}s")
            if self.on_wait:
                self.on_wait(wait, 429)
            await asyncio.sleep(wait)

    def _observe(self, model: str, resp: httpx.Response) -> None:
        st = self._state_of(model)
        if resp.status_code < 400:
            st.fails = 0
        try:
            st.remaining_tokens = int(resp.headers["x-ratelimit-remaining-tokens"])
            reset = parse_duration(resp.headers.get("x-ratelimit-reset-tokens")) or 0.0
            st.window_resets_at = time.monotonic() + reset
        except (KeyError, ValueError):
            pass

    def _penalize(self, model: str, resp: httpx.Response, attempt: int, body: str, est: int) -> float:
        """Block `model` after a retryable error; returns the block time in seconds.

        The 429 body says "Limit L, Used U, Requested R ... try again in Xs". If R > L the request can
        never fit this model, so it is marked unfit for requests of that size instead of being retried.
        """
        LLM.stats["retries"] += 1
        st = self._state_of(model)
        limit = _LIMIT_BODY.search(body)
        too_big = bool(limit and int(limit.group(3)) > int(limit.group(1)))
        if too_big:
            st.unfit_est = min(st.unfit_est, est)
            logger.warning(f"{model}: request (~{limit.group(3)} tokens) exceeds its limit {limit.group(1)}")
        delay = parse_duration(resp.headers.get("retry-after"))
        if delay is None and (again := _TRY_AGAIN.search(body)):
            delay = parse_duration(again.group(1))
        if delay is None:
            delay = parse_duration(resp.headers.get("x-ratelimit-reset-tokens"))
        if delay is None:
            delay = 2.0 ** min(attempt, 5)
        # Servers round retry-after down to 1s while the budget is still short: back off exponentially
        # on consecutive failures instead of probing the same model every second.
        st.fails += 1
        delay = min(max(delay, 1.0, 2.0 ** min(st.fails, 6)), 3600.0)
        provider = split_target(model)[0]
        if resp.status_code == 429 and "free-models-per-day" in body:
            # Daily quota of the whole account: no point retrying any model of this provider until it resets.
            delay = min(health.reset_seconds(resp.headers) or 3600.0, 86400.0)
            health.mark_daily_limit(provider, delay)
            for t in settings.model_pool:
                if split_target(t)[0] == provider:
                    self._state_of(t).blocked_until = time.monotonic() + delay
        st.blocked_until = time.monotonic() + delay
        if not too_big and "free-models-per-day" not in body:
            health.mark(model, "limited" if resp.status_code == 429 else "error",
                        "sin cuota" if resp.status_code == 429 else f"HTTP {resp.status_code}", retry_in=delay)
        snippet = " ".join(body.split())[:230]
        logger.warning(f"{model}: HTTP {resp.status_code}, blocked for {delay:.0f}s (fail #{st.fails}) {snippet}")
        return delay

    def _penalize_transport(self, model: str, attempt: int, err: Exception) -> None:
        """Network hiccup (timeout, dropped connection): back the model off and let the loop retry."""
        LLM.stats["retries"] += 1
        st = self._state_of(model)
        st.fails += 1
        delay = min(2.0 ** min(st.fails, 6), 60.0)
        st.blocked_until = time.monotonic() + delay
        health.mark(model, "error", "no responde")
        logger.warning(f"{model}: {type(err).__name__}, blocked for {delay:.0f}s (fail #{st.fails})")

    # --- public API ---------------------------------------------------------

    async def chat(self, messages: list[dict], model: Optional[str] = None, **kw: Any) -> dict:
        LLM.stats["calls"] += 1
        est, waited, first = self._estimate(messages, kw), [0.0], None
        for attempt in range(MAX_ATTEMPTS):
            chosen = await self._acquire(model, est, waited)
            first = first or chosen
            if chosen != first:
                LLM.stats["switches"] += 1
                first = chosen
            try:
                resp = await self._client_for(chosen).post(
                    "/chat/completions", json=self._body(messages, chosen, **kw)
                )
            except httpx.TransportError as e:
                self._penalize_transport(chosen, attempt, e)
                continue
            self._observe(chosen, resp)
            if resp.status_code in RETRY_STATUS:
                self._penalize(chosen, resp, attempt, resp.text, est)
                continue
            if resp.status_code >= 400:
                raise LLMError(f"{chosen}: {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            if not data.get("choices"):  # some free providers answer 200 with {"error": ...}: treat as transient
                self._penalize_transport(chosen, attempt, LLMError(str(data.get("error"))[:200]))
                continue
            health.mark(chosen, "ok")
            return data["choices"][0]["message"]
        raise LLMError("LLM unavailable or rate limited after retries")

    async def stream(
        self, messages: list[dict], model: Optional[str] = None, **kw: Any
    ) -> AsyncIterator[str]:
        LLM.stats["calls"] += 1
        est, waited, first = self._estimate(messages, kw), [0.0], None
        for attempt in range(MAX_ATTEMPTS):
            chosen = await self._acquire(model, est, waited)
            first = first or chosen
            if chosen != first:
                LLM.stats["switches"] += 1
                first = chosen
            emitted = False
            try:
                async with self._client_for(chosen).stream(
                    "POST", "/chat/completions", json=self._body(messages, chosen, stream=True, **kw)
                ) as resp:
                    self._observe(chosen, resp)
                    if resp.status_code in RETRY_STATUS:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        self._penalize(chosen, resp, attempt, body, est)
                        continue
                    if resp.status_code >= 400:
                        raise LLMError(f"{chosen}: {resp.status_code}: {(await resp.aread())[:300]!r}")
                    health.mark(chosen, "ok")
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: ") or line.endswith("[DONE]"):
                            continue
                        choices = json.loads(line[6:]).get("choices") or []
                        delta = choices[0].get("delta", {}).get("content") if choices else None
                        if delta:
                            emitted = True
                            yield delta
                    return
            except httpx.TransportError as e:
                if emitted:  # text already reached the consumer; a retry would duplicate it
                    raise LLMError(f"stream interrupted mid-answer: {type(e).__name__}") from e
                self._penalize_transport(chosen, attempt, e)
        raise LLMError("LLM unavailable or rate limited after retries")

    async def aclose(self) -> None:
        await self._client.aclose()
        for c in self._extra.values():
            await c.aclose()
