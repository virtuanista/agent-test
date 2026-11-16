import json

import httpx
import pytest

import app.llm as llm_mod
from app.config import settings
from app.llm import LLM, LLMError, parse_duration
from tests.conftest import run

POOL = ("m-a", "m-b", "m-c")


@pytest.fixture(autouse=True)
def pool(monkeypatch):
    monkeypatch.setattr(type(settings), "model_pool", POOL, raising=False)
    object.__setattr__(settings, "model_pool", POOL)
    object.__setattr__(settings, "model", "m-a")
    LLM.stats.update(calls=0, retries=0, switches=0)
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)
    return sleeps


def make_llm(handler):
    llm = LLM()
    llm._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")
    return llm


def ok(text="hi", remaining="5000", reset="30s"):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": text}}]},
        headers={"x-ratelimit-remaining-tokens": remaining, "x-ratelimit-reset-tokens": reset},
    )


def limited(retry="20"):
    return httpx.Response(429, json={"error": "rate"}, headers={"retry-after": retry})


def test_parse_duration():
    assert parse_duration("697ms") == pytest.approx(0.697)
    assert parse_duration("1m26.4s") == pytest.approx(86.4)
    assert parse_duration("4h27m50.4s") == pytest.approx(4 * 3600 + 27 * 60 + 50.4)
    assert parse_duration("12") == 12.0
    assert parse_duration(None) is None and parse_duration("soon") is None


def test_429_moves_to_next_model_without_sleeping(pool):
    seen = []

    def handler(req):
        model = json.loads(req.content)["model"]
        seen.append(model)
        return limited() if model == "m-a" else ok("from " + model)

    msg = run(make_llm(handler).chat([{"role": "user", "content": "x"}]))
    assert msg["content"] == "from m-b" and seen == ["m-a", "m-b"]
    assert pool == []  # no sleeping: another model was available
    assert LLM.stats["retries"] == 1 and LLM.stats["switches"] == 1


def test_blocked_model_is_skipped_on_later_calls():
    seen = []

    def handler(req):
        model = json.loads(req.content)["model"]
        seen.append(model)
        return limited() if model == "m-a" else ok()

    llm = make_llm(handler)
    run(llm.chat([{"role": "user", "content": "1"}]))
    seen.clear()
    run(llm.chat([{"role": "user", "content": "2"}]))
    assert seen == ["m-b"]  # m-a is still blocked: not even tried


def test_waits_only_when_every_model_is_exhausted(pool, monkeypatch):
    state = {"n": 0}

    def handler(req):
        state["n"] += 1
        return limited("5") if state["n"] <= 3 else ok()  # all three models get blocked for 5s

    llm = make_llm(handler)
    waits = []
    llm.on_wait = lambda secs, status: waits.append((secs, status))

    # Virtual clock: sleeping advances time, so the 5s blocks expire.
    clock = {"t": 1000.0}

    class FakeTime:
        @staticmethod
        def monotonic():
            return clock["t"]

    async def fake_sleep(s):
        pool.append(s)
        clock["t"] += s

    monkeypatch.setattr(llm_mod, "time", FakeTime)
    monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)

    msg = run(llm.chat([{"role": "user", "content": "x"}]))
    assert msg["content"] == "hi"
    assert pool and waits and waits[0][1] == 429  # it did wait, and told the UI why
    assert state["n"] == 4  # three blocked attempts, then the retry that worked


def test_preferred_model_is_used_first():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content)["model"])
        return ok()

    run(make_llm(handler).chat([{"role": "user", "content": "x"}], model="m-c"))
    assert seen == ["m-c"]


def test_low_remaining_tokens_prefer_another_model():
    seen = []

    def handler(req):
        model = json.loads(req.content)["model"]
        seen.append(model)
        # m-a reports almost no budget left for the next minute
        return ok(remaining="10", reset="45s") if model == "m-a" else ok()

    llm = make_llm(handler)
    big = [{"role": "user", "content": "palabra " * 600}]  # ~1.5k tokens estimated
    run(llm.chat(big))
    seen.clear()
    run(llm.chat(big))
    assert seen == ["m-b"]


def too_big(model="m-a"):
    body = (
        f"Rate limit reached for model `{model}` on input tokens per minute (ITPM): "
        "Limit 7000, Used 100, Requested 9000. Please try again in 1.2s."
    )
    return httpx.Response(429, text=body, headers={"retry-after": "1"})


def test_request_too_big_for_a_model_skips_it_instead_of_retrying():
    seen = []

    def handler(req):
        model = json.loads(req.content)["model"]
        seen.append(model)
        return too_big(model) if model == "m-a" else ok(remaining="100000")

    llm = make_llm(handler)
    huge = [{"role": "user", "content": "palabra " * 3000}]
    run(llm.chat(huge))
    seen.clear()
    run(llm.chat(huge))
    assert seen == ["m-b"]  # m-a is never retried for requests of that size
    seen.clear()
    llm._state_of("m-a").blocked_until = 0.0  # its 1s cool-down is over
    run(llm.chat([{"role": "user", "content": "corta"}]))
    assert seen[0] == "m-a"  # but it still serves small requests


def test_request_too_big_for_every_model_fails_fast_with_a_clear_message():
    calls = []

    def handler(req):
        calls.append(1)
        return too_big(json.loads(req.content)["model"])

    huge = [{"role": "user", "content": "palabra " * 3000}]
    with pytest.raises(LLMError, match="has to be made smaller"):
        run(make_llm(handler).chat(huge))
    assert len(calls) == 3  # one probe per model, then it stops (no 24 futile retries)


def test_try_again_hint_in_body_sets_the_block_time(pool):
    seen = []

    def handler(req):
        model = json.loads(req.content)["model"]
        seen.append(model)
        return httpx.Response(429, text="... Please try again in 7.5s. Need more tokens?") if model != "m-c" else ok()

    llm = make_llm(handler)
    run(llm.chat([{"role": "user", "content": "x"}]))
    assert 7.0 < llm._state_of("m-a").blocked_until - llm_mod.time.monotonic() <= 7.5


def test_client_errors_are_not_retried():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(400, text="bad request")

    with pytest.raises(LLMError, match="400"):
        run(make_llm(handler).chat([{"role": "user", "content": "x"}]))
    assert len(calls) == 1


def test_stream_switches_model_before_first_byte():
    seen = []

    def handler(req):
        model = json.loads(req.content)["model"]
        seen.append(model)
        if model == "m-a":
            return limited()
        sse = 'data: {"choices":[{"delta":{"content":"ho"}}]}\n\ndata: {"choices":[{"delta":{"content":"la"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text=sse, headers={"x-ratelimit-remaining-tokens": "4000"})

    async def go():
        return "".join([t async for t in make_llm(handler).stream([{"role": "user", "content": "x"}])])

    assert run(go()) == "hola" and seen == ["m-a", "m-b"]


def test_transport_error_is_retried_not_fatal(pool):
    state = {"n": 0}

    def handler(req):
        state["n"] += 1
        if state["n"] <= 2:
            raise httpx.ReadTimeout("slow", request=req)
        return ok("recovered")

    msg = run(make_llm(handler).chat([{"role": "user", "content": "x"}]))
    assert msg["content"] == "recovered" and state["n"] == 3


def test_waits_far_longer_than_old_cap_before_giving_up(monkeypatch):
    clock = {"t": 1000.0}

    class FakeTime:
        @staticmethod
        def monotonic():
            return clock["t"]

    async def fake_sleep(s):
        clock["t"] += s

    monkeypatch.setattr(llm_mod, "time", FakeTime)
    monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)
    start = clock["t"]

    def handler(req):  # quota only comes back after 10 minutes (old cap was 4)
        return limited("30") if clock["t"] - start < 600 else ok("finally")

    msg = run(make_llm(handler).chat([{"role": "user", "content": "x"}]))
    assert msg["content"] == "finally"


def _with_openrouter(handler_default, handler_or, monkeypatch):
    object.__setattr__(settings, "model_pool", ("m-a", "openrouter::v/free-1:free"))
    object.__setattr__(settings, "openrouter_api_key", "test-key")
    llm = LLM()
    llm._client = httpx.AsyncClient(transport=httpx.MockTransport(handler_default), base_url="http://x")
    llm._extra["openrouter"] = httpx.AsyncClient(transport=httpx.MockTransport(handler_or), base_url="http://or")
    return llm


@pytest.fixture
def restore_or():
    yield
    object.__setattr__(settings, "openrouter_api_key", "")


def test_openrouter_is_the_fallback_when_the_default_endpoint_is_limited(restore_or, monkeypatch):
    seen = {}

    def default(req):
        return limited("60")

    def or_(req):
        seen["model"] = json.loads(req.content)["model"]  # provider prefix is stripped from the request
        return ok("from-or")

    llm = _with_openrouter(default, or_, monkeypatch)
    msg = run(llm.chat([{"role": "user", "content": "x"}]))
    assert msg["content"] == "from-or" and seen["model"] == "v/free-1:free"


def test_openrouter_targets_are_ignored_without_a_key(restore_or, monkeypatch):
    object.__setattr__(settings, "model_pool", ("m-a", "openrouter::v/free-1:free"))
    object.__setattr__(settings, "openrouter_api_key", "")
    assert LLM()._order(None) == ["m-a"]


def test_200_with_error_body_is_retried_elsewhere(restore_or, monkeypatch):
    def default(req):
        return httpx.Response(200, json={"error": {"message": "provider overloaded"}})

    llm = _with_openrouter(default, lambda r: ok("recovered"), monkeypatch)
    assert run(llm.chat([{"role": "user", "content": "x"}]))["content"] == "recovered"


def test_pinned_model_goes_first_and_the_rest_stay_as_fallback():
    llm = LLM()
    assert llm._order("m-b")[0] == "m-b"
    llm.pin = "m-c"
    assert llm._order("m-b") == ["m-c", "m-a", "m-b"]  # pin wins over the per-call preference; pool is the backup


def test_available_targets_hides_providers_without_credentials(restore_or):
    from app.llm import available_targets

    object.__setattr__(settings, "model_pool", ("m-a", "openrouter::v/x:free", "m-a"))
    object.__setattr__(settings, "openrouter_api_key", "")
    assert available_targets() == ["m-a"]
    object.__setattr__(settings, "openrouter_api_key", "k")
    assert available_targets() == ["m-a", "openrouter::v/x:free"]
