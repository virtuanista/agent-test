import pytest

import app.search as search
from app.agent import run_agent
from tests.conftest import FakeLLM, run


@pytest.fixture(autouse=True)
def clean_cache():
    search._cache.clear()
    yield
    search._cache.clear()


def fake_query(monkeypatch, responses):
    """Replace the DuckDuckGo call with a queue of answers: a result list, or an exception to raise."""
    calls = []

    async def q(query):
        calls.append(query)
        answer = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def no_sleep(_):
        return None

    monkeypatch.setattr(search, "_query", q)
    monkeypatch.setattr(search.asyncio, "sleep", no_sleep)
    return calls


THROTTLED = search.SearchUnavailable("DuckDuckGo is rate limiting this machine")
ONE = [{"url": "https://a.com", "title": "A", "content": "x"}]


def test_throttling_raises_instead_of_returning_empty(monkeypatch):
    fake_query(monkeypatch, [THROTTLED])
    with pytest.raises(search.SearchUnavailable) as e:
        run(search.search_raw("anything"))
    assert "rate limiting" in str(e.value)


def test_genuinely_empty_results_are_not_an_outage(monkeypatch):
    fake_query(monkeypatch, [[]])
    assert run(search.search_raw("zzzz qqqq")) == []


def test_recovers_on_the_retry(monkeypatch):
    calls = fake_query(monkeypatch, [THROTTLED, ONE])
    assert [r.url for r in run(search.search_raw("q"))] == ["https://a.com"] and len(calls) == 2


def test_cache_avoids_repeating_queries_and_ignores_case_and_spacing(monkeypatch):
    calls = fake_query(monkeypatch, [ONE])
    run(search.search_raw("Shadcn  UI"))
    run(search.search_raw("shadcn ui"))
    assert len(calls) == 1


def test_failures_are_never_cached(monkeypatch):
    calls = fake_query(monkeypatch, [THROTTLED])
    for _ in range(2):
        with pytest.raises(search.SearchUnavailable):
            run(search.search_raw("q"))
    assert len(calls) == 4  # 2 attempts per call, nothing served from cache


def test_blocked_and_duplicate_urls_are_filtered(monkeypatch):
    raw = [
        {"url": "https://a.com", "title": "A", "content": ""},
        {"url": "https://a.com", "title": "A again", "content": ""},
        {"url": "https://x.org/paper.pdf", "title": "P", "content": ""},
        {"url": "https://www.instagram.com/p/1", "title": "I", "content": ""},
        {"url": "https://b.com", "title": "B", "content": ""},
    ]
    fake_query(monkeypatch, [raw])
    assert [r.url for r in run(search.search_raw("q"))] == ["https://a.com", "https://b.com"]


def test_agent_stops_and_says_so_when_search_is_down(monkeypatch):
    fake_query(monkeypatch, [THROTTLED])

    async def collect():
        return [e async for e in run_agent("one two three four five six seven", "English", 8, llm=FakeLLM())]

    events = run(collect())
    kinds = [e["type"] for e in events]
    assert kinds.count("task_start") == 1  # did not burn the remaining loops
    assert any(e["type"] == "notice" and "unavailable" in e["text"] for e in events)
    done = next(e for e in events if e["type"] == "task_done")
    assert "unavailable" in done["result"]


def test_ddg_empty_is_not_an_outage_but_rate_limit_and_timeout_are(monkeypatch):
    import sys
    import types

    from app.search import _ddg_text

    ex = types.ModuleType("ddgs.exceptions")

    class DDGSException(Exception): ...

    class RatelimitException(DDGSException): ...

    class TimeoutException(DDGSException): ...

    ex.DDGSException, ex.RatelimitException, ex.TimeoutException = DDGSException, RatelimitException, TimeoutException
    mod = types.ModuleType("ddgs")
    behavior = {}

    class DDGS:
        def text(self, query, max_results):
            if isinstance(behavior["answer"], Exception):
                raise behavior["answer"]
            return behavior["answer"]

    mod.DDGS = DDGS
    monkeypatch.setitem(sys.modules, "ddgs", mod)
    monkeypatch.setitem(sys.modules, "ddgs.exceptions", ex)

    behavior["answer"] = DDGSException("No results found.")
    assert _ddg_text("zzz") == []
    for exc in (RatelimitException("202 Ratelimit"), TimeoutException("slow"), DDGSException("boom")):
        behavior["answer"] = exc
        with pytest.raises(search.SearchUnavailable):
            _ddg_text("zzz")
    behavior["answer"] = [{"href": "https://a.com", "title": "A", "body": "text"}]
    assert _ddg_text("q") == [{"url": "https://a.com", "title": "A", "content": "text"}]
