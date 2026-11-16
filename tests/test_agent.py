import asyncio

from app.agent import run_agent
from app.search import Result
from app.tools import Ctx
from tests.conftest import FakeLLM, run

GOAL = "what are the best ai design tools for websites today"  # >= 6 words: triggers the scout


async def collect(llm, goal=GOAL, loops=8):
    return [ev async for ev in run_agent(goal, "English", loops, llm=llm)]


def kinds(events):
    return [e["type"] for e in events]


def test_ctx_cite_numbers_sources_globally():
    ctx = Ctx(llm=None, goal="g", language="en")
    a = ctx.cite(Result("A", "https://a.com", "x"))
    b = ctx.cite(Result("B", "https://b.com", "y"))
    again = ctx.cite(Result("A", "https://a.com", "x", page="long page"))
    assert (a.index, b.index, again.index) == (1, 2, 1)
    assert len(ctx.sources) == 2 and ctx.sources[0].page == "long page"


def test_full_run_wires_scout_entities_into_planning(fake_search):
    llm = FakeLLM(relevance="ALL2")
    events = run(collect(llm))
    k = kinds(events)
    assert k[:3] == ["goal", "scouting", "scout"]
    assert next(e for e in events if e["type"] == "scout")["entities"] == ["shadcn/ui", "v0"]
    planner = next(p for p in llm.prompts if "research planner" in p)
    assert "shadcn/ui, v0" in planner  # entities reached the planning prompt
    assert "AI generated sites look generic" in planner  # and so did the underlying problem
    scout_ev = next(e for e in events if e["type"] == "scout")
    assert scout_ev["problem"] == "AI generated sites look generic"
    # 2 web + 2 github, plus round 2 with the vocabulary seen in the results (one repeats a first query)
    assert len(scout_ev["queries"]) == 4 + 2
    assert "unslop ui skill" in scout_ev["queries"]
    assert any("REUSE that exact vocabulary" in p for p in llm.prompts)
    assert k[-1] == "done" and "error" not in k


def test_summary_only_cites_real_sources(fake_search):
    events = run(collect(FakeLLM(relevance="ALL2")))
    summary = next(e for e in events if e["type"] == "summary")
    assert "https://ex.com/a" in summary["text"]
    assert "evil.com" not in summary["text"]  # hallucinated source removed
    assert summary["stats"]["links_removed"] >= 1
    sources = next(e for e in events if e["type"] == "sources")["sources"]
    assert {s["url"] for s in sources} == {"https://ex.com/a", "https://ex.com/b", "https://ex.com/c"}


def test_irrelevant_results_are_not_summarized(fake_search):
    llm = FakeLLM(relevance="[0,0,0]")
    events = run(collect(llm, loops=1))
    done = next(e for e in events if e["type"] == "task_done")
    assert "No relevant results" in done["result"]
    assert any("returned nothing relevant" in p for p in llm.prompts)  # tried to reformulate once


def test_stops_when_searches_add_no_new_sources(fake_search):
    # Every query returns the same 3 URLs: task 1 adds them, tasks 2 and 3 add nothing -> stop before task 4.
    events = run(collect(FakeLLM(relevance="ALL2")))
    assert kinds(events).count("task_start") == 3
    assert any(e["type"] == "notice" and "No new sources" in e["text"] for e in events)


def test_duplicate_new_task_is_not_queued(fake_search):
    llm = FakeLLM(relevance="ALL2", new_task="v0 design quality")  # same as an existing task
    llm.plan = ["shadcn/ui ai workflow", "v0 design quality"]
    events = run(collect(llm))
    assert kinds(events).count("task_start") == 2


def test_rate_limit_wait_becomes_an_event(fake_search):
    events = run(collect(FakeLLM(relevance="ALL2", wait_once=True), loops=1))
    waiting = [e for e in events if e["type"] == "waiting"]
    assert waiting and waiting[0]["status"] == 429 and waiting[0]["seconds"] == 1.5


def test_short_goal_skips_scout(fake_search):
    events = run(collect(FakeLLM(relevance="ALL2"), goal="who won the 2022 cup", loops=1))  # 5 words < MIN_SCOUT_WORDS
    assert "scouting" not in kinds(events)


def test_closing_the_stream_early_does_not_hang(fake_search):
    async def go():
        gen = run_agent(GOAL, "English", 8, llm=FakeLLM(relevance="ALL2"))
        async for ev in gen:
            if ev["type"] == "tasks":
                break
        await asyncio.wait_for(gen.aclose(), timeout=5)

    run(go())


def test_failing_task_is_skipped_and_run_still_finishes(fake_search, monkeypatch):
    import app.tools as tools_mod

    async def boom(self, ctx, task, arg):
        raise RuntimeError("boom")
        yield ""

    monkeypatch.setattr(tools_mod.Search, "run", boom)
    events = run(collect(FakeLLM(relevance="ALL2")))
    k = kinds(events)
    assert "error" not in k and k[-1] == "done"
    assert any(e["type"] == "notice" and "skipped" in e["text"] for e in events)
    assert "summary" in k
