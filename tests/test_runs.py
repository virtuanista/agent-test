import asyncio

import pytest

from app import db, runs
from app.config import settings


@pytest.fixture(autouse=True)
def tmp_db(tmp_path):
    runs._runs.clear()
    original = settings.db_path
    object.__setattr__(settings, "db_path", str(tmp_path / "t.db"))  # frozen dataclass: bypass __setattr__
    yield
    object.__setattr__(settings, "db_path", original)


def fake_agent(gate: asyncio.Event | None = None, fail: bool = False):
    async def agent(goal, language, max_loops, model=None):
        yield {"type": "goal", "goal": goal}
        yield {"type": "token", "text": "a"}
        if gate:
            await gate.wait()
        if fail:
            raise RuntimeError("boom")
        yield {"type": "summary", "text": "ok"}
        yield {"type": "done", "loops": 1}

    return agent


async def collect(run, start=0):
    return [ev async for _, ev in run.follow(start) if ev is not None]


def test_run_survives_without_clients_and_replays_from_any_index():
    async def main():
        r = runs.start("goal", "Spanish", 3, agent=fake_agent())
        await r.task
        assert r.finished and r.status == "done"
        all_events = await collect(r)
        assert [e["type"] for e in all_events] == ["goal", "token", "summary", "done"]
        assert [e["type"] for e in await collect(r, 2)] == ["summary", "done"]
        stored = db.get_run(r.id)
        assert stored["status"] == "done"
        assert "token" not in [e["type"] for e in stored["events"]]  # tokens are not persisted

    asyncio.run(main())


def test_late_client_gets_buffer_then_live_events():
    async def main():
        gate = asyncio.Event()
        r = runs.start("goal", "Spanish", 3, agent=fake_agent(gate))
        await asyncio.sleep(0.05)
        follower = asyncio.create_task(collect(r))  # attaches mid-run
        await asyncio.sleep(0.05)
        assert not follower.done()
        gate.set()
        events = await asyncio.wait_for(follower, 2)
        assert [e["type"] for e in events] == ["goal", "token", "summary", "done"]

    asyncio.run(main())


def test_only_one_active_run_and_stop_marks_stopped():
    async def main():
        gate = asyncio.Event()
        r = runs.start("goal", "Spanish", 3, agent=fake_agent(gate))
        with pytest.raises(runs.Busy) as busy:
            runs.start("other", "Spanish", 3, agent=fake_agent())
        assert busy.value.run_id == r.id and runs.active() is r
        await asyncio.sleep(0.05)
        assert runs.stop(r.id)
        await asyncio.wait_for(r.task, 2)
        assert r.finished and r.status == "stopped" and runs.active() is None
        assert (await collect(r))[-1]["type"] == "stopped"
        assert db.get_run(r.id)["status"] == "stopped"
        runs.start("next", "Spanish", 3, agent=fake_agent())  # a finished run frees the slot

    asyncio.run(main())


def test_stop_before_first_step_still_frees_the_slot():
    async def main():
        r = runs.start("goal", "Spanish", 3, agent=fake_agent(asyncio.Event()))
        assert runs.stop(r.id)  # cancelled before _drive ever ran
        await asyncio.gather(r.task, return_exceptions=True)
        assert r.finished and runs.active() is None and db.get_run(r.id)["status"] == "stopped"

    asyncio.run(main())


def test_agent_crash_is_reported_and_frees_the_slot():
    async def main():
        r = runs.start("goal", "Spanish", 3, agent=fake_agent(fail=True))
        await r.task
        assert r.status == "error" and (await collect(r))[-1]["type"] == "error"
        assert runs.active() is None

    asyncio.run(main())


def test_heartbeat_when_idle(monkeypatch):
    monkeypatch.setattr(runs, "HEARTBEAT_SECONDS", 0.05)

    async def main():
        gate = asyncio.Event()
        r = runs.start("goal", "Spanish", 3, agent=fake_agent(gate))
        beats = 0
        async for _, ev in r.follow(0):
            if ev is None:
                beats += 1
                if beats == 2:
                    break
        gate.set()
        await r.task

    asyncio.run(main())


def test_mark_interrupted_only_touches_running():
    a, b = db.create_run("x1", "es"), db.create_run("x2", "es")
    db.finish_run(b, "done")
    assert db.mark_interrupted() == 1
    assert db.get_run(a)["status"] == "interrupted" and db.get_run(b)["status"] == "done"


def test_selected_model_reaches_the_agent():
    got = {}

    async def agent(goal, language, max_loops, model=None):
        got["model"] = model
        yield {"type": "done", "loops": 0}

    async def main():
        r = runs.start("goal", "Spanish", 3, agent=agent, model="m-x")
        await r.task
        assert r.model == "m-x"

    asyncio.run(main())
    assert got["model"] == "m-x"


def test_default_agent_accepts_the_arguments_the_runner_passes():
    import inspect

    from app.agent import run_agent

    # regression: model was once passed positionally and landed in run_agent's `llm` parameter
    bound = inspect.signature(run_agent).bind("goal", "Spanish", 3, model="m-x")
    assert bound.arguments["model"] == "m-x" and "llm" not in bound.arguments
