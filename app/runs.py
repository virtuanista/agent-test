"""Background runs: the agent keeps working on the server whether or not a browser is attached.

A run buffers every event in memory so a client can (re)attach at any point and replay from an index.
Persistence to SQLite is unchanged (token events are not stored; the finished record has the full text).
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from loguru import logger

from . import db
from .agent import run_agent

Event = dict[str, Any]
# Streamed token events are not stored one by one: task_done / summary carry the full text.
SKIP_PERSIST = {"token", "summary_token", "waiting"}
KEEP_FINISHED_SECONDS = 600  # a finished run stays replayable from memory for reconnecting clients
HEARTBEAT_SECONDS = 15

AgentFactory = Callable[..., AsyncIterator[Event]]  # (goal, language, max_loops, *, model=None)


class Busy(RuntimeError):
    def __init__(self, run_id: str):
        super().__init__("a run is already in progress")
        self.run_id = run_id


@dataclass
class Run:
    id: str
    goal: str
    language: str
    created: float
    max_loops: int
    model: str | None = None
    events: list[Event] = field(default_factory=list)
    finished: bool = False
    status: str = "running"
    task: asyncio.Task | None = None
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def publish(self, ev: Event) -> None:
        async with self.cond:
            self.events.append(ev)
            self.cond.notify_all()

    async def follow(self, start: int = 0) -> AsyncIterator[tuple[int, Event | None]]:
        """Yield (index, event) from `start`, then live ones. Yields (index, None) as a heartbeat when idle."""
        i = max(0, start)
        while True:
            while i < len(self.events):
                yield i, self.events[i]
                i += 1
            if self.finished:
                return
            async with self.cond:
                try:
                    await asyncio.wait_for(
                        self.cond.wait_for(lambda: i < len(self.events) or self.finished), HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    yield i, None


_runs: dict[str, Run] = {}


def get(run_id: str) -> Run | None:
    return _runs.get(run_id)


def active() -> Run | None:
    return next((r for r in _runs.values() if not r.finished), None)


async def _drive(run: Run, agent: AgentFactory, max_loops: int) -> None:
    status = "stopped"
    try:
        async for ev in agent(run.goal, run.language, max_loops, model=run.model):
            if ev["type"] not in SKIP_PERSIST:
                db.add_event(run.id, ev)
            if ev["type"] == "done":
                status = "done"
            elif ev["type"] == "error":
                status = "error"
            await run.publish(ev)
    except asyncio.CancelledError:  # stop requested: tell attached clients, then end normally
        status = "stopped"
        ev = {"type": "stopped"}
        db.add_event(run.id, ev)
        await run.publish(ev)
    except Exception as e:  # the agent reports its own errors; this is a last resort
        logger.exception("background run crashed")
        status = "error"
        ev = {"type": "error", "message": str(e)}
        db.add_event(run.id, ev)
        await run.publish(ev)
    finally:
        db.finish_run(run.id, status)
        async with run.cond:
            run.status = status
            run.finished = True
            run.cond.notify_all()
        asyncio.get_running_loop().call_later(KEEP_FINISHED_SECONDS, _runs.pop, run.id, None)


def _ensure_finished(run: Run) -> None:
    """Safety net: a task cancelled before its first step never reaches _drive's cleanup."""
    if not run.finished:
        db.finish_run(run.id, "stopped")
        run.status, run.finished = "stopped", True


def start(
    goal: str, language: str, max_loops: int, agent: AgentFactory | None = None, model: str | None = None
) -> Run:
    """Create the run and start it in the background. Only one run at a time (shared free-tier model quota)."""
    if (cur := active()) is not None:
        raise Busy(cur.id)
    run_id = db.create_run(goal, language)
    run = Run(run_id, goal, language, db.get_run(run_id)["created"], max_loops, model)
    _runs[run_id] = run
    agent = agent or run_agent
    run.task = asyncio.create_task(_drive(run, agent, max_loops))
    run.task.add_done_callback(lambda _t: _ensure_finished(run))
    return run


def stop(run_id: str) -> bool:
    run = _runs.get(run_id)
    if not run or run.finished or not run.task:
        return False
    run.task.cancel()
    return True
