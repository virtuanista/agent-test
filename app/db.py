import json
import sqlite3
import time
import uuid
from contextlib import closing
from typing import Any

from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, goal TEXT NOT NULL, language TEXT, created REAL, status TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, ts REAL, data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id);
"""
_RUN_COLS = ("id", "goal", "language", "created", "status")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(settings.db_path)
    c.executescript(_SCHEMA)
    return c


def create_run(goal: str, language: str) -> str:
    run_id = uuid.uuid4().hex[:12]
    with closing(_conn()) as c, c:
        c.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (run_id, goal, language, time.time(), "running"))
    return run_id


def add_event(run_id: str, event: dict[str, Any]) -> None:
    with closing(_conn()) as c, c:
        c.execute(
            "INSERT INTO events (run_id, ts, data) VALUES (?,?,?)",
            (run_id, time.time(), json.dumps(event, ensure_ascii=False)),
        )


def finish_run(run_id: str, status: str) -> None:
    with closing(_conn()) as c, c:
        c.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))


def mark_interrupted() -> int:
    """Runs still 'running' at startup died with the previous server process."""
    with closing(_conn()) as c, c:
        return c.execute("UPDATE runs SET status='interrupted' WHERE status='running'").rowcount


def list_runs(limit: int = 50) -> list[dict]:
    with closing(_conn()) as c:
        rows = c.execute(
            "SELECT id, goal, language, created, status FROM runs ORDER BY created DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(zip(_RUN_COLS, r)) for r in rows]


def get_run(run_id: str) -> dict | None:
    with closing(_conn()) as c:
        run = c.execute("SELECT id, goal, language, created, status FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            return None
        events = [json.loads(r[0]) for r in c.execute("SELECT data FROM events WHERE run_id=? ORDER BY id", (run_id,))]
    return {**dict(zip(_RUN_COLS, run)), "events": events}
