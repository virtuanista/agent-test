import asyncio
import contextlib
import json
from collections import deque
from typing import Any, AsyncIterator

from loguru import logger

from . import prompts
from .config import settings, today
from .llm import LLM
from .parsing import (  # noqa: F401  (re-exported for callers/tests)
    clean_task,
    ground_summary,
    is_duplicate,
    parse_new_task,
    parse_task_list,
)
from .scout import scout
from .tools import DEFAULT_TOOL, TOOLS, Ctx

Event = dict[str, Any]
RESULT_CHARS = 1500  # what create_task sees of the last result (keeps token use, and 429s, down)
SUMMARY_CHARS = 10000  # research-notes budget of the final summary (whole prompt must fit ~7k tokens/min)
MAX_LISTED_SOURCES = 30  # sources shown to the summarizer; the numbering stays global
MIN_SCOUT_WORDS = 6  # short factual questions skip the scouting pass
STALE_LIMIT = 2  # consecutive retrieval tasks adding no new source before the loop stops
FAIL_LIMIT = 3  # consecutive tasks that crashed (LLM already waited out its quota) before giving up


# --- agent steps -------------------------------------------------------------


def _tool_specs(language: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {
                    "type": "object",
                    "required": ["reasoning", "arg"],
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "How the task will be accomplished with the current function. "
                            f"Detail the plan and any concerns. Write it in {language}.",
                        },
                        "arg": {"type": "string", "description": t.arg_description},
                    },
                },
            },
        }
        for t in TOOLS.values()
    ]


async def start_goal(
    llm: LLM, goal: str, language: str, entities: list[str] | None = None, problem: str = ""
) -> list[str]:
    msg = await llm.chat(
        [
            {
                "role": "user",
                "content": prompts.START_GOAL.format(
                    goal=goal,
                    language=language,
                    today=today(),
                    problem=problem or "(unknown)",
                    entities=", ".join(entities) if entities else "(none found)",
                ),
            }
        ],
        model=settings.strong_model,
        max_tokens=1200,
    )
    tasks: list[str] = []
    for t in parse_task_list(msg.get("content") or ""):
        if not is_duplicate(t, tasks):
            tasks.append(t)
    return tasks or [goal]


async def analyze(llm: LLM, goal: str, task: str, language: str) -> dict:
    default = {"action": DEFAULT_TOOL, "arg": task, "reasoning": "I'll try searching it up."}
    try:
        msg = await llm.chat(
            [
                {
                    "role": "user",
                    "content": prompts.ANALYZE.format(goal=goal, task=task, language=language, today=today()),
                }
            ],
            tools=_tool_specs(language),
            tool_choice="required",
            temperature=0.2,
            max_tokens=1024,
        )
        call = msg["tool_calls"][0]["function"]
        args = json.loads(call["arguments"])
        action = call["name"]
        if action not in TOOLS:
            return default
        arg = (args.get("arg") or "").strip()
        if TOOLS[action].retrieves and not arg:
            arg = task
        return {"action": action, "arg": arg, "reasoning": args.get("reasoning", "")}
    except Exception as e:  # same fallback AgentGPT uses: just search the task
        logger.warning(f"analyze failed ({e!r}); defaulting to search")
        return default


async def create_task(
    llm: LLM, goal: str, language: str, remaining: list[str], last_task: str, result: str, done: list[str]
) -> str:
    msg = await llm.chat(
        [
            {
                "role": "user",
                "content": prompts.CREATE_TASKS.format(
                    goal=goal,
                    language=language,
                    today=today(),
                    tasks="\n".join(remaining) or "(none)",
                    last_task=last_task,
                    result=result[:RESULT_CHARS],
                ),
            }
        ],
        max_tokens=512,
    )
    task = parse_new_task(msg.get("content") or "")
    # Semantic check: a rephrasing of anything already done or queued is not a new task.
    return "" if not task or is_duplicate(task, done + remaining) else task


def _sources_block(ctx: Ctx) -> str:
    lines = [
        f"[{i}] {r.title[:70]} - {r.url}" for i, r in enumerate(ctx.sources[:MAX_LISTED_SOURCES], start=1)
    ]
    return "Source list:\n" + "\n".join(lines) if lines else "Source list: (none; do not cite anything)"


async def _run(goal: str, language: str, max_loops: int, llm: LLM, ctx: Ctx) -> AsyncIterator[Event]:
    """start -> (analyze -> execute -> create)* -> summarize, yielding UI events."""
    try:
        yield {"type": "goal", "goal": goal}

        entities: list[str] = []
        problem = ""
        if len(goal.split()) >= MIN_SCOUT_WORDS:
            yield {"type": "scouting"}
            try:
                found = await scout(llm, goal)
            except Exception as e:  # scouting only improves the plan; the run works without it
                logger.warning(f"scout failed ({e!r}); planning without it")
                found = {"queries": [], "entities": [], "problem": ""}
            entities, problem = found["entities"], found["problem"]
            yield {"type": "scout", "queries": found["queries"], "entities": entities, "problem": problem}

        try:
            plan = await start_goal(llm, goal, language, entities, problem)
        except Exception as e:  # no plan: treat the goal itself as the single first task
            logger.warning(f"start_goal failed ({e!r}); using the goal as the only task")
            plan = [goal]
        queue: deque[str] = deque(plan)
        done: list[str] = []
        results: list[tuple[str, str]] = []
        yield {"type": "tasks", "remaining": list(queue), "completed": list(done)}

        loops = stale = failures = 0
        while queue and loops < max_loops:
            loops += 1
            task = queue.popleft()
            yield {"type": "task_start", "task": task, "loop": loops}

            known = len(ctx.sources)
            text = ""
            try:
                analysis = await analyze(llm, goal, task, language)
                yield {"type": "analysis", "task": task, **analysis}
                tool = TOOLS[analysis["action"]]
                async for tok in tool().run(ctx, task, analysis["arg"]):
                    text += tok
                    yield {"type": "token", "text": tok}
                failures = 0
            except Exception as e:  # one broken task must not end the run: skip it and keep going
                logger.warning(f"task failed ({e!r}); skipping it")
                failures += 1
                tool = TOOLS[DEFAULT_TOOL]
                text += f"\n(Task could not be completed: {e})"
                yield {"type": "notice", "text": f"Task skipped after an error: {e}"}
            yield {"type": "task_done", "task": task, "result": text}
            done.append(task)
            results.append((task, text))

            if failures >= FAIL_LIMIT:
                yield {"type": "notice", "text": f"{failures} tasks failed in a row; wrapping up with what was found."}
                break
            if ctx.search_down:
                yield {"type": "notice", "text": f"Search backend unavailable ({ctx.search_down}); stopping."}
                break
            if tool.retrieves:
                stale = stale + 1 if len(ctx.sources) == known else 0
            if stale >= STALE_LIMIT:
                yield {"type": "notice", "text": "No new sources in the last searches; stopping early."}
                break

            try:
                new = await create_task(llm, goal, language, list(queue), task, text, done)
            except Exception as e:  # AgentGPT ignores errors here: just stop adding tasks
                logger.warning(f"create_task failed: {e!r}")
                new = ""
            if new:
                queue.append(new)
            yield {"type": "tasks", "remaining": list(queue), "completed": list(done)}

        if loops >= max_loops and queue:
            yield {"type": "notice", "text": f"Reached the loop limit ({max_loops}); {len(queue)} task(s) left."}

        per_task = max(600, SUMMARY_CHARS // max(len(results), 1))
        body = "\n\n".join(f"## {t}\n{r[:per_task]}" for t, r in results)
        yield {"type": "summary_start"}
        yield {
            "type": "sources",
            "sources": [{"n": i, "url": r.url, "title": r.title} for i, r in enumerate(ctx.sources, start=1)],
        }
        content = prompts.SUMMARIZE.format(
            goal=goal, language=language, today=today(), sources=_sources_block(ctx), text=body
        )
        raw = ""
        try:
            async for tok in llm.stream(
                [{"role": "user", "content": content}], model=settings.strong_model, max_tokens=3000
            ):
                raw += tok
                yield {"type": "summary_token", "text": tok}
        except Exception as e:  # never lose the research: fall back to the raw task notes
            logger.warning(f"summary failed ({e!r}); delivering the raw notes")
            yield {"type": "notice", "text": f"Summary generation failed ({e}); showing the raw notes."}
            raw += ("\n\n" if raw else "") + body
        grounded, stats = ground_summary(raw, {r.url for r in ctx.sources}, language, [r.url for r in ctx.sources])
        yield {"type": "summary", "text": grounded, "stats": stats}
        yield {"type": "done", "loops": loops}
    except Exception as e:
        logger.exception("run failed")
        yield {"type": "error", "message": str(e)}


async def run_agent(
    goal: str, language: str = "Spanish", max_loops: int = 8, llm: LLM | None = None, model: str | None = None
) -> AsyncIterator[Event]:
    """Public entry point. Wraps _run so that rate-limit waits inside the LLM client
    surface as `waiting` events instead of the stream looking frozen."""
    own = llm is None
    llm = llm or LLM()
    llm.pin = model or None
    max_loops = max(1, min(max_loops, settings.max_loops))
    ctx = Ctx(llm, goal, language)

    queue: asyncio.Queue[Event | None] = asyncio.Queue()
    llm.on_wait = lambda secs, status: queue.put_nowait(
        {"type": "waiting", "seconds": round(secs, 1), "status": status}
    )

    async def pump() -> None:
        try:
            async for ev in _run(goal, language, max_loops, llm, ctx):
                await queue.put(ev)
        except Exception as e:  # _run reports its own errors; this is a last resort
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(None)

    task = asyncio.create_task(pump())
    try:
        while (ev := await queue.get()) is not None:
            yield ev
    finally:
        task.cancel()  # client disconnected or finished: stop any in-flight work
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        llm.on_wait = None
        if own:
            await llm.aclose()
