"""Scouting pass: find out *what exists* before planning, so queries can name real tools and repos.

1. The strong model restates the objective as an underlying problem and writes queries in the vocabulary
   practitioners use (web queries and GitHub keyword queries).
2. Those run against DuckDuckGo and the GitHub repository API.
3. The model extracts the named things (tools, repos, techniques) from titles and descriptions.
"""
import asyncio

from loguru import logger

from . import prompts
from .config import settings, today
from .github import search_repos
from .llm import LLM
from .parsing import parse_scout_plan, parse_task_list
from .search import Result, search_raw

MAX_SCAN_LINES = 60
LINE_CHARS = 170
GITHUB_GAP = 6.5  # seconds between GitHub searches: 10/min without a token


def _line(r: Result) -> str:
    return f"{r.title} :: {r.content}"[:LINE_CHARS]


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


async def _github_lines(queries: list[str]) -> list[Result]:
    found: list[Result] = []
    for i, q in enumerate(queries):
        if i:
            await asyncio.sleep(GITHUB_GAP if not settings.github_token else 0.5)
        try:
            found += await search_repos(q, k=6)
        except Exception as e:  # rate limit, network: scouting is best effort
            logger.warning(f"scout github search failed for {q!r}: {e}")
            if "rate limit" in str(e).lower():
                break
    return found


async def scout(llm: LLM, goal: str) -> dict:
    """Returns {"problem", "queries", "entities"}. Never raises: scouting is best effort."""
    empty = {"problem": "", "queries": [], "entities": []}
    try:
        msg = await llm.chat(
            [{"role": "user", "content": prompts.SCOUT_QUERIES.format(goal=goal, today=today())}],
            model=settings.strong_model,
            temperature=0.3,
            max_tokens=900,
        )
        plan = parse_scout_plan(msg.get("content") or "")
        web_q, gh_q = plan["web_queries"], plan["github_queries"]
        queries = web_q + gh_q
        if not queries:
            return empty

        web_batches, gh_results = await asyncio.gather(
            asyncio.gather(*(search_raw(q, k=8) for q in web_q), return_exceptions=True),
            _github_lines(gh_q),
        )

        seen: set[str] = set()
        lines: list[str] = []
        for r in gh_results:  # repos first: they carry the proper names we want
            if r.url not in seen:
                seen.add(r.url)
                lines.append(f"- [github] {_line(r)}")
        for batch in web_batches:
            if isinstance(batch, Exception):
                logger.warning(f"scout web search failed: {batch!r}")
                continue
            for r in batch:
                if r.url not in seen:
                    seen.add(r.url)
                    lines.append(f"- {_line(r)}")
        if not lines:
            return {"problem": plan["problem"], "queries": queries, "entities": []}

        # Round 2: reuse the vocabulary the community actually uses (the model rarely guesses the slang).
        problem = plan["problem"] or goal
        try:
            refine = await llm.chat(
                [
                    {
                        "role": "user",
                        "content": prompts.SCOUT_REFINE.format(
                            problem=problem,
                            results="\n".join(lines[:30]),
                            previous="; ".join(queries),
                        ),
                    }
                ],
                model=settings.strong_model,
                temperature=0.3,
                max_tokens=500,
            )
            second = [q for q in parse_task_list(refine.get("content") or "") if q.lower() not in {x.lower() for x in queries}][:3]
        except Exception as e:
            logger.warning(f"scout refine failed ({e!r})")
            second = []
        if second:
            queries += second
            for r in await _github_lines(second):
                if r.url not in seen:
                    seen.add(r.url)
                    lines.append(f"- [github] {_line(r)}")

        msg = await llm.chat(
            [
                {
                    "role": "user",
                    "content": prompts.SCOUT_ENTITIES.format(
                        goal=problem, results="\n".join(lines[:MAX_SCAN_LINES])
                    ),
                }
            ],
            model=settings.strong_model,
            temperature=0,
            max_tokens=900,
        )
        entities = _dedupe(parse_task_list(msg.get("content") or ""))[:14]
        return {"problem": plan["problem"], "queries": queries, "entities": entities}
    except Exception as e:
        logger.warning(f"scout failed ({e!r}); planning without entities")
        return empty
