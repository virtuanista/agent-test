"""Relevance filtering of search results with a cheap model call."""
import re

from loguru import logger

from . import prompts
from .config import settings
from .llm import LLM
from .search import Result

SNIPPET_CHARS = 220


def parse_scores(text: str, n: int) -> list[int] | None:
    """Pull a JSON-ish list of n small integers out of the model output."""
    for match in re.finditer(r"\[[\d\s,]*\]", text or ""):
        nums = [int(x) for x in re.findall(r"\d+", match.group(0))]
        if len(nums) == n:
            return [min(x, 2) for x in nums]
    return None


async def filter_relevant(llm: LLM, goal: str, task: str, results: list[Result]) -> list[Result]:
    """Keep results scored >= 1 (best first). If scoring fails, keep everything."""
    if not results:
        return []
    listing = "\n".join(
        f"[{i}] {r.title} | {r.url} | {r.content[:SNIPPET_CHARS]}" for i, r in enumerate(results, start=1)
    )
    prompt = prompts.RELEVANCE.format(goal=goal, task=task, n=len(results), results=listing)
    try:
        msg = await llm.chat(
            [{"role": "user", "content": prompt}], model=settings.model, temperature=0, max_tokens=800
        )
        scores = parse_scores(msg.get("content") or "", len(results))
    except Exception as e:
        logger.warning(f"relevance scoring failed ({e!r}); keeping all results")
        return results
    if scores is None:
        logger.warning("relevance scores unparsable; keeping all results")
        return results
    kept = [(s, i, r) for i, (s, r) in enumerate(zip(scores, results)) if s >= 1]
    kept.sort(key=lambda t: (-t[0], t[1]))
    return [r for _, _, r in kept]


async def reformulate_query(llm: LLM, goal: str, task: str, query: str) -> str:
    """One better English query after a search returned nothing relevant ('' if none)."""
    prompt = prompts.REFORMULATE.format(goal=goal, task=task, query=query)
    try:
        msg = await llm.chat([{"role": "user", "content": prompt}], model=settings.model, max_tokens=300)
    except Exception as e:
        logger.warning(f"reformulate failed ({e!r})")
        return ""
    line = (msg.get("content") or "").strip().splitlines()
    return line[0].strip().strip('"').strip("`") if line else ""
