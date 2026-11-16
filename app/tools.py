from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable, ClassVar, Type

from loguru import logger

from . import prompts
from .config import settings, today
from .github import search_repos
from .llm import LLM
from .relevance import filter_relevant, reformulate_query
from .search import CitedSnippet, Result, SearchUnavailable, read_pages, search_raw

MAX_SOURCES_PER_TASK = 5


@dataclass
class Ctx:
    llm: LLM
    goal: str
    language: str
    # Global source ledger: citation number n is sources[n - 1]. Shared by every task of a run,
    # so the final summary can only cite what was really retrieved.
    sources: list[Result] = field(default_factory=list)
    _index: dict[str, int] = field(default_factory=dict)
    # Set when the search backend is down; the loop stops instead of burning calls on empty searches.
    search_down: str = ""

    def cite(self, result: Result) -> CitedSnippet:
        n = self._index.get(result.url)
        if n is None:
            self.sources.append(result)
            n = self._index[result.url] = len(self.sources)
        elif result.page and not self.sources[n - 1].page:
            self.sources[n - 1].page = result.page
        return CitedSnippet(n, result.text(), result.url)


class Tool:
    name: ClassVar[str]
    description: ClassVar[str]
    arg_description: ClassVar[str] = "The argument to the function."
    # Tools that retrieve external evidence; the loop uses this to detect "no new information".
    retrieves: ClassVar[bool] = False

    async def run(self, ctx: Ctx, task: str, arg: str) -> AsyncIterator[str]:
        raise NotImplementedError
        yield ""  # pragma: no cover


async def _prompt_stream(ctx: Ctx, template: str, **kw: str) -> AsyncIterator[str]:
    content = template.format(goal=ctx.goal, language=ctx.language, today=today(), **kw)
    async for tok in ctx.llm.stream([{"role": "user", "content": content}]):
        yield tok


async def _find_relevant(
    ctx: Ctx, task: str, query: str, fetch: Callable[[str], Awaitable[list[Result]]]
) -> list[Result]:
    """fetch -> relevance filter -> (one reformulated retry if nothing useful) -> top results."""
    relevant = await filter_relevant(ctx.llm, ctx.goal, task, await fetch(query))
    if not relevant:
        better = await reformulate_query(ctx.llm, ctx.goal, task, query)
        if better and better.lower() != query.lower():
            logger.info(f"nothing relevant for {query!r}; retrying with {better!r}")
            relevant = await filter_relevant(ctx.llm, ctx.goal, task, await fetch(better))
    return relevant[:MAX_SOURCES_PER_TASK]


async def _summarize_results(ctx: Ctx, task: str, results: list[Result]) -> AsyncIterator[str]:
    snippets = [ctx.cite(r) for r in results]
    async for tok in _prompt_stream(
        ctx, prompts.SUMMARIZE_WITH_SOURCES, query=task, snippets=str(snippets)
    ):
        yield tok


class Reason(Tool):
    name = "reason"
    description = (
        "Reason about task via existing information or understanding. "
        "Make decisions / selections from options."
    )

    async def run(self, ctx: Ctx, task: str, arg: str) -> AsyncIterator[str]:
        async for tok in _prompt_stream(ctx, prompts.EXECUTE, task=task):
            yield tok


class Code(Tool):
    name = "code"
    description = (
        "Should only be used to write code, refactor code, fix code bugs, "
        "and explain programming concepts."
    )

    async def run(self, ctx: Ctx, task: str, arg: str) -> AsyncIterator[str]:
        async for tok in _prompt_stream(ctx, prompts.CODE, task=task):
            yield tok


class Search(Tool):
    name = "search"
    retrieves = True
    description = (
        "Search the web for up to date information about tools, products, techniques, news, "
        "people and public facts."
    )
    arg_description = (
        "The web search query, IN ENGLISH, 3-10 words, naming the specific things to look for. "
        "No years unless the user asked for one. Never empty."
    )

    async def run(self, ctx: Ctx, task: str, arg: str) -> AsyncIterator[str]:
        query = arg or task
        try:
            relevant = await _find_relevant(ctx, task, query, search_raw)
            await read_pages(relevant, settings.read_pages)
        except SearchUnavailable as e:
            logger.warning(f"search unavailable: {e}")
            ctx.search_down = str(e)
            yield f"Web search is unavailable right now ({e}). This task has no web evidence."
            return
        except Exception as e:
            logger.warning(f"search failed ({e!r}); falling back to reasoning")
            async for tok in Reason().run(ctx, task, arg):
                yield tok
            return

        if not relevant:
            yield "No relevant results were found for this task."
            return
        async for tok in _summarize_results(ctx, task, relevant):
            yield tok


class GithubSearch(Tool):
    name = "github_search"
    retrieves = True
    description = (
        "Search GitHub repositories to find which open source projects, libraries, templates, skills "
        "and tools exist for a topic, with their stars, recent activity and README. Use it for questions "
        "about what people have built or published."
    )
    arg_description = (
        "A short GitHub keyword query IN ENGLISH (2-6 words), e.g. 'shadcn ai design skill'. "
        "No operators, no years. Never empty."
    )

    async def run(self, ctx: Ctx, task: str, arg: str) -> AsyncIterator[str]:
        query = arg or task
        try:
            relevant = await _find_relevant(ctx, task, query, search_repos)
        except Exception as e:
            logger.warning(f"github search failed ({e!r}); falling back to web search")
            async for tok in Search().run(ctx, task, f"{query} github"):
                yield tok
            return

        if not relevant:
            yield "No relevant repositories were found for this task."
            return
        async for tok in _summarize_results(ctx, task, relevant):
            yield tok


TOOLS: dict[str, Type[Tool]] = {t.name: t for t in (Search, GithubSearch, Reason, Code)}
DEFAULT_TOOL = "search"
