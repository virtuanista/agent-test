import asyncio
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import trafilatura
from loguru import logger

from .config import settings

UA = "Mozilla/5.0 (compatible; agent-test/0.1)"


@dataclass
class Result:
    """One web/GitHub hit. `page` is filled lazily by read_pages (or by the source)."""

    title: str
    url: str
    content: str = ""
    page: str = ""

    def text(self) -> str:
        text = f"{self.title}. {self.content}".strip()
        if self.page:
            text += f"\nPage excerpt: {self.page}"
        return text


@dataclass
class CitedSnippet:
    index: int
    text: str
    url: str = ""

    def __repr__(self) -> str:  # what the model sees
        return f"{{i: {self.index}, text: {self.text}, url: {self.url}}}"


def is_blocked(url: str) -> bool:
    """PDFs and low-signal hosts never reach the model."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.path.lower().endswith(".pdf"):
        return True
    return any(host == d or host.endswith("." + d) for d in settings.block_domains)


async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        resp = await client.get(url, follow_redirects=True)
        if "html" not in resp.headers.get("content-type", ""):
            return ""
        text = await asyncio.to_thread(trafilatura.extract, resp.text, include_comments=False)
        return (text or "")[: settings.page_chars]
    except Exception as e:  # pages fail all the time; the snippet is the fallback
        logger.debug(f"read failed {url}: {e}")
        return ""


class SearchUnavailable(RuntimeError):
    """DuckDuckGo is throttling or failing (not 'no results')."""


_gate: asyncio.Semaphore | None = None
_cache: dict[str, tuple[float, list[dict]]] = {}


def _semaphore() -> asyncio.Semaphore:
    # A public engine throttles a single IP quickly: never hit it with a burst.
    global _gate
    if _gate is None:
        _gate = asyncio.Semaphore(2)
    return _gate


def _ddg_text(query: str) -> list[dict]:
    from ddgs import DDGS  # imported lazily: the library is only needed once a search actually runs
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

    try:
        found = DDGS().text(query, max_results=10)
    except RatelimitException as e:
        raise SearchUnavailable(f"DuckDuckGo is rate limiting this machine: {e}") from e
    except TimeoutException as e:
        raise SearchUnavailable(f"DuckDuckGo timed out: {e}") from e
    except DDGSException as e:
        if "no results" in str(e).lower():  # their way of saying "empty", not an outage
            return []
        raise SearchUnavailable(f"DuckDuckGo failed: {e}") from e
    return [{"url": r.get("href", ""), "title": r.get("title", ""), "content": r.get("body", "")} for r in found]


async def _query(query: str) -> list[dict]:
    async with _semaphore():
        results = await asyncio.to_thread(_ddg_text, query)
        await asyncio.sleep(0.4)  # spacing between consecutive upstream queries
    return results


async def _raw_results(query: str) -> list[dict]:
    """Cached query. Raises SearchUnavailable when DuckDuckGo is blocking us, not on empty results."""
    key = " ".join(query.lower().split())
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < settings.search_cache_hours * 3600:
        return hit[1]

    try:
        raw = await _query(query)
    except SearchUnavailable:  # throttled: one short retry before giving up
        await asyncio.sleep(3)
        raw = await _query(query)
    if raw:  # never cache failures or empties
        _cache[key] = (time.time(), raw)
    return raw


async def search_raw(query: str, k: int = 8) -> list[Result]:
    """DuckDuckGo (no API key) -> up to k deduplicated, non-blocked results (no page reading)."""
    raw = await _raw_results(query)

    results: list[Result] = []
    seen: set[str] = set()
    for r in raw:
        url = r.get("url", "")
        if not url or url in seen or is_blocked(url):
            continue
        seen.add(url)
        results.append(Result(r.get("title", ""), url, r.get("content", "")))
        if len(results) >= k:
            break
    return results


async def read_pages(results: list[Result], n: int) -> None:
    """Fill `page` for the first n results that do not have one yet (in place)."""
    todo = [r for r in results[:n] if not r.page]
    if not todo:
        return
    async with httpx.AsyncClient(timeout=8, headers={"User-Agent": UA}) as client:
        pages = await asyncio.gather(*(_fetch_text(client, r.url) for r in todo))
    for r, page in zip(todo, pages):
        r.page = page
