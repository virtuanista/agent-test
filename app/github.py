"""GitHub repository search (REST API, works without a token: 10 searches/min)."""
import asyncio
from datetime import date, timedelta

import httpx
from loguru import logger

from .config import settings
from .search import Result

API = "https://api.github.com/search/repositories"
README_CHARS = 1200


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "agent-test/0.1"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


async def _readme(client: httpx.AsyncClient, full_name: str, branch: str) -> str:
    # raw.githubusercontent.com does not count against the API rate limit.
    for name in ("README.md", "readme.md", "README.MD"):
        try:
            resp = await client.get(f"https://raw.githubusercontent.com/{full_name}/{branch}/{name}")
        except httpx.HTTPError:
            return ""
        if resp.status_code == 200:
            return resp.text[:README_CHARS]
    return ""


async def search_repos(query: str, k: int = 6, months: int = 18) -> list[Result]:
    """Best-match repos pushed in the last `months`, with stars/recency and a README excerpt."""
    since = (date.today() - timedelta(days=30 * months)).isoformat()
    params = {"q": f"{query} pushed:>{since}", "per_page": str(k)}
    async with httpx.AsyncClient(timeout=20, headers=_headers()) as client:
        resp = await client.get(API, params=params)
        if resp.status_code in (403, 429):
            raise RuntimeError("GitHub rate limit reached (set GITHUB_TOKEN to raise it)")
        resp.raise_for_status()
        items = resp.json().get("items", [])[:k]
        readmes = await asyncio.gather(
            *(_readme(client, i["full_name"], i.get("default_branch", "main")) for i in items),
            return_exceptions=True,
        )

    results = []
    for item, readme in zip(items, readmes):
        if isinstance(readme, Exception):
            logger.debug(f"readme failed for {item['full_name']}: {readme}")
            readme = ""
        topics = ", ".join(item.get("topics") or [])
        meta = f"stars: {item['stargazers_count']}, last push: {item['pushed_at'][:10]}"
        content = f"{item.get('description') or ''} ({meta}{'; topics: ' + topics if topics else ''})"
        results.append(Result(item["full_name"], item["html_url"], content, readme))
    return results
