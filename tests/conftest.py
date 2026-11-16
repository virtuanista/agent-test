import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.search import Result  # noqa: E402


def run(coro):
    return asyncio.run(coro)


class FakeLLM:
    """Answers each prompt type of the agent with canned output and records the prompts."""

    def __init__(self, relevance="[2,2,0]", new_task="", wait_once=False):
        self.on_wait = None
        self.prompts: list[str] = []
        self.relevance = relevance
        self.new_task = new_task
        self.wait_once = wait_once
        self.plan = ["shadcn/ui ai workflow", "v0 design quality", "taste-skill design github", "scroll animation github"]

    async def chat(self, messages, model=None, **kw):
        content = messages[0]["content"]
        self.prompts.append(content)
        if self.wait_once and self.on_wait:
            self.wait_once = False
            self.on_wait(1.5, 429)
        if "preparing a web scan" in content:
            return {
                "content": json.dumps(
                    {
                        "problem": "AI generated sites look generic",
                        "web_queries": ["ai design slop fix", "llm website design quality"],
                        "github_queries": ["ai slop design skill", "unslop ui"],
                    }
                )
            }
        if "REUSE that exact vocabulary" in content:
            return {"content": '["unslop ui skill", "anti slop design", "ai slop design skill"]'}
        if "Below are titles and snippets" in content:
            return {"content": '["shadcn/ui", "v0"]'}
        if "research planner" in content:
            return {"content": str(self.plan).replace("'", '"')}
        if "High level objective" in content:
            task = re.search(r'Current task: "(.*)"', content).group(1)
            args = f'{{"reasoning": "buscar", "arg": "{task}"}}'
            return {"tool_calls": [{"function": {"name": "search", "arguments": args}}]}
        if "Rate how useful" in content:
            n = int(re.search(r"exactly (\d+) integers", content).group(1))
            if self.relevance == "ALL2":
                return {"content": str([2] * n)}
            return {"content": self.relevance}
        if "task creation agent" in content:
            return {"content": self.new_task}
        if "returned nothing relevant" in content:
            return {"content": "a better query"}
        return {"content": ""}

    async def stream(self, messages, model=None, **kw):
        content = messages[0]["content"]
        self.prompts.append(content)
        if "Write a cohesive document" in content:
            text = "Resumen[1](https://ex.com/a) y falso[2](https://evil.com/x)."
        else:
            text = "Hallazgo[1](https://ex.com/a)."
        for word in re.findall(r"\S+\s*", text):
            yield word

    async def aclose(self):
        pass


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_search(monkeypatch):
    """Same three results for every query, no network, no page reading."""
    results = [
        Result("A", "https://ex.com/a", "about a"),
        Result("B", "https://ex.com/b", "about b"),
        Result("C", "https://ex.com/c", "about c"),
    ]

    async def search_raw(query, k=8):
        return [Result(r.title, r.url, r.content) for r in results]

    async def read_pages(rs, n):
        return None

    async def search_repos(query, k=6, months=18):
        return [Result("owner/unslop-ui", "https://github.com/owner/unslop-ui", "anti slop skill (stars: 50)")]

    for module in ("app.tools", "app.scout"):
        monkeypatch.setattr(f"{module}.search_raw", search_raw)
    monkeypatch.setattr("app.tools.search_repos", search_repos)
    monkeypatch.setattr("app.scout.search_repos", search_repos)
    monkeypatch.setattr("app.scout.GITHUB_GAP", 0)
    monkeypatch.setattr("app.tools.read_pages", read_pages)
    return results
