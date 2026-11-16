"""Pure text helpers: task parsing (ported from AgentGPT's task_output_parser), semantic dedup of tasks
and grounding of the final summary to the real source list. No network, no LLM: easy to test."""
import ast
import json
import re
import unicodedata
from typing import Any
from urllib.parse import urlparse

# --- task parsing ------------------------------------------------------------

_NO_TASK = re.compile(
    r"^(No( (new|further|additional|extra|other))? tasks?( is)? (required|needed|added|created|inputted)"
    r"|Task (complete|completed|finished|done|over|success)|Do nothing|\s*$)",
    re.IGNORECASE,
)
_PREFIX = re.compile(r"^(Task\s*\d*[.\-:]?\s*|Step\s*\d*[\-:.]?\s*|\d+[.)]\s*|[-*•]\s*)", re.IGNORECASE)


def clean_task(text: str) -> str:
    return _PREFIX.sub("", text.strip()).strip().strip('"').strip("`").strip()


def parse_task_list(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    items: list[Any] | None = None
    if match:
        for loader in (json.loads, ast.literal_eval):
            try:
                items = loader(match.group(0))
                break
            except Exception:
                continue
    if items is None:  # no parsable array: numbered / bulleted lines fallback
        items = [ln for ln in text.splitlines() if ln.strip()]
    tasks = [clean_task(str(i)) for i in items]
    return [t for t in tasks if t and not _NO_TASK.match(t)]


def parse_new_task(text: str) -> str:
    lines = [clean_task(ln) for ln in text.strip().splitlines() if ln.strip()]
    task = lines[0] if lines else ""
    return "" if _NO_TASK.match(task) else task


def parse_scout_plan(text: str) -> dict:
    """{"problem": str, "web_queries": [...], "github_queries": [...]} from the scout model output.

    Accepts the JSON object asked for, or (fallback) a plain array, which then counts as web queries.
    """
    plan: dict = {"problem": "", "web_queries": [], "github_queries": []}
    obj = re.search(r"\{.*\}", text or "", re.DOTALL)
    if obj:
        try:
            data = json.loads(obj.group(0))
            plan["problem"] = str(data.get("problem") or "").strip()
            for key in ("web_queries", "github_queries"):
                plan[key] = [clean_task(str(q)) for q in (data.get(key) or []) if str(q).strip()][:3]
            if plan["web_queries"] or plan["github_queries"]:
                return plan
        except Exception:
            pass
    plan["web_queries"] = parse_task_list(text or "")[:3]
    return plan


# --- semantic dedup (no embeddings) -------------------------------------------

_STOP = set(
    "the and for with from that this are was were how what which who when where why can use using into about "
    "del las los una uno unos unas por para con sin sobre como que cual cuales quien cuando donde esta este esto "
    "estos estas son fue ser mas muy entre hacia desde hasta tambien search find investigar investigate analizar "
    "analyze identificar identify buscar".split()
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", normalize(text)) if len(t) > 2 and t not in _STOP}


def similarity(a: str, b: str) -> float:
    """max(Jaccard, containment of the shorter in the longer) over content tokens."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    jaccard = inter / len(ta | tb)
    containment = inter / min(len(ta), len(tb)) if min(len(ta), len(tb)) >= 3 else 0.0
    return max(jaccard, containment)


def is_duplicate(task: str, existing: list[str], threshold: float = 0.6) -> bool:
    if normalize(task).strip() in {normalize(e).strip() for e in existing}:
        return True
    return any(similarity(task, e) >= threshold for e in existing)


# --- grounding of the summary --------------------------------------------------

_INVISIBLE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")  # break copy/paste and word search
_LINK = re.compile(r"\[+([^\[\]]*)\]+\(([^()\s]*)\)")
_MSG = {
    "Spanish": "⚠ {u} de {n} líneas no llevan fuente: verifícalas antes de usarlas.",
    "English": "⚠ {u} of {n} lines have no source: verify them before relying on them.",
}


def norm_url(url: str) -> str:
    p = urlparse(url.strip())
    return f"{(p.hostname or '').lower()}{p.path.rstrip('/')}{'?' + p.query if p.query else ''}"


_BARE_CITE = re.compile(r"\[(\d{1,3})\](?![\(\[:])")


def ground_summary(
    text: str, allowed_urls: set[str], language: str = "Spanish", numbered: list[str] | None = None
) -> tuple[str, dict]:
    """Drop links that are not in the real source list and flag uncited lines.

    A removed numeric citation ([3](bad-url)) disappears; a removed text link keeps its text.
    `numbered` is the source list in the order the model saw it: weaker models write a bare [3] instead of
    [3](url), so those are turned into real links (a number outside the list is left alone).
    """
    if numbered:
        text = _BARE_CITE.sub(
            lambda m: f"[{m.group(1)}]({numbered[int(m.group(1)) - 1]})" if 1 <= int(m.group(1)) <= len(numbered) else m.group(0),
            text,
        )
    allowed = {norm_url(u) for u in allowed_urls}
    stats = {"links_kept": 0, "links_removed": 0, "uncited_lines": 0, "claim_lines": 0}

    def fix(m: re.Match) -> str:
        label, url = m.group(1).strip(), m.group(2)
        if url.startswith("http") and norm_url(url) in allowed:
            stats["links_kept"] += 1
            return f"[{label}]({url})"
        stats["links_removed"] += 1
        return "" if re.fullmatch(r"\d+", label) or not label else label

    grounded = _LINK.sub(fix, _INVISIBLE.sub("", text))

    for line in grounded.splitlines():
        s = line.strip()
        if len(s) < 60 or s.startswith(("#", ">", "---", "```", "|--", "| --", "|:")):
            continue
        stats["claim_lines"] += 1
        if "](http" not in s:
            stats["uncited_lines"] += 1

    if stats["uncited_lines"] and stats["uncited_lines"] / max(stats["claim_lines"], 1) > 0.25:
        msg = _MSG.get(language, _MSG["English"]).format(u=stats["uncited_lines"], n=stats["claim_lines"])
        grounded = f"{grounded.rstrip()}\n\n> {msg}"
    return grounded, stats
