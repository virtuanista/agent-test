"""Evaluacion de calidad de investigacion.

Uso: python eval/run_eval.py <etiqueta> [id_objetivo ...] [--loops N]
Guarda eval/results/<fecha>-<etiqueta>.json y imprime una tabla.
Es una evaluacion orientativa (pocos objetivos), pensada para comparar antes/despues.
"""
import asyncio
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import run_agent  # noqa: E402
from app.llm import LLM  # noqa: E402

URL_RE = re.compile(r"https?://[^\s)\]>\"'`]+")
MD_LINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", norm(text)) if len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def clean_url(url: str) -> str:
    return url.rstrip(".,;:").rstrip("/")


async def run_one(item: dict, loops: int) -> dict:
    stats0 = dict(LLM.stats)
    t0 = time.time()
    tasks: list[str] = []
    results: list[str] = []
    summary, error, ledger = "", None, set()
    waiting = 0
    async for ev in run_agent(item["goal"], "Spanish", loops):
        t = ev["type"]
        if t == "task_start":
            tasks.append(ev["task"])
        elif t == "task_done":
            results.append(ev["result"])
        elif t == "summary":
            summary = ev["text"]
        elif t == "sources":  # ledger emitted by the agent (after the grounding phase)
            ledger |= {clean_url(s["url"]) for s in ev["sources"]}
        elif t == "waiting":
            waiting += 1
        elif t == "error":
            error = ev["message"]

    corpus = norm(summary + "\n" + "\n".join(results))
    found = [alts[0] for alts in item["expected"] if any(norm(a) in corpus for a in alts)]

    # Citation validity: every markdown link in the summary should be a known source.
    # Before the ledger exists, "known" = any URL that appeared in a task result.
    known = ledger or {clean_url(u) for r in results for u in URL_RE.findall(r)}
    cited = [clean_url(u) for u in MD_LINK_RE.findall(summary)]
    valid = [u for u in cited if u in known]
    broken = len(re.findall(r"\]\((?!https?://)[^)]*\)", summary))

    # Duplicate tasks: near-duplicate (Jaccard >= 0.6) of an earlier task.
    dups = sum(
        1 for i, t in enumerate(tasks) if any(jaccard(tokens(t), tokens(p)) >= 0.6 for p in tasks[:i])
    )
    pdfs = sum(1 for r in results for u in URL_RE.findall(r) if u.lower().split("?")[0].endswith(".pdf"))
    year_queries = sum(1 for t in tasks if re.search(r"\b20(1\d|2[0-4])\b", t))

    return {
        "id": item["id"],
        "elapsed_s": round(time.time() - t0, 1),
        "tasks": len(tasks),
        "entities_found": found,
        "entities_recall": f"{len(found)}/{len(item['expected'])}",
        "citations": len(cited),
        "citations_valid": len(valid),
        "citations_broken_relative": broken,
        "duplicate_tasks": dups,
        "pdf_sources": pdfs,
        "stale_year_tasks": year_queries,
        "llm_calls": LLM.stats["calls"] - stats0["calls"],
        "llm_retries_429": LLM.stats["retries"] - stats0["retries"],
        "model_switches": LLM.stats.get("switches", 0) - stats0.get("switches", 0),
        "waiting_events": waiting,
        "summary_chars": len(summary),
        "error": error,
        "task_list": tasks,
        "summary": summary,
    }


async def main() -> None:
    args = sys.argv[1:]
    loops = 5
    if "--loops" in args:
        i = args.index("--loops")
        loops = int(args[i + 1])
        del args[i : i + 2]
    label = args[0] if args else "run"
    only = set(args[1:])

    goals = json.loads((ROOT / "eval" / "goals.json").read_text(encoding="utf-8"))
    goals = [g for g in goals if not only or g["id"] in only]

    out = []
    for g in goals:
        print(f"-> {g['id']} ...", flush=True)
        res = await run_one(g, loops)
        out.append(res)
        print(
            f"   {res['elapsed_s']}s tareas={res['tasks']} entidades={res['entities_recall']} "
            f"citas_validas={res['citations_valid']}/{res['citations']} dup={res['duplicate_tasks']} "
            f"pdf={res['pdf_sources']} año_viejo={res['stale_year_tasks']} 429={res['llm_retries_429']} "
            f"err={res['error']}",
            flush=True,
        )

    path = ROOT / "eval" / "results" / f"{datetime.now():%Y%m%d-%H%M}-{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"label": label, "loops": loops, "results": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nguardado: {path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
