import asyncio
import sys

from .agent import run_agent


async def main(goal: str, loops: int) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    async for ev in run_agent(goal, max_loops=loops):
        t = ev["type"]
        if t in ("token", "summary_token"):
            print(ev["text"], end="", flush=True)
        elif t == "tasks":
            print(f"\n[cola] pendientes={ev['remaining']} hechas={len(ev['completed'])}")
        elif t == "analysis":
            print(f"\n[{ev['action']}] arg={ev['arg']!r}\n  razon: {ev['reasoning'][:120]}")
        elif t == "task_start":
            print(f"\n=== tarea {ev['loop']}: {ev['task']}")
        elif t == "scout":
            print(f"\n[scout] entidades: {ev['entities']}")
        elif t == "waiting":
            print(f"\n[espera] {ev['seconds']}s (HTTP {ev['status']})", flush=True)
        elif t == "summary_start":
            print("\n\n##### RESUMEN FINAL #####")
        elif t == "summary":
            print(f"\n\n[citas] {ev['stats']}")
        elif t in ("error", "notice"):
            print(f"\n[{t}] {ev.get('message') or ev.get('text')}")
    print()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3))
