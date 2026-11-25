<p align="center"><a href="README.md">Español</a> · <b>English</b></p>

# agent-test

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)
![Electron](https://img.shields.io/badge/Electron-desktop-47848f?logo=electron&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?logo=fastapi&logoColor=white)
![MIT License](https://img.shields.io/badge/license-MIT-blue)

Autonomous web research agent with no paid services: LLM through any OpenAI-compatible endpoint
(Groq by default, OpenRouter as fallback) + search with DuckDuckGo (no server or key).

Loop: `scout -> plan -> (analyze -> execute -> create)* -> summarize`.
Tools: `search` (DuckDuckGo + page reading), `reason`, `code`. The final summary only keeps the
links that are in the real list of sources.

## Quick start (desktop, Electron)
Requirements: **Node 20+** and **Python 3.10+**.
```
git clone https://github.com/virtuanista/agent-test.git && cd agent-test
npm install        # installs Electron and sets up .venv + Python dependencies (scripts/setup.js)
# edit .env: GROQ_API_KEY=... (and/or OPENROUTER_API_KEY=...). Keys are always read from .env
npm start          # opens the window and starts the backend
npm test           # Python tests
```
- Without Python, on Windows: `winget install -e --id Python.Python.3.12` and then `npm run setup`.
- The window talks to the backend over a random local port and a secret cookie per launch (`APP_TOKEN`):
  no other local process or page can use the API. Closing the window shuts down the backend
  (if a run is in progress it asks you first).
- Without Electron: `start.bat` starts the server at http://127.0.0.1:8000, and the CLI is
  `.venv\Scripts\python -m app.cli "tu objetivo" 3` (`tu objetivo`: your goal).

## Models
`LLM_MODEL_POOL` is the fallback order (if one gets a 429, the next one is used). With `OPENROUTER_API_KEY`,
OpenRouter `:free` models are added (`OPENROUTER_MODELS` to change them); a `provider::model` entry uses that
provider. Note: OpenRouter's free tier is ~50 requests per day **per account** (1000 with 10 USD of credit).

In the interface, the model selector only marks as available what actually answered a real test
(a tool call + a non-empty text response). Models that are down appear greyed out with the reason. The automatic
checks do not touch OpenRouter (they use up its quota): use the recheck button to verify it.

## API
Runs execute in the background on the server: closing or reloading the window does not stop them (they are only lost
if the server restarts; they are then marked as `interrupted`). Only one runs at a time.
- `POST /api/run` {goal, language, max_loops, model?} -> `{run_id}` (409 with `detail.run_id` if one is already in progress)
- `GET /api/runs/{id}/stream?from=N` -> SSE: replays events from index N and continues live
- `GET /api/runs/active` -> the run in progress or `null`; `POST /api/runs/{id}/stop` stops it
- `GET /api/runs`, `GET /api/runs/{id}` -> history (SQLite, `agent.db`)
- `GET /api/health` -> status of the model key and the fallback
- `GET /api/models[?check=1[&manual=1]]` -> models with verified status (`ok`/`limited`/`error`/`unknown`)

## Evaluation
`python eval/run_eval.py <label> [id ...] [--loops N]` runs the goals in `eval/goals.json` and saves a
table in `eval/results/`. It is indicative (few goals), meant for comparing before and after a change.

## License

[MIT](LICENSE) © 2026 virtuanista

<p align="center">
	Repository made by <a href="https://github.com/virtuanista" target="_blank">virtu 🎣</a>
</p>

<p align="center">
	<img src="https://open.soniditos.com/cat_footer.svg" alt="Cat" />
</p>

<p align="center">
	Copyright &copy; 2026
</p>

<p align="center">
	<a href="LICENSE"><img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=License&message=MIT&logoColor=d9e0ee&colorA=363a4f&colorB=b7bdf8" alt="MIT License" /></a>
</p>
