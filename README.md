# agent-test

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)
![Electron](https://img.shields.io/badge/Electron-escritorio-47848f?logo=electron&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?logo=fastapi&logoColor=white)
![Licencia MIT](https://img.shields.io/badge/licencia-MIT-blue)

Agente autónomo de investigación web, sin servicios de pago: LLM por cualquier endpoint compatible con OpenAI
(Groq por defecto, OpenRouter como respaldo) + búsqueda con DuckDuckGo (sin servidor ni key).

Bucle: `scout -> plan -> (analyze -> execute -> create)* -> summarize`.
Herramientas: `search` (DuckDuckGo + lectura de páginas), `reason`, `code`. El resumen final solo conserva los
enlaces que están en la lista real de fuentes.

## Inicio rápido (escritorio, Electron)
Requisitos: **Node 20+** y **Python 3.10+**.
```
git clone https://github.com/virtuanista/agent-test.git && cd agent-test
npm install        # instala Electron y prepara .venv + dependencias de Python (scripts/setup.js)
# edita .env: GROQ_API_KEY=... (y/o OPENROUTER_API_KEY=...). Las keys se leen siempre de .env
npm start          # abre la ventana y arranca el backend
npm test           # tests de Python
```
- Sin Python, en Windows: `winget install -e --id Python.Python.3.12` y luego `npm run setup`.
- La ventana habla con el backend por un puerto local aleatorio y una cookie secreta por arranque (`APP_TOKEN`):
  ningún otro proceso o página local puede usar la API. Al cerrar la ventana el backend se apaga
  (si hay una ejecución en curso te pregunta antes).
- Sin Electron: `start.bat` arranca el servidor en http://127.0.0.1:8000, y la CLI es
  `.venv\Scripts\python -m app.cli "tu objetivo" 3`.

## Modelos
`LLM_MODEL_POOL` es el orden de respaldo (si uno recibe un 429 se pasa al siguiente). Con `OPENROUTER_API_KEY` se
añaden modelos `:free` de OpenRouter (`OPENROUTER_MODELS` para cambiarlos); una entrada `proveedor::modelo` usa ese
proveedor. Ojo: el tier gratuito de OpenRouter es ~50 peticiones al día **por cuenta** (1000 con 10 USD de crédito).

En la interfaz, el selector de modelo solo marca como disponible lo que respondió de verdad a una prueba real
(llamada con herramientas + respuesta de texto no vacía). Los caídos salen en gris con el motivo. Las comprobaciones
automáticas no tocan OpenRouter (gastan su cuota): usa el botón de recomprobar para verificarlo.

## API
Las ejecuciones corren en segundo plano en el servidor: cerrar o recargar la ventana no las detiene (solo se pierden
si se reinicia el servidor; quedan marcadas como `interrupted`). Solo hay una a la vez.
- `POST /api/run` {goal, language, max_loops, model?} -> `{run_id}` (409 con `detail.run_id` si ya hay una en curso)
- `GET /api/runs/{id}/stream?from=N` -> SSE: repite los eventos desde el índice N y sigue en vivo
- `GET /api/runs/active` -> la ejecución en curso o `null`; `POST /api/runs/{id}/stop` la detiene
- `GET /api/runs`, `GET /api/runs/{id}` -> historial (SQLite, `agent.db`)
- `GET /api/health` -> estado de la clave del modelo y del respaldo
- `GET /api/models[?check=1[&manual=1]]` -> modelos con estado verificado (`ok`/`limited`/`error`/`unknown`)

## Evaluación
`python eval/run_eval.py <etiqueta> [id ...] [--loops N]` ejecuta los objetivos de `eval/goals.json` y guarda una
tabla en `eval/results/`. Es orientativa (pocos objetivos), pensada para comparar antes y después de un cambio.

## Licencia

[MIT](LICENSE) © 2026 virtuanista

<p align="center">
	Repositorio generado por <a href="https://github.com/virtuanista" target="_blank">virtu 🎣</a>
</p>

<p align="center">
	<img src="https://open.soniditos.com/cat_footer.svg" alt="Gatito" />
</p>

<p align="center">
	Copyright &copy; 2026
</p>

<p align="center">
	<a href="LICENSE"><img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=License&message=MIT&logoColor=d9e0ee&colorA=363a4f&colorB=b7bdf8" alt="Licencia MIT" /></a>
</p>
