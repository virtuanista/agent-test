// Desktop shell. The Python backend (FastAPI) is the app; Electron only starts it, opens a window on it
// and stops it on quit. Development mode: uses the project's .venv (see scripts/setup.js).
const { app, BrowserWindow, dialog, shell, session } = require("electron");
const { spawn, spawnSync } = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const isWin = process.platform === "win32";
const HOST = "127.0.0.1";
const token = crypto.randomBytes(24).toString("hex"); // only this window knows it: see require_app_token in app/main.py

let backend = null;
let backendLog = [];
let win = null;
let origin = "";
let quitting = false;

function pythonPath() {
  const venv = path.join(root, ".venv", isWin ? "Scripts" : "bin", isWin ? "python.exe" : "python");
  return fs.existsSync(venv) ? venv : null;
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once("error", reject);
    srv.listen(0, HOST, () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

async function waitForBackend(port, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (backend && backend.exitCode !== null) throw new Error("El backend terminó antes de arrancar.");
    try {
      const res = await fetch(`http://${HOST}:${port}/api/health`, { headers: { cookie: `agent_token=${token}` } });
      if (res.ok) return;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("El backend no respondió en 30 s.");
}

function startBackend(port) {
  const py = pythonPath();
  if (!py) throw new Error("No existe .venv. Ejecuta: npm run setup (si falta Python: winget install -e --id Python.Python.3.12)");
  backend = spawn(py, ["-m", "uvicorn", "app.main:app", "--host", HOST, "--port", String(port)], {
    cwd: root,
    env: { ...process.env, APP_TOKEN: token, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
    windowsHide: true,
  });
  const keep = (d) => { backendLog.push(String(d)); backendLog = backendLog.slice(-60); };
  backend.stdout.on("data", keep);
  backend.stderr.on("data", keep);
}

function stopBackend() {
  if (!backend || backend.exitCode !== null) return;
  if (isWin) spawnSync("taskkill", ["/pid", String(backend.pid), "/T", "/F"]); // uvicorn may have children
  else backend.kill("SIGTERM");
}

async function activeRun() {
  try {
    const res = await fetch(`${origin}/api/runs/active`, { headers: { cookie: `agent_token=${token}` } });
    return res.ok ? await res.json() : null;
  } catch { return null; }
}

function createWindow() {
  win = new BrowserWindow({
    width: 1280, height: 860, minWidth: 720, minHeight: 520, show: false,
    title: "Agent Test",
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true, spellcheck: false },
  });
  win.once("ready-to-show", () => win.show());
  // Links in model/web content open in the real browser, never inside the app window.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//i.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (e, url) => {
    if (!url.startsWith(origin)) { e.preventDefault(); if (/^https?:\/\//i.test(url)) shell.openExternal(url); }
  });
  win.on("close", async (e) => {
    if (quitting) return;
    e.preventDefault();
    const run = await activeRun();
    if (run) {
      const { response } = await dialog.showMessageBox(win, {
        type: "warning", buttons: ["Seguir aquí", "Cerrar y detener"], defaultId: 0, cancelId: 0,
        title: "Ejecución en curso",
        message: "Hay una investigación en marcha.",
        detail: "Si cierras la aplicación se detiene (el backend se apaga con ella). Lo generado hasta ahora queda en el historial.",
      });
      if (response === 0) return;
    }
    quitting = true;
    app.quit();
  });
  win.loadURL(origin);
}

async function boot() {
  const port = await freePort();
  origin = `http://${HOST}:${port}`;
  startBackend(port);
  await waitForBackend(port);
  await session.defaultSession.cookies.set({
    url: origin, name: "agent_token", value: token, httpOnly: true, sameSite: "strict",
  });
  createWindow();
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => { if (win) { if (win.isMinimized()) win.restore(); win.focus(); } });
  app.whenReady().then(() =>
    boot().catch((err) => {
      dialog.showErrorBox("No se pudo arrancar Agent Test", `${err.message}\n\n${backendLog.join("").slice(-1500)}`);
      quitting = true;
      stopBackend();
      app.quit();
    }),
  );
  app.on("before-quit", () => { quitting = true; stopBackend(); });
  app.on("window-all-closed", () => app.quit());
}
