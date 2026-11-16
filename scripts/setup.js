// One-command setup: finds Python 3.10+, creates .venv, installs the requirements and a starter .env. Idempotent.
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const isWin = process.platform === "win32";
const venvPython = path.join(root, ".venv", isWin ? "Scripts" : "bin", isWin ? "python.exe" : "python");

function run(cmd, args, label) {
  const r = spawnSync(cmd, args, { cwd: root, stdio: "inherit" });
  if (r.status !== 0) {
    console.error(`\n[setup] Fallo: ${label}.`);
    process.exit(1);
  }
}

function findPython() {
  const candidates = isWin ? [["py", ["-3"]], ["python", []]] : [["python3", []], ["python", []]];
  return (
    candidates.find(([cmd, pre]) => {
      const r = spawnSync(cmd, [...pre, "-c", "import sys, venv; print(sys.version_info >= (3, 10))"], { encoding: "utf8" });
      return r.status === 0 && r.stdout.trim() === "True";
    }) || null
  );
}

if (!fs.existsSync(venvPython)) {
  const python = findPython();
  if (!python) {
    console.error("\n[setup] No hay Python 3.10+ instalado. Instálalo y repite: npm run setup");
    if (isWin) console.error("[setup]   winget install -e --id Python.Python.3.12   (o https://www.python.org/downloads/)");
    // Do not fail `npm install` for this: Electron is already installed, only the backend setup is pending.
    process.exit(process.env.npm_lifecycle_event === "postinstall" ? 0 : 1);
  }
  console.log("[setup] Creando entorno virtual .venv ...");
  run(python[0], [...python[1], "-m", "venv", ".venv"], "crear el entorno virtual");
}

console.log("[setup] Instalando dependencias de Python ...");
run(venvPython, ["-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", "requirements-dev.txt"], "instalar requirements-dev.txt");

const env = path.join(root, ".env");
if (!fs.existsSync(env)) {
  fs.copyFileSync(path.join(root, ".env.example"), env);
  console.log("[setup] Creado .env desde .env.example: edita GROQ_API_KEY (u OPENROUTER_API_KEY) antes de ejecutar.");
}
console.log("[setup] Listo. Arranca con: npm start");
