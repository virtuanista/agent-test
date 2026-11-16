// `npm test`: runs the Python suite with the project's venv, whatever the OS.
const { spawnSync } = require("node:child_process");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const win = process.platform === "win32";
const py = path.join(root, ".venv", win ? "Scripts" : "bin", win ? "python.exe" : "python");
process.exit(spawnSync(py, ["-m", "pytest", "-q"], { cwd: root, stdio: "inherit" }).status ?? 1);
