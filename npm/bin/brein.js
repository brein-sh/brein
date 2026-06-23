#!/usr/bin/env node
// Thin Node shim. Verifies python3 + uv, installs the brein Python package
// from git, then execs the real `brein` CLI. No deps.

const { spawnSync, spawn } = require("node:child_process");
const { platform } = require("node:os");

const REPO_URL = "git+https://github.com/brein-sh/brein.git";
const MIN_PY = [3, 11];

const C = {
  dim: (s) => `\x1b[2m${s}\x1b[0m`,
  red: (s) => `\x1b[31m${s}\x1b[0m`,
  yellow: (s) => `\x1b[33m${s}\x1b[0m`,
  green: (s) => `\x1b[32m${s}\x1b[0m`,
  bold: (s) => `\x1b[1m${s}\x1b[0m`,
};

function which(cmd) {
  const r = spawnSync(platform() === "win32" ? "where" : "which", [cmd], {
    encoding: "utf8",
  });
  return r.status === 0 ? r.stdout.trim().split(/\r?\n/)[0] : null;
}

function pythonVersion(bin) {
  const r = spawnSync(bin, ["-c", "import sys; print(sys.version_info[0], sys.version_info[1])"], { encoding: "utf8" });
  if (r.status !== 0) return null;
  const [maj, min] = r.stdout.trim().split(/\s+/).map(Number);
  return [maj, min];
}

function findPython() {
  for (const cmd of ["python3.13", "python3.12", "python3.11", "python3", "python"]) {
    if (!which(cmd)) continue;
    const v = pythonVersion(cmd);
    if (!v) continue;
    if (v[0] > MIN_PY[0] || (v[0] === MIN_PY[0] && v[1] >= MIN_PY[1])) {
      return { bin: cmd, version: v };
    }
  }
  return null;
}

function pythonInstallHint() {
  const p = platform();
  if (p === "darwin") return "brew install python@3.12  # or visit https://www.python.org/downloads/";
  if (p === "linux") return "sudo apt install python3.12  # or your distro's equivalent";
  if (p === "win32") return "winget install Python.Python.3.12  # or https://www.python.org/downloads/";
  return "Install Python 3.11+ from https://www.python.org/downloads/";
}

function uvInstallHint() {
  if (platform() === "win32") return "powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"";
  return "curl -LsSf https://astral.sh/uv/install.sh | sh";
}

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  return r.status === 0;
}

function execReplace(cmd, args) {
  // No exec() in Node; spawn with inherited stdio and forward the exit code.
  const child = spawn(cmd, args, { stdio: "inherit" });
  child.on("exit", (code) => process.exit(code ?? 0));
}

function preflight() {
  const py = findPython();
  if (!py) {
    console.error(C.red("✗") + " Python " + MIN_PY.join(".") + "+ not found.");
    console.error("  → " + pythonInstallHint());
    process.exit(2);
  }
  console.log(C.green("✓") + ` python ${py.version.join(".")} (${py.bin})`);

  const uv = which("uv");
  if (!uv) {
    console.error(C.red("✗") + " uv not found.");
    console.error("  → " + uvInstallHint());
    console.error("    (uv is a single static binary, no Python prereq.)");
    process.exit(2);
  }
  console.log(C.green("✓") + ` uv (${uv})`);
}

function installBrein(branch) {
  const url = branch ? `${REPO_URL}@${branch}` : REPO_URL;
  console.log(C.bold("\nInstalling brein from") + " " + C.dim(url));
  const ok = run("uv", ["tool", "install", "--force", url]);
  if (!ok) {
    console.error(C.red("\nuv tool install failed."));
    process.exit(1);
  }
}

function ensureOnPath() {
  if (!which("brein")) {
    console.error(C.yellow("!") + " `brein` not on PATH after install.");
    console.error("  → run: " + C.bold("uv tool update-shell") + "  (then restart your shell)");
    console.error("    or add ~/.local/bin to PATH manually.");
    process.exit(1);
  }
}

function usage() {
  console.log(`brein — npm wrapper for the brein MCP server

Usage:
  npx brein init [--branch <name>]    Install Python package + run setup wizard
  npx brein <subcommand> [args...]    Forward to the installed brein CLI
                                      (setup | doctor | mcp | config)

Examples:
  npx brein init
  npx brein doctor
  npx brein mcp claude
`);
}

function main() {
  const [, , cmd, ...rest] = process.argv;

  if (!cmd || cmd === "--help" || cmd === "-h") {
    usage();
    process.exit(cmd ? 0 : 1);
  }

  if (cmd === "init") {
    const branchIdx = rest.indexOf("--branch");
    const branch = branchIdx >= 0 ? rest[branchIdx + 1] : null;
    preflight();
    installBrein(branch);
    ensureOnPath();
    console.log(C.bold("\nRunning ") + C.bold("brein setup") + C.bold("...\n"));
    execReplace("brein", ["setup"]);
    return;
  }

  // Forward anything else to the installed CLI.
  if (!which("brein")) {
    console.error(C.red("✗") + " brein is not installed. Run: npx brein init");
    process.exit(1);
  }
  execReplace("brein", [cmd, ...rest]);
}

main();
