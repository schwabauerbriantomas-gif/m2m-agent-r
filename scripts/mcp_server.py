#!/usr/bin/env python3
"""Servidor MCP (stdio) que expone m2mctl.py — control de los dos núcleos M2M.

Cada tool mapea 1:1 a un subcomando de scripts/m2mctl.py (contrato congelado,
SOLO LECTURA), invocado vía subprocess con --json y salida JSON estable.

Requisitos: SDK `mcp` 1.x (FastMCP) — venv ~/venvs/m2m-mcp (creado con uv).
Arranque: ~/venvs/m2m-mcp/bin/python scripts/mcp_server.py  (transporte stdio)

Reglas de operación (también embebidas en las descriptions de las tools):
  - Benches SIEMPRE efímeros (AGENT_R_STATE_PATH= vacío — RAM pura).
  - bench 1m usa puerto propio 19004 y levanta SU propio server (harness).
  - Long jobs (bench 300k/1m, builds Rust) correrlos en background.
  - Verificar puertos con `ss -tln` antes de levantar/detener servers.
  - PIDs y logs en ~/.m2mctl/ (rust.pid, py.pid, *_serve.log).

Contrato de errores (importante para el consumidor MCP):
  - Errores de dominio de m2mctl (rc!=0 o "ok": false) se devuelven como
    resultado NORMAL cuyo contenido ES ese JSON (siempre parseable, 1:1 con
    el contrato --json de m2mctl; sin prefijos ni tracebacks).
  - Errores de infraestructura (timeout, salida no-JSON) o de uso (argumento
    inválido) se lanzan como ToolError → isError=true en el cliente.
"""
import json
import os
import subprocess
import sys

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

# ── Paths absolutos al repo (independiente del cwd del cliente MCP) ────────
REPO = "/home/user/swarm_agent_r/wt_mcp"
M2MCTL = f"{REPO}/scripts/m2mctl.py"

# Timeouts generosos (segundos) por familia de operación.
T_FAST = 60        # status / call / stop / serve (serve espera health ~10 s)
T_TEST = 1200      # cargo test --lib completo (suite 276 tests)
T_BUILD = 1200     # builds Rust desde frío: 600 s+ => margen x2
T_BENCH = 3600     # bench 1m medido ~19 min + overhead del harness

mcp = FastMCP(
    "m2mctl",
    instructions=(
        "Control de los dos núcleos M2M (Rust prod + Python lab) vía m2mctl. "
        "Empezá con `status` (read-only) para ver repos, binario y servers vivos. "
        "REGLAS: benches SIEMPRE efímeros (AGENT_R_STATE_PATH=); bench 1m en "
        "puerto propio 19004 (levanta su server); long jobs (bench 300k/1m, "
        "builds) en background; verificar puertos con `ss -tln`; PIDs/logs en "
        "~/.m2mctl/. Errores de dominio llegan como JSON {\"ok\": false} en el "
        "resultado de la tool (no como isError)."
    ),
)

# Reglas comunes que van en TODAS las descriptions (exigencia del contrato).
RULES = (
    " REGLAS DE OPERACIÓN: (1) benches SIEMPRE efímeros con AGENT_R_STATE_PATH= "
    "vacío (RAM pura, sin snapshot); (2) bench 1m usa puerto propio 19004 y "
    "levanta SU propio server desde el harness; (3) long jobs (bench 300k/1m, "
    "builds Rust) correrlos en background, nunca bloqueando la sesión; "
    "(4) verificar puertos con `ss -tln` antes de levantar o detener servers; "
    "(5) PIDs y logs viven en ~/.m2mctl/. Errores de dominio = JSON {\"ok\": false} "
    "normal; isError solo para args inválidos o fallos de infraestructura."
)

DESC = {
    "status": (
        "Estado de los dos núcleos M2M: repos (branch/commit/dirty), binario "
        "Rust (existe/fresh), venv Python y servers vivos (PID + stats/health). "
        "Read-only, sin efectos secundarios: la tool segura para empezar." + RULES
    ),
    "build": (
        "Build de un núcleo: `rust` (cargo build --release, ~600s+ desde frío) "
        "o `py` (crea venv ~/venvs/m2m-py e instala -e .[all]). deploy=True "
        "copia el binario Rust a ~/swarm_agent_r/bin_integrate (requerido antes "
        "de serve/bench que usan ese binario). Long job: correlo en background. "
        "Timeout interno 1200s." + RULES
    ),
    "test": (
        "Tests de un núcleo: `rust` (cargo test --lib, baseline 276/276), `py` "
        "(pytest tests/) o `all`. filter opcional (nombre de test / -k PAT). "
        "m2mctl ya fuerza TMPDIR=/tmp y --capture=no (el scratch de Hermes rompe "
        "los tmpfiles en vivo). Long job: background. Timeout interno 1200s." + RULES
    ),
    "serve": (
        "Levanta un server de un núcleo en background (PID en ~/.m2mctl/<t>.pid, "
        "log en ~/.m2mctl/<t>_serve.log). Rust: default puerto 19103 persistente; "
        "ephemeral=true = AGENT_R_STATE_PATH= (RAM pura, para benches — REGLA); "
        "state=PATH = snapshot O4 explícito. Python: default 8080. Verificá el "
        "puerto con `ss -tln` antes. NUNCA dejes servers vivos al terminar: usá "
        "`stop`. NO usar 19004 (reservado al harness de bench 1m)." + RULES
    ),
    "stop": (
        "Detiene servers: `rust`, `py` o `all` (SIGTERM, luego SIGKILL si no "
        "muere en ~2s; borra el pidfile). Idempotente. Verificá con `status` o "
        "`ss -tln` que el puerto quedó libre. Usalo SIEMPRE antes de terminar "
        "una sesión que levantó servers." + RULES
    ),
    "bench": (
        "Bench del núcleo Rust con el harness rambench: scale `100k`, `300k` o "
        "`1m`. SIEMPRE efímero (AGENT_R_STATE_PATH= — el harness ya lo fuerza; "
        "nunca benches contra estado persistente). Escala 1m: usa puerto propio "
        "19004 y el harness levanta SU propio server (no compite con el 19103). "
        "Long job (300k/1m, hasta ~20 min): correlo en background. Timeout "
        "interno 3600s. Reportes en reports/*<label>*." + RULES
    ),
    "call": (
        "HTTP directo a un endpoint vivo de un núcleo (default rust:19103). "
        "endpoint p.ej. /agent-r/stats; body = JSON string (opcional); method "
        "GET|POST (AUTO según body). Sin server vivo o endpoint inexistente → "
        "resultado normal JSON {\"ok\": false, ...} (error limpio, sin "
        "traceback). Para probar salud: call rust /agent-r/stats." + RULES
    ),
}


def _toolerr(msg: str) -> ToolError:
    """ToolError con payload JSON (args inválidos / infra). → isError=true."""
    return ToolError(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))


def run_ctl(args: list[str], timeout_s: int) -> str:
    """Corre m2mctl.py <args...> --json y devuelve contenido JSON estable.

    rc==0 y ok:true → contenido normal (JSON indentado).
    rc!=0 u ok:false → TAMBIÉN contenido normal: el JSON de m2mctl con
      ok:false (contrato 1:1; el cliente MCP lo parsea y decide).
    timeout / salida no-JSON / fallo de exec → ToolError (isError=true).
    """
    cmd = [sys.executable, M2MCTL, *args, "--json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                           env=dict(os.environ, PYTHONUNBUFFERED="1"))
    except subprocess.TimeoutExpired:
        raise _toolerr(f"m2mctl {' '.join(args)}: timeout tras {timeout_s}s "
                       "(si es long job: correlo en background)")
    except OSError as e:
        raise _toolerr(f"no pude ejecutar {M2MCTL}: {e}")
    out = (r.stdout or "").strip()
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        raise _toolerr(f"salida de m2mctl no es JSON (rc={r.returncode}); "
                       f"stdout_tail={out[-400:]!r} stderr_tail={(r.stderr or '')[-400:]!r}")
    # Errores de dominio: se devuelven como resultado normal (JSON estable).
    return json.dumps(payload, indent=1, ensure_ascii=False)


# ── Tools (1:1 con subcomandos de m2mctl) ─────────────────────────────────

@mcp.tool(description=DESC["status"])
def status() -> str:
    return run_ctl(["status"], T_FAST)


@mcp.tool(description=DESC["build"])
def build(target: str, deploy: bool = False) -> str:
    if target not in ("rust", "py"):
        raise _toolerr("target debe ser 'rust' o 'py'")
    args = ["build", target] + (["--deploy"] if deploy else [])
    return run_ctl(args, T_BUILD)


@mcp.tool(description=DESC["test"])
def test(target: str, filter: str = "") -> str:
    if target not in ("rust", "py", "all"):
        raise _toolerr("target debe ser 'rust', 'py' o 'all'")
    args = ["test", target] + (["--filter", filter] if filter else [])
    return run_ctl(args, T_TEST)


@mcp.tool(description=DESC["serve"])
def serve(target: str, port: int = 0, ephemeral: bool = False,
          state: str = "") -> str:
    if target not in ("rust", "py"):
        raise _toolerr("target debe ser 'rust' o 'py'")
    args = ["serve", target]
    if port:
        args += ["--port", str(port)]
    if ephemeral:
        args += ["--ephemeral"]
    elif state:
        args += ["--state", state]
    return run_ctl(args, T_FAST)


@mcp.tool(description=DESC["stop"])
def stop(target: str = "all") -> str:
    if target not in ("rust", "py", "all"):
        raise _toolerr("target debe ser 'rust', 'py' o 'all'")
    return run_ctl(["stop", target], T_FAST)


@mcp.tool(description=DESC["bench"])
def bench(scale: str, abort_at: int = 0) -> str:
    if scale not in ("100k", "300k", "1m"):
        raise _toolerr("scale debe ser '100k', '300k' o '1m'")
    args = ["bench", "rust", scale] + (["--abort-at", str(abort_at)] if abort_at else [])
    return run_ctl(args, T_BENCH)


@mcp.tool(description=DESC["call"])
def call(target: str, endpoint: str, body: str = "", port: int = 0,
         method: str = "AUTO") -> str:
    if target not in ("rust", "py"):
        raise _toolerr("target debe ser 'rust' o 'py'")
    if body:
        try:
            json.loads(body)
        except json.JSONDecodeError as e:
            raise _toolerr(f"body no es JSON válido: {e}")
    args = ["call", target, endpoint] + ([body] if body else [])
    if port:
        args += ["--port", str(port)]
    if method != "AUTO":
        args += ["--method", method]
    return run_ctl(args, T_FAST)


if __name__ == "__main__":
    mcp.run()  # transporte stdio (default)
