#!/usr/bin/env python3
"""Smoke test del servidor MCP m2mctl (scripts/mcp_server.py).

Automatable: exit 0 = OK, exit 1 = fallo (con diagnóstico).

Prueba, SIN levantar ningún server long-running:
  1. Conexión stdio + initialize (handshake MCP).
  2. tools/list: las 7 tools presentes (status/build/test/serve/stop/bench/call)
     y las reglas de operación embebidas en las descriptions.
  3. tools/call `status` real (read-only; funciona sin servers vivos).
  4. tools/call `call` a un endpoint inexistente → error limpio estructurado
     (isError, JSON con ok:false; sin traceback python).

Uso:
  ~/venvs/m2m-mcp/bin/python scripts/test_mcp_server.py
"""
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "mcp_server.py")
VENV_PY = os.path.expanduser("~/venvs/m2m-mcp/bin/python")

EXPECTED_TOOLS = {"status", "build", "test", "serve", "stop", "bench", "call"}
# Reglas que DEBEN aparecer en las descriptions de las tools.
REQUIRED_RULE_MARKERS = ("efímer", "19004", "background", "ss -tln")


async def main() -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    failures: list[str] = []

    params = StdioServerParameters(
        command=VENV_PY,
        args=[SERVER],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
             "HOME": os.environ.get("HOME", "/home/user")},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. handshake
            init = await session.initialize()
            print(f"[1] initialize OK: server={init.serverInfo.name} "
                  f"v{init.serverInfo.version} proto={init.protocolVersion}")

            # 2. tools/list
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            print(f"[2] tools/list: {len(names)} tools → {sorted(names)}")
            if names != EXPECTED_TOOLS:
                failures.append(f"tools mismatch: esperadas {sorted(EXPECTED_TOOLS)}, "
                                f"obtenidas {sorted(names)}")
            for t in tools.tools:
                missing = [m for m in REQUIRED_RULE_MARKERS if m not in t.description]
                if missing:
                    failures.append(f"tool '{t.name}': description sin reglas {missing}")

            # 3. status real (read-only, sin servers vivos)
            r = await session.call_tool("status", {})
            text = r.content[0].text if r.content else "{}"
            st = json.loads(text)
            print(f"[3] call status: ok={st.get('ok')} "
                  f"rust.repo.exists={st.get('rust', {}).get('exists')} "
                  f"rust.server_alive={st.get('rust', {}).get('server_alive')} "
                  f"python.server_alive={st.get('python', {}).get('server_alive')}")
            if r.isError or not st.get("ok"):
                failures.append(f"status devolvió error: {text[:300]}")

            # 4. call a endpoint inexistente → error de dominio LIMPIO:
            #    resultado normal (no isError), JSON parseable con ok:false,
            #    sin traceback python.
            r = await session.call_tool(
                "call", {"target": "rust", "endpoint": "/no/siste",
                         "port": 19399})  # puerto de humo, nada escucha ahí
            etext = r.content[0].text if r.content else ""
            print(f"[4] call /no/siste → isError={r.isError} text={etext[:180]}")
            if r.isError:
                failures.append(f"error de dominio llegó como isError (debe ser "
                                f"resultado normal): {etext[:200]}")
            else:
                try:
                    err = json.loads(etext)
                    if err.get("ok") is not False:
                        failures.append(f"resultado sin ok:false: {etext[:200]}")
                    if "Traceback" in etext:
                        failures.append("error contiene traceback python (no limpio)")
                except json.JSONDecodeError:
                    failures.append(f"error no es JSON parseable: {etext[:200]}")

            # 5. args inválidos → isError=true (validación del server MCP)
            r = await session.call_tool("bench", {"scale": "9m"})
            btext = r.content[0].text if r.content else ""
            print(f"[5] bench scale=9m → isError={r.isError} text={btext[:120]}")
            if not r.isError:
                failures.append("args inválidos deberían dar isError=true")

    if failures:
        print("\nSMOKE FAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nSMOKE OK: 7 tools, reglas en descriptions, status real, error limpio.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
