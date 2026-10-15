#!/usr/bin/env python3
"""Prueba de durabilidad del debounce (P2-validate).

Contrato: bajo tráfico, pérdida máx = FLUSH_INTERVAL (2 s) o FLUSH_MAX_OPS
desde el último flush exitoso. Escenario TRÁFICO: escrituras que cruzan el
intervalo => el writer background compromete el snapshot SIN flush()
explícito; kill -9 inmediato => delta acotado por el intervalo.

(Punto fino del diseño, documentado en agent_r.rs: el flush por tiempo se
dispara EN el camino de escritura — un proceso que muere en reposo absoluto
dentro de la ventana final <2 s pierde las ops de esa ventana quietas. Para
apagado controlado: flush() / Drop.)

Puerto 19105 (exclusivo de P2-validate).
"""
import json, subprocess, time, urllib.request, os, signal, sys

BIN = "target/release/m2m-vector-search"
STATE = "/tmp/p2_graceful.json"
PORT = 19105

def post(i):
    d = json.dumps({"category": "other", "severity": "warning",
        "operation": "op", "what": f"w {i}", "why": "y", "context_tags": [],
        "prevention": "p"}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/agent-r/record-error",
                               data=d, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=5))

if os.path.exists(STATE):
    os.remove(STATE)

env = dict(os.environ)
env["AGENT_R_STATE_PATH"] = STATE
proc = subprocess.Popen([BIN, "serve", "--port", str(PORT)], env=env)
for _ in range(100):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
        break
    except Exception:
        time.sleep(0.1)

# Tráfico que cruza el intervalo varias veces (>= 2 ventanas completas).
t0 = time.time()
i = 0
while time.time() - t0 < 4.0:
    post(i); i += 1
n_written = i
time.sleep(0.3)  # margen para que el writer comprometa el último job
proc.send_signal(signal.SIGKILL)  # peor caso: ni Drop ni handler
proc.wait(timeout=10)

d = json.load(open(STATE))
n_persisted = len(d["errors"])
seeds = 3
delta = n_written - (n_persisted - seeds)
print(f"escrituras: {n_written}, snapshot: {n_persisted} ({seeds} seeds), delta: {delta}")
# delta acotado por la ventana del intervalo: a ~700 rps, 2 s < 2000 ops.
ok = 0 <= delta <= 2000 and n_persisted > seeds
print("CONTRATO OK" if ok else "CONTRATO VIOLADO")
sys.exit(0 if ok else 1)
