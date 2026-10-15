#!/usr/bin/env python3
"""Smoke del self-tuning wire + endpoints nuevos (puerto 19105):
1. /agent-r/preflight-hybrid con situation → hybrid:true + w_content.
2. /agent-swarm/preflight con situation+task_id → graba regla en task.
3. outcome fail → tuning hit (W sube); outcome complete → miss (W baja).
4. /agent-r/flush → ok:true.
"""
import json, subprocess, time, urllib.request, os, sys

BIN = "target/release/m2m-vector-search"
STATE = "/tmp/p2_tuning.json"
PORT = 19105

def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=5))

if os.path.exists(STATE):
    os.remove(STATE)
env = dict(os.environ); env["AGENT_R_STATE_PATH"] = STATE
proc = subprocess.Popen([BIN, "serve", "--port", str(PORT)], env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
for _ in range(100):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); break
    except Exception:
        time.sleep(0.1)

fails = []
try:
    # Lección: la operación "index_rebuild" falló 2 veces (crítica recurrente).
    for i in range(2):
        post("/agent-r/record-error", {"category": "timeout", "severity": "critical",
            "operation": "index_rebuild", "what": "index rebuild stalled on large corpus",
            "why": "no batching", "context_tags": ["swarm"], "prevention": "batch in chunks of 10k"})
    w0 = post("/agent-r/preflight-hybrid", {"operation": "index_rebuild",
        "context_tags": ["swarm"], "situation": "rebuilding index, corpus grew, stalled"})["w_content"]
    assert abs(w0 - 3.0) < 1e-6, f"W fábrica esperado 3.0, got {w0}"

    # Task del swarm con preflight híbrido (graba regla top-1).
    post("/agent-swarm/register", {"name": "a1", "capabilities": ["index"], "role": None})
    tid = post("/agent-swarm/tasks", {"operation": "index_rebuild",
        "description": "rebuild big index", "depends_on": [], "preferred_agent": None})["task_id"]
    post("/agent-swarm/tasks/claim", {"task_id": tid, "agent": "a1"})
    pf = post("/agent-swarm/preflight", {"operation": "index_rebuild",
        "context_tags": ["swarm"], "situation": "rebuilding index stalled", "task_id": tid})
    assert pf.get("hybrid") is True and pf.get("task_rule_recorded") is not None, pf

    # OUTCOME: fail → la regla anticipó la falla → HIT (W sube).
    post("/agent-swarm/tasks/fail", {"task_id": tid, "agent": "a1", "reason": "stalled again"})
    w1 = post("/agent-r/preflight-hybrid", {"operation": "index_rebuild",
        "context_tags": ["swarm"], "situation": "rebuilding index stalled"})["w_content"]
    assert w1 > 3.0, f"fail debía subir W: {w1}"

    # Nueva task que completa → la señal era FP → MISS (W baja).
    tid2 = post("/agent-swarm/tasks", {"operation": "index_rebuild",
        "description": "rebuild small", "depends_on": [], "preferred_agent": None})["task_id"]
    post("/agent-swarm/tasks/claim", {"task_id": tid2, "agent": "a1"})
    post("/agent-swarm/preflight", {"operation": "index_rebuild",
        "context_tags": ["swarm"], "situation": "small index, quick", "task_id": tid2})
    post("/agent-swarm/tasks/complete", {"task_id": tid2, "agent": "a1", "result": "ok fast"})
    w2 = post("/agent-r/preflight-hybrid", {"operation": "index_rebuild",
        "context_tags": ["swarm"], "situation": "rebuilding index"})["w_content"]
    assert w2 < w1, f"complete debía bajar W: {w1} -> {w2}"

    # Flush explícito.
    fl = post("/agent-r/flush", {})
    assert fl.get("ok") is True, fl
    print(f"W: {w0} -> {w1:.3f} (fail/hit) -> {w2:.3f} (complete/miss) | flush OK")
except AssertionError as e:
    fails.append(str(e))
finally:
    proc.kill(); proc.wait()

if fails:
    print("FALLOS:", fails); sys.exit(1)
print("SMOKE SELF-TUNING OK")
