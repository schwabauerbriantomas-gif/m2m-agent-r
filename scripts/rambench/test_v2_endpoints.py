#!/usr/bin/env python3
"""Smoke test de los endpoints v2 (P2-validate): preflight (top-3/scoring)
y validate (N0-N3 con evidencia). Puerto 19105."""
import json, subprocess, time, urllib.request, os, sys

BIN = "target/release/m2m-vector-search"
STATE = "/tmp/p2_smoke.json"
PORT = 19105

def post(path, body):
    d = json.dumps(body).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=d,
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=5))

if os.path.exists(STATE):
    os.remove(STATE)
env = dict(os.environ)
env["AGENT_R_STATE_PATH"] = STATE
proc = subprocess.Popen([BIN, "serve", "--port", str(PORT)], env=env,
                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
for _ in range(100):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
        break
    except Exception:
        time.sleep(0.1)

fails = []
try:
    # Cargar lecciones: una critical recurrente (veto), una warning.
    for _ in range(3):
        post("/agent-r/record-error", {"category": "fabrication", "severity": "critical",
            "operation": "generate_data", "what": "Made up numbers", "why": "No testing",
            "context_tags": ["data"], "prevention": "Test first"})
    post("/agent-r/record-error", {"category": "other", "severity": "warning",
        "operation": "hnsw", "what": "index corrupt", "why": "y",
        "context_tags": ["index"], "prevention": "rebuild from source"})

    # ── preflight v2 ──
    pf = post("/agent-r/preflight", {"operation": "generate_data", "context_tags": ["data"]})
    assert pf["proceed"] is False, "veto esperado"
    assert any("BLOCKED" in w for w in pf["warnings"]), pf["warnings"]
    assert len(pf["top_rules"]) >= 1 and pf["top_rules"][0]["score"] > 0, pf["top_rules"]
    assert pf["suggestion"] is not None
    assert pf["veto_reason"] is not None
    print("preflight v2 veto:", json.dumps(pf["veto_reason"]))

    # No substring cruzado: query por otra op que CONTIENE "hnsw".
    pf2 = post("/agent-r/preflight", {"operation": "rebuild_hnsw_index", "context_tags": []})
    no_exact = all(r["operation"] != "rebuild_hnsw_index" or True for r in pf2["top_rules"])
    # la lección 'hnsw' puede aparecer por recencia pero sin bonus exact (+10):
    hnsw_rule = next((r for r in pf2["top_rules"] if r["operation"] == "hnsw"), None)
    if hnsw_rule:
        assert hnsw_rule["score"] < 10.0, f"score sin exact-match debe ser < 10: {hnsw_rule}"
    print("preflight v2 no-substring OK (top:", [r['operation'] for r in pf2['top_rules']], ")")

    # ── validate v2 ──
    # N1 sin FP por substring:
    v = post("/agent-r/validate", {"operation": "compute_distance",
        "output": {"score": "NaN", "note": "Infinity last run"}})
    assert v["verdicts"]["n1_numeric_consistency"] == "pass", v["verdicts"]
    assert v["overall"] == "pass" and v["valid"] is True
    assert v["confidence"] == 1.0
    print("validate N1 sin FP:", v["overall"])

    # N0 null:
    v2 = post("/agent-r/validate", {"operation": "x", "output": None})
    assert v2["verdicts"]["n0_form"] == "fail" and v2["overall"] == "fail"
    assert v2["confidence"] == 0.0 and v2["valid"] is False
    assert "N0" in v2["suggestion"]
    print("validate N0 null:", v2["overall"])

    # N2 con evidencia:
    v3 = post("/agent-r/validate", {"operation": "build", "output": {"ok": 1},
        "evidence": {"exit_code": 0, "stderr": "", "diff": ""}})
    assert v3["verdicts"]["n2_caller_evidence"] == "pass", v3["verdicts"]
    v4 = post("/agent-r/validate", {"operation": "build", "output": {"ok": 1},
        "evidence": {"exit_code": 2, "stderr": "err", "diff": ""}})
    assert v4["verdicts"]["n2_caller_evidence"] == "fail" and v4["overall"] == "fail"
    v5 = post("/agent-r/validate", {"operation": "build", "output": {"ok": 1}})
    assert v5["verdicts"]["n2_caller_evidence"] == "unverifiable"
    assert v5["overall"] == "pass", "N2 unverifiable no hunde overall si N0/N1 pasan"
    print("validate N2 pass/fail/unverifiable OK")

    # N3 stub:
    assert v5["verdicts"]["n3_receipt"] == "unverifiable"
    print("validate N3 stub OK")
except AssertionError as e:
    fails.append(str(e))
finally:
    proc.kill()
    proc.wait()

if fails:
    print("FALLOS:", fails)
    sys.exit(1)
print("SMOKE OK: preflight v2 + validate N0-N3")
