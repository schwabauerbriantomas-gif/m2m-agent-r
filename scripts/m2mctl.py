#!/usr/bin/env python3
"""m2mctl — control total de los dos núcleos M2M (Rust prod + Python lab).

Uso:
  m2mctl.py status [--json]
  m2mctl.py build rust|py [--deploy] [--json]
  m2mctl.py test  rust|py|all [--filter PAT] [--json]
  m2mctl.py serve rust|py --port N [--ephemeral|--state PATH] [--json]
  m2mctl.py stop  rust|py|all [--json]
  m2mctl.py bench rust 100k|300k|1m [--abort-at N] [--json]
  m2mctl.py call rust <endpoint> [json_body] [--method GET|POST] [--json]

Toda salida soporta --json (formato estable, pensado para exposición MCP:
cada subcomando mapea 1:1 a una tool). PIDs y logs en ~/.m2mctl/.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
import time
import urllib.request
import urllib.error

HOME = os.path.expanduser("~")
# Repo override por env; default: padre de scripts/ (repo clonado).
REPO = os.environ.get("AGENT_R_REPO", str(Path(__file__).resolve().parent.parent))
RUST_DIR = REPO
RUST_TARGET = os.environ.get("AGENT_R_TARGET_DIR", os.path.join(REPO, "target"))
RUST_BIN = os.environ.get("AGENT_R_BIN", os.path.join(REPO, "target", "release", "agentr"))
RUST_DEFAULT_PORT = 19103
PY_DIR = os.environ.get("AGENT_R_PY_DIR", os.path.join(HOME, "swarm_m2m_unify/work/m2m-vector-search"))
PY_VENV = f"{HOME}/venvs/m2m-py"
PY_DEFAULT_PORT = 8080
CTL_DIR = f"{HOME}/.m2mctl"

RUST_HEALTH = "/agent-r/stats"   # endpoint vivo del núcleo Rust
PY_HEALTH = "/health"


def out(data: dict, as_json: bool, pretty_key: str = None):
    if as_json:
        print(json.dumps(data, indent=1, ensure_ascii=False))
    else:
        if pretty_key and pretty_key in data:
            print(json.dumps(data[pretty_key], indent=1, ensure_ascii=False))
        else:
            print(json.dumps(data, indent=1, ensure_ascii=False))
    return 0 if data.get("ok", True) else 1


def pidfile(name: str) -> str:
    os.makedirs(CTL_DIR, exist_ok=True)
    return f"{CTL_DIR}/{name}.pid"


def read_pid(name: str):
    try:
        with open(pidfile(name)) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def port_pid(port: int):
    """PID escuchando un puerto (vía ss -tlnp, mismo usuario)."""
    try:
        r = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if f":{port} " in line and "pid=" in line:
                for tok in line.split():
                    if tok.startswith("pid="):
                        return int(tok[4:].split(",")[0])
    except Exception:
        pass
    return None


def http(port: int, path: str, body=None, method=None, timeout=10):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ── status ───────────────────────────────────────────────────────────────

def git_state(repo: str) -> dict:
    if not os.path.isdir(repo):
        return {"exists": False}
    def g(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                              text=True).stdout.strip()
    return {
        "exists": True,
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": g("rev-parse", "--short", "HEAD"),
        "dirty": bool(g("status", "--porcelain")),
    }


def cmd_status(args):
    rust = {"repo": RUST_DIR, **git_state(RUST_DIR)}
    rust["bin_exists"] = os.path.isfile(RUST_BIN)
    rust["bin_fresh"] = False
    if rust["bin_exists"]:
        cargo = f"{RUST_DIR}/Cargo.toml"
        src_newest = max(
            (os.path.getmtime(os.path.join(dp, f))
             for dp, _, fs in os.walk(f"{RUST_DIR}/src") for f in fs),
            default=0)
        rust["bin_fresh"] = os.path.getmtime(RUST_BIN) > src_newest
    rp = read_pid("rust")
    rust["server_pid"] = rp if pid_alive(rp) else (port_pid(RUST_DEFAULT_PORT))
    rust["server_alive"] = pid_alive(rust["server_pid"])
    if rust["server_alive"]:
        try:
            rust["stats"] = http(RUST_DEFAULT_PORT, RUST_HEALTH, timeout=3)
        except Exception as e:
            rust["stats_error"] = str(e)

    py = {"repo": PY_DIR, **git_state(PY_DIR)}
    py["venv_exists"] = os.path.isdir(f"{PY_VENV}")
    if py["venv_exists"]:
        py["venv_python"] = f"{PY_VENV}/bin/python"
    pp = read_pid("py")
    py["server_pid"] = pp if pid_alive(pp) else (port_pid(PY_DEFAULT_PORT))
    py["server_alive"] = pid_alive(py["server_pid"])
    if py["server_alive"]:
        try:
            py["health"] = http(PY_DEFAULT_PORT, PY_HEALTH, timeout=3)
        except Exception as e:
            py["health_error"] = str(e)

    return out({"ok": True, "rust": rust, "python": py}, args.json)


# ── build ────────────────────────────────────────────────────────────────

def cmd_build(args):
    t0 = time.time()
    if args.target == "rust":
        env = dict(os.environ, CARGO_TARGET_DIR=RUST_TARGET)
        r = subprocess.run(["cargo", "build", "--release"], cwd=RUST_DIR,
                           env=env, capture_output=True, text=True)
        ok = r.returncode == 0
        res = {"ok": ok, "target": "rust", "elapsed_s": round(time.time() - t0, 1)}
        if ok and args.deploy:
            shutil.copy2(f"{RUST_TARGET}/release/agentr", RUST_BIN)
            res["deployed_to"] = RUST_BIN
        if not ok:
            res["stderr_tail"] = r.stderr[-2000:]
        return out(res, args.json)
    if args.target == "py":
        if not os.path.isdir(PY_VENV):
            r1 = subprocess.run([sys.executable, "-m", "venv", PY_VENV],
                                capture_output=True, text=True)
            if r1.returncode != 0:
                return out({"ok": False, "target": "py",
                            "error": f"venv fallo: {r1.stderr[-500:]}"}, args.json)
        pip = f"{PY_VENV}/bin/pip"
        r = subprocess.run([pip, "install", "-e", f"{PY_DIR}[all]"],
                           capture_output=True, text=True)
        ok = r.returncode == 0
        res = {"ok": ok, "target": "py", "venv": PY_VENV,
               "elapsed_s": round(time.time() - t0, 1)}
        if not ok:
            res["stderr_tail"] = r.stderr[-2000:]
        return out(res, args.json)
    return out({"ok": False, "error": "target invalido"}, args.json)


# ── test ─────────────────────────────────────────────────────────────────

def cmd_test(args):
    results = {}
    if args.target in ("rust", "all"):
        cmd = ["cargo", "test", "--lib"]
        if args.filter:
            cmd.append(args.filter)
        # TMPDIR=/tmp: el TMPDIR heredado (scratch de Hermes) borra tmpfiles
        # en vivo y rompe tests de persistencia en paralelo (flaky 274-276).
        env = dict(os.environ, CARGO_TARGET_DIR=RUST_TARGET, TMPDIR="/tmp")
        r = subprocess.run(cmd, cwd=RUST_DIR, env=env, capture_output=True, text=True)
        tail = r.stdout.strip().splitlines()
        line = next((l for l in tail if l.startswith("test result:")), None)
        results["rust"] = {"ok": r.returncode == 0, "result_line": line,
                           "cmd": " ".join(cmd)}
    if args.target in ("py", "all"):
        if not os.path.isdir(PY_VENV):
            results["python"] = {"ok": False, "error": f"sin venv; correr: m2mctl.py build py"}
        else:
            # TMPDIR=/tmp + capture=no: el TMPDIR heredado (scratch de Hermes)
            # borra tmpfiles en vivo => FileNotFoundError en capture de pytest
            # (0 tests colectados). Con capture=no corre normal.
            cmd = [f"{PY_VENV}/bin/python", "-m", "pytest", "tests/", "-q",
                   "--no-header", "-p", "no:cacheprovider", "--capture=no"]
            if args.filter:
                cmd += ["-k", args.filter]
            env = dict(os.environ, TMPDIR="/tmp")
            r = subprocess.run(cmd, cwd=PY_DIR, env=env, capture_output=True, text=True)
            tail = [l for l in r.stdout.strip().splitlines() if l][-3:]
            results["python"] = {"ok": r.returncode == 0, "tail": tail,
                                 "cmd": " ".join(cmd)}
    ok = all(v.get("ok") for v in results.values())
    return out({"ok": ok, **results}, args.json)


# ── serve / stop ─────────────────────────────────────────────────────────

def cmd_serve(args):
    log = f"{CTL_DIR}/{args.target}_serve.log"
    os.makedirs(CTL_DIR, exist_ok=True)
    if args.target == "rust":
        port = args.port or RUST_DEFAULT_PORT
        env = dict(os.environ)
        if args.ephemeral:
            env["AGENT_R_STATE_PATH"] = ""   # RAM pura (bench)
        elif args.state:
            env["AGENT_R_STATE_PATH"] = args.state
        # sin flags => default persistente del server (data/agent_r_state.json)
        cmd = [RUST_BIN, "serve", "--port", str(port)]
        with open(log, "ab") as lf:
            p = subprocess.Popen(cmd, cwd=RUST_DIR, env=env,
                                 stdout=lf, stderr=lf,
                                 start_new_session=True)
        with open(pidfile("rust"), "w") as f:
            f.write(str(p.pid))
        health_path = RUST_HEALTH
    elif args.target == "py":
        if not os.path.isdir(PY_VENV):
            return out({"ok": False, "error": "sin venv; correr: m2mctl.py build py"}, args.json)
        port = args.port or PY_DEFAULT_PORT
        env = dict(os.environ, M2M_HOST="127.0.0.1", M2M_PORT=str(port))
        cmd = [f"{PY_VENV}/bin/python", "-m", "m2m"]
        with open(log, "ab") as lf:
            p = subprocess.Popen(cmd, cwd=PY_DIR, env=env,
                                 stdout=lf, stderr=lf,
                                 start_new_session=True)
        with open(pidfile("py"), "w") as f:
            f.write(str(p.pid))
        health_path = PY_HEALTH
    else:
        return out({"ok": False, "error": "target invalido"}, args.json)

    for _ in range(50):
        time.sleep(0.2)
        try:
            http(port, health_path, timeout=2)
            return out({"ok": True, "target": args.target, "port": port,
                        "pid": p.pid, "log": log,
                        "mode": "ephemeral" if (args.target == "rust" and args.ephemeral)
                        else ("state" if (args.target == "rust" and args.state) else "default")},
                       args.json)
        except Exception:
            continue
    return out({"ok": False, "error": f"server no levanto; ver {log}"}, args.json)


def cmd_stop(args):
    stopped = {}
    targets = ["rust", "py"] if args.target == "all" else [args.target]
    for t in targets:
        pid = read_pid(t) or port_pid(RUST_DEFAULT_PORT if t == "rust" else PY_DEFAULT_PORT)
        if pid and pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(20):
                    time.sleep(0.1)
                    if not pid_alive(pid):
                        break
                else:
                    os.kill(pid, signal.SIGKILL)
                stopped[t] = {"stopped": pid}
            except OSError as e:
                stopped[t] = {"error": str(e)}
        else:
            stopped[t] = {"stopped": None}
        try:
            os.remove(pidfile(t))
        except OSError:
            pass
    return out({"ok": True, **stopped}, args.json)


# ── bench (harness rambench del núcleo Rust) ─────────────────────────────

def cmd_bench(args):
    script = f"{RUST_DIR}/scripts/rambench"
    label = f"ctl{int(time.time()) % 100000}"
    if args.scale == "100k":
        cmd = (f"BIN={RUST_BIN} PORT=19103 OUT_DIR=reports/ctl_{label} "
               f"AGG_OUT=reports/ctl_{label}.json RAMBENCH_MAX_ERRORS=2000000 "
               f"AGENT_R_STATE_PATH= bash {script}/full_baseline.sh 100000 {label}")
    elif args.scale == "300k":
        cmd = (f"BIN={RUST_BIN} PORT=19103 OUT_DIR=reports/ctl_nocap_{label} "
               f"RAMBENCH_MAX_ERRORS=2000000 AGENT_R_STATE_PATH= "
               f"ABORT_AT={args.abort_at or 300000} "
               f"bash {script}/nocap_1m_abort.sh")
    else:  # 1m
        cmd = (f"BIN={RUST_BIN} PORT=19004 OUT_DIR=reports/ctl_1m_{label} "
               f"RAMBENCH_MAX_ERRORS=2000000 AGENT_R_STATE_PATH= "
               f"ABORT_AT={args.abort_at or 2000000} "
               f"bash {script}/nocap_1m_abort.sh")
    r = subprocess.run(["bash", "-c", cmd], cwd=RUST_DIR, capture_output=True, text=True)
    return out({"ok": r.returncode == 0, "scale": args.scale,
                "label": label, "rc": r.returncode,
                "reports_dir": f"reports/*{label}*", "note": "sin server propio: el harness levanta el suyo"}, args.json)


# ── call (HTTP directo a los endpoints) ──────────────────────────────────

def cmd_call(args):
    port = args.port or RUST_DEFAULT_PORT
    body = json.loads(args.body) if args.body else None
    try:
        r = http(port, args.endpoint if args.endpoint.startswith("/") else "/" + args.endpoint,
                 body=body, method=args.method if args.method != "AUTO" else None)
        return out({"ok": True, "response": r}, args.json)
    except urllib.error.HTTPError as e:
        return out({"ok": False, "http_error": e.code,
                    "body": e.read().decode()[:500]}, args.json)
    except Exception as e:
        return out({"ok": False, "error": str(e)}, args.json)


def main():
    ap = argparse.ArgumentParser(prog="m2mctl")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_json(p):
        p.add_argument("--json", action="store_true", help="salida JSON estable")
        return p

    s = add_json(sub.add_parser("status")); s.set_defaults(fn=cmd_status)

    s = add_json(sub.add_parser("build")); s.add_argument("target", choices=["rust", "py"])
    s.add_argument("--deploy", action="store_true", help="cp binario a bin_integrate")
    s.set_defaults(fn=cmd_build)

    s = add_json(sub.add_parser("test")); s.add_argument("target", choices=["rust", "py", "all"])
    s.add_argument("--filter", default=None)
    s.set_defaults(fn=cmd_test)

    s = add_json(sub.add_parser("serve")); s.add_argument("target", choices=["rust", "py"])
    s.add_argument("--port", type=int, default=None)
    g = s.add_mutually_exclusive_group()
    g.add_argument("--ephemeral", action="store_true", help="Rust: RAM pura (bench)")
    g.add_argument("--state", default=None, help="Rust: path del snapshot O4")
    s.set_defaults(fn=cmd_serve)

    s = add_json(sub.add_parser("stop")); s.add_argument("target", choices=["rust", "py", "all"])
    s.set_defaults(fn=cmd_stop)

    s = add_json(sub.add_parser("bench")); s.add_argument("target", choices=["rust"])
    s.add_argument("scale", choices=["100k", "300k", "1m"])
    s.add_argument("--abort-at", type=int, default=None)
    s.set_defaults(fn=cmd_bench)

    s = add_json(sub.add_parser("call")); s.add_argument("target", choices=["rust", "py"])
    s.add_argument("endpoint"); s.add_argument("body", nargs="?", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--method", default="AUTO")
    s.set_defaults(fn=cmd_call)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
