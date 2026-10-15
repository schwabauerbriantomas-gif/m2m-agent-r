#!/usr/bin/env bash
# bench_persist.sh — bench persistente P2-validate (puerto 19105, JAMÁS 19004/19103).
# Mide la ingesta 100k con --state PATH (write-through antes / debounce después).
# Uso: bench_persist.sh <BIN> <STATE_PATH> <OUTDIR> <LABEL> [N]
set -euo pipefail
N=${5:-100000}
SEED=20261012
PORT=19105
BIN=$1; STATE=$2; OUT=$3; LABEL=$4
BASE_URL="http://127.0.0.1:${PORT}"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"

port_busy() { ss -tln 2>/dev/null | grep -q ":${PORT} "; }
free_port() {
  local i=0
  while port_busy && [ $i -lt 50 ]; do
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    sleep 0.2; i=$((i+1))
  done
  if port_busy; then echo "FATAL: puerto $PORT ocupado" >&2; exit 1; fi
}

free_port
rm -f "$STATE" "$STATE.tmp"
AGENT_R_STATE_PATH="$STATE" "$BIN" serve --port "$PORT" &
SERVER_PID=$!
for i in $(seq 1 300); do
  curl -sf --max-time 1 "$BASE_URL/health" >/dev/null 2>&1 && break
  sleep 0.1
done
curl -sf --max-time 2 "$BASE_URL/health" >/dev/null || { echo "FATAL: server no responde /health" >&2; kill "$SERVER_PID" 2>/dev/null || true; exit 1; }

# Ingesta N (determinista, concurrencia 10) — mide rps del pipeline persistente
python3 "$SCRIPTS/gen_errors.py" --url "$BASE_URL" --n "$N" --seed "$SEED" \
  --concurrency 10 > "$OUT/persist_${LABEL}_n${N}.json"

sleep 2.5   # margen flusher (después) / irrelevante (antes)
STATS=$(curl -sf --max-time 10 "$BASE_URL/agent-r/stats" || echo '{}')
SNAP_BYTES=$(stat -c %s "$STATE" 2>/dev/null || echo 0)

kill "$SERVER_PID" 2>/dev/null || true
sleep 0.5
kill -9 "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
free_port

python3 - "$OUT" "$LABEL" "$N" "$SEED" "$STATS" "$SNAP_BYTES" "$STATE" <<'PYEOF'
import json, sys
outdir, label, n, seed, stats, snap_bytes, state = sys.argv[1:8]
gen = json.load(open(f"{outdir}/persist_{label}_n{n}.json"))
rec = {
  "label": label, "n": int(n), "seed": int(seed),
  "elapsed_s": gen.get("elapsed_s"), "throughput_rps": gen.get("throughput_rps"),
  "codes": gen.get("codes"),
  "snapshot_bytes_on_disk": int(snap_bytes),
  "stats_after_ingest": json.loads(stats),
  "state_path": state,
}
path = f"{outdir}/persist_result_{label}_n{n}.json"
open(path, "w").write(json.dumps(rec, indent=2))
print(json.dumps({k: rec[k] for k in ("label","n","throughput_rps","elapsed_s","snapshot_bytes_on_disk")}))
PYEOF
