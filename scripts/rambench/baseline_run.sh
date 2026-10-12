#!/usr/bin/env bash
# baseline_run.sh — Una corrida completa (N errores) del baseline de RAM de Agent-R.
# Levanta server limpio con /usr/bin/time -v, genera N errores con seed fija,
# mide peak RSS, latencia p50/p95 de 100 preflights, y costo de restart.
#
# Uso: baseline_run.sh <N> <SEED> <PORT> <BIN> <OUTDIR> [ITER_LABEL]
set -euo pipefail

N=$1; SEED=$2; PORT=$3; BIN=$4; OUTDIR=$5; LABEL=${6:-run}
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
BASE_URL="http://127.0.0.1:${PORT}"
TIME_LOG="$OUTDIR/time_${LABEL}_n${N}.txt"
mkdir -p "$OUTDIR"

port_busy() { ss -tln 2>/dev/null | grep -q ":${PORT} "; }

# Asegura puerto libre antes de levantar (mata stray servers del puerto)
free_port() {
  local i=0
  while port_busy && [ $i -lt 50 ]; do
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    sleep 0.2; i=$((i+1))
  done
  if port_busy; then echo "FATAL: puerto $PORT ocupado y no liberable" >&2; exit 1; fi
}

# start_server <time_log> => deja WRAP (pid del /usr/bin/time) y SERVER_PID (child)
start_server() {
  free_port
  /usr/bin/time -v -o "$1" "$BIN" serve --port "$PORT" &
  WRAP=$!
  SERVER_PID=""
  for i in $(seq 1 100); do
    SERVER_PID=$(pgrep -P "$WRAP" -f 'serve --port' || true)
    [ -n "$SERVER_PID" ] && break
    sleep 0.05
  done
  [ -n "$SERVER_PID" ] || { echo "FATAL: server child no aparece" >&2; kill "$WRAP" 2>/dev/null || true; exit 1; }
  for i in $(seq 1 300); do
    curl -sf --max-time 1 "$BASE_URL/health" >/dev/null 2>&1 && break
    sleep 0.1
  done
  curl -sf --max-time 2 "$BASE_URL/health" >/dev/null || { echo "FATAL: server no responde /health" >&2; exit 1; }
}

# stop_server: mata al CHILD (asi time -v escribe su log) y espera al wrapper
stop_server() {
  kill ${SERVER_PID} 2>/dev/null || true
  local i=0
  while kill -0 ${SERVER_PID} 2>/dev/null && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done
  kill -0 ${SERVER_PID} 2>/dev/null && kill -9 ${SERVER_PID} 2>/dev/null || true
  wait "$WRAP" 2>/dev/null || true
  free_port
}

rss_kb() { awk '/VmRSS/{print $2}' "/proc/$1/status"; }

# ══ 1) Server limpio bajo /usr/bin/time -v ══
start_server "$TIME_LOG"
RSS_EMPTY=$(rss_kb "$SERVER_PID")

# ══ 2) Generación de N errores (determinista, concurrente x10) ══
python3 "$SCRIPTS/gen_errors.py" --url "$BASE_URL" --n "$N" --seed "$SEED" \
  --concurrency 10 > "$OUTDIR/gen_${LABEL}_n${N}.json"

# ══ 3) Estabilización + RSS cargado (dos lecturas separadas 1s) ══
sleep 1
RSS_LOADED=$(rss_kb "$SERVER_PID")
sleep 1
RSS_LOADED2=$(rss_kb "$SERVER_PID")

# ══ 4) Latencia de 100 preflights secuenciales (curl -w) ══
LAT_RAW="$OUTDIR/preflight_latencies_${LABEL}_n${N}.csv"
: > "$LAT_RAW"
for i in $(seq 1 100); do
  curl -s --max-time 10 -o /dev/null -w '%{time_total}\n' -X POST "$BASE_URL/agent-r/preflight" \
    -H 'Content-Type: application/json' \
    -d '{"operation":"distill_lessons","context_tags":["gpu","ci"]}' >> "$LAT_RAW"
done
P50=$(sort -g "$LAT_RAW" | awk '{a[NR]=$1} END{if(NR%2==1) printf "%.3f", a[(NR+1)/2]*1000; else printf "%.3f", (a[NR/2]+a[NR/2+1])/2*1000}')
P95=$(sort -g "$LAT_RAW" | awk '{a[NR]=$1} END{printf "%.3f", a[int(NR*0.95+0.9999)]*1000}')

# ══ 5) Stats pre-restart (verifica N aceptado o cap 10k) ══
STATS_PRE=$(curl -sf --max-time 5 "$BASE_URL/agent-r/stats" || echo '{}')

# ══ 6) Restart: baseline sin snapshot => arranque en frío; medir hasta /health ══
stop_server
T0=$(date +%s.%N)
start_server "$OUTDIR/time_restart_${LABEL}_n${N}.txt"
T1=$(date +%s.%N)
RESTART_S=$(echo "$T1 $T0" | awk '{printf "%.3f", $1-$2}')
STATS_POST=$(curl -sf --max-time 5 "$BASE_URL/agent-r/stats" || echo '{}')
stop_server

# ══ 7) Peak RSS de time -v ══
PEAK_KB=$(grep 'Maximum resident set size' "$TIME_LOG" | awk '{print $NF}')
[ -n "$PEAK_KB" ] || { echo "FATAL: time -v sin peak RSS" >&2; cat "$TIME_LOG" >&2; exit 1; }

# ══ 8) JSON de la corrida ══
python3 - "$OUTDIR" "$N" "$SEED" "$LABEL" "$RSS_EMPTY" "$RSS_LOADED" "$RSS_LOADED2" \
  "$PEAK_KB" "$P50" "$P95" "$RESTART_S" "$STATS_PRE" "$STATS_POST" <<'PYEOF'
import json, sys
outdir, n, seed, label = sys.argv[1:5]
rss_e, rss_l, rss_l2, peak, p50, p95, restart = sys.argv[5:12]
stats_pre, stats_post = sys.argv[12], sys.argv[13]
rec = {
  "n": int(n), "seed": int(seed), "label": label,
  "rss_empty_kb": int(rss_e), "rss_loaded_kb": int(rss_l), "rss_loaded2_kb": int(rss_l2),
  "peak_rss_kb_time_v": int(peak),
  "rss_delta_loaded_minus_empty_kb": int(rss_l) - int(rss_e),
  "preflight_p50_ms": float(p50), "preflight_p95_ms": float(p95),
  "restart_to_health_s": float(restart),
  "stats_pre_restart": json.loads(stats_pre),
  "stats_post_restart": json.loads(stats_post),
}
path = f"{outdir}/run_{label}_n{n}.json"
open(path, "w").write(json.dumps(rec, indent=2))
print(json.dumps(rec, indent=2))
PYEOF
