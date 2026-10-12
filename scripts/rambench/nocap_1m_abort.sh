#!/usr/bin/env bash
# nocap_1m_abort.sh — Datapoint de escala 1M con aborto controlado.
# Levanta server en 19004, dispara la generación de 1M, muestrea stats+RSS
# cada 30s y aborta al superar 300k aceptados (record_error es O(n) por POST:
# la corrida completa seria O(n^2) ~15h). Deja timeline crudo + peak RSS real.
set -euo pipefail
cd "$(dirname "$0")/../.."
PORT=${PORT_1M:-19004}
OUT=${OUT_DIR:-reports/rambench_raw_nocap}
BIN=${BIN:-$HOME/swarm_agent_r/target/release/m2m-vector-search}
SEED=${SEED:-20261012}
LIMIT=${ABORT_AT:-300000}
mkdir -p "$OUT"

free_port() {
  local i=0
  while ss -tln 2>/dev/null | grep -q ":${PORT} " && [ $i -lt 50 ]; do
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true; sleep 0.2; i=$((i+1))
  done
}
free_port

/usr/bin/time -v -o "$OUT/time_1m_abort.txt" "$BIN" serve --port "$PORT" &
WRAP=$!
SERVER_PID=""
for i in $(seq 1 100); do
  SERVER_PID=$(pgrep -P "$WRAP" -f 'serve --port' || true)
  [ -n "$SERVER_PID" ] && break; sleep 0.05
done
for i in $(seq 1 300); do curl -sf --max-time 1 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break; sleep 0.1; done

python3 scripts/rambench/gen_errors.py --url "http://127.0.0.1:${PORT}" --n 1000000 \
  --seed "$SEED" --concurrency 10 > "$OUT/gen_1m_inflight.json" 2>/dev/null &
GEN=$!

: > "$OUT/rss_1m_abort_timeline.txt"
while kill -0 "$GEN" 2>/dev/null; do
  ACC=$(curl -sf --max-time 5 "http://127.0.0.1:${PORT}/agent-r/stats" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['total_errors_recorded'])" 2>/dev/null || echo 0)
  RSS=$(awk '/VmRSS/{print $2}' "/proc/$SERVER_PID/status" 2>/dev/null || echo 0)
  echo "$(date +%H:%M:%S) aceptados=$ACC rss_kb=$RSS" >> "$OUT/rss_1m_abort_timeline.txt"
  if [ "${ACC:-0}" -ge "$LIMIT" ]; then
    echo "$(date +%H:%M:%S) ABORT: >= $LIMIT aceptados" >> "$OUT/rss_1m_abort_timeline.txt"
    kill "$GEN" 2>/dev/null || true
    break
  fi
  sleep 30
done
wait "$GEN" 2>/dev/null || true

# lecturas sincronizadas finales (antes de matar el server)
ACC=$(curl -sf --max-time 5 "http://127.0.0.1:${PORT}/agent-r/stats" | python3 -c "import json,sys; print(json.load(sys.stdin)['total_errors_recorded'])")
RSS=$(awk '/VmRSS/{print $2}' "/proc/$SERVER_PID/status")
kill "$SERVER_PID" 2>/dev/null || true
for i in $(seq 1 100); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.1; done
kill -0 "$SERVER_PID" 2>/dev/null && kill -9 "$SERVER_PID" 2>/dev/null || true
wait "$WRAP" 2>/dev/null || true
PEAK=$(grep 'Maximum resident set size' "$OUT/time_1m_abort.txt" | awk '{print $NF}')

python3 - "$ACC" "$RSS" "$PEAK" "$OUT" "$LIMIT" <<'PYEOF'
import json, sys
acc, rss, peak, outdir, limit = sys.argv[1:6]
rec = {
  "n_target": 1000000,
  "abort_en_aceptados": int(acc),
  "limite_abort": int(limit),
  "rss_vivo_abort_kb": int(rss),
  "peak_rss_kb_time_v": int(peak),
  "bytes_por_incidente_a_escala": round(int(rss)*1024/int(acc), 1) if int(acc) else None,
  "motivo": "record_error hace 2 scans O(n) por POST (recurrence+pattern_id): rps colapsa ~O(n^2) total; rps@live medido 13.2/s a ~288k aceptados (corrida previa). Corrida completa ~15h => aborto controlado.",
  "timeline": "rss_1m_abort_timeline.txt (stats+RSS cada 30s)",
}
open(f"{outdir}/run_1m_abort.json", "w").write(json.dumps(rec, indent=2))
print(json.dumps(rec, indent=2))
PYEOF
