#!/usr/bin/env bash
# full_baseline.sh — Matriz de corridas: NS x 3 repeticiones + agregación.
#
# Uso: full_baseline.sh [NS_OUT] [LABEL_PREFIX]
#   NS_OUT           lista de N separada por comas (default "10000,50000,100000")
#   LABEL_PREFIX     prefijo del label de corrida (default "r" -> run_r1_n10000.json)
# Env vars: SEED (20261012), PORT (19003), BIN, OUT_DIR, AGG_OUT
#
# Ejemplo nocap:
#   BIN=.../m2m-vector-search NS="10000 50000 100000 1000000" \
#     LABEL_PREFIX=nc OUT_DIR=reports/rambench_raw_nocap AGG_OUT=reports/rambench_nocap.json \
#     bash scripts/rambench/full_baseline.sh "10000,50000,100000,1000000" nc
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
SEED=${SEED:-20261012}
PORT=${PORT:-19003}
BIN=${BIN:-$HOME/swarm_agent_r/target/release/m2m-vector-search}
OUT=${OUT_DIR:-$ROOT/reports/rambench_raw}
AGG=${AGG_OUT:-$ROOT/reports/baseline_rambench.json}
NS=${1:-"10000,50000,100000"}
PREFIX=${2:-r}
mkdir -p "$OUT" "$ROOT/reports"

IFS=',' read -ra NS_LIST <<< "$NS"
for N in "${NS_LIST[@]}"; do
  for R in 1 2 3; do
    echo "=== N=$N rep=$R $(date +%H:%M:%S) ===" >&2
    bash scripts/rambench/baseline_run.sh "$N" "$SEED" "$PORT" "$BIN" "$OUT" "${PREFIX}${R}" >/dev/null 2>&1 || {
      echo "FALLO corrida N=$N rep=$R — ver $OUT" >&2; exit 1; }
  done
done

python3 scripts/rambench/aggregate.py --raw-dir "$OUT" --seed "$SEED" --out "$AGG"
echo "OK -> $AGG" >&2
