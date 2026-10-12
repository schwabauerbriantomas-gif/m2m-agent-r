#!/usr/bin/env bash
# full_baseline.sh — Corrida completa del baseline: 3 repeticiones x N in {10k,50k,100k}.
# Produce reports/baseline_rambench.json (números crudos por corrida + medianas).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
SEED=${SEED:-20261012}
PORT=${PORT:-19003}
BIN=${BIN:-$HOME/swarm_agent_r/target/release/m2m-vector-search}
OUT="$ROOT/reports/rambench_raw"
mkdir -p "$OUT" "$ROOT/reports"

for N in 10000 50000 100000; do
  for R in 1 2 3; do
    echo "=== N=$N rep=$R $(date +%H:%M:%S) ===" >&2
    bash scripts/rambench/baseline_run.sh "$N" "$SEED" "$PORT" "$BIN" "$OUT" "r${R}" >/dev/null 2>&1 || {
      echo "FALLO corrida N=$N rep=$R — ver $OUT" >&2; exit 1; }
  done
done

python3 scripts/rambench/aggregate.py --raw-dir "$OUT" --seed "$SEED" \
  --out "$ROOT/reports/baseline_rambench.json"
echo "OK -> reports/baseline_rambench.json" >&2
