#!/usr/bin/env python3
"""Agrega las corridas crudas de baseline_run.sh en baseline_rambench.json.

Toma run_r{1,2,3}_n{N}.json por cada N, calcula mediana de cada métrica y
embebe las corridas crudas. Uso:
  aggregate.py --raw-dir reports/rambench_raw --seed 20261012 --out reports/baseline_rambench.json
"""
import argparse
import glob
import json
import os
import re
import statistics
from datetime import datetime, timezone

METRICS = [
    "rss_empty_kb", "rss_loaded_kb", "peak_rss_kb_time_v",
    "rss_delta_loaded_minus_empty_kb", "preflight_p50_ms", "preflight_p95_ms",
    "restart_to_health_s",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    by_n = {}
    gens = {}
    for path in sorted(glob.glob(os.path.join(args.raw_dir, "run_*_n*.json"))):
        rec = json.load(open(path))
        by_n.setdefault(rec["n"], []).append(rec)
    for path in sorted(glob.glob(os.path.join(args.raw_dir, "gen_*_n*.json"))):
        g = json.load(open(path))
        gens.setdefault(g["n"], []).append(g)

    out = {
        "meta": {
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "harness": "scripts/rambench/{gen_errors.py,baseline_run.sh,full_baseline.sh}",
            "seed": args.seed,
            "generador": "splitmix64 determinista (ver gen_errors.py); mismos payloads para cualquier N",
            "binario": "target/release/m2m-vector-search (commit del branch ar2/rambench, baseline sin tocar src/)",
            "build": "cargo build --release, rustc default (sin LTO custom)",
            "medicion_rss": "/usr/bin/time -v (Maximum resident set size) + /proc/<pid>/status VmRSS",
            "medicion_latencia": "100 POST /agent-r/preflight secuenciales, curl -w %{time_total}, op=distill_lessons tags=[gpu,ci]",
            "concurrencia_generacion": "10 (asyncio semaphore)",
            "repeticiones": 3,
            "nota_cap": "MAX_ERRORS=10000 en el baseline: con N=50k/100k el Vec satura en 10k registros (eviction remove(0)); total_errors_recorded sigue contando N",
        },
        "por_N": {},
        "corridas_crudas": by_n,
    }
    for n, recs in sorted(by_n.items()):
        agg = {"repeticiones": len(recs)}
        for m in METRICS:
            vals = [r[m] for r in recs]
            agg[m] = {
                "vals": vals,
                "mediana": statistics.median(vals),
            }
        # bytes/incidente observados (delta RSS / registros RESIDENTES, no N).
        # residentes = min(stats total_errors_recorded, MAX_ERRORS efectivo del binario);
        # RAMBENCH_MAX_ERRORS la pasa el wrapper cuando el binario relevo el cap.
        max_eff = int(os.environ.get("RAMBENCH_MAX_ERRORS", "10000"))
        residentes = min(int(statistics.median(
            [r["stats_pre_restart"]["total_errors_recorded"] for r in recs])), max_eff)
        agg["bytes_por_incidente_residente"] = round(
            statistics.median([r["rss_delta_loaded_minus_empty_kb"] for r in recs]) * 1024 / residentes, 1
        )
        agg["residentes_en_vec"] = residentes
        agg["total_aceptados_mediana"] = statistics.median(
            [r["stats_pre_restart"]["total_errors_recorded"] for r in recs]
        )
        if n in gens:
            agg["gen_throughput_rps"] = [g["throughput_rps"] for g in gens[n]]
            agg["gen_elapsed_s"] = [g["elapsed_s"] for g in gens[n]]
            agg["gen_codes_ok"] = all(g["codes"].get("200") == g["n"] for g in gens[n])
        # sanity: post-restart SIEMPRE vacío en baseline (sin persistencia)
        agg["memoria_perdida_en_restart"] = all(
            r["stats_post_restart"]["total_errors_recorded"] == 0 for r in recs
        )
        out["por_N"][str(n)] = agg

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: {m: v["mediana"] for m, v in agg.items() if isinstance(v, dict)}
                      for k, agg in out["por_N"].items()}, indent=2))


if __name__ == "__main__":
    main()
