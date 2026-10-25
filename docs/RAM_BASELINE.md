# RAM BASELINE — Agent-R v1 (baseline RAM-only)

**Fecha:** 2026-10-12 · **Agente:** R03 (oleada 0, track rambench) · **Branch:** `ar2/rambench`
**Propósito:** línea de base MEDIDA contra la cual las oleadas 3-5 (reducción de firma
de RAM estilo DBs embebidas) comparan sus números. Regla: número sin medición
reproducible = rechazado.

## TL;DR (medianas de 3 corridas, seed 20261012, release build)

| N generado | ΔRSS cargado (KB) | peak RSS (KB) | preflight p50 (ms) | p95 (ms) | restart hasta /health (s) | memoria tras restart |
|---|---|---|---|---|---|---|
| 10 000 | 8 112 | 16 176 | 2.62 | 3.10 | 0.62 | **0 errores (todo perdido)** |
| 50 000 | 9 640 | 17 268 | 2.84 | 3.35 | 0.60 | **0 errores (todo perdido)** |
| 100 000 | 10 440 | 18 260 | 3.01 | 3.51 | 0.57 | **0 errores (todo perdido)** |

- Ingesta sostenida: ~700–790 POST/s (concurrencia 10, asyncio), 200 OK en el 100% de los registros.
- **El cap `MAX_ERRORS=10000` NO estabiliza la RAM**: el Vec satura en 10k registros, pero el RSS sigue creciendo con N (fragmentación/churn del allocator por `errors.remove(0)` O(n) en cada inserción post-cap).
- Preflight empeora ~15% p50 de 10k→100k aunque el residente quede clavado en 10k.
- Anatomía: **~680 B teóricos y ~830 B observados por incidente residente**; 1M errores sin cap ≈ **0.65 GiB** solo el `Vec<ErrorRecord>`.

## Cómo reproducir (3 comandos)

```bash
cd ~/swarm_agent_r/wt_rambench
CARGO_TARGET_DIR=$HOME/swarm_agent_r/target cargo build --release --bin m2m-vector-search
bash scripts/rambench/full_baseline.sh          # 9 corridas (~10 min) -> docs/benchmarks/baseline_rambench.json
CARGO_TARGET_DIR=$HOME/swarm_agent_r/target cargo run --release --example rambench_anatomy
```

Corrida individual (una sola N/repetición):

```bash
bash scripts/rambench/baseline_run.sh <N> <SEED> <PORT> <BIN> <OUTDIR> <LABEL>
# ej: bash scripts/rambench/baseline_run.sh 10000 20261012 19003 \
#       $HOME/swarm_agent_r/target/release/m2m-vector-search /tmp/out r1
```

## Metodología

1. **Server limpio** por corrida: `m2m-vector-search serve --port 19003` bajo
   `/usr/bin/time -v` (peak RSS del proceso). VmRSS por `/proc/<pid>/status`.
2. **Generación determinista**: `scripts/rambench/gen_errors.py` — splitmix64
   sobre (índice, seed). Mismo seed ⇒ mismos payloads exactos (9 categorías ×
   3 severidades × 15 operaciones × pool de 20 tags, 1–4 tags/registro).
   POST concurrente con asyncio+semáforo(10). Longitudes medias del payload:
   operation 15.4 B, what 85.3 B, why 99.4 B, prevention 82.4 B, 2.51 tags.
3. **RSS estabilizado**: 2 lecturas VmRSS separadas 1 s tras la ingesta.
4. **Latencia preflight**: 100 POST secuenciales `/agent-r/preflight`
   (`operation=distill_lessons`, `tags=[gpu,ci]`), `curl -w %{time_total}`,
   p50/p95 por ordenamiento.
5. **Restart**: kill del server + medición de arranque en frío hasta `/health`
   OK, y `GET /agent-r/stats` post-restart (verifica pérdida total).
6. **3 repeticiones por N**, mediana reportada. JSON crudo por corrida en
   (raw runs not archived; aggregate: docs/benchmarks/baseline_rambench.json).

### Condiciones de la medición

- Hardware: AMD Ryzen 5 3400G (8 hilos), 13 GiB RAM, WSL2 kernel 6.18.35.
- rustc 1.97.1, `cargo build --release` (sin flags custom). Seed 20261012.
- El baseline NO toca `src/` (agente R03 = baseline de referencia); binario del
  branch `ar2/rambench` idéntico al auditado por R1.
- El servidor crea `SplatStore::new()` al arrancar (~8 MB de RSS vacío ya
  incluyen ese estado base, no solo Agent-R).

## Anatomía de memoria (examples/rambench_anatomy.rs)

```
ErrorRecord inline: 160 B (id u64, timestamp f64, 2 enums, 4 Strings de 24 B,
                           Vec<String> 24 B, recurrence u32, Option<u64> 16 B)
Heap por incidente : ~512 B (4 strings + tags, con bins de malloc glibc)
Slot del Vec       : 8 B
TOTAL              : ~680 B teóricos · ~830 B observados en RSS (N=10k)
Proyección sin cap : 100k → 66.8 MiB · 1M → 668.5 MiB (0.65 GiB)
```

Además de ese residente permanente, cada `recall_errors` materializa un
`Vec<(f64, &ErrorRecord)>` de hasta 16 B × errores evaluados (transitorio).

## Línea de base para las oleadas 3-5 (qué debe bajar)

Métrica de contrato: **ΔRSS cargado (mediana de 3)** a N=100k generado.
- Hoy: **10 440 KB** con solo 10k residentes (cap) — el objetivo "1M errores en
  RAM acotada" hoy ni siquiera se puede medir sin quitar el cap.
- Secundarias: preflight p50/p95 (no debe empeorar), restart con snapshot
  (debe restaurar memoria; hoy pierde el 100%), ingesta ≥700 POST/s.

Reglas para comparar: mismo seed, mismas longitudes de payload (o reportar las
nuevas), mismo método (`baseline_run.sh` sirve igual contra un binario nuevo:
solo cambia `<BIN>`), y **siempre** medianas de 3 corridas mínimas.

## Archivos

- `docs/benchmarks/baseline_rambench.json` — números crudos + medianas + anatomía + hallazgos.
- 9 corridas crudas (RSS, latencias por curl, gen stats, time -v) — raw runs not archived; aggregate: docs/benchmarks/baseline_rambench.json.
- `scripts/rambench/gen_errors.py` — generador determinista (asyncio, concurrencia 10).
- `scripts/rambench/baseline_run.sh` — una corrida (server limpio → N errores → RSS/latencia/restart).
- `scripts/rambench/full_baseline.sh` — 3×N completo + agregación.
- `scripts/rambench/aggregate.py` — medianas y JSON final.
- `examples/rambench_anatomy.rs` — size_of + proyección de escala.

## Extensión nocap (branch `ar2/rambench-nocap`) — la firma real sin el cap

Mismo harness, binario con el único cambio `MAX_ERRORS: 10_000 → 2_000_000` en
`src/agent_r.rs`. Datos en `docs/benchmarks/rambench_nocap.json`,
(raw runs not archived; aggregate: docs/benchmarks/rambench_nocap.json).

| N | ΔRSS (KB) | preflight p50 (ms) | p95 (ms) | B/registro |
|---|---|---|---|---|
| 10 000 | 8 000 | 2.53 | 2.91 | 819 (piso ~1 MB) |
| 50 000 | 34 564 | 10.02 | 11.51 | 707 |
| 100 000 | 67 388 | 19.50 | 21.77 | 690 |
| 301 175 (abort controlado) | 207 316 vivo / 206 644 peak | — | — | 705 |

- **RAM lineal ~700 B/registro** en los 4 puntos ⇒ 1M completo ≈ **0.67–0.70 GiB**.
- **La latencia también escala lineal**: p50 ×7.7 de 10k→100k (recall O(n) + clon del Vec de scores).
- **La ingesta es O(n²) total**: `record_error` escanea el Vec 2× por POST
  (recurrence en agent_r.rs:176, pattern_id en :181). RPS medidos: ~730 limpio
  → **13.2 rps @288k aceptados** ⇒ la corrida 1M completa tardaría ~15 h; se
  dejó datapoint con aborto en 301k (timeline `rss_1m_abort_timeline.txt`, stats+RSS cada 30 s).
- Contraste con el baseline capado: con cap, el preflight casi no se degrada
  (2.6→3.0 ms) pero la RAM crece igual por churn; sin cap, la RAM es honesta
  (~700 B/err) y la latencia explota. **Ninguna configuración escala a 1M**:
  esa es la justificación cuantitativa de las oleadas 3-5 (SQLite + top-N residente).
- Reproducir: `bash scripts/rambench/nocap_1m_abort.sh` (env `PORT_1M`, `ABORT_AT`, `BIN`, `OUT_DIR`).

## Hallazgos del baseline (para oleadas siguientes)

1. **La eviction actual no controla memoria** (R1 la marcó O(n)); su churn
   post-cap *añade* ~2.3 MB de RSS al pasar de 10k a 100k generados.
2. **Sin persistencia cableada**: `serialize_errors`/`load_errors` existen pero
   nadie los llama — restart = amnesia total (verificado en 9/9 corridas).
3. **El preflight arrastra el bug de substring**: en pleno bench, un preflight
   de `distill_lessons` devuelve warnings de `route_query`/`merge_splats`.
4. **El Mutex global** no limita la ingesta a este volumen (~730 POST/s);
   sí hace que preflight compita con record-error por el mismo lock.
