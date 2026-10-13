# INTEGRACIÓN OPT — O1 + O2 + O3 + O4 + SWARM sobre Agent-R (`opt/integrate`)

Base: `511ea9d` (rambench-nocap). Fusiona `opt/opt_ingesta` (O1), `opt/opt_recall` (O2),
`opt/opt_alloc` (O3) y el port de `ar2/persist` (O4); agrega la capa de
coordinación multi-agente (`agent_swarm.rs`).
Binario: `~/swarm_agent_r/bin_integrate` (target-integrate, CARGO_TARGET_DIR propio).
Todos los números de este informe son corridas reales en esta máquina
(seed 20261012, 127.0.0.1).

## Qué integra cada pieza

| Pieza | Cambio | Efecto medido |
|-----|--------|--------|
| O1 (ingesta) | Índice `HashMap<(ErrorCategory, InternedId), VecDeque<u64>>`: recurrence = len(bucket) pre-insert, pattern_id = back(bucket). Elimina los 2 scans O(n) por POST | Ingesta O(n²)→O(n) total |
| O3 (layout) | `operation`/`context_tags` internadas a `InternedId=u32` (pool dedup), textos como `Box<str>` | ~693 → ~510 B/reg (−26%) |
| O2 (recall) | top-20 con min-heap acotado en un pase (sin sort O(n log n)); paridad testeada | Preflight ~2.4× más rápido @100k |
| O4 (persist, port `beff9c0` anterior: `8cb3179`) | Dedup ADD/UPDATE/NOOP por fingerprint FNV-1a u64 (category, operation, what normalizados) + snapshot write-through atómico (tmp+rename) + seeds + `deprecate_error` | Memoria persistente; sin filas duplicadas |
| SWARM (`beff9c0`) | Cola de tareas con claim CAS, dependencias, barrera/contador, fallback por epoch, roles (Coordinator/Proposer/Validator/Executor) y consenso con quórum | Multi-agente colaborativo sobre la misma memoria |

**Sinergias**: clave de O1 usa el `InternedId` de O3 (ingesta O(1) sin allocs);
O4 dedup en u64 (8 B) en vez de String (~124 B); complete/fail del swarm
escriben lecciones en el Agent-R común (dedup O4) y `preflight_shared`
consulta la memoria de TODO el swarm.

## Tests

- **276/276** de la lib completa (`cargo test --lib`), de los cuales:
  - agent_r: 32 (baseline + O1 paridad + O2 paridad + O3 interning/JSON + 8 de O4:
    roundtrip, dedup, seeds, no-reseed, deprecate, corrupto→seeds, legacy, atómico)
  - agent_swarm: 16 (CAS 1-ganador, dependencias, fallback determinista, barrera,
    TTL/death, contador de fase, roles-gate, consenso accept/reject/empate/stalled,
    quórum parcial, status)
- E2E HTTP verificados sobre server release:
  - **O4**: seeds cargan (ids 1-3) → POST Add id=4 → duplicado Noop rec=1 →
    restart: memoria sobrevive (total=5, únicos=4).
  - **Swarm**: dependencias bloquean → CAS `claimed`/`taken` → fail→re-claim →
    lección del fallo visible en preflight de otro agente.
  - **Consenso**: propuesta REJECT con razones → contra-propuesta ACCEPTED 2/2 →
    materializada en tarea → executor la completa (el executor no puede proponer) →
    lección en memoria compartida.

## Números medidos

### Matriz 100k nocap ×3 (`reports/int_rambench.json`; integrado pre-O4)
| métrica | integrado | base 511ea9d |
|---|---|---|
| rps ingesta (mediana) | **755.9** | 571.3 |
| RAM Δ (mediana) | **48.7 MB** | 67.4 MB |
| B/registro | **510.4** | 687.5 |
| preflight p50 / p95 | **16.9 / 18.7 ms** | 19.5 / 21.8 ms |
| integridad | 200000/200000 OK ×3 | idem |

### Sanity O4 post-port ×3 @100k (`reports/o4_sanity.json`)
Con el harness ya en modo efímero (server sin state path):
- rps ingesta mediana **727.8** [736.8, 683.6, 754.8] — en línea con el 755.9
  del integrado pre-O4 (la dedup O4 no degrada la ingesta cuando los payloads
  no duplican).
- RAM **55.7 MB Δ = 570 B/incidente** residente (100k únicos; +12% vs 510 por los
  4 campos nuevos de O4 + índice de fingerprints).
- preflight p50 **18.3 ms** / p95 19.5 (sin colas).
- restart frío 0.735 s (efímero: sin carga de snapshot).
- 100000/100000 aceptados, distribución por categoría uniforme (9×~11k).

### Escala 300k (pre-O4, `reports/int_raw_nocap/`)
309,096 aceptados, 529.4 B/reg, ~5.5 min (base: 301k en ~17.3 min, rps 770→13.2).

### **1M COMPLETO** (pre-O4, `reports/int_raw_1m/`)
- 1,000,000/1,000,000 aceptados, ~20 min (proyección base: ~15 h ⇒ ~45× menos).
- rps plano: mediana 827 primeras 5 ventanas vs 829 últimas (755-863).
- Peak RSS 478 MiB = **501.9 B/reg** (base proyectada ~680 MB ⇒ −26%).
- Re-validación a 1M CON O4+swarm: pendiente (los deltas esperados son mínimos:
  O4 agrega 1 lookup HashMap + hash FNV por POST; swarm no toca el hot path de
  ingesta).

### Persistencia O4: costo del write-through (`/tmp` datapoint 13-oct, borrado)
POSTs secuenciales con `AGENT_R_STATE_PATH` activo (sin dedup: incidentes únicos):
| estado | rps | snapshot |
|---|---|---|
| 1k | 120.1 | 0.4 MB |
| 6k | 62.6 | 2.3 MB |
| 11k | 31.3 | 4.2 MB |
| 16k | 19.8 | 6.1 MB |

Firma **O(n) por POST** confirmada (duplicar estado ⇒ mitad de rps): el
write-through serializa el estado completo en cada escritura. Es el diseño del
O4 original, mantenido a propósito. Consecuencias prácticas:
- **Benchmarks en modo efímero** (`AGENT_R_STATE_PATH=` en `baseline_run.sh`):
  miden el pipeline en RAM; el write-through contaminaría todo (corrida inicial
  abortada: 32 min sin terminar 100k, 15.6 MB por write).
- **Producción**: use con estados chicos o agregue batch/debounce de snapshots
  (trabajo futuro; no bloquea nada actual).

## Veredicto

El integrado convierte Agent-R de "ingesta O(n²), 15 h para 1M" a **1M errores
en 20 minutos con RAM −26%**, con memoria persistente y dedup (O4) y una capa
multi-agente colaborativa (roles + consenso + cola CAS) sobre la misma base.
Siguientes cuellos conocidos (no bloquean 1M): `errors.remove(0)` O(n) si se
supera MAX_ERRORS=2M; scoring O(n) residual por preflight (~2.4× más barato pero
crece con n); write-through O(n) por POST si se usa persistencia con estados
grandes.
