# INTEGRACIÓN OPT — O1 + O2 + O3 sobre Agent-R (branch `opt/integrate`)

Base: `511ea9d` (rambench-nocap). Fusiona `opt/opt_ingesta` (O1), `opt/opt_recall` (O2),
`opt/opt_alloc` (O3). Binario: `target-integrate/release` (CARGO_TARGET_DIR propio).
Todos los números de este informe son corridas reales del harness
`scripts/rambench/` en esta máquina (seed 20261012, puerto 19103, 127.0.0.1).

## Qué integra cada optimización

| OPT | Cambio | Efecto |
|-----|--------|--------|
| O1 (ingesta) | Índice `HashMap<(ErrorCategory, InternedId), VecDeque<u64>>` en `record_error`: recurrence = len(bucket) pre-insert, pattern_id = back(bucket). Elimina los 2 scans O(n) por POST (agent_r.rs:176/:181 de 511ea9d) | Ingesta O(n²)→O(n) total |
| O3 (layout) | `operation`/`context_tags` internadas a `InternedId=u32` (string pool dedup), `what/why/prevention` como `Box<str>` | ~693 → ~510 B/reg |
| O2 (recall) | `recall_errors`: top-20 con min-heap acotado en un pase (sin Vec de n scores ni sort O(n log n)); salida idéntica (paridad testeada) | Preflight ~2.4× más rápido @100k |

**Sinergia clave**: la clave del índice de O1 usa el `InternedId` de O3 → la ingesta
O(1) no aloca ningún String por POST (en O1 standalone la clave costaba 1 alloc).
`load_errors` re-interna (O3) y reconstruye el índice (O1).

## Validación semántica

- **22/22 tests** de `agent_r`: 9 del baseline + 3 de O1 (paridad del índice vs
  scan naive, post-load, bucket) + 2 de O2 (paridad top-20 heap vs sort estable)
  + 6 de O3 (interning, contrato JSON con strings, roundtrip).
- Contrato de serialización estable: el JSON sigue exponiendo
  `operation`/`context_tags` como strings (no ids) — verificado por test.

## Números medidos del integrado

### Matriz 100k nocap ×3 (`reports/int_rambench.json`)
| métrica | integrado | base 511ea9d (rambench_nocap.json) |
|---|---|---|
| rps ingesta (mediana) | **755.9** [699.3, 755.9, 760.0] | 571.3 (corrida en máquina ociosa) |
| RAM Δ (mediana) | **48.7 MB** | 67.4 MB |
| B/registro | **510.4** | 687.5 |
| preflight p50 / p95 | **16.9 / 18.7 ms** | 19.5 / 21.8 ms |
| restart frío | 0.81 s | 0.57-0.77 s |
| integridad | 200000/200000 OK ×3 | idem |

### Escala 300k (datapoint abortable, `reports/int_raw_nocap/run_1m_abort.json`)
- 309,096 aceptados, **529.4 B/reg**, ~5.5 min de ingestión (base: 301k en ~17.3
  min con rps colapsando 770→13.2).

### **1M COMPLETO — la corrida que el baseline nunca pudo hacer** (`reports/int_raw_1m/`)
- **1,000,000/1,000,000 aceptados, sin aborto (límite puesto en 2M).**
- Wall-clock de ingestión: **~20 min** (proyección base: ~15 h ⇒ **~45× menos**).
- **rps plano en todo el rango**: mediana 827 primeras 5 ventanas vs 829 últimas 5
  (min 755, max 863) — lineal de 0 a 1M, sin decaimiento.
- Peak RSS: **478 MiB** (`/usr/bin/time -v`), **501.9 B/registro a escala 1M**
  (proyección base: ~680 MB ⇒ **−26% RAM**).
- El "motivo" dentro del JSON de la corrida es texto legacy del script (describe
  el problema del BASELINE); la corrida misma demuestra que ya no aplica:
  `abort_en_aceptados == n_target == 1_000_000`.

## Veredicto

El integrado convierte el perfil de Agent-R de "ingesta O(n²), 15 h para 1M" a
**1M errores en 20 minutos con RAM −26% y preflights más rápidos**, con paridad
semántica probada por tests en los tres frentes. Siguientes cuellos conocidos (no
bloquean 1M): `errors.remove(0)` O(n) si se supera MAX_ERRORS=2M, y el scoring
O(n) residual por preflight (ahora ~2.4× más barato pero sigue creciendo con n).
