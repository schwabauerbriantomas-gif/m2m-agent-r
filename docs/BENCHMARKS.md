# Agent-R — Benchmarks (consolidated, verified)

Every number below points at a file in this repository (or a dated doc in it).
Hardware/software per section as noted; all ingest runs use deterministic
splitmix64 payloads (seed 20261012), concurrency 10, release build, and
verify 100% HTTP 200 acceptance. Reproduction harness: `scripts/rambench/`.

Reference machine: AMD Ryzen 5 3400G (8 threads), 13–32 GiB RAM, WSL2,
rustc 1.97/1.94.

## Ingest throughput

| Metric | Value | Conditions | Source |
|---|---|---|---|
| Persistent ingest @100k | **790.3 rps** (runs: 790.3 / 789.9), 126.5 s total, snapshot 61.5 MB | write-through ON, 100k/100k accepted, 9 categories x 3 severities x 15 ops | docs/benchmarks/persist_after_n100000.json:8, persist_after2_n100000.json:8 |
| Ephemeral ingest @100k (integration, pre-O4) | **755.9 rps** median x3; RAM delta 48.7 MB; **510.4 B/record**; preflight p50 16.9 / p95 18.7 ms | nocap, base 511ea9d = 571.3 rps / 687.5 B | docs/OPT_INTEGRACION.md:47-51 |
| O4 sanity @100k (ephemeral) | **727.8 rps** median [736.8, 683.6, 754.8]; 570 B/incident; preflight p50 18.3 ms; cold restart 0.735 s | dedup ON, no duplicate payloads | docs/OPT_INTEGRACION.md:55-62 |
| Full 1M ingest (O4 + swarm) | **1,000,000 / 1M in 19:10**, median **901 rps** (min 651, max 967), decay 0.1%, peak RSS 547 MiB = **574.7 B/incident** | final validation run | docs/OPT_INTEGRACION.md:68-72 |
| 1M pre-O4 reference | ~20 min, 755-863 rps, 501.9 B/reg (base 511ea9d projection: ~15 h) | | docs/OPT_INTEGRACION.md:73-74 |
| 300k (pre-O4) | 309,096 accepted, 529.4 B/reg, ~5.5 min | | docs/OPT_INTEGRACION.md:64-65 |

## Persistence write-through cost (O(n) per POST, by design)

| State size | rps | Snapshot |
|---|---|---|
| 1k | 120.1 | 0.4 MB |
| 6k | 62.6 | 2.3 MB |
| 11k | 31.3 | 4.2 MB |
| 16k | 19.8 | 6.1 MB |

Sequential POSTs with AGENT_R_STATE_PATH active. Doubling state halves rps
(full snapshot serialized per write). docs/OPT_INTEGRACION.md:76-92.
**Benchmarks run ephemeral**; production should batch/debounce snapshots.

## v1 baseline (RAM-only, before optimizations)

- Sustained ingest ~700-790 POST/s @10k-100k (cap MAX_ERRORS=10000, 200 OK 100%).
- ~680 B theoretical / **~830 B observed per resident incident**; 1M uncapped = 0.65 GiB.
- Preflight p50 2.62->3.01 ms (10k->100k); memory fully lost on restart (pre-O4).
- docs/RAM_BASELINE.md:10-19, 114.

## Quality metrics — UNVERIFIED (owner's logbook only)

| Metric | Value | Status |
|---|---|---|
| 2-head taxonomy accuracy | 54.2% vs 20.8% baseline | **unverified — owner logbook, re-verify** — not in any repo file |
| Hybrid preflight hit-rate | top-1 29.0% / top-3 41.9% | **unverified — owner logbook, re-verify** — harness exists (examples/far_benchtrace.rs:159-164) but numbers were never committed |

These two rows come from the owner's private logbook and could not be
reproduced from any artifact in the repository. Do not cite them as measured
results until the harness is re-run and the output committed.

## Historical / legacy

The original `BENCHMARKS.md` and `BENCHMARKS_V2.md` (GPU CUDA kernels,
KMeans, RTX 3090 speedups, "226/34 tests") describe the **legacy
vector-search engine**, not the Agent-R server path; they were moved to
`docs/history/` and are superseded by this file. Test counts over time:
276/276 lib at integration time (docs/OPT_INTEGRACION.md:27), 324 lib + 3
integration passing today (baseline run, 2026-10-25).

## Mechanism comparison (in-process, deterministic)

Comparación de **mecanismos de memoria de errores** sobre un corpus sintético
determinista — **no** una comparación contra productos externos. Harness:
`tests/comparison.rs` (reproducible con
`cargo test --test comparison -- --nocapture`).

### Metodología

- **Corpus** (seed `0x20261025`, ChaCha8, fingerprint FNV-64 pinneado
  `0x4e8743339e66234d`): 40 operaciones × 4–5 firmas de falla recurrente
  (184 firmas, 1 incidente canónico c/u) + 46 incidentes únicos de ruido
  (20% del total, nunca consultados) = **230 incidentes**. Los textos
  what/why/prevention se generan de plantillas con vocabulario cerrado (26
  arquetipos de falla); el orden de ingestión es un shuffle seedeado.
- **Queries**: 408 = 368 positivas (2 por firma, **paráfrasis** del
  what/why — plantillas distintas, ~30% sinónimos, ~25% keyword distractor de
  otro tema, para castigar matching ingenuo; ground truth = la firma) + 40
  negativas (operaciones **sin historial**, miden falsos vetos).
- **Estrategias**, todas in-process sobre el MISMO corpus y las MISMAS
  queries: `no_memory` (piso), `exact_match` (v1: `HashMap<op, lección más
  reciente>`), `keyword_overlap` (bag-of-words con IDF simple del corpus sobre
  what+why vs situación — **proxy honesto de similarity-search sin
  embeddings**), y `agent_r_hybrid` (el sistema real vía lib pública:
  `AgentR::preflight_v2_hybrid`, el mismo call que `api_server` sirve en
  `POST /agent-r/preflight-hybrid`; memoria limpia con `new_ephemeral`).
- **Latencias** medidas en build **debug** (CI corre `cargo test` sin
  optimizar; release ≈ 10× más rápido). **Memoria**: `exact_match`/
  `keyword_overlap` son deep-size estructural estimado; `agent_r_hybrid` es
  medición real del snapshot JSON flushed (ΔVmRSS reportado como ruidoso).
- **Determinismo**: 3 corridas — corpus, hits, vetos y fingerprint
  **idénticos**; solo latencia y bytes del snapshot oscilan (timestamps).

### Resultados medidos (debug build)

| strategy | top1_hit | top3_hit | false_veto_rate | p50 µs | p95 µs | bytes/incident |
|---|---|---|---|---|---|---|
| no_memory | 0.000 | 0.000 | n/a (sin veto) | 0.0 | 0.1 | 0 |
| exact_match | 0.185 | 0.185 | n/a (sin veto) | 0.4 | 0.5 | 10 B (est.; guarda ~1 lección/op) |
| keyword_overlap | 0.584 | 0.742 | n/a (sin veto) | ~1100 | ~1500 | 794 B (est.) |
| agent_r_hybrid | 0.139 | 0.503 | **0.500** | ~9300 | ~12000 | ~558 B (snapshot JSON medido; ΔVmRSS ≈ 1 KB/inc, ruidoso) |
| agent_r_hybrid (diagnóstico repeat-quote) | **0.000** | — | — | — | — | situación = what+why VERBATIM |
| keyword_overlap (diagnóstico repeat-quote) | 1.000 | — | — | — | — | ídem |

### Hallazgos (la comparación honesta manda)

1. **El híbrido queda DEBAJO de exact_match en top1** (0.139 vs 0.185) y
   **debajo de keyword_overlap en top3** (0.503 vs 0.742) en este corpus. Las
   aserciones objetivo del plan (`hybrid ≥ exact` top1, `hybrid ≥ keyword`
   top3, `false_veto < 5%`, `p95 < 10 ms`) **no se cumplen** con los números
   reales y NO se forzaron: el test pinnea el baseline medido con margen y
   este documento es el registro de la verdad.
2. **Causa raíz** (análisis de `preflight_v2_hybrid_w`, confirmado por el
   dump de misses que imprime el test): el canal de fallback de lección
   completa (`W_FULL_FALLBACK = 25`) aplica `25 × ΣIDF` a **cualquier** regla
   cuyo `why` comparta ≥1 token distintivo con la situación **aunque su
   `what` no tenga overlap** (c_idf == 0), sin gate de overlap mínimo. La
   regla CORRECTA, que matchea el canal fino (`c_idf > 0` sobre `what`), es
   **inelegible** para el fallback y queda en `base + 3·c_idf`. Un solo token
   compartido en el `why` de una regla ajena (~IDF 4–5 ⇒ ~100–125) supera el
   match fino correcto (~40–60): el fallback domina el ranking completo.
3. **El diagnóstico repeat-quote lo prueba en el caso de diseño**: citando
   VERBATIM el what+why de la lección correcta, el híbrido da top1 = 0.000
   (las reglas ajenas ganan por el fallback) mientras keyword_overlap da
   1.000. El canal fallback fue calibrado para repetición-cita (c_full
   mediana ~27 en los datos de linaje); en corpus con vocabulario más chico
   y colisiones accidentales invierte la dominancia.
4. **Falsos vetos = 50%** (20/40 queries negativas vetadas): el veto (regla
   Critical con recurrencia ≥ 1 en el top-3) hereda el ranking envenenado —
   reglas Critical ajenas a la operación alcanzan el top-3 por el fallback.
5. **Latencia**: el híbrido re-tokeniza what+why de TODAS las reglas activas
   en cada query (O(n·tokens)); en debug p50 ≈ 9.3 ms @ 230 reglas (para
   referencia, README reporta p50 16.9 ms @ 100k en release).
6. Nota positiva: la fila "Hybrid preflight hit-rate" de la sección *Quality
   metrics — UNVERIFIED* (top-1 29% / top-3 42% del logbook) ahora tiene un
   harness reproducible; este test mide 0.139/0.503 en su corpus propio.

Los umbrales del test se calibraron con margen BAJO los valores medidos
(floors 0.10/0.45, pin de false_veto 0.60, presupuesto de latencia 30 ms
debug / 10 ms release) para que cualquier **degradación adicional** rompa CI;
cuando el mecanismo mejore, tightear los floors y restaurar las aserciones
objetivo (`hybrid ≥ exact` top1, `hybrid ≥ keyword` top3, `false_veto < 5%`).
