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
