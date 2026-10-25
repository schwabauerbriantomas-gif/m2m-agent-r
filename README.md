# Agent-R

**Error memory & self-tuning server for LLM agents, written in Rust.**

[![CI](https://github.com/schwabauerbriantomas-gif/m2m-agent-r/actions/workflows/ci.yml/badge.svg)](https://github.com/schwabauerbriantomas-gif/m2m-agent-r/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Rust Edition 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://blog.rust-lang.org/2021/10/21/Rust-1.56.0.html)

Agent-R gives AI agents a durable memory of their own failures.

- Every error an agent hits is recorded as an **incident** (with category, severity, operation, context and prevention).
- Incidents that repeat — after validation at levels **N0–N3** — are distilled into **rules**.
- Before an agent runs an operation, a **hybrid preflight** retrieves the rules that apply and can **veto** the run outright.
- Agents report real **outcomes** (`hit`/`miss`), and the memory self-tunes: rules that help get stronger, noisy signals fade.

Everything is exposed over an HTTP API (axum), a zero-latency local CLI (`agentr`), and a built-in MCP stdio server — so any LLM agent, orchestrator or swarm can share one error memory.

> New here? [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the two-layer pipeline, the N0–N3 validation levels and the module map.

## How it works — two layers

```
            ┌────────────────────────────────────────────────────────┐
            │                     LAYER 1: INCIDENTS                 │
            │  /agent-r/record-error  →  ErrorRecord                 │
            │  raw failures, deduped by fingerprint (O4)             │
            │  validation levels N0–N3 attached to each incident     │
            └───────────────────────────┬────────────────────────────┘
                                        │ recurrence ≥ 2  (gate N≥2)
            ┌───────────────────────────▼────────────────────────────┐
            │                     LAYER 2: RULES                     │
            │  distilled lessons: operation → what/why/prevention    │
            │  active vs deprecated, per-rule stats                  │
            └───────────────────────────┬────────────────────────────┘
                                        │
            ┌───────────────────────────▼────────────────────────────┐
            │                HYBRID PREFLIGHT (scoring)              │
            │  composite score: exact op match + similarity +        │
            │  recency + frequency → top-3 rules                     │
            │  global VETO if any active rule (N≥2) hard-matches     │
            └───────────────────────────┬────────────────────────────┘
                                        │ outcome hit / miss
            ┌───────────────────────────▼────────────────────────────┐
            │                 SELF-TUNING + RECEIPTS                 │
            │  tuning feedback re-weights rules on real outcomes     │
            │  HMAC receipts make agent claims tamper-evident        │
            └────────────────────────────────────────────────────────┘
```

- **Incidents → rules**: an incident escalates validation levels N0 (self-report) → N3 (externally verified). Only patterns backed by **N≥2** evidence distill into rules — one-off bad luck never becomes a rule.
- **Hybrid preflight**: a composite score (`exact operation match` + similarity + recency + `log(1+frequency)`) ranks rules and returns the **top-3** with concrete prevention advice. Separately, any *active, non-deprecated* rule with N≥2 can trigger a **global veto** (`proceed: false` + reason).
- **Self-tuning**: after acting, the agent reports whether the lesson applied. Repeated misses decay a rule's influence; hits confirm it.
- **Receipts**: HMAC-signed receipts (`/agent-r/receipts/issue`, `/agent-r/receipts/verify`) let you verify that an agent actually did what it claims.

## Quickstart

Requirements: Rust stable (edition 2021). No GPU, database or external services needed.

```bash
git clone https://github.com/schwabauerbriantomas-gif/m2m-agent-r.git agent-r && cd agent-r
cargo build --release

# 1) Start the HTTP server (axum) — binds 127.0.0.1
cargo run --release -- serve --port 19105
# M2M server listening on 127.0.0.1:19105
```

Record an error (Layer 1):

```bash
curl -s -X POST http://127.0.0.1:19105/agent-r/record-error \
  -H 'content-type: application/json' -d '{
    "category": "fabrication",
    "severity": "error",
    "operation": "deploy_edge",
    "what": "curl to the edge API timed out because HTTP_PROXY was not set",
    "why": "requests to 10.x hosts require the proxy env vars in this shell",
    "context_tags": ["deploy", "network"],
    "prevention": "export HTTP_PROXY before any curl to the edge cluster"
  }'
# → { "id": 4, "outcome": "Add", "recurrence": 0 }
```

`outcome` is `Add` (new incident), `Update` (merged new info into an existing
fingerprint) or `Noop` (exact duplicate). The ids start at 4 because a fresh
memory is **seeded with three starter lessons** (visible in `agentr stats`) —
remove them with `agentr deprecate <id> "…"`, or start the server with
`AGENT_R_NO_SEED=1` for an empty memory.

Ask for a preflight before running that operation again (Layer 2 + scoring):

```bash
curl -s -X POST http://127.0.0.1:19105/agent-r/preflight-hybrid \
  -H 'content-type: application/json' -d '{
    "operation": "deploy_edge",
    "context_tags": ["deploy"],
    "situation": "about to curl the edge API from a fresh shell"
  }'
# → { "proceed": false,
#     "warnings": ["BLOCKED: recurring critical rule [1] …"],
#     "top_rules": [ { "id": 4, "operation": "deploy_edge", "what": "…", "score": 31.2 }, … ],
#     "suggestion": "[deploy_edge] export HTTP_PROXY before any curl to the edge cluster",
#     "veto_reason": "rule 1 [generate_benchmarks] critical, seen 3x, no open FPs",
#     "hybrid": true, "w_content": 3.0, "tuning": { "hits": 0, "misses": 0 } }
```

`top_rules` carries the top-3 lessons by composite score — your fresh
`deploy_edge` incident outranks the seeds thanks to the exact-operation-match
bonus. The veto above comes from a **seeded critical rule**: any active,
non-deprecated rule with N≥2 incidents, Critical severity and no open false
positives vetoes **globally** (every operation) until deprecated. Prefer a
clean first run? Start with `AGENT_R_NO_SEED=1` and the same preflight
returns `proceed: true`, `veto_reason: null` and your rule as the only
`top_rule` (score ≈ 23).

Then close the loop — report the outcome so the memory tunes itself:

```bash
curl -s http://127.0.0.1:19105/agent-r/stats
# → { "total_errors_recorded": 4, "unique_incidents": 4, "total_preflights": 0,
#     "prefights_blocked": 1, "errors_by_category": { "Fabrication": 2, … },
#     "top_recurring_patterns": [] }
```

Persistence is a single JSON snapshot (`AGENT_R_STATE_PATH`, default `~/.agentr/state.json`), shared by the server and the CLI, written atomically on each change with fingerprint dedup.

## CLI — `agentr`

Daily use without the HTTP server — the CLI talks straight to the library (process-level latency, no network) and shares state with the server via `AGENT_R_STATE_PATH`:

```bash
agentr remember deploy_edge "HTTP_PROXY not set" \
  --why "10.x hosts need the proxy" \
  --severity error --category fabrication \
  --tags deploy,network \
  --prevention "export HTTP_PROXY before curl"

agentr check deploy_edge --tags deploy          # hybrid preflight
agentr validate --op deploy_edge --output "…"   # N0–N3 validation of a result
agentr show 4                                    # inspect an incident/rule
agentr deprecate 4 "no longer applies"           # retire a wrong rule
agentr stats                                     # memory + self-tuning counters
agentr flush                                     # wipe memory (seeds reload)
agentr mcp                                       # MCP stdio server (JSON-RPC 2.0)
```

Add `--json` to any command for machine-readable output.

## HTTP API

Server: `cargo run --release -- serve --port 19105`. Full inventory in [`docs/HTTP_API.md`](docs/HTTP_API.md).

### Agent-R core

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/agent-r/record-error` | record an incident (dedup by fingerprint) |
| POST | `/agent-r/preflight` | rule preflight for an operation |
| POST | `/agent-r/preflight-hybrid` | composite scoring + global veto + top-3 rules |
| POST | `/agent-r/validate` | validate an output at levels N0–N3 |
| GET  | `/agent-r/stats` | memory and self-tuning counters |
| POST | `/agent-r/flush` | reset the memory |
| POST | `/agent-r/refine` | distill/refine rules from incidents |
| POST | `/agent-r/receipts/issue` | issue an HMAC receipt for a claim |
| POST | `/agent-r/receipts/verify` | verify a receipt |
| GET  | `/agent-r/questions` · POST `/agent-r/questions/ask`, `/agent-r/questions/:id/answer` | curiosity loop: open questions |
| POST | `/agent-r/insights`, `/agent-r/skills/induce`, `/agent-r/self-debug`, GET `/agent-r/capacity` | introspection & meta endpoints |

### Swarm coordination

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/agent-swarm/register`, `/agent-swarm/heartbeat` | register agents, liveness |
| POST | `/agent-swarm/tasks`, `/tasks/claim`, `/tasks/complete`, `/tasks/fail` | task queue with **atomic claim** (READY→CLAIMED via CAS) and deterministic fallback |
| GET  | `/agent-swarm/tasks/result` | fetch a task result |
| POST | `/agent-swarm/barrier`, `/barrier/arrive` | phase barriers |
| POST | `/agent-swarm/preflight` | preflight against the **whole swarm's** shared memory |
| POST | `/agent-swarm/roles` | assign roles: Coordinator / Proposer / Validator / Executor |
| POST | `/agent-swarm/proposals`, `/proposals/vote`, `/proposals/materialize`, `/proposals/close-stalled`; GET `/proposals/status` | consensus with quorum |
| GET  | `/agent-swarm/status` | swarm overview |

Legacy memory endpoints (`/health`, `/status`, `/store`, `/search`, `/rebuild-index`) are also served for backwards compatibility.

## MCP

`agentr mcp` starts a stdio MCP server (JSON-RPC 2.0) with four tools:

| Tool | Purpose |
|------|---------|
| `agentr_check` | preflight: which registered lessons apply to this operation (vetos and warnings included) |
| `agentr_remember` | record an error/lesson (auto-dedup by fingerprint) |
| `agentr_outcome` | self-tuning feedback: `hit` (lesson applied and the task still failed) or `miss` (signal did not apply) |
| `agentr_stats` | memory and self-tuning counters |

Client configuration example and tool schemas: [`docs/MCP.md`](docs/MCP.md).

## Swarm mode

Multiple agents can share one memory and coordinate through the server: a task queue with **atomic claim** (`READY → CLAIMED` via compare-and-swap, exactly one winner), dependencies, barriers and epoch-based fallback; **roles** gate who proposes vs. executes; proposals go through **consensus voting with quorum** before materializing into tasks. Task completions and failures are written back to the shared error memory, so what one agent learns, every agent's preflight sees. Details: [`docs/SWARM.md`](docs/SWARM.md).

## Benchmarks (summary)

All numbers measured on the reference machine (AMD Ryzen 5 3400G, 8 threads, WSL2, release build, seed 20261012, concurrency 10, synthetic splitmix64 payloads). Full methodology and raw sources: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| Metric | Value | Conditions |
|--------|-------|-----------|
| Ingest throughput @100k (persistent) | **790 rps** | write-through snapshot active, 100k/100k accepted |
| Ingest throughput @100k (ephemeral) | **755.9 rps** median ×3 | pre-persistence integration baseline |
| Full 1M-error ingest | **19:10 min**, median **901 rps**, 0.1% decay | with dedup + swarm; linear from 0 to 1M |
| Memory per incident | **~510–575 B** | interned layout + fingerprint index |
| Preflight latency @100k | **p50 16.9 ms / p95 18.7 ms** | top-20 min-heap, single pass |
| Write-through cost | 120 → 20 rps @ 1k → 16k state | full snapshot serialized per POST (by design) |

> ⚠️ Persistence serializes the whole state on every write (O(n) per POST). It is fine for small working sets; run high-volume ingestion ephemeral (`AGENT_R_STATE_PATH=`) — batched/debounced snapshots are future work.

## Feature flags

| Flag | Effect |
|------|--------|
| `critic` | optional external-LLM critic for meta/refinement (blocking HTTP via `ureq`) |

No GPU is required or used: everything runs on CPU, and the server path has no database dependency.

## Development & tests

```bash
cargo fmt --all --check    # enforced in CI — the tree is rustfmt-clean
cargo test                 # unit + integration + doc-tests
cargo clippy --lib --bins  # see note below
```

The doc-tests that historically failed without the `critic` feature have been rewritten against the real API, so a plain `cargo test` is expected to pass. Clippy still reports warnings in the **legacy vector-search engine** modules (splats, hrm2, quantization, …) — the Agent-R core (`agent_r`, `agent_swarm`, `api_server`, `receipts`, `meta*`) is warning-free; CI therefore does not pass `-D warnings` yet and clippy is tightened progressively. CI runs fmt, clippy and tests on Rust stable — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

Python test drivers under `tests/` and the dev control plane (`scripts/m2mctl.py`, `scripts/mcp_server.py`) are development tools for this repo, not part of the shipped server.

## Project status

**Beta.** The error-memory pipeline (incidents → rules → hybrid preflight → self-tuning → receipts) and the swarm layer are complete and covered by 320+ unit/integration tests plus E2E HTTP checks; the HTTP API may still change. The repository also carries a **legacy vector-search engine** (Gaussian-splat embeddings, HRM2 retrieval, quantization) from an earlier phase of the project — it is not part of the Agent-R server path and is being split out.

## License

[AGPL-3.0](LICENSE) — full GNU text. Agent-R traces its lineage to the AGPL m2m-memory project.
