# Agent-R — Architecture

Agent-R is an **error memory & self-tuning server** for LLM agents. The core
idea: agents fail in recurring ways; if you record the failures, validate
them, and distill the repeated ones into rules, every future agent (and the
same agent, after a restart) can check those rules *before* acting.

## The two-layer pipeline

```
 record-error              refine (gate N≥2)             preflight-hybrid
      │                          │                              │
┌─────▼──────────┐      ┌────────▼─────────┐      ┌──────────────▼───────────┐
│  LAYER 1       │      │  LAYER 2         │      │  HYBRID PREFLIGHT        │
│  incidents     ├─────▶│  rules           ├─────▶│  composite scoring +     │
│  dedup by      │      │  operation →     │      │  global veto             │
│  fingerprint   │      │  what/why/       │      │  top-3 rules +           │
│  (ADD/UPDATE/  │      │  prevention      │      │  suggestion              │
│   NOOP)        │      │  active/deprecated│     │                          │
└─────┬──────────┘      └────────┬─────────┘      └──────────────┬───────────┘
      │ persist                   │                              │ run + report
      ▼                           │                              ▼
┌───────────────┐                 │                      ┌─────────────────────┐
│ snapshot JSON │                 └──────────────────────│ SELF-TUNING         │
│ (write-through│                                        │ outcome hit/miss    │
│  O(n), atomic)│                                        │ re-weights w_content │
└───────────────┘                                        │ + rule influence    │
       ▲                                                 └──────────┬──────────┘
       │                       HMAC receipts (N3)  ◀────────────────┘
       └────────────── receipts store (tamper-evident claims)
```

1. **Layer 1 — incidents.** `POST /agent-r/record-error` writes a raw failure
   (category, severity, operation, what/why, context tags, prevention).
   Writes follow ADD/UPDATE/NOOP semantics keyed by a dedup fingerprint
   `(category, operation, normalized what)` (OPT-O4): a repeated failure
   bumps recurrence instead of duplicating rows.
2. **Layer 2 — rules.** Only patterns with **recurrence ≥ 2** (gate N≥2)
   distill into rules — one-off bad luck never becomes a rule. Rules carry
   the distilled lesson (what/why/prevention), can be active or deprecated,
   and keep per-rule stats (false positives decay them).
3. **Hybrid preflight.** Before an agent runs an operation,
   `POST /agent-r/preflight-hybrid` scores every rule with a composite
   score — exact operation match + tag/recency/recurrence/severity bonuses,
   plus a *content* signal (binary-IDF overlap between the caller's live
   `situation` and the rule's `what`) weighted by the self-tuned `w_content`.
   It returns the **top-3** rules and a suggestion (the best rule's
   prevention). Separately, an *active, non-deprecated* rule with N≥2,
   Critical severity and no open false positives can trigger a **global
   veto** (`proceed: false` + `veto_reason`) — global means it blocks every
   operation, not only matching ones, until the rule is deprecated.
4. **Self-tuning.** After acting, the agent reports `hit`/`miss` outcomes
   (CLI `agentr outcome`, MCP `agentr_outcome`). Feedback adjusts `w_content`
   and rule influence: signals that help get stronger, noisy ones fade.
5. **Receipts.** Claims can be backed by HMAC-signed receipts
   (`/agent-r/receipts/issue` → `/verify`), making agent reports
   tamper-evident (validation level N3).

## Validation levels N0–N3

`POST /agent-r/validate` runs an output through four levels; a level with no
oracle reports `unverifiable`, never `fail`. `overall` = fail if any oracle
level fails, unverifiable if none could verify, pass otherwise.

| Level | Name | Oracle | Evidence required |
|---|---|---|---|
| N0 | form | structural | non-null output, well-formed shape (e.g. uniform arrays) |
| N1 | numeric consistency | internal | numbers in the output are self-consistent |
| N2 | caller evidence | external (caller-supplied) | programmatic evidence from the caller: `{"exit_code", "stderr", "diff", "expected_exit_code"}` — nothing is ever executed by the server |
| N3 | receipt | cryptographic | HMAC receipt verified via the receipts store |

Incidents escalate as they gain evidence; rules require **N≥2** (evidence
from at least two sources/incidents) before they can influence a veto.

## Core module map (`src/`)

| Module | Lines (approx) | Responsibility |
|---|---|---|
| `agent_r.rs` | 4300 | the engine: incidents, dedup fingerprints, rules, preflight v2 + hybrid, self-tuning, stats, snapshot persist/load |
| `agent_swarm.rs` | 1200 | multi-agent coordination: registry + heartbeats, task queue with atomic claim (CAS), dependencies, barriers, roles, proposals/voting |
| `api_server.rs` | 1300 | axum HTTP server, 38 routes (see [HTTP_API.md](HTTP_API.md)) |
| `receipts.rs` | — | HMAC receipt store (issue / verify / verify-claim), file-backed |
| `meta.rs`, `meta_evolve.rs`, `meta_curiosity.rs` | — | meta layer: refine/self-debug loops with the optional `critic`, open questions (curiosity), insights and skill induction |
| `bin/agentr.rs` | 370 | the `agentr` CLI (remember/check/validate/show/deprecate/stats/flush/mcp) + built-in MCP stdio server |

### Legacy engine (not on the Agent-R server path)

The crate still carries the previous project phase: a Gaussian-splat vector
search engine (`splats`, `hrm2_engine`, `clustering`, `quantization`,
`encoding`, `hnsw_index`, `lsh_index`, …). `api_server` keeps five legacy
endpoints (`/health`, `/status`, `/store`, `/search`, `/rebuild-index`)
wrapping it for backwards compatibility. It is being split out; do not build
new Agent-R features against it.

## Persistence

Single JSON snapshot at `AGENT_R_STATE_PATH` (default `~/.agentr/state.json`),
shared by server and CLI, written atomically on each change. Records are
stored as human-readable `PlainRecord`s with an interned string pool
reconstructed on load. Cost model: the full snapshot is serialized per write
(**O(n) per POST — by design**, see [BENCHMARKS.md](BENCHMARKS.md)); run
high-volume ingestion ephemeral and batch snapshots.

## Runtime shape

- One process, in-memory state behind an `axum` handler state
  (`Arc<Mutex<AgentR>>`), CPU-only, no database, no GPU.
- Horizontal sharing happens at the HTTP layer: many agents (or a swarm)
  talk to one server; swarm task completions/failures are written back into
  the shared error memory so every agent's preflight sees them
  (see [SWARM.md](SWARM.md)).
