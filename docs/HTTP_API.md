# Agent-R — HTTP API

The server (`agentr serve --port N`, axum) binds `127.0.0.1` and speaks JSON
on every endpoint. All request bodies and responses are `application/json`.
Route inventory: `src/api_server.rs` (router builder in `run_server`).

Base URL for the examples: `http://127.0.0.1:19105`.

## Endpoint inventory (38 routes)

### Agent-R core (17)

| # | Method | Endpoint | Purpose |
|---|--------|----------|---------|
| 1 | POST | `/agent-r/record-error` | record an incident (dedup by fingerprint, ADD/UPDATE/NOOP semantics) |
| 2 | POST | `/agent-r/preflight` | rule preflight for an operation (v2 scoring) |
| 3 | POST | `/agent-r/preflight-hybrid` | composite scoring + global veto + top-3 rules, hybrid structural+content signals |
| 4 | POST | `/agent-r/flush` | reset the memory (write-through snapshot) |
| 5 | POST | `/agent-r/validate` | validate an output at levels N0–N3 |
| 6 | GET | `/agent-r/stats` | memory + self-tuning counters |
| 7 | POST | `/agent-r/receipts/issue` | issue an HMAC receipt for a claim payload |
| 8 | POST | `/agent-r/receipts/verify` | verify a receipt (optionally against a claim) |
| 9 | POST | `/agent-r/refine` | distill/refine rules from incidents (critic) |
| 10 | GET | `/agent-r/questions` | open questions of the curiosity loop |
| 11 | POST | `/agent-r/questions/ask` | ask a new question |
| 12 | POST | `/agent-r/questions/:id/answer` | answer a question |
| 13 | POST | `/agent-r/insights` | record an insight |
| 14 | POST | `/agent-r/skills/induce` | induce a skill from recurring patterns |
| 15 | GET | `/agent-r/capacity` | live capacity: incidents, successes, insights, skills, open questions, `w_content` |
| 16 | POST | `/agent-r/self-debug` | critic loop with budget: N0–N3 re-validation until pass |
| 17 | GET | `/agent-r/stats` *(see 6)* | — |

### Swarm coordination (16)

| # | Method | Endpoint | Purpose |
|---|--------|----------|---------|
| 18 | POST | `/agent-swarm/register` | register an agent (role defaults to Executor) |
| 19 | POST | `/agent-swarm/heartbeat` | liveness ping |
| 20 | POST | `/agent-swarm/tasks` | submit a task (optional `depends_on`, `preferred_agent`) |
| 21 | POST | `/agent-swarm/tasks/claim` | **atomic claim** (READY→CLAIMED via CAS, one winner) |
| 22 | POST | `/agent-swarm/tasks/complete` | complete a task (result written to shared memory) |
| 23 | POST | `/agent-swarm/tasks/fail` | fail a task (reason written to shared memory) |
| 24 | GET | `/agent-swarm/tasks/result?task_id=N` | fetch a task result |
| 25 | POST | `/agent-swarm/barrier` | create a phase barrier for N participants |
| 26 | POST | `/agent-swarm/barrier/arrive` | arrive at a barrier (blocks the phase until quorum) |
| 27 | POST | `/agent-swarm/preflight` | preflight against the whole swarm's shared memory |
| 28 | GET | `/agent-swarm/status` | swarm overview (agents, tasks, barriers, proposals) |
| 29 | POST | `/agent-swarm/roles` | assign role: Coordinator / Proposer / Validator / Executor |
| 30 | POST | `/agent-swarm/proposals` | submit a proposal (Coordinator) |
| 31 | POST | `/agent-swarm/proposals/vote` | vote a proposal (Validator/Coordinator only) |
| 32 | POST | `/agent-swarm/proposals/materialize` | turn an accepted proposal into tasks |
| 33 | GET | `/agent-swarm/proposals/status` | proposal status / votes |
| 34 | POST | `/agent-swarm/proposals/close-stalled` | close proposals stalled past their epoch |

### Legacy memory (5) — backwards compatibility

| # | Method | Endpoint | Purpose |
|---|--------|----------|---------|
| 35 | GET | `/health` | liveness |
| 36 | POST | `/status` | legacy store status |
| 37 | POST | `/store` | legacy vector store (old engine) |
| 38 | POST | `/search` · POST `/rebuild-index` | legacy search / index rebuild (old engine) |

> The legacy endpoints wrap the previous vector-search engine and are kept
> for compatibility. They are not part of the Agent-R pipeline.

## Examples (shapes verified against the `Deserialize`/`Serialize` structs)

### 1. `POST /agent-r/record-error` — Layer 1

Request (`RecordErrorRequest`, all fields required):

```json
{
  "category": "fabrication",
  "severity": "error",
  "operation": "deploy_edge",
  "what": "curl to the edge API timed out because HTTP_PROXY was not set",
  "why": "requests to 10.x hosts require the proxy env vars in this shell",
  "context_tags": ["deploy", "network"],
  "prevention": "export HTTP_PROXY before any curl to the edge cluster"
}
```

`category` one of: `circular`, `drift`, `fabrication`, `assumption`,
`skipped`, `timeout`, `corruption`, `recurring`, `other`.
`severity` one of: `warning`, `error`, `critical`.

Response (`RecordErrorResponse`):

```json
{ "id": 4, "outcome": "Add", "recurrence": 0 }
```

- `outcome`: `"Add"` (new fingerprint) / `"Update"` (merged new info into an
  existing incident) / `"Noop"` (exact duplicate, bookkeeping only).
- `recurrence`: recurrence count after this write — `0` on `Add`.
- A **fresh memory is seeded** with three starter lessons, so the first
  user-recorded incident gets `id` 4. Remove the seeds with
  `agentr deprecate <id> "demo seed"` (or via the CLI `show`/`deprecate`).

### 2. `POST /agent-r/preflight-hybrid` — Layer 2 + scoring

Request (`PreflightHybridRequest`):

```json
{
  "operation": "deploy_edge",
  "context_tags": ["deploy"],
  "situation": "about to curl the edge API from a fresh shell"
}
```

`situation` (required) is the caller's current state (observation, command,
stderr): it activates the content signal (binary-IDF overlap against the
`what` of each rule).

Response (measured against a fresh seeded memory after recording the
`deploy_edge` incident above):

```json
{
  "proceed": false,
  "warnings": ["BLOCKED: recurring critical rule [1] generate_benchmarks: Invented benchmark data for M2M project"],
  "precautions": [
    "[deploy_edge] export HTTP_PROXY before any curl to the edge cluster",
    "[generate_benchmarks] Only publish validated data from real test runs",
    "[image2svg_pipeline] Run MASFactory analysis before implementation"
  ],
  "relevant_errors": 3,
  "top_rules": [
    { "id": 4, "operation": "deploy_edge", "what": "curl to the edge API timed out because HTTP_PROXY was not set", "score": 31.22 },
    { "id": 1, "operation": "generate_benchmarks", "what": "Invented benchmark data for M2M project", "score": 2.77 },
    { "id": 2, "operation": "image2svg_pipeline", "what": "Image2SVG failed 3x by skipping MASFactory analysis step", "score": 2.77 }
  ],
  "suggestion": "[deploy_edge] export HTTP_PROXY before any curl to the edge cluster",
  "veto_reason": "rule 1 [generate_benchmarks] critical, seen 3x, no open FPs",
  "hybrid": true,
  "w_content": 3.0,
  "tuning": { "hits": 0, "misses": 0 }
}
```

- `top_rules`: top-3 by composite score (exact operation match dominates —
  note `deploy_edge` at 31.2 vs the seeds at 2.8; hybrid adds the content
  signal weighted by `w_content`).
- `veto_reason` / `proceed: false`: the veto is **global** — any *active,
  non-deprecated* rule with N≥2 incidents, Critical severity and no open
  false positives blocks **every** operation (not just matching ones) until
  the rule is deprecated. On a fresh memory the seeded critical rule
  (`generate_benchmarks`) triggers it; deprecate the seeds to restore
  `proceed: true`.

### 3. `GET /agent-r/stats`

Response (`AgentRStats`, same fresh memory as above):

```json
{
  "total_errors_recorded": 4,
  "unique_incidents": 4,
  "total_preflights": 0,
  "total_validations": 0,
  "prefights_blocked": 1,
  "validations_failed": 0,
  "errors_by_category": { "BadAssumption": 1, "SkippedValidation": 1, "Fabrication": 2 },
  "top_recurring_patterns": []
}
```

Notes: `errors_by_category` keys are the PascalCase enum names
(`Fabrication`, not `fabrication`). `total_preflights` counts plain
`/agent-r/preflight` calls (the hybrid endpoint increments
`prefights_blocked` when it blocks). `top_recurring_patterns` fills once
incidents actually recur (recurrence ≥ 2).

(`prefights_blocked` is the on-the-wire name — historical spelling.)

### 4. `POST /agent-swarm/tasks/claim` — atomic claim

Request (`SwarmClaimRequest`):

```json
{ "task_id": 7, "agent": "worker-1" }
```

Response (`SwarmClaimResponse`):

```json
{ "result": "claimed" }
```

`result` (`ClaimResult`, snake_case): `claimed` (won the CAS),
`taken` (another agent claimed first), `not_ready` (dependencies pending),
`unknown_agent` (not registered / heartbeat expired), `reserved`
(preferred agent mismatch).

### 5. `POST /agent-r/validate` — N0–N3

Request (`ValidateOutputRequest`):

```json
{
  "operation": "deploy_edge",
  "output": { "ok": true },
  "evidence": { "exit_code": 0, "expected_exit_code": 0, "stderr": "" }
}
```

Response: `verdicts` per level (`n0_form`, `n1_numeric_consistency`,
`n2_caller_evidence`, `n3_receipt` — each `pass`/`fail`/`unverifiable`),
`overall`, `suggestion`, `details`, plus legacy-compat `valid`,
`confidence`, `issues`, `checks_run`, `elapsed_ms`.

## Environment

| Variable | Effect | Default |
|---|---|---|
| `AGENT_R_STATE_PATH` | JSON snapshot path (shared server/CLI). Empty ⇒ ephemeral. | `~/.agentr/state.json` |
| `AGENT_R_NO_SEED` | set to `1` to start with an empty memory (skips the three starter lessons) | unset (seeds load) |
| `AGENT_R_RECEIPTS_DIR` | directory for HMAC receipts | `data/receipts` |
