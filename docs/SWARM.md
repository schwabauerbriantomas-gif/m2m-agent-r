# Agent-R — Swarm coordination

Several agents can share one Agent-R server: a **task queue with atomic
claims**, **dependencies**, **barriers**, **roles** and **consensus
voting** — all writing lessons back into the shared error memory, so what one
agent learns, every agent's preflight sees.

Implementation: `src/agent_swarm.rs` (~1,200 lines, 16 tests). HTTP surface:
the `/agent-swarm/*` routes (see [HTTP_API.md](HTTP_API.md)).

## Task lifecycle

```
 PENDING ──(all depends_on COMPLETED)──▶ READY ──CAS──▶ CLAIMED ──▶ COMPLETED
    │                                     ▲   │                     │
    │                        epoch >      │   │ agent dies          │
    │                        holder's     ├───┘ (fallback)          ▼
    └── deps never finish ◀───────────────┘   fail() ──▶ FAILED (lesson recorded)
```

- **Submit** (`POST /agent-swarm/tasks`): operation + description, optional
  `depends_on: [task_id]` and `preferred_agent`. No dependencies ⇒ `READY`
  immediately; otherwise `PENDING` until every dependency is `COMPLETED`.
- **Atomic claim** (`POST /agent-swarm/tasks/claim`): the transition
  `READY → CLAIMED` is a **compare-and-swap** on the task state
  (`compare_exchange(ST_READY, ST_CLAIMED)`) — under concurrency exactly one
  caller gets `{"result":"claimed"}`; everyone else gets `taken` (lost the
  CAS), `not_ready` (dependencies pending), `unknown_agent` (not registered
  or heartbeat expired) or `reserved` (task prefers another agent).
- **Deterministic fallback**: a `CLAIMED` task whose holder dies (no
  heartbeat within the liveness TTL) becomes re-claimable. Re-claims are
  decided by **epoch** — each claimant presents `claim_epoch = now`; the
  highest epoch wins (`fetch_max`). Recovery is reproducible, not
  first-come-first-served racing.
- **Complete / fail**: `complete` stores the result (and extracts a
  `receipt_id` when the result JSON carries one); `fail` records the reason.
  **Both write a lesson into the shared Agent-R memory** (dedup by
  fingerprint), so the failure is visible to every agent's preflight.

## Liveness

Agents `POST /agent-swarm/register` (role defaults to Executor) and must
`POST /agent-swarm/heartbeat` regularly. An expired heartbeat marks the
agent dead: its claimed tasks fall back (above) and it stops counting for
quorum.

## Barriers

`POST /agent-swarm/barrier { "participants": N }` creates a phase barrier;
each agent arrives via `POST /agent-swarm/barrier/arrive`. A phase only
advances when all N participants have arrived — used to keep generations of
work (e.g. all-proposals-before-any-vote) separated.

## Roles

`POST /agent-swarm/roles` assigns one of four roles (only the Coordinator
may set roles authoritatively):

| Role | May do |
|---|---|
| **Coordinator** | set roles, submit proposals, vote, orchestrate |
| **Proposer** | submit proposals |
| **Validator** | vote on proposals |
| **Executor** | claim and execute tasks |

Role gates are enforced server-side: an Executor submitting a proposal or a
non-Validator casting a vote gets `not_a_validator` / role rejection.

## Consensus (proposals)

1. `POST /agent-swarm/proposals` — the Coordinator submits a plan of action.
   `min_votes` defaults to the number of **live validators** (≥ 1).
2. `POST /agent-swarm/proposals/vote {proposal_id, agent, accept, reason}` —
   Validators (and the Coordinator) cast ACCEPT/REJECT votes with reasons.
3. Once `votes ≥ min_votes`, the decision is **strict majority of cast
   votes** (`accepts * 2 > votes.len()`); ties reject.
4. Accepted proposals are **materialized** into executable tasks via
   `POST /agent-swarm/proposals/materialize`; stalled proposals (quorum never
   reached) are closed by `POST /agent-swarm/proposals/close-stalled`.

## Verified E2E sequence (release server, integration report)

From the integration report (docs/OPT_INTEGRACION.md, §Tests / E2E), all
exercised over real HTTP against a release build:

- Dependencies block a task until ready; **CAS** returns `claimed` to one
  agent and `taken` to the loser.
- A failed task is re-claimable after fallback, and the **failure lesson
  becomes visible in another agent's preflight** (`preflight_shared` reads
  the whole swarm's memory).
- Consensus: a proposal REJECTed with reasons → counter-proposal ACCEPTED
  2/2 → materialized into a task → an Executor completes it (the Executor
  cannot propose — role gate) → lesson lands in the shared memory.

## Why CAS + epoch instead of a lock

The queue is in-process (single server), but claims must stay correct under
concurrent HTTP handlers: the CAS guarantees exactly-one-winner without a
coarse lock over the whole swarm state, and epoch-based fallback makes
recovery after agent death deterministic (highest epoch wins) instead of
race-dependent.
