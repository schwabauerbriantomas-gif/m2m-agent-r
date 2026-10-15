//! Minimal HTTP API server for Python bindings
//!
//! Endpoints:
//!   POST /store   - Store a memory (text + optional metadata)
//!   POST /search  - Search memories (query text, top-k)
//!   POST /status  - Get store stats
//!   GET  /health  - Health check
//!   Agent-R:
//!     POST /agent-r/record-error  - Record an error for learning
//!     POST /agent-r/preflight     - Pre-flight check before an operation
//!     POST /agent-r/validate      - Validate output against error patterns
//!     GET  /agent-r/stats         - Get Agent-R statistics

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::splats::SplatStore;
use crate::config::M2MConfig;
use crate::agent_r::{AgentR, ErrorCategory, ErrorSeverity, WriteOutcome};

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<Mutex<SplatStore>>,
    pub next_id: Arc<Mutex<usize>>,
    pub agent_r: Arc<Mutex<AgentR>>,
    pub index_dirty: Arc<tokio::sync::Notify>,
    /// Coordinación multi-agente (OPT-SWARM): cola con claim atómico +
    /// lecciones compartidas via agent_r.
    pub swarm: std::sync::Arc<crate::agent_swarm::AgentSwarm>,
}

#[derive(Deserialize)]
pub struct StoreRequest {
    pub text: String,
    pub category: Option<String>,
    pub id: Option<String>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    pub top_k: Option<usize>,
}

#[derive(Serialize)]
pub struct StoreResponse {
    pub id: String,
    pub status: String,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub index: usize,
    pub score: f32,
    pub metadata: Option<String>,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub n_active: usize,
    pub max_splats: usize,
    pub dimension: usize,
    pub has_hnsw: bool,
    pub has_lsh: bool,
    pub has_quantization: bool,
    pub has_semantic_memory: bool,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        version: "2.1.0".into(),
    })
}

async fn status(State(state): State<AppState>) -> Json<StatusResponse> {
    let store = state.store.lock().await;
    let stats = store.get_statistics();
    Json(StatusResponse {
        n_active: store.n_active(),
        max_splats: store.max_splats(),
        dimension: stats.embedding_dim,
        has_hnsw: store.has_hnsw(),
        has_lsh: store.has_lsh(),
        has_quantization: store.has_quantization(),
        has_semantic_memory: store.has_semantic_memory(),
    })
}

async fn store_memory(
    State(state): State<AppState>,
    Json(req): Json<StoreRequest>,
) -> Result<(StatusCode, Json<StoreResponse>), (StatusCode, String)> {
    let dim = {
        let store = state.store.lock().await;
        store.get_statistics().embedding_dim
    };
    
    let embedding = req.embedding.unwrap_or_else(|| {
        simple_hash_embedding(&req.text, dim)
    });
    
    // Hold a single lock for the entire insert to prevent TOCTOU races
    let mut store = state.store.lock().await;
    let arr = Array2::from_shape_vec((1, dim), embedding.clone())
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Bad embedding shape: {}", e)))?;
    
    let added = store.add_splat(&arr);
    if !added {
        return Err((StatusCode::INSUFFICIENT_STORAGE, "Store full".into()));
    }
    
    // Mark index as dirty — will be rebuilt lazily or on demand
    drop(store);
    state.index_dirty.notify_one();
    
    let mut id_counter = state.next_id.lock().await;
    *id_counter += 1;
    let id = req.id.unwrap_or_else(|| format!("mem_{}", *id_counter));
    
    Ok((StatusCode::OK, Json(StoreResponse {
        id,
        status: "stored".into(),
    })))
}

async fn rebuild_index(State(state): State<AppState>) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut store = state.store.lock().await;
    store.build_index();
    let stats = store.get_statistics();
    Ok(Json(serde_json::json!({
        "status": "rebuilt",
        "n_active": store.n_active(),
        "has_hnsw": store.has_hnsw(),
        "has_lsh": store.has_lsh(),
        "dimension": stats.embedding_dim,
    })))
}

async fn search_memories(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let store = state.store.lock().await;
    let dim = store.get_statistics().embedding_dim;
    let k = req.top_k.unwrap_or(10).min(store.n_active());
    
    if store.n_active() == 0 {
        return Ok(Json(SearchResponse { results: vec![] }));
    }
    
    let embedding = req.embedding.unwrap_or_else(|| {
        simple_hash_embedding(&req.query, dim)
    });
    
    let query = Array1::from_vec(embedding);
    let neighbors = store.find_neighbors(&query.view(), k);
    
    let results: Vec<SearchResult> = neighbors
        .into_iter()
        .map(|n| SearchResult {
            index: n.index,
            score: n.distance,
            metadata: None,
        })
        .collect();
    
    Ok(Json(SearchResponse { results }))
}

/// Deterministic hash-based pseudo-embedding for text.
/// NOT a real embedding - placeholder for structure.
/// Replace with actual model (DINOv2, BGE, etc.) in production.
fn simple_hash_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut result = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        i.hash(&mut hasher);
        let raw = hasher.finish() as f32;
        let val = (raw / u32::MAX as f32) * 2.0 - 1.0;
        result.push(val);
    }
    // Normalize
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in result.iter_mut() {
            *x /= norm;
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// Agent-R endpoints
// ═══════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct RecordErrorRequest {
    category: String,
    severity: String,
    operation: String,
    what: String,
    why: String,
    context_tags: Vec<String>,
    prevention: String,
}

// ══════════════════════════════════════════════════════════════════════
// Agent-Swarm: coordinación multi-agente (OPT-SWARM)
// ══════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct SwarmRegisterRequest {
    name: String,
    #[serde(default)]
    capabilities: Vec<String>,
    /// Rol inicial (bootstrap de confianza). Para cambios en caliente,
    /// /agent-swarm/roles (sólo Coordinator).
    #[serde(default)]
    role: Option<String>,
}

#[derive(Serialize)]
struct SwarmRegisterResponse {
    name: String,
    ok: bool,
    role: Option<crate::agent_swarm::SwarmRole>,
}

async fn swarm_register(
    State(state): State<AppState>,
    Json(req): Json<SwarmRegisterRequest>,
) -> Json<SwarmRegisterResponse> {
    let caps: Vec<&str> = req.capabilities.iter().map(|s| s.as_str()).collect();
    let role = req.role.as_deref().and_then(crate::agent_swarm::SwarmRole::from_str);
    let name = state.swarm.register_agent(&req.name, &caps, role);
    let agents_role = state
        .swarm
        .live_agents()
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.role);
    Json(SwarmRegisterResponse { name, ok: true, role: agents_role })
}

#[derive(Deserialize)]
struct SwarmHeartbeatRequest {
    name: String,
}

#[derive(Serialize)]
struct SwarmHeartbeatResponse {
    ok: bool,
}

async fn swarm_heartbeat(
    State(state): State<AppState>,
    Json(req): Json<SwarmHeartbeatRequest>,
) -> Json<SwarmHeartbeatResponse> {
    Json(SwarmHeartbeatResponse { ok: state.swarm.heartbeat(&req.name) })
}

#[derive(Deserialize)]
struct SwarmSubmitTaskRequest {
    operation: String,
    description: String,
    #[serde(default)]
    depends_on: Vec<u64>,
    #[serde(default)]
    preferred_agent: Option<String>,
}

#[derive(Serialize)]
struct SwarmSubmitTaskResponse {
    task_id: u64,
}

async fn swarm_submit_task(
    State(state): State<AppState>,
    Json(req): Json<SwarmSubmitTaskRequest>,
) -> Json<SwarmSubmitTaskResponse> {
    let task_id = state.swarm.submit_task(
        &req.operation,
        &req.description,
        req.depends_on,
        req.preferred_agent.as_deref(),
    );
    Json(SwarmSubmitTaskResponse { task_id })
}

#[derive(Deserialize)]
struct SwarmClaimRequest {
    task_id: u64,
    agent: String,
}

#[derive(Serialize)]
struct SwarmClaimResponse {
    result: crate::agent_swarm::ClaimResult,
}

async fn swarm_claim_task(
    State(state): State<AppState>,
    Json(req): Json<SwarmClaimRequest>,
) -> Json<SwarmClaimResponse> {
    let result = state.swarm.claim_task(req.task_id, &req.agent);
    Json(SwarmClaimResponse { result })
}

#[derive(Deserialize)]
struct SwarmCompleteRequest {
    task_id: u64,
    agent: String,
    result: String,
}

#[derive(Serialize)]
struct SwarmCompleteResponse {
    ok: bool,
}

async fn swarm_complete_task(
    State(state): State<AppState>,
    Json(req): Json<SwarmCompleteRequest>,
) -> Json<SwarmCompleteResponse> {
    let mut agent_r = state.agent_r.lock().await;
    Json(SwarmCompleteResponse {
        ok: state.swarm.complete_task(&mut agent_r, req.task_id, &req.agent, &req.result),
    })
}

#[derive(Deserialize)]
struct SwarmFailRequest {
    task_id: u64,
    agent: String,
    reason: String,
}

#[derive(Serialize)]
struct SwarmFailResponse {
    ok: bool,
}

async fn swarm_fail_task(
    State(state): State<AppState>,
    Json(req): Json<SwarmFailRequest>,
) -> Json<SwarmFailResponse> {
    let mut agent_r = state.agent_r.lock().await;
    Json(SwarmFailResponse {
        ok: state.swarm.fail_task(&mut agent_r, req.task_id, &req.agent, &req.reason),
    })
}

#[derive(Deserialize)]
struct SwarmResultQuery {
    task_id: u64,
}

async fn swarm_task_result(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<SwarmResultQuery>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "result": state.swarm.task_result(q.task_id) }))
}

#[derive(Deserialize)]
struct SwarmBarrierRequest {
    participants: usize,
}

#[derive(Serialize)]
struct SwarmBarrierResponse {
    barrier_id: u64,
}

async fn swarm_barrier(
    State(state): State<AppState>,
    Json(req): Json<SwarmBarrierRequest>,
) -> Json<SwarmBarrierResponse> {
    let barrier_id = state.swarm.create_barrier(req.participants.max(1));
    Json(SwarmBarrierResponse { barrier_id })
}

#[derive(Deserialize)]
struct SwarmBarrierArriveRequest {
    barrier_id: u64,
}

#[derive(Serialize)]
struct SwarmBarrierArriveResponse {
    /// true cuando TODOS los participantes llegaron (barrera abierta).
    open: Option<bool>,
}

async fn swarm_barrier_arrive(
    State(state): State<AppState>,
    Json(req): Json<SwarmBarrierArriveRequest>,
) -> Json<SwarmBarrierArriveResponse> {
    Json(SwarmBarrierArriveResponse { open: state.swarm.barrier_arrive(req.barrier_id) })
}

#[derive(Deserialize)]
struct SwarmPreflightRequest {
    operation: String,
    #[serde(default)]
    context_tags: Vec<String>,
    /// Texto del estado actual (opcional): activa el híbrido y graba la
    /// regla top-1 en la task para el self-tuning por outcome.
    #[serde(default)]
    situation: Option<String>,
    /// Task del swarm ya sometida cuyo preflight es éste (grabar regla).
    #[serde(default)]
    task_id: Option<u64>,
}

async fn swarm_preflight(
    State(state): State<AppState>,
    Json(req): Json<SwarmPreflightRequest>,
) -> Json<serde_json::Value> {
    let mut agent_r = state.agent_r.lock().await;
    let tags: Vec<&str> = req.context_tags.iter().map(|s| s.as_str()).collect();
    let mut out = match &req.situation {
        Some(s) if !s.trim().is_empty() => {
            let mut r = agent_r.preflight_v2_hybrid(&req.operation, &tags, s);
            serde_json::json!({
                "proceed": r.proceed,
                "relevant_errors": r.relevant_errors,
                "warnings": r.warnings,
                "precautions": r.precautions,
                "top_rules": r.top_rules,
                "suggestion": r.suggestion,
                "veto_reason": r.veto_reason,
                "hybrid": true,
                "w_content": agent_r.w_content(),
            })
        }
        _ => state.swarm.preflight_shared(&mut agent_r, &req.operation, &tags),
    };
    // Self-tuning: grabar la regla top-1 en la task para que el outcome
    // (complete/fail) la use como etiqueta — SOLO si hay una regla real
    // (score >= 10 = exact-match de operación; semillas/genericidad no
    // cuentan como predicción).
    if let Some(tid) = req.task_id {
        let top = out.get("top_rules").and_then(|t| t.get(0));
        let score_ok = top.and_then(|r| r.get("score")).and_then(|v| v.as_f64()).unwrap_or(0.0) >= 10.0;
        if score_ok {
            if let Some(rule) = top.and_then(|r| r.get("id")).and_then(|v| v.as_u64()) {
                state.swarm.set_task_rule(tid, rule);
                out["task_rule_recorded"] = serde_json::json!(rule);
            }
        }
    }
    Json(out)
}

async fn swarm_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let agent_r = state.agent_r.lock().await;
    Json(state.swarm.status(&agent_r))
}

// ── Acuerdo con roles: propuestas + consenso (API) ─────────────────────

#[derive(Deserialize)]
struct SwarmSetRoleRequest {
    coordinator: String,
    agent: String,
    role: String,
}

#[derive(Serialize)]
struct SwarmSetRoleResponse {
    ok: bool,
    role: Option<crate::agent_swarm::SwarmRole>,
}

async fn swarm_set_role(
    State(state): State<AppState>,
    Json(req): Json<SwarmSetRoleRequest>,
) -> Json<SwarmSetRoleResponse> {
    let role = crate::agent_swarm::SwarmRole::from_str(&req.role)
        .map(|r| state.swarm.set_role(&req.coordinator, &req.agent, r))
        .unwrap_or(None);
    Json(SwarmSetRoleResponse { ok: role.is_some(), role })
}

#[derive(Deserialize)]
struct SwarmSubmitProposalRequest {
    proposer: String,
    title: String,
    body: String,
    operation: String,
    #[serde(default)]
    min_votes: Option<usize>,
}

#[derive(Serialize)]
struct SwarmSubmitProposalResponse {
    ok: bool,
    proposal_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn swarm_submit_proposal(
    State(state): State<AppState>,
    Json(req): Json<SwarmSubmitProposalRequest>,
) -> Json<SwarmSubmitProposalResponse> {
    match state.swarm.submit_proposal(
        &req.proposer, &req.title, &req.body, &req.operation, req.min_votes,
    ) {
        Ok(id) => Json(SwarmSubmitProposalResponse { ok: true, proposal_id: Some(id), error: None }),
        Err(e) => Json(SwarmSubmitProposalResponse { ok: false, proposal_id: None, error: Some(e) }),
    }
}

#[derive(Deserialize)]
struct SwarmVoteRequest {
    proposal_id: u64,
    voter: String,
    /// ACCEPT | REJECT
    decision: String,
    reason: String,
}

#[derive(Serialize)]
struct SwarmVoteResponse {
    result: crate::agent_swarm::VoteResult,
}

async fn swarm_vote(
    State(state): State<AppState>,
    Json(req): Json<SwarmVoteRequest>,
) -> Json<SwarmVoteResponse> {
    let result = state
        .swarm
        .vote_proposal(req.proposal_id, &req.voter, req.decision.eq_ignore_ascii_case("accept"), &req.reason);
    Json(SwarmVoteResponse { result })
}

#[derive(Deserialize)]
struct SwarmMaterializeRequest {
    proposal_id: u64,
}

#[derive(Serialize)]
struct SwarmMaterializeResponse {
    ok: bool,
    task_id: u64,
    state: Option<crate::agent_swarm::ProposalState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn swarm_materialize(
    State(state): State<AppState>,
    Json(req): Json<SwarmMaterializeRequest>,
) -> Json<SwarmMaterializeResponse> {
    match state.swarm.materialize_proposal(req.proposal_id) {
        Ok((tid, st)) => Json(SwarmMaterializeResponse {
            ok: tid != 0,
            task_id: tid,
            state: Some(st),
            error: None,
        }),
        Err(e) => Json(SwarmMaterializeResponse {
            ok: false,
            task_id: 0,
            state: None,
            error: Some(e),
        }),
    }
}

#[derive(Deserialize)]
struct SwarmProposalQuery {
    proposal_id: u64,
}

async fn swarm_proposal_status(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<SwarmProposalQuery>,
) -> Json<serde_json::Value> {
    Json(state
        .swarm
        .proposal_status(q.proposal_id)
        .unwrap_or(serde_json::json!({"error": "propuesta inexistente"})))
}

#[derive(Serialize)]
struct SwarmCloseStalledResponse {
    closed: usize,
}

async fn swarm_close_stalled(State(state): State<AppState>) -> Json<SwarmCloseStalledResponse> {
    Json(SwarmCloseStalledResponse { closed: state.swarm.close_stalled() })
}

#[derive(Serialize)]
struct RecordErrorResponse {
    id: u64,
    /// Dedup outcome: "Add" (new incident), "Update" (merged new info into
    /// an existing incident), "Noop" (duplicate, bookkeeping only). (OPT-O4)
    outcome: WriteOutcome,
    /// Recurrence count of the incident after this write (0 on Add). (OPT-O4)
    recurrence: u32,
}

fn parse_category(s: &str) -> ErrorCategory {
    match s {
        "circular_reasoning" | "CircularReasoning" => ErrorCategory::CircularReasoning,
        "output_drift" | "OutputDrift" => ErrorCategory::OutputDrift,
        "fabrication" | "Fabrication" => ErrorCategory::Fabrication,
        "bad_assumption" | "BadAssumption" => ErrorCategory::BadAssumption,
        "skipped_validation" | "SkippedValidation" => ErrorCategory::SkippedValidation,
        "timeout" | "Timeout" => ErrorCategory::Timeout,
        "state_corruption" | "StateCorruption" => ErrorCategory::StateCorruption,
        "recurring_pattern" | "RecurringPattern" => ErrorCategory::RecurringPattern,
        _ => ErrorCategory::Other,
    }
}

fn parse_severity(s: &str) -> ErrorSeverity {
    match s {
        "critical" | "Critical" => ErrorSeverity::Critical,
        "error" | "Error" => ErrorSeverity::Error,
        _ => ErrorSeverity::Warning,
    }
}

async fn record_error(
    State(state): State<AppState>,
    Json(req): Json<RecordErrorRequest>,
) -> Result<Json<RecordErrorResponse>, StatusCode> {
    let mut agent = state.agent_r.lock().await;
    let tags: Vec<&str> = req.context_tags.iter().map(|s| s.as_str()).collect();
    let (id, outcome, recurrence) = agent.record_error_dedup(
        parse_category(&req.category),
        parse_severity(&req.severity),
        &req.operation,
        &req.what,
        &req.why,
        &tags,
        &req.prevention,
    );
    Ok(Json(RecordErrorResponse { id, outcome, recurrence }))
}

#[derive(Deserialize)]
struct PreflightRequest {
    operation: String,
    context_tags: Vec<String>,
}

async fn preflight_check(
    State(state): State<AppState>,
    Json(req): Json<PreflightRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut agent = state.agent_r.lock().await;
    let tags: Vec<&str> = req.context_tags.iter().map(|s| s.as_str()).collect();
    // Preflight v2 (pendiente 2a): score compuesto + top-3 + sugerencia.
    // Campos v1 (proceed/warnings/precautions/relevant_errors) preservados.
    let result = agent.preflight_v2(&req.operation, &tags);
    Ok(Json(serde_json::json!({
        "proceed": result.proceed,
        "warnings": result.warnings,
        "precautions": result.precautions,
        "relevant_errors": result.relevant_errors,
        "top_rules": result.top_rules,
        "suggestion": result.suggestion,
        "veto_reason": result.veto_reason,
    })))
}

#[derive(Deserialize)]
struct PreflightHybridRequest {
    operation: String,
    #[serde(default)]
    context_tags: Vec<String>,
    /// Estado actual del caller (obs/comando/stderr): activa la señal de
    /// contenido (overlap binario IDF contra el `what` de las reglas).
    situation: String,
}

/// Preflight HÍBRIDO (bench 15-oct): selección global estructural+contenido.
/// Misma forma de respuesta que /agent-r/preflight + w_content vigente.
async fn preflight_hybrid(
    State(state): State<AppState>,
    Json(req): Json<PreflightHybridRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut agent = state.agent_r.lock().await;
    let tags: Vec<&str> = req.context_tags.iter().map(|s| s.as_str()).collect();
    let result = agent.preflight_v2_hybrid(&req.operation, &tags, &req.situation);
    Ok(Json(serde_json::json!({
        "proceed": result.proceed,
        "warnings": result.warnings,
        "precautions": result.precautions,
        "relevant_errors": result.relevant_errors,
        "top_rules": result.top_rules,
        "suggestion": result.suggestion,
        "veto_reason": result.veto_reason,
        "hybrid": true,
        "w_content": agent.w_content(),
        "tuning": { "hits": agent.tuning_stats().0, "misses": agent.tuning_stats().1 },
    })))
}

/// Flush sincrónico del snapshot de Agent-R (apagado controlado de los
/// agentes; el debounce escribe por su cuenta pero esto garantiza disco).
async fn agent_r_flush(State(state): State<AppState>) -> Json<serde_json::Value> {
    let agent = state.agent_r.lock().await;
    match agent.flush() {
        Ok(()) => Json(serde_json::json!({"ok": true, "flushed": true})),
        Err(e) => Json(serde_json::json!({"ok": false, "error": e})),
    }
}

#[derive(Deserialize)]
struct ValidateOutputRequest {
    operation: String,
    output: serde_json::Value,
    /// Evidencia programática del llamador para N2 (pendiente 2a):
    /// {"exit_code": 0, "stderr": "...", "diff": "...", "expected_exit_code": 0}.
    #[serde(default)]
    evidence: Option<serde_json::Value>,
}

async fn validate_output(
    State(state): State<AppState>,
    Json(req): Json<ValidateOutputRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut agent = state.agent_r.lock().await;
    // Validate v2 (pendiente 2a): veredictos N0-N3 + overall + suggestion.
    // Compat: valid/confidence mapeados desde overall; issues en formato legado.
    let r = agent.validate_output_v2_evidence(&req.operation, &req.output, req.evidence.as_ref());
    Ok(Json(serde_json::json!({
        "verdicts": {
            "n0_form": r.verdicts.n0_form,
            "n1_numeric_consistency": r.verdicts.n1_numeric_consistency,
            "n2_caller_evidence": r.verdicts.n2_caller_evidence,
            "n3_receipt": r.verdicts.n3_receipt,
        },
        "overall": r.overall,
        "suggestion": r.suggestion,
        "details": r.details,
        // ── compat campos viejos ──
        "valid": r.valid,
        "confidence": r.confidence,
        "issues": r.issues,
        "checks_run": r.checks_run,
        "elapsed_ms": r.elapsed_ms,
    })))
}

async fn agent_r_stats(
    State(state): State<AppState>,
) -> Result<Json<crate::agent_r::AgentRStats>, StatusCode> {
    let agent = state.agent_r.lock().await;
    Ok(Json(agent.stats().clone()))
}

// ============================================================================
// Receipts HMAC (rediseño v2, Pendiente 1 — N3). Handlers NUEVOS de P1:
// el ReceiptStore es file-backed e independiente del AppState, así que vive
// en un OnceLock global (dir según AGENT_R_RECEIPTS_DIR o default
// data/receipts). Los tests de endpoints lo apuntan a un tmpdir via env.
// ============================================================================

static RECEIPTS: std::sync::OnceLock<std::sync::Arc<crate::receipts::ReceiptStore>> =
    std::sync::OnceLock::new();

fn receipts_store() -> std::sync::Arc<crate::receipts::ReceiptStore> {
    RECEIPTS
        .get_or_init(|| {
            let dir = std::env::var("AGENT_R_RECEIPTS_DIR")
                .ok()
                .filter(|d| !d.is_empty())
                .unwrap_or_else(|| crate::receipts::DEFAULT_RECEIPTS_DIR.to_string());
            match crate::receipts::ReceiptStore::open(&dir) {
                Ok(s) => std::sync::Arc::new(s),
                // Sin disco disponible no caemos: receipts efímeros en RAM
                // (dir temporal). La clave sigue sin loggearse.
                Err(_) => {
                    let tmp = std::env::temp_dir().join(format!("receipts_ram_{}", std::process::id()));
                    std::sync::Arc::new(
                        crate::receipts::ReceiptStore::open(&tmp)
                            .expect("receipts en tmp debe poder abrirse"),
                    )
                }
            }
        })
        .clone()
}

#[derive(Deserialize)]
struct ReceiptsIssueRequest {
    payload: serde_json::Value,
}

async fn receipts_issue(
    Json(req): Json<ReceiptsIssueRequest>,
) -> Result<Json<crate::receipts::Receipt>, (StatusCode, String)> {
    let store = receipts_store();
    store
        .issue(&req.payload)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

#[derive(Deserialize)]
struct ReceiptsVerifyRequest {
    receipt_id: String,
    #[serde(default)]
    claim: Option<serde_json::Value>,
}

async fn receipts_verify(
    Json(req): Json<ReceiptsVerifyRequest>,
) -> Result<Json<crate::receipts::VerifyResult>, (StatusCode, String)> {
    let store = receipts_store();
    let result = match &req.claim {
        Some(claim) => store.verify_claim(&req.receipt_id, claim),
        None => store.verify_receipt(&req.receipt_id),
    };
    Ok(Json(result))
}

/// Flush de snapshots Agent-R (ruta agregada por P1 para el debounce de P2;
/// el handler delega en AgentR::flush que hoy es write-through inmediato).
async fn agent_r_flush(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let agent = state.agent_r.lock().await;
    match agent.flush() {
        Ok(()) => Ok(Json(serde_json::json!({"flushed": true}))),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

pub async fn run_server(port: u16) -> anyhow::Result<()> {
    let config = M2MConfig::default();
    let store = SplatStore::new(config);

    // Agent-R state: snapshot path from AGENT_R_STATE_PATH or default
    // "data/agent_r_state.json" relative to CWD. Load persisted state if the
    // file exists (memory survives restarts); otherwise AgentR seeds the 3
    // historical incidents and persists them (write-through from then on).
    // Empty string => ephemeral (RAM only, no disk writes). (OPT-O4)
    let agent_r = match std::env::var("AGENT_R_STATE_PATH") {
        Ok(ref p) if p.is_empty() => AgentR::new_ephemeral(),
        Ok(p) => AgentR::with_state_path(std::path::PathBuf::from(p)),
        Err(_) => AgentR::with_state_path(std::path::PathBuf::from(AgentR::DEFAULT_STATE_PATH)),
    };
    let agent_r_arc = Arc::new(Mutex::new(agent_r));

    let state = AppState {
        store: Arc::new(Mutex::new(store)),
        next_id: Arc::new(Mutex::new(0)),
        agent_r: agent_r_arc.clone(),
        index_dirty: Arc::new(tokio::sync::Notify::new()),
        swarm: std::sync::Arc::new(crate::agent_swarm::AgentSwarm::new()),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/status", post(status))
        .route("/store", post(store_memory))
        .route("/search", post(search_memories))
        .route("/rebuild-index", post(rebuild_index))
        .route("/agent-r/record-error", post(record_error))
        .route("/agent-r/preflight", post(preflight_check))
        .route("/agent-r/preflight-hybrid", post(preflight_hybrid))
        .route("/agent-r/flush", post(agent_r_flush))
        .route("/agent-r/validate", post(validate_output))
        .route("/agent-r/stats", get(agent_r_stats))
        .route("/agent-r/receipts/issue", post(receipts_issue))
        .route("/agent-r/receipts/verify", post(receipts_verify))
        .route("/agent-r/flush", post(agent_r_flush))
        .route("/agent-swarm/register", post(swarm_register))
        .route("/agent-swarm/heartbeat", post(swarm_heartbeat))
        .route("/agent-swarm/tasks", post(swarm_submit_task))
        .route("/agent-swarm/tasks/claim", post(swarm_claim_task))
        .route("/agent-swarm/tasks/complete", post(swarm_complete_task))
        .route("/agent-swarm/tasks/fail", post(swarm_fail_task))
        .route("/agent-swarm/tasks/result", get(swarm_task_result))
        .route("/agent-swarm/barrier", post(swarm_barrier))
        .route("/agent-swarm/barrier/arrive", post(swarm_barrier_arrive))
        .route("/agent-swarm/preflight", post(swarm_preflight))
        .route("/agent-swarm/status", get(swarm_status))
        .route("/agent-swarm/roles", post(swarm_set_role))
        .route("/agent-swarm/proposals", post(swarm_submit_proposal))
        .route("/agent-swarm/proposals/vote", post(swarm_vote))
        .route("/agent-swarm/proposals/materialize", post(swarm_materialize))
        .route("/agent-swarm/proposals/status", get(swarm_proposal_status))
        .route("/agent-swarm/proposals/close-stalled", post(swarm_close_stalled))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("127.0.0.1:{}", port);
    println!("M2M server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    Ok(axum::serve(listener, app).await?)
}

#[cfg(test)]
mod receipts_endpoint_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// HTTP/1.1 POST crudo sobre TcpStream (sin deps extra) -> body del response.
    async fn http_post(port: u16, path: &str, body: &str) -> String {
        let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{}", port))
            .await
            .expect("conectar al server de test");
        let req = format!(
            "POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            path,
            body.len(),
            body
        );
        stream.write_all(req.as_bytes()).await.unwrap();
        let mut raw = String::new();
        stream.read_to_string(&mut raw).await.unwrap();
        raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string()
    }

    #[tokio::test]
    async fn test_receipts_issue_verify_endpoints() {
        // Dir propio para el OnceLock del store (primer uso en este proceso).
        let dir = std::env::temp_dir().join(format!("receipts_ep_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("AGENT_R_RECEIPTS_DIR", &dir);

        // Router SOLO con las rutas nuevas de P1 (in-process, puerto asignado P1).
        let app = Router::new()
            .route("/agent-r/receipts/issue", post(receipts_issue))
            .route("/agent-r/receipts/verify", post(receipts_verify));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:18555").await.unwrap();
        let srv = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        // 1) issue
        let body = r#"{"payload":{"cmd":"echo hi","output":{"exit_code":0},"exit_code":0}}"#;
        let resp = http_post(18555, "/agent-r/receipts/issue", body).await;
        let v: serde_json::Value = serde_json::from_str(&resp).expect("issue responde JSON");
        let rid = v["id"].as_str().expect("id en receipt").to_string();
        assert!(v["sig_hex"].as_str().unwrap().len() == 64);
        assert!(dir.join(".key").exists(), "clave persistida en el dir de test");

        // 2) verify con claim correcto (exit_code=0 está en el payload firmado)
        let ok_body = format!(
            r#"{{"receipt_id":"{}","claim":{{"metric":"exit_code","value":0}}}}"#,
            rid
        );
        let resp2 = http_post(18555, "/agent-r/receipts/verify", &ok_body).await;
        assert!(resp2.contains("\"valid\":true"), "verify ok: {}", resp2);

        // 3) verify con claim FALSO (exit_code=1 no está) → invalid
        let bad_body = format!(
            r#"{{"receipt_id":"{}","claim":{{"metric":"exit_code","value":1}}}}"#,
            rid
        );
        let resp3 = http_post(18555, "/agent-r/receipts/verify", &bad_body).await;
        assert!(resp3.contains("\"valid\":false"), "claim falso: {}", resp3);

        // 4) verify de receipt inexistente → invalid
        let none_body = r#"{"receipt_id":"rcpt_nope_0","claim":{"metric":"x","value":1}}"#;
        let resp4 = http_post(18555, "/agent-r/receipts/verify", none_body).await;
        assert!(resp4.contains("\"valid\":false"), "ausente: {}", resp4);

        // 5) n3_check vía store (contrato del integrador) sobre el mismo receipt
        let store = crate::receipts::ReceiptStore::open(&dir).unwrap();
        assert_eq!(
            store.n3_check(&serde_json::json!({
                "receipt_id": rid,
                "claim": {"metric": "exit_code", "value": 0}
            })),
            crate::receipts::N3Verdict::Pass
        );

        srv.abort();
    }
}
