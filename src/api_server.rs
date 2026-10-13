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
) -> Result<Json<crate::agent_r::PreflightResult>, StatusCode> {
    let mut agent = state.agent_r.lock().await;
    let tags: Vec<&str> = req.context_tags.iter().map(|s| s.as_str()).collect();
    let result = agent.preflight(&req.operation, &tags);
    Ok(Json(result))
}

#[derive(Deserialize)]
struct ValidateOutputRequest {
    operation: String,
    output: serde_json::Value,
}

async fn validate_output(
    State(state): State<AppState>,
    Json(req): Json<ValidateOutputRequest>,
) -> Result<Json<crate::agent_r::ValidationResult>, StatusCode> {
    let mut agent = state.agent_r.lock().await;
    let result = agent.validate_output(&req.operation, &req.output);
    Ok(Json(result))
}

async fn agent_r_stats(
    State(state): State<AppState>,
) -> Result<Json<crate::agent_r::AgentRStats>, StatusCode> {
    let agent = state.agent_r.lock().await;
    Ok(Json(agent.stats().clone()))
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

    let state = AppState {
        store: Arc::new(Mutex::new(store)),
        next_id: Arc::new(Mutex::new(0)),
        agent_r: Arc::new(Mutex::new(agent_r)),
        index_dirty: Arc::new(tokio::sync::Notify::new()),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/status", post(status))
        .route("/store", post(store_memory))
        .route("/search", post(search_memories))
        .route("/rebuild-index", post(rebuild_index))
        .route("/agent-r/record-error", post(record_error))
        .route("/agent-r/preflight", post(preflight_check))
        .route("/agent-r/validate", post(validate_output))
        .route("/agent-r/stats", get(agent_r_stats))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("127.0.0.1:{}", port);
    println!("M2M server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    Ok(axum::serve(listener, app).await?)
}
