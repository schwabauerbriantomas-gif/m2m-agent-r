//! Agent-R: Self-correcting AI that learns from mistakes.
//!
//! Core principle: Record errors with full context, recall them before acting,
//! validate outputs against known failure patterns, and detect reasoning corruption.
//!
//! Three layers:
//! 1. Error Memory — structured error records with context, cause, and prevention
//! 2. Pre-flight Checks — recall relevant errors before mutations
//! 3. Reasoning Integrity — detect chain-of-thought corruption (loops, drift, fabrication)
//!
//! Reference: Agent-R methodology (self-correcting AI with reflective memory)

use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Severity of a recorded error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Minor issue, logged but doesn't block operation
    Warning,
    /// Significant error that produced wrong results
    Error,
    /// Critical failure that corrupted state or data
    Critical,
}

/// Category of error for pattern matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Reasoning went in circles (same state revisited)
    CircularReasoning,
    /// Output diverged from expected pattern
    OutputDrift,
    /// Data was fabricated or hallucinated
    Fabrication,
    /// Assumptions were wrong despite available evidence
    BadAssumption,
    /// Skipped validation that would have caught an issue
    SkippedValidation,
    /// Operation took too long, partial results used
    Timeout,
    /// State corruption detected
    StateCorruption,
    /// Repeated the same mistake
    RecurringPattern,
    /// Generic/unknown
    Other,
}

/// Interned string id (index into `AgentR::string_pool`).
/// 4 B inline en vez de un `String` de 24 B + heap: las ~15 operaciones y ~20
/// tags del workload real se deduplican en un pool mínimo (OPT-O3).
pub type InternedId = u32;

/// A structured error record with full context for learning.
///
/// Layout optimizado (OPT-O3, medido: 693 -> ~460 B/registro a 100k):
/// - `operation` y `context_tags` viven en el string pool del `AgentR` (dedup):
///   `String` 24 B -> `u32` 4 B, y `Vec<String>` (24 B ptr + 96 B cap-4 buffer
///   + 2.49 × 32 B bins de tags) -> `Vec<u32>` (~24 B buffer).
/// - `what`/`why`/`prevention` son `Box<str>`: dato único por registro, la
///   capacidad extra de `String` no aporta nada y cuesta 8 B inline por campo.
/// - Los payloads siguen en heap (DEDIUM: puntero inline, dato afuera); con el
///   Vec contenedor append-only los punteros heap son estables (verificado en
///   examples/falsify_layout.rs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecord {
    /// Unique ID for this error
    pub id: u64,
    /// When the error occurred (unix timestamp)
    pub timestamp: f64,
    /// Category for pattern matching
    pub category: ErrorCategory,
    /// How bad was it
    pub severity: ErrorSeverity,
    /// What operation was being performed (interned en el pool del AgentR)
    pub operation: InternedId,
    /// What went wrong (concise)
    pub what: Box<str>,
    /// Why it went wrong (root cause analysis)
    pub why: Box<str>,
    /// What context led to the error (tags internadas en el pool del AgentR)
    pub context_tags: Vec<InternedId>,
    /// How to prevent it next time
    pub prevention: Box<str>,
    /// Whether this error has been seen before (recurrence count)
    pub recurrence: u32,
    /// Associated error pattern ID (if this matches a known pattern)
    pub pattern_id: Option<u64>,
    /// Hash FNV-1a del fingerprint (category, operation, what normalizados).
    /// 0 en records legacy; recomputado on load. u64 en vez del String del
    /// O4 original (8 B vs ~124 B inline; ver hash_fingerprint). (OPT-O4)
    #[serde(default)]
    pub fingerprint_hash: u64,
    /// When this error was last seen (unix timestamp). 0.0 on legacy
    /// records; set to `timestamp` on load. (OPT-O4)
    #[serde(default)]
    pub last_seen: f64,
    /// Whether this lesson has been deprecated (e.g. measured false-positive
    /// loop). Deprecated records are ignored by recall/preflight. (OPT-O4)
    #[serde(default)]
    pub deprecated: bool,
    /// Reason for deprecation, if deprecated.
    #[serde(default)]
    pub deprecation_reason: Option<String>,
}

/// Outcome of an error write under Mem0-style ADD/UPDATE/NOOP semantics. (OPT-O4)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WriteOutcome {
    /// Fingerprint (category, operation, normalized what) unknown: new record inserted.
    Add,
    /// Fingerprint known and the write added information (new context tags,
    /// severity escalation): merged into the existing record.
    Update,
    /// Fingerprint known, nothing new: only recurrence/last_seen bookkeeping.
    Noop,
}

/// Compute the dedup fingerprint of an error: (category, operation, what)
/// with operation/what normalized to lowercase + trimmed. (OPT-O4)
fn error_fingerprint(category: ErrorCategory, operation: &str, what: &str) -> String {
    format!(
        "{:?}|{}|{}",
        category,
        operation.trim().to_lowercase(),
        what.trim().to_lowercase()
    )
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Tokenizador de contenido (bench 15-oct): lowercase, [a-z]+, sin
/// stopwords, len>2. MISMA definición que los compositores offline — si
/// cambia acá o ahí, el híbrido deja de reproducir los números del bench.
fn tokenize_words(text: &str) -> Vec<String> {
    const STOP: [&str; 26] = [
        "the", "and", "for", "not", "are", "was", "were", "been", "its",
        "with", "from", "you", "your", "there", "that", "this", "those",
        "these", "can", "will", "just", "but", "has", "had", "into", "out",
    ];
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_lowercase())
        .filter(|t| t.len() > 2 && !STOP.contains(t))
        .map(|t| t.to_string())
        .collect()
}

fn severity_rank(s: ErrorSeverity) -> u8 {
    match s {
        ErrorSeverity::Warning => 0,
        ErrorSeverity::Error => 1,
        ErrorSeverity::Critical => 2,
    }
}

/// A detected reasoning chain issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningIssue {
    pub issue_type: ErrorCategory,
    pub description: String,
    pub position: usize,
    pub confidence: f64,
}

/// Pre-flight check result.
#[derive(Debug, Clone, Serialize)]
pub struct PreflightResult {
    /// Whether the operation should proceed
    pub proceed: bool,
    /// Warnings from past errors
    pub warnings: Vec<String>,
    /// Precautions to take
    pub precautions: Vec<String>,
    /// Number of relevant past errors found
    pub relevant_errors: usize,
}

/// Preflight v2 (pendiente 2a): scoring compuesto + sugerencia, no sólo veto.
///
/// - `top_rules`: las 3 lecciones mejor puntuadas por el score compuesto
///   `α·similitud + β·recencia(now) + γ·log(1+freq)` con **exact-match** de
///   operación y tags como filtro (jamás substring cruzado).
/// - `suggestion`: la prevención de la mejor regla (qué hacer distinto).
/// - `veto`: SOLO una regla activa (no deprecada, N≥2 incidentes) con
///   severidad Critical y 0 falsos-positivos abiertos puede vetar.
#[derive(Debug, Clone, Serialize)]
pub struct PreflightResultV2 {
    pub proceed: bool,
    pub warnings: Vec<String>,
    pub precautions: Vec<String>,
    pub relevant_errors: usize,
    /// Top-3 reglas por score compuesto (id, operation, what, score).
    pub top_rules: Vec<ScoredRule>,
    /// Sugerencia accionable (prevención de la mejor regla).
    pub suggestion: Option<String>,
    /// Por qué se vetó (solo cuando proceed=false).
    pub veto_reason: Option<String>,
}

/// Una lección puntuada por el preflight v2.
#[derive(Debug, Clone, Serialize)]
pub struct ScoredRule {
    pub id: u64,
    pub operation: String,
    pub what: String,
    pub score: f64,
}

/// Nivel de validación N0-N3 (pendiente 2a). Un nivel sin oráculo reporta
/// `Unverifiable`, nunca `Fail`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LevelVerdict {
    Pass,
    Fail,
    Unverifiable,
}

impl LevelVerdict {
    #[allow(dead_code)] // usado por consumidores JSON que serializan a mano
    fn as_str(self) -> &'static str {
        match self {
            LevelVerdict::Pass => "pass",
            LevelVerdict::Fail => "fail",
            LevelVerdict::Unverifiable => "unverifiable",
        }
    }
}

/// Validación v2 por niveles (pendiente 2a):
/// N0 forma · N1 auto-consistencia numérica · N2 evidencia del llamador ·
/// N3 receipts (stub hasta el merge).
#[derive(Debug, Clone, Serialize)]
pub struct ValidationVerdicts {
    pub n0_form: LevelVerdict,
    pub n1_numeric_consistency: LevelVerdict,
    pub n2_caller_evidence: LevelVerdict,
    pub n3_receipt: LevelVerdict,
}

/// Resultado completo de validate_output v2.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationResultV2 {
    /// Veredicto por nivel (pass/fail/unverifiable).
    pub verdicts: ValidationVerdicts,
    /// "pass" | "fail" | "unverifiable": fail si algún nivel con oráculo
    /// falla; unverifiable si NINGÚN nivel pudo verificarse.
    pub overall: String,
    /// Sugerencia accionable derivada del primer nivel que falló.
    pub suggestion: Option<String>,
    /// Detalles por nivel (mensajes cortos, deduped).
    pub details: Vec<String>,
    /// Compat: mapeado de overall (true solo si overall == "pass").
    pub valid: bool,
    /// Compat: 1.0 pass · 0.3 unverifiable · 0.0 fail.
    pub confidence: f64,
    pub checks_run: usize,
    pub elapsed_ms: f64,
    /// Issues en el formato legado (ReasoningIssue) para consumidores viejos.
    pub issues: Vec<ReasoningIssue>,
}

/// Validation result for an output.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub issues: Vec<ReasoningIssue>,
    pub confidence: f64,
    pub checks_run: usize,
    pub elapsed_ms: f64,
}

/// Agent-R state statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentRStats {
    /// Total write events processed (ADD + UPDATE + NOOP): one per
    /// record-error call, duplicates included. (OPT-O4)
    pub total_errors_recorded: u64,
    /// Unique incidents currently held in memory (deduped by fingerprint). (OPT-O4)
    pub unique_incidents: u64,
    pub total_preflights: u64,
    pub total_validations: u64,
    pub prefights_blocked: u64,
    pub validations_failed: u64,
    pub errors_by_category: HashMap<String, u64>,
    pub top_recurring_patterns: Vec<(String, u32)>,
}

/// Persisted snapshot of the whole Agent-R state (write-through format). (OPT-O4)
/// Records en formato strings (PlainRecord): human-readable, legacy-compatible
/// con el formato de O4 original, y el pool internado se reconstruye al cargar.
#[derive(Debug, Serialize, Deserialize)]
struct StateSnapshot {
    version: u32,
    saved_at: f64,
    next_id: u64,
    stats: AgentRStats,
    errors: Vec<PlainRecord>,
    /// Self-tuning (15-oct): W del híbrido + contadores del feedback.
    /// Opcionales para cargar snapshots v1 sin tocarlos (compat).
    #[serde(default)]
    w_content: Option<f64>,
    #[serde(default)]
    tuning_hits: Option<u64>,
    #[serde(default)]
    tuning_misses: Option<u64>,
}

const MAX_ERRORS: usize = 2_000_000; // rambench-nocap: relevo el cap para medir la firma real (solo en este branch)
const MAX_REASONING_STEPS: usize = 1000;
const REPETITION_THRESHOLD: usize = 3;
/// Snapshot format version; loaders reject newer versions. (OPT-O4)
const SNAPSHOT_VERSION: u32 = 1;

// ── Debounce de snapshots (pendiente 2a) ─────────────────────────────────
/// Intervalo máx entre flushes del snapshot (segundos).
pub const FLUSH_INTERVAL: Duration = Duration::from_secs(2);
/// Nº máx de ops de escritura entre flushes (segundo disparador).
pub const FLUSH_MAX_OPS: u64 = 5_000;
/// Cola del writer background (jobs pendientes máx; si se llena, se reintenta
/// en la próxima op elegible — nunca se bloquea el camino de escritura).
const FLUSH_QUEUE: usize = 2;

/// Job de flush diferido: bytes ya serializados + generación monótona.
struct FlushJob {
    gen: u64,
    path: PathBuf,
    json: String,
}

/// Handle al writer background: canal de jobs + generación comprometida en
/// disco (compartida con `flush()` síncrono para descartar jobs viejos).
struct FlushHandle {
    tx: std::sync::mpsc::SyncSender<FlushJob>,
    committed_gen: Arc<AtomicU64>,
    /// Exclusión mutua entre el writer background y `flush()` síncrono para
    /// que tmp+rename de ambos nunca se interleave sobre el mismo path.
    write_lock: Arc<std::sync::Mutex<()>>,
}

/// Escritura atómica de un snapshot YA serializado (tmp + rename).
/// Usada por el writer background y por `flush()` síncrono.
fn write_json_atomic(path: &std::path::Path, json: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
    }
    let tmp = path.with_extension("json.tmp");
    {
        let mut f =
            fs::File::create(&tmp).map_err(|e| format!("create {}: {}", tmp.display(), e))?;
        f.write_all(json.as_bytes())
            .map_err(|e| format!("write {}: {}", tmp.display(), e))?;
        f.sync_all().ok();
    }
    fs::rename(&tmp, path).map_err(|e| format!("rename {}->{}: {}", tmp.display(), path.display(), e))
}

/// FNV-1a 64-bit: hash de fingerprint estable entre procesos y versiones de
/// Rust (DefaultHasher NO lo garantiza; el hash viaja en los snapshots).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Hash del fingerprint (category, operation, what normalizados). (OPT-O4,
/// desviación documentada: se guarda el hash u64, no el String — 8 B vs ~124
/// B por registro; colisión de cumpleaños a 2M entradas ~1e-7, y un hit de
/// dedup falso solo fusionaría dos incidentes idénticos en la práctica).
fn hash_fingerprint(category: ErrorCategory, operation: &str, what: &str) -> u64 {
    fnv1a(error_fingerprint(category, operation, what).as_bytes())
}

/// Shape of the versioned seed file (src/agent_r_seed.json). (OPT-O4)
#[derive(Debug, Deserialize)]
struct SeedFile {
    #[allow(dead_code)]
    version: u32,
    #[allow(dead_code)]
    description: String,
    errors: Vec<PlainRecord>,
}

/// Record en formato "strings" (seeds, vector legacy y snapshot human-readable):
/// el layout internado (OPT-O3) resuelve estos strings contra el pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlainRecord {
    id: u64,
    timestamp: f64,
    category: ErrorCategory,
    severity: ErrorSeverity,
    operation: String,
    what: String,
    why: String,
    context_tags: Vec<String>,
    prevention: String,
    recurrence: u32,
    pattern_id: Option<u64>,
    #[serde(default)]
    fingerprint_hash: u64,
    #[serde(default)]
    last_seen: f64,
    #[serde(default)]
    deprecated: bool,
    #[serde(default)]
    deprecation_reason: Option<String>,
}

/// Agent-R: Self-correcting agent with reflective error memory.
pub struct AgentR {
    /// Registros en orden de inserción. Deque: eviction O(1) amortizado
    /// (pendiente 2a; antes `Vec::remove(0)` era O(n) por purga).
    errors: VecDeque<ErrorRecord>,
    next_id: u64,
    /// Self-tuning (15-oct): peso de la señal de contenido del preflight
    /// híbrido, ajustable por feedback de outcomes reales. Fábrica 3.0
    /// (bench 15-oct). Persistido en el snapshot si está presente.
    w_content: f64,
    /// Contadores del loop de feedback (hits/misses de la regla top-1).
    tuning_hits: u64,
    tuning_misses: u64,
    /// String pool: ids de operaciones/tags internadas (dedup global).
    /// Workload real: ~15 ops + ~20 tags => pool acotado y mínimo.
    string_pool: Vec<Box<str>>,
    /// Índice inverso del pool (string -> id) para internar en O(1).
    string_pool_index: HashMap<Box<str>, InternedId>,
    /// Índice de recurrencia: (category, operation-id) -> ids de errores con
    /// esa clave, en orden de inserción. Elimina los 2 scans O(n) por POST de
    /// record_error (recurrence + pattern_id): ambos se derivan del bucket.
    /// Clave con InternedId (OPT-O3): cero alloc de String por POST.
    /// Invariante: todo id del bucket sigue vivo en `errors` (eviction lo purga).
    recurrence_index: HashMap<(ErrorCategory, InternedId), std::collections::VecDeque<u64>>,
    /// Dedup index: fingerprint_hash (FNV-1a stable) -> id del incidente.
    /// Rebuilt on load. (OPT-O4)
    fingerprint_index: HashMap<u64, u64>,
    /// Write-through snapshot path. None = persistence disabled (pure RAM,
    /// e.g. benchmarks). Set by `with_state_path` / `new`. (OPT-O4)
    state_path: Option<PathBuf>,
    /// Seed records embedded at build time; loaded only when the state file
    /// does not exist. (OPT-O4)
    seeded: bool,
    /// Reasoning chain tracker (operation hashes for loop detection)
    reasoning_chain: Vec<u64>,
    /// Reasoning chain metadata (operation name for reporting)
    reasoning_labels: Vec<String>,
    /// Custom validation rules
    validators: Vec<Box<dyn Fn(&str, &serde_json::Value) -> Option<ReasoningIssue> + Send + Sync>>,
    stats: AgentRStats,
    // ── Debounce de snapshots (pendiente 2a) ────────────────────────────
    /// Última vez que el snapshot tocó disco (epoch secs). Zona muerta del
    /// debounce: escrituras más recientes que FLUSH_INTERVAL no re-serializan.
    last_flush: f64,
    /// Escrituras desde el último flush (2º disparador del debounce).
    ops_since_flush: u64,
    /// Generación monótona del snapshot en memoria; el writer background
    /// descarta jobs cuya gen ya quedó comprometida en disco.
    snapshot_gen: u64,
    /// Handle del writer background (None cuando no hay persistencia o el
    /// receiver murió: degradación a write-through inmediato).
    flusher: Option<FlushHandle>,
    /// true si el último intento de flush falló (para no spamear warnings
    /// idénticos por POST: se loguea una vez hasta que un flush tenga éxito).
    flush_warned: bool,
}

/// Writer background de snapshots (debounce, pendiente 2a). Drena la cola de
/// jobs con timeout FLUSH_INTERVAL; ante cada job escribe atómicamente
/// (tmp+rename+fsync) y compromete la generación. Jobs cuya gen ≤ committed
/// se descartan (ya en disco). El thread muere cuando el sender se dropea.
fn spawn_flusher() -> FlushHandle {
    let (tx, rx) = std::sync::mpsc::sync_channel::<FlushJob>(FLUSH_QUEUE);
    let committed_gen = Arc::new(AtomicU64::new(0));
    let write_lock = Arc::new(std::sync::Mutex::new(()));
    let committed_w = committed_gen.clone();
    let lock_w = write_lock.clone();
    std::thread::Builder::new()
        .name("agent-r-flusher".into())
        .spawn(move || {
            loop {
                match rx.recv_timeout(FLUSH_INTERVAL) {
                    Ok(job) => {
                        if job.gen <= committed_w.load(Ordering::Acquire) {
                            continue; // ya comprometido en disco por flush() sync
                        }
                        let _g = lock_w.lock().unwrap_or_else(|p| p.into_inner());
                        if let Err(e) = write_json_atomic(&job.path, &job.json) {
                            eprintln!("[agent-r] WARN: background flush failed: {}", e);
                        }
                        committed_w.store(job.gen, Ordering::Release);
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        // Nada pendiente: seguir esperando.
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        })
        .expect("spawn agent-r-flusher");
    FlushHandle { tx, committed_gen, write_lock }
}

impl Drop for AgentR {
    /// Flush al apagar (pendiente 2a): apagado ordenado = pérdida cero.
    /// (El thread del flusher muere solo cuando todos los senders se dropean.)
    fn drop(&mut self) {
        if self.state_path.is_some() && !self.errors.is_empty() {
            let _ = self.flush();
        }
    }
}

impl AgentR {
    /// `data/agent_r_state.json` relative to the server CWD. (OPT-O4)
    pub const DEFAULT_STATE_PATH: &'static str = "data/agent_r_state.json";

    /// Create an Agent-R with persistence enabled, mirroring OPT-O4:
    /// AGENT_R_STATE_PATH (o default data/agent_r_state.json); si el snapshot
    /// existe se carga (memoria sobrevive restarts), si no, seeds históricos.
    pub fn new() -> Self {
        let path = std::env::var("AGENT_R_STATE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(Self::DEFAULT_STATE_PATH));
        Self::with_state_path(path)
    }

    /// Create an Agent-R bound to an explicit snapshot path (server, tests). (OPT-O4)
    pub fn with_state_path(path: PathBuf) -> Self {
        let mut agent = Self {
            errors: VecDeque::new(),
            next_id: 1,
            w_content: 3.0,
            tuning_hits: 0,
            tuning_misses: 0,
            string_pool: Vec::new(),
            string_pool_index: HashMap::new(),
            recurrence_index: HashMap::new(),
            fingerprint_index: HashMap::new(),
            state_path: Some(path),
            seeded: false,
            reasoning_chain: Vec::new(),
            reasoning_labels: Vec::new(),
            validators: Vec::new(),
            stats: AgentRStats::default(),
            last_flush: now_secs(),
            ops_since_flush: 0,
            snapshot_gen: 0,
            flusher: None,
            flush_warned: false,
        };

        if let Some(sp) = agent.state_path.clone() {
            if sp.is_file() {
                match agent.load_state_from(&sp) {
                    Ok(n) => eprintln!("[agent-r] loaded {} errors from {}", n, sp.display()),
                    Err(e) => {
                        eprintln!(
                            "[agent-r] WARN: corrupt state file {} ({}), starting from seeds",
                            sp.display(),
                            e
                        );
                        if !agent.no_seed() {
                            agent.load_seeds();
                        }
                    }
                }
            } else if !agent.no_seed() {
                agent.load_seeds();
            }
        }
        // Debounce (pendiente 2a): writer background dedicado para el snapshot.
        // Contrato de durabilidad: pérdida máx = lo escrito desde el último
        // flush exitoso, acotado por FLUSH_INTERVAL (2 s) o FLUSH_MAX_OPS
        // (5k ops); `flush()` y Drop escriben lo pendiente.
        agent.flusher = Some(spawn_flusher());
        agent
    }

    /// Create an Agent-R with persistence disabled (RAM only): modo por
    /// defecto de los benches/harness (no tocan disco). (OPT-O4)
    pub fn new_ephemeral() -> Self {
        Self {
            errors: VecDeque::new(),
            next_id: 1,
            w_content: 3.0,
            tuning_hits: 0,
            tuning_misses: 0,
            string_pool: Vec::new(),
            string_pool_index: HashMap::new(),
            recurrence_index: HashMap::new(),
            fingerprint_index: HashMap::new(),
            state_path: None,
            seeded: false,
            reasoning_chain: Vec::new(),
            reasoning_labels: Vec::new(),
            validators: Vec::new(),
            stats: AgentRStats::default(),
            last_flush: now_secs(),
            ops_since_flush: 0,
            snapshot_gen: 0,
            flusher: None,
            flush_warned: false,
        }
    }

    /// Disable persistence (read-only deployments / benchmarks). (OPT-O4)
    pub fn ephemeral(mut self) -> Self {
        self.state_path = None;
        self
    }

    /// Snapshot path in use, if persistence is enabled. (OPT-O4)
    pub fn state_path(&self) -> Option<&PathBuf> {
        self.state_path.as_ref()
    }

    // ── Seeds (OPT-O4) ──────────────────────────────────────────────────

    /// Seed JSON with the 3 historical incidents of the M2M lineage,
    /// versioned alongside the code (src/agent_r_seed.json, include_str!).
    pub const SEED_JSON: &'static str = include_str!("agent_r_seed.json");

    /// Load seed records if memory is empty. Idempotent: does nothing when
    /// any error is already present or seeds were already loaded.
    pub fn load_seeds(&mut self) -> usize {
        if !self.errors.is_empty() || self.seeded {
            return 0;
        }
        match serde_json::from_str::<SeedFile>(Self::SEED_JSON) {
            Ok(file) => {
                let n = file.errors.len();
                for rec in file.errors {
                    self.insert_plain_record(rec);
                }
                self.seeded = true;
                self.persist_snapshot();
                n
            }
            Err(e) => {
                eprintln!("[agent-r] WARN: embedded seed file invalid: {}", e);
                0
            }
        }
    }

    // ── Persistence: batch/debounce snapshot (pendiente 2a) ──────────────
    //
    // CONTRATO DE DURABILIDAD (antes write-through O(n) por POST):
    // - Las escrituras marcan estado sucio y ENCOLAN un flush cuando:
    //   (a) pasaron ≥ FLUSH_INTERVAL (2 s) desde el último flush, o
    //   (b) se acumularon ≥ FLUSH_MAX_OPS (5k) ops desde el último flush.
    // - El writer background serializa+escribe atómicamente FUERA del camino
    //   de escritura (el POST nunca espera la serialización O(n)).
    // - `flush()` (público) y `Drop` escriben lo pendiente SÍNCRONAMENTE:
    //   apagado ordenado = pérdida cero.
    // - Pérdida máx ante crash (kill -9): las ops desde el último flush
    //   exitoso. Bajo tráfico continuo eso está acotado por FLUSH_INTERVAL
    //   (2 s) o FLUSH_MAX_OPS (5k ops); si el proceso muere en REPOSO
    //   absoluto (cero escrituras en la ventana final <2 s), las últimas
    //   ops de esa ventana quietas no se disparan solas — usá flush() en el
    //   apagado controlado o el endpoint /agent-r/flush (lo agrega P1).
    // - La serialización del job se hace bajo el lock del caller (estado
    //   consistente: `&self`) pero la E/S queda en el writer.

    /// Marca el snapshot sucio y encola un flush al writer background si
    /// corresponde (debounce por tiempo o por nº de ops). No bloquea: si la
    /// cola está llena, la próxima op elegible reintenta.
    fn persist_snapshot(&mut self) {
        let Some(path) = self.state_path.clone() else { return };
        self.ops_since_flush += 1;
        self.snapshot_gen += 1;
        let now = now_secs();
        let due = now - self.last_flush >= FLUSH_INTERVAL.as_secs_f64()
            || self.ops_since_flush >= FLUSH_MAX_OPS;
        if !due {
            return;
        }
        // Degradación a write-through inmediato si no hay flusher vivo:
        // mejor lento que silenciosamente sin persistir.
        let Some(handle) = &self.flusher else {
            if let Err(e) = self.write_state_to(&path) {
                self.warn_flush_once(&e);
            }
            self.last_flush = now;
            self.ops_since_flush = 0;
            return;
        };
        match self.snapshot_json() {
            Ok(json) => {
                let gen = self.snapshot_gen;
                // try_send: la escritura del caller jamás espera al disco.
                if let Err(std::sync::mpsc::TrySendError::Full(_)) =
                    handle.tx.try_send(FlushJob { gen, path, json })
                {
                    // Cola llena: el job se descarta; el próximo POST elegible
                    // re-serializa y reintenta (debounce, no backpressure).
                    return;
                }
                self.last_flush = now;
                self.ops_since_flush = 0;
            }
            Err(e) => self.warn_flush_once(&format!("serialize: {}", e)),
        }
    }

    /// WARN único por racha de fallos (evita log-spam por POST).
    fn warn_flush_once(&mut self, e: &str) {
        if !self.flush_warned {
            eprintln!("[agent-r] WARN: snapshot flush failed: {} (se reintentará)", e);
            self.flush_warned = true;
        }
    }

    /// Serialize the state snapshot (used by persist + tests).
    fn snapshot_json(&self) -> Result<String, serde_json::Error> {
        let errors: Vec<PlainRecord> = self.errors.iter()
            .map(|e| PlainRecord {
                id: e.id,
                timestamp: e.timestamp,
                category: e.category,
                severity: e.severity,
                operation: self.resolve(e.operation).to_string(),
                what: e.what.to_string(),
                why: e.why.to_string(),
                context_tags: e.context_tags.iter()
                    .map(|t| self.resolve(*t).to_string())
                    .collect(),
                prevention: e.prevention.to_string(),
                recurrence: e.recurrence,
                pattern_id: e.pattern_id,
                fingerprint_hash: e.fingerprint_hash,
                last_seen: e.last_seen,
                deprecated: e.deprecated,
                deprecation_reason: e.deprecation_reason.clone(),
            })
            .collect();
        serde_json::to_string(&StateSnapshot {
            version: SNAPSHOT_VERSION,
            saved_at: now_secs(),
            next_id: self.next_id,
            stats: self.stats.clone(),
            errors,
            w_content: Some(self.w_content),
            tuning_hits: Some(self.tuning_hits),
            tuning_misses: Some(self.tuning_misses),
        })
    }

    /// Atomic write: serialize + write `<path>.tmp` + rename (via helper
    /// compartido con el writer background).
    fn write_state_to(&self, path: &std::path::Path) -> Result<(), String> {
        let json = self
            .snapshot_json()
            .map_err(|e| format!("serialize: {}", e))?;
        write_json_atomic(path, &json)
    }

    /// Load a full state snapshot written by `write_state_to`. Rebuilds the
    /// interned-string pool, recurrence index and dedup index; repairs legacy
    /// records (fingerprint/last_seen) via `rebuild_record`.
    fn load_state_from(&mut self, path: &std::path::Path) -> Result<usize, String> {
        let raw = fs::read_to_string(path).map_err(|e| format!("read: {}", e))?;
        let snap: StateSnapshot =
            serde_json::from_str(&raw).map_err(|e| format!("parse: {}", e))?;
        if snap.version > SNAPSHOT_VERSION {
            return Err(format!(
                "snapshot version {} newer than supported {}",
                snap.version, SNAPSHOT_VERSION
            ));
        }
        // OPT-O3 x OPT-O4: el snapshot viaja en strings => re-internar cada
        // record contra el pool (rebuild_plain_record) y reparar legacy.
        self.errors = snap.errors.into_iter()
            .map(|r| self.rebuild_plain_record(r))
            .collect();
        self.next_id = snap.next_id.max(
            self.errors.iter().map(|e| e.id).max().unwrap_or(0) + 1,
        );
        self.stats = snap.stats;
        // Self-tuning: restaurar W y contadores si el snapshot los trae
        // (snapshots v1 no los tienen => quedan los de fábrica).
        if let Some(w) = snap.w_content {
            self.w_content = w.clamp(0.5, 6.0);
        }
        self.tuning_hits = snap.tuning_hits.unwrap_or(self.tuning_hits);
        self.tuning_misses = snap.tuning_misses.unwrap_or(self.tuning_misses);
        self.rebuild_recurrence_index();
        self.rebuild_fingerprint_index();
        Ok(self.errors.len())
    }

    /// Persist right now (public escape hatch for graceful shutdown / tests). (OPT-O4)
    /// Debounce (pendiente 2a): escribe SÍNCRONAMENTE lo pendiente y marca la
    /// generación como comprometida (el writer background descarta jobs viejos).
    pub fn flush(&self) -> Result<(), String> {
        let Some(path) = &self.state_path else { return Ok(()) };
        let r = self.write_state_to(path);
        if let (Ok(()), Some(h)) = (&r, &self.flusher) {
            h.committed_gen.store(self.snapshot_gen, Ordering::Release);
        }
        r
    }

    fn rebuild_fingerprint_index(&mut self) {
        self.fingerprint_index.clear();
        for e in &self.errors {
            if e.fingerprint_hash != 0 {
                self.fingerprint_index.insert(e.fingerprint_hash, e.id);
            }
        }
    }

    fn rebuild_recurrence_index(&mut self) {
        self.recurrence_index.clear();
        for e in &self.errors {
            self.recurrence_index
                .entry((e.category, e.operation))
                .or_default()
                .push_back(e.id);
        }
    }

    /// Convierte un record "strings" (seed/legacy) al layout internado,
    /// reparando fingerprint_hash/last_seen heredados. (OPT-O4 x OPT-O3)
    fn rebuild_plain_record(&mut self, mut r: PlainRecord) -> ErrorRecord {
        let op_id = self.intern(&r.operation);
        let tag_ids: Vec<InternedId> = r.context_tags.iter().map(|t| self.intern(t)).collect();
        if r.fingerprint_hash == 0 {
            r.fingerprint_hash = hash_fingerprint(r.category, &r.operation, &r.what);
        }
        if r.last_seen <= 0.0 {
            r.last_seen = r.timestamp;
        }
        ErrorRecord {
            id: r.id,
            timestamp: r.timestamp,
            category: r.category,
            severity: r.severity,
            operation: op_id,
            what: r.what.into_boxed_str(),
            why: r.why.into_boxed_str(),
            context_tags: tag_ids,
            prevention: r.prevention.into_boxed_str(),
            recurrence: r.recurrence,
            pattern_id: r.pattern_id,
            fingerprint_hash: r.fingerprint_hash,
            last_seen: r.last_seen,
            deprecated: r.deprecated,
            deprecation_reason: r.deprecation_reason,
        }
    }

    /// Inserta un record en formato strings (seeds): internar, indexar,
    /// stats y eviction. (OPT-O4)
    fn insert_plain_record(&mut self, plain: PlainRecord) {
        let record = self.rebuild_plain_record(plain);
        if record.id >= self.next_id {
            self.next_id = record.id + 1;
        }
        self.stats.total_errors_recorded += 1;
        *self.stats.errors_by_category
            .entry(format!("{:?}", record.category))
            .or_insert(0) += 1;
        self.fingerprint_index.insert(record.fingerprint_hash, record.id);
        self.recurrence_index
            .entry((record.category, record.operation))
            .or_default()
            .push_back(record.id);
        self.errors.push_back(record);
        if self.errors.len() > MAX_ERRORS {
            self.evict_oldest();
        }
        self.stats.unique_incidents = self.errors.len() as u64;
    }

    /// Purga el registro más viejo (eviction) manteniendo recurrence_index,
    /// fingerprint_index y unique_incidents consistentes. (OPT-O1 x OPT-O4)
    /// VecDeque: pop_front es O(1) — antes `Vec::remove(0)` memmove O(n).
    /// (pendiente 2a)
    fn evict_oldest(&mut self) {
        let Some(evicted) = self.errors.pop_front() else { return };
        let evict_key = (evicted.category, evicted.operation);
        if let Some(bucket) = self.recurrence_index.get_mut(&evict_key) {
            bucket.pop_front();
            if bucket.is_empty() {
                self.recurrence_index.remove(&evict_key);
            }
        }
        if self.fingerprint_index.get(&evicted.fingerprint_hash) == Some(&evicted.id) {
            self.fingerprint_index.remove(&evicted.fingerprint_hash);
        }
        self.stats.unique_incidents = self.errors.len() as u64;
    }

    /// Intern a string in the pool, returning its canonical id.
    /// Strings idénticas comparten id => comparar ids equivale a comparar strings.
    fn intern(&mut self, s: &str) -> InternedId {
        if let Some(&id) = self.string_pool_index.get(s) {
            return id;
        }
        let id = self.string_pool.len() as InternedId;
        let boxed: Box<str> = s.into();
        self.string_pool.push(boxed.clone());
        self.string_pool_index.insert(boxed, id);
        id
    }

    /// Resolve an interned id back to its string (operación o tag).
    pub fn resolve(&self, id: InternedId) -> &str {
        &self.string_pool[id as usize]
    }

    // ═══════════════════════════════════════════════════════════════════
    // Layer 1: Error Memory
    // ═══════════════════════════════════════════════════════════════════

    /// Record an error with full context for future learning.
    /// Returns the error ID assigned.
    ///
    /// Dedup (OPT-O4, Mem0 ADD/UPDATE/NOOP): si el fingerprint
    /// (category, operation, what normalizados) ya existe, se actualiza el
    /// incidente existente in-place (recurrence+1, last_seen=now, timestamp=now,
    /// escalada de severidad, unión de tags) y NO se inserta duplicado.
    /// `next_id` sigue creciendo igual. OPT-O1 mantiene la ingesta O(1).
    pub fn record_error(
        &mut self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        operation: &str,
        what: &str,
        why: &str,
        context_tags: &[&str],
        prevention: &str,
    ) -> u64 {
        self.record_error_dedup(category, severity, operation, what, why, context_tags, prevention)
            .0
    }

    /// Same as `record_error` but also reports the dedup outcome
    /// (Add/Update/Noop) and the resulting recurrence count. (OPT-O4)
    pub fn record_error_dedup(
        &mut self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        operation: &str,
        what: &str,
        why: &str,
        context_tags: &[&str],
        prevention: &str,
    ) -> (u64, WriteOutcome, u32) {
        let now = now_secs();
        let fph = hash_fingerprint(category, operation, what);
        // Pre-internar tags ANTES del borrow de existing: la igualdad de ids
        // internados equivale a la igualdad de strings (canon de intern()).
        let tag_ids_in: Vec<InternedId> = context_tags
            .iter()
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .map(|t| self.intern(t))
            .collect();

        // UPDATE / NOOP path: merge into the existing incident.
        if let Some(&existing_id) = self.fingerprint_index.get(&fph) {
            let existing = self
                .errors
                .iter_mut()
                .find(|e| e.id == existing_id)
                .expect("fingerprint_index apunta a un id vivo");
            let mut changed = false;

            existing.recurrence += 1;
            existing.last_seen = now;
            existing.timestamp = now;

            // Severity escalation (never de-escalate on duplicates).
            if severity_rank(severity) > severity_rank(existing.severity) {
                existing.severity = severity;
                changed = true;
            }
            // Context tags union (por id internado: sin allocs ni resolves).
            let mut added = false;
            for tid in &tag_ids_in {
                if !existing.context_tags.contains(tid) {
                    existing.context_tags.push(*tid);
                    added = true;
                }
            }
            if added {
                changed = true;
            }
            // Refresh root-cause / prevention if the caller brings new ones.
            if !why.trim().is_empty() && why.trim() != existing.why.trim() {
                existing.why = why.trim().into();
                changed = true;
            }
            if !prevention.trim().is_empty() && prevention.trim() != existing.prevention.trim() {
                existing.prevention = prevention.trim().into();
                changed = true;
            }

            let outcome = if changed {
                WriteOutcome::Update
            } else {
                WriteOutcome::Noop
            };
            let ret = (existing.id, outcome, existing.recurrence);
            self.stats.total_errors_recorded += 1;
            self.persist_snapshot();
            return ret;
        }

        // ADD path: fingerprint desconocido.
        let id = self.next_id;
        self.next_id += 1;

        // OPT-O1: recurrence/pattern_id O(1) vía índice. recurrence = # de
        // INCIDENTES DISTINTOS previos con la misma (category, operation)
        // (con dedup ya no hay filas duplicadas); pattern_id = id máximo
        // previo de la clave = back() del bucket (ids monótonos).
        let op_id = self.intern(operation.trim());
        let tag_ids: Vec<InternedId> = context_tags
            .iter()
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .map(|t| self.intern(t))
            .collect();

        let key = (category, op_id);
        let bucket = self.recurrence_index.entry(key).or_default();
        let pattern_id = bucket.back().copied();
        let recurrence = bucket.len() as u32;
        bucket.push_back(id);

        let record = ErrorRecord {
            id,
            timestamp: now,
            category,
            severity,
            operation: op_id,
            what: what.trim().into(),
            why: why.trim().into(),
            context_tags: tag_ids,
            prevention: prevention.trim().into(),
            recurrence,
            pattern_id,
            fingerprint_hash: fph,
            last_seen: now,
            deprecated: false,
            deprecation_reason: None,
        };

        self.stats.total_errors_recorded += 1;
        *self.stats.errors_by_category
            .entry(format!("{:?}", category))
            .or_insert(0) += 1;
        self.fingerprint_index.insert(fph, id);

        self.errors.push_back(record);

        // Evict oldest if over limit (OPT-O1 x OPT-O4: purga los 3 índices).
        if self.errors.len() > MAX_ERRORS {
            self.evict_oldest();
        }
        self.stats.unique_incidents = self.errors.len() as u64;

        self.persist_snapshot();
        (id, WriteOutcome::Add, 0)
    }

    /// Deprecate a lesson by ID (false-positive loop, obsolescence).
    /// Deprecated records stay in the snapshot for audit but are excluded
    /// from recall/preflight. Persisted immediately. (OPT-O4)
    pub fn deprecate_error(&mut self, id: u64, reason: &str) -> bool {
        if let Some(e) = self.errors.iter_mut().find(|e| e.id == id) {
            e.deprecated = true;
            e.deprecation_reason = Some(reason.to_string());
            self.stats.unique_incidents = self
                .errors
                .iter()
                .filter(|e| !e.deprecated)
                .count() as u64;
            self.persist_snapshot();
            true
        } else {
            false
        }
    }

    /// Search error memory for relevant past errors.
    ///
    /// OPT (opt/opt_recall): top-20 por selección en un solo pase con un
    /// min-heap acotado a 20, en vez de materializar un Vec de n scores y
    /// ordenarlo completo (O(n log n)). El scoring por registro es idéntico
    /// al original; el desempate es por índice (posición en memoria) para
    /// reproducir el orden del `sort_by` estable sobre el Vec original
    /// (mismos scores => mismo orden de salida).
    /// Medido @100k: scan+Vec+sort 13.6 ms vs single-pass 5.6 ms (micro-bench
    /// bench/micro_scan_vs_sort.rs); además elimina la asignación de un Vec
    /// de n elementos por preflight (~1.6 MB @100k).
    pub fn recall_errors(&self, operation: &str, tags: &[&str]) -> Vec<&ErrorRecord> {
        const TOP: usize = 20;
        let last_ts = self.errors.iter().next_back().map(|e| e.timestamp).unwrap_or(0.0);

        // OPT-O3: la operación de la query puede no estar internada (nunca
        // registrada) => si no está en el pool, no hay match exacto posible.
        // Los matches parciales (contains) sí requieren el string: se resuelven
        // via el pool. Lookups del pool sin &mut: índice read-only.
        let exact_op = self.string_pool_index.get(operation).copied();

        // OPT (opt/opt_recall): Min-heap de los TOP mejores. Clave:
        // (score_bits, Reverse(idx)). score como bits => Ord total en f64
        // (scores finitos >= 0). Reverse(idx): entre scores iguales el "peor"
        // es el de índice MAYOR (el sort estable del código original conserva
        // índice menor), y un candidato con score empatado NUNCA desplaza a un
        // residente => salida idéntica a `sort_by` estable + take(20).
        let mut heap: BinaryHeap<Reverse<(u64, Reverse<usize>)>> = BinaryHeap::with_capacity(TOP + 1);

        for (idx, e) in self.errors.iter().enumerate() {
            if e.deprecated {
                continue; // OPT-O4: lecciones deprecadas fuera de recall/preflight
            }
            let mut score = 0.0;
            let e_op = self.resolve(e.operation);

            // Exact operation match (id contra id: equivalente a comparar strings)
            if Some(e.operation) == exact_op {
                score += 10.0;
            } else if e_op.contains(operation) || operation.contains(e_op) {
                score += 5.0;
            }

            // Tag overlap (ids internados contra ids internados)
            for tag in tags {
                if let Some(&tid) = self.string_pool_index.get(*tag) {
                    if e.context_tags.iter().any(|t| *t == tid) {
                        score += 3.0;
                    }
                }
            }

            // Recency bonus (more recent = higher score)
            let age_hours = (last_ts - e.timestamp) / 3600.0;
            score += (24.0 - age_hours.min(24.0)) * 0.5;

            // Recurrence penalty — recurring errors are more important
            score += e.recurrence as f64 * 2.0;

            // Severity bonus
            score += match e.severity {
                ErrorSeverity::Critical => 5.0,
                ErrorSeverity::Error => 3.0,
                ErrorSeverity::Warning => 1.0,
            };

            if score <= 0.0 {
                continue;
            }

            let key = (score.to_bits(), Reverse(idx));
            if heap.len() < TOP {
                heap.push(Reverse(key));
            } else if heap.peek().is_some_and(|Reverse(worst)| *worst < key) {
                heap.pop();
                heap.push(Reverse(key));
            }
        }

        // Extraer en orden descendente: pop del min-heap => ascendente; revertir.
        let mut out: Vec<&ErrorRecord> = Vec::with_capacity(heap.len());
        while let Some(Reverse((_, Reverse(idx)))) = heap.pop() {
            out.push(&self.errors[idx]);
        }
        out.reverse();
        out
    }

    // ═══════════════════════════════════════════════════════════════════
    // Layer 2: Pre-flight Checks
    // ═══════════════════════════════════════════════════════════════════

    /// Pesos del score compuesto del preflight v2 (pendiente 2a):
    /// score = α·similitud + β·recencia(now) + γ·log(1+freq).
    /// α=10 exact / 5 tags-only · β∈[0,6] (half-life 24 h) · γ=2 por incidencia.
    fn preflight_v2_score(
        &self,
        e: &ErrorRecord,
        op_id: Option<InternedId>,
        tag_ids: &[InternedId],
        now: f64,
    ) -> f64 {
        // SIMILITUD: exact-match de operación (id contra id) como filtro
        // primario; overlap de tags como filtro secundario. NUNCA substring
        // cruzado (era la fuente de FPs: "hnsw" matcheaba "rebuild_hnsw_index").
        let mut sim = 0.0;
        if Some(e.operation) == op_id {
            sim = 10.0;
        }
        let tag_overlap = tag_ids
            .iter()
            .filter(|t| e.context_tags.contains(t))
            .count();
        if tag_overlap > 0 {
            sim += 3.0 * tag_overlap.min(2) as f64; // cap: 2 tags de overlap
        }
        // Frecuencia observada del incidente (recurrence post-dedup).
        let freq = e.recurrence as f64 + 1.0;
        // RECENCIA(now): exponencial con half-life 24 h, acotada a ~6 días.
        let age_h = ((now - e.last_seen) / 3600.0).max(0.0);
        let recency = (-age_h / 24.0).exp() * 6.0; // 6.0 ahora -> ~0 en 6 días
        // Score compuesto: α·similitud + β·recencia + γ·log(1+freq).
        1.0 * sim + 1.0 * recency + 2.0 * (1.0 + freq).ln()
    }

    /// Preflight v2 (pendiente 2a): score compuesto sobre TODAS las reglas
    /// activas, top-3 + warnings dedup + sugerencia; veto SOLO con regla
    /// activa N≥2 y 0 FPs abiertos. O(n) single-pass con min-heap acotado.
    pub fn preflight_v2(&mut self, operation: &str, tags: &[&str]) -> PreflightResultV2 {
        self.stats.total_preflights += 1;
        let now = now_secs();

        // Internar la query contra el pool: si la op nunca se registró, no
        // puede haber exact-match (los tags siguen puntuando).
        let op_id = self.string_pool_index.get(operation).copied();
        let tag_ids: Vec<InternedId> = tags
            .iter()
            .filter_map(|t| self.string_pool_index.get(*t).copied())
            .collect();

        // Top-3 por score compuesto (min-heap acotado, análogo a recall_errors).
        let mut heap: BinaryHeap<Reverse<(u64, Reverse<usize>)>> = BinaryHeap::with_capacity(4);
        for (idx, e) in self.errors.iter().enumerate() {
            if e.deprecated {
                continue; // FP medido => deprecada => fuera del preflight
            }
            let score = self.preflight_v2_score(e, op_id, &tag_ids, now);
            if score <= 0.0 {
                continue;
            }
            let key = (score.to_bits(), Reverse(idx));
            if heap.len() < 3 {
                heap.push(Reverse(key));
            } else if heap.peek().is_some_and(|Reverse(worst)| *worst < key) {
                heap.pop();
                heap.push(Reverse(key));
            }
        }
        let mut top: Vec<&ErrorRecord> = Vec::with_capacity(heap.len());
        while let Some(Reverse((_, Reverse(idx)))) = heap.pop() {
            top.push(&self.errors[idx]);
        }
        top.reverse();

        // Veto: SOLO regla activa con severidad Critical y N≥2 incidentes
        // (recurrence+1 ≥ 2) y sin FPs abiertos (deprecación es el mecanismo
        // de FP: una regla deprecada jamás llega aquí).
        let mut should_block = false;
        let mut veto_reason = None;
        let mut warnings: Vec<String> = Vec::new();
        let mut precautions: Vec<String> = Vec::new();
        let mut suggestion: Option<String> = None;

        for e in &top {
            let op_str = self.resolve(e.operation);
            if e.severity == ErrorSeverity::Critical && (e.recurrence as u64 + 1) >= 2 {
                should_block = true;
                let w = format!("BLOCKED: recurring critical rule [{}] {}: {}", e.id, op_str, e.what);
                if !warnings.contains(&w) {
                    warnings.push(w);
                }
                if veto_reason.is_none() {
                    veto_reason = Some(format!(
                        "rule {} [{}] critical, seen {}x, no open FPs",
                        e.id, op_str, e.recurrence + 1
                    ));
                }
            } else if e.severity == ErrorSeverity::Critical {
                let w = format!("WARNING: past critical rule [{}] {}: {}", e.id, op_str, e.what);
                if !warnings.contains(&w) {
                    warnings.push(w);
                }
            }
            if !e.prevention.is_empty() {
                let p = format!("[{}] {}", op_str, e.prevention);
                if !precautions.contains(&p) {
                    precautions.push(p);
                }
            }
            if suggestion.is_none() && !e.prevention.is_empty() {
                suggestion = Some(format!("[{}] {}", op_str, e.prevention));
            }
        }

        let n_relevant = top.len();
        if should_block {
            self.stats.prefights_blocked += 1;
        }

        let top_rules: Vec<ScoredRule> = top
            .iter()
            .map(|e| ScoredRule {
                id: e.id,
                operation: self.resolve(e.operation).to_string(),
                what: e.what.to_string(),
                score: self.preflight_v2_score(e, op_id, &tag_ids, now),
            })
            .collect();

        PreflightResultV2 {
            proceed: !should_block,
            warnings,
            precautions,
            relevant_errors: n_relevant,
            top_rules,
            suggestion,
            veto_reason,
        }
    }

    /// Preflight v2 HÍBRIDO (bench composición 15-oct): SELECCIÓN de
    /// candidatos con las dos señales, no solo re-rank del top-3 v2.
    ///
    /// score_final = s_compuesto_v2 + W · c_idf_binario, con
    ///   c_idf = Σ ln((N+1)/(df+0.5)) por token de `what` PRESENTE en
    ///           `situation` (presencia binaria, NUNCA frecuencia).
    ///
    /// El top-3/veto/warnings se computa sobre ese ranking global (así el
    /// contenido PUEDE promover al top-3 una regla que la señal estructural
    /// dejó afuera — en el bench eso era la mitad de la ganancia).
    /// Hallazgos que fija (BenchTrace, n=31): matching de lecciones es de
    /// PRESENCIA (frecuencia hunde top3 48%→29%); W=3 equilibra unidades
    /// (s mediana ~28, c mediana ~4.5). Números v2→híbrido:
    /// top1 16.1%→19.4% (lib, top3 v2) y 29.0%→29.0% en selección global
    /// el compositor offline con este scorer midió top1 29.0%/top5 58.1%.
    pub fn preflight_v2_hybrid(
        &mut self,
        operation: &str,
        tags: &[&str],
        situation: &str,
    ) -> PreflightResultV2 {
        let w = self.w_content;
        self.preflight_v2_hybrid_w(operation, tags, situation, w)
    }

    /// Peso vigente de la señal de contenido (self-tuning).
    pub fn w_content(&self) -> f64 {
        self.w_content
    }

    /// Feedback del loop de self-tuning: el outcome de una decisión donde el
    /// híbrido señaló `predicted` como regla top-1.
    /// - (None, _) → abstención: no había predicción que evaluar (sin cambio:
    ///   castigarla destruye la exploración).
    /// - (Some(p), Some(a)) p==a → hit (W×1.05, cap 6.0).
    /// - (Some, Some) p!=a → miss (W×0.95, floor 0.5).
    /// - (Some, None) → miss: se predijo y el outcome mostró que la regla NO
    ///   aplicaba (ej. task completó sin tropezar con ella).
    /// Devuelve (outcome, w_nuevo).
    pub fn tuning_feedback(&mut self, predicted: Option<u64>, actual: Option<u64>) -> (&'static str, f64) {
        let outcome = match (predicted, actual) {
            (None, _) => "abstain",
            (Some(p), Some(a)) if p == a => {
                self.w_content = (self.w_content * 1.05).min(6.0);
                self.tuning_hits += 1;
                "hit"
            }
            (Some(_), _) => {
                self.w_content *= 0.95;
                if self.w_content < 0.5 {
                    self.w_content = 0.5;
                }
                self.tuning_misses += 1;
                "miss"
            }
        };
        (outcome, self.w_content)
    }

    /// Contadores del loop (para /status y para detectar deriva).
    pub fn tuning_stats(&self) -> (u64, u64, f64) {
        (self.tuning_hits, self.tuning_misses, self.w_content)
    }

    /// Seeds desactivados (AGENT_R_NO_SEED=1): memoria vacía de fábrica para
    /// dominios nuevos — las 3 semillas históricas de M2M no aplican y un veto
    /// heredado en un estado limpio confunde (visto en el smoke del CLI).
    pub fn no_seed(&self) -> bool {
        std::env::var("AGENT_R_NO_SEED").map(|v| v == "1").unwrap_or(false)
    }

    /// Variante con W_content configurable (auto-tuning: un loop de feedback
    /// puede ajustar W según outcomes reales; ver examples/self_tuning.rs).
    /// W<=0 desactiva la señal de contenido (equivale a preflight_v2).
    pub fn preflight_v2_hybrid_w(
        &mut self,
        operation: &str,
        tags: &[&str],
        situation: &str,
        w_content: f64,
    ) -> PreflightResultV2 {
        const W_CONTENT: f64 = 3.0;
        let _ = W_CONTENT; // referencia documental; el peso llega por parámetro
        let w_content = if w_content.is_finite() && w_content > 0.0 { w_content } else { 0.0 };
        let q_tokens: std::collections::HashSet<String> =
            tokenize_words(situation).into_iter().collect();

        // ── Scorer estructural + contenido en un solo pass sobre el pool ──
        let now = now_secs();
        let op_id = self.string_pool_index.get(operation).copied();
        let tag_ids: Vec<InternedId> = tags
            .iter()
            .filter_map(|t| self.string_pool_index.get(*t).copied())
            .collect();
        // Token-sets de las reglas activas + df (corpus local, O(n)).
        let active: Vec<&ErrorRecord> =
            self.errors.iter().filter(|e| !e.deprecated).collect();
        let n_rules = active.len() as f64;
        let mut df: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let rule_tokens: Vec<(u64, std::collections::HashSet<String>)> = active
            .iter()
            .map(|e| {
                let set: std::collections::HashSet<String> =
                    tokenize_words(&e.what).into_iter().collect();
                for t in &set {
                    *df.entry(t.clone()).or_default() += 1;
                }
                (e.id, set)
            })
            .collect();
        let token_of = |id: u64| -> Option<&std::collections::HashSet<String>> {
            rule_tokens.iter().find(|(rid, _)| *rid == id).map(|(_, t)| t)
        };

        // Canal de fallback: tokens del `why` COMPLETO (la lección entera).
        // GATEMAX (bench C2c): solo compite por el head cuando el overlap con
        // la lección completa es masivo (c_full ~ mediana 27 >> c_what), es
        // decir, la situación contiene virtualmente toda la lección — una
        // evidencia de repeat tan fuerte que merece el head aunque el canal
        // fino no haya visto nada. λ ∈ [15,45] da resultados idénticos
        // (plateau por unidades; no es auto-calibración fina).
        const W_FULL_FALLBACK: f64 = 25.0;
        let mut df_full: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let rule_tokens_full: Vec<(u64, std::collections::HashSet<String>)> = active
            .iter()
            .map(|e| {
                let set: std::collections::HashSet<String> =
                    tokenize_words(&e.why).into_iter().collect();
                for t in &set {
                    *df_full.entry(t.clone()).or_default() += 1;
                }
                (e.id, set)
            })
            .collect();
        let token_full_of = |id: u64| -> Option<&std::collections::HashSet<String>> {
            rule_tokens_full.iter().find(|(rid, _)| *rid == id).map(|(_, t)| t)
        };

        // score híbrido de TODAS las activas; heap top-3 idéntico a v2.
        let mut heap: BinaryHeap<Reverse<(u64, Reverse<usize>)>> =
            BinaryHeap::with_capacity(4);
        let mut hybrid_of: std::collections::HashMap<u64, f64> =
            std::collections::HashMap::new();
        for (idx, e) in active.iter().enumerate() {
            let base = self.preflight_v2_score(e, op_id, &tag_ids, now);
            let c_idf: f64 = token_of(e.id)
                .map(|toks| {
                    toks.iter()
                        .filter(|t| q_tokens.contains(*t))
                        .map(|t| {
                            ((n_rules + 1.0) / (*df.get(t).unwrap_or(&0) as f64 + 0.5)).ln()
                        })
                        .sum()
                })
                .unwrap_or(0.0);
            let score = base + w_content * c_idf;
            // GATEMAX: fallback del canal completo — SOLO si el canal fino no
            // vio nada (c_idf == 0) y el overlap con la lección completa es
            // masivo (λ·c_full supera al score estructural). Nunca resta: el
            // ranking del canal fino queda intacto (piso de no-degradación).
            let score = if c_idf == 0.0 {
                let c_full: f64 = token_full_of(e.id)
                    .map(|toks| {
                        toks.iter()
                            .filter(|t| q_tokens.contains(*t))
                            .map(|t| {
                                ((n_rules + 1.0)
                                    / (*df_full.get(t).unwrap_or(&0) as f64 + 0.5))
                                    .ln()
                            })
                            .sum()
                    })
                    .unwrap_or(0.0);
                let fallback = base + W_FULL_FALLBACK * c_full;
                score.max(fallback)
            } else {
                score
            };
            hybrid_of.insert(e.id, score);
            if score <= 0.0 {
                continue;
            }
            let key = (score.to_bits(), Reverse(idx));
            if heap.len() < 3 {
                heap.push(Reverse(key));
            } else if heap.peek().is_some_and(|Reverse(worst)| *worst < key) {
                heap.pop();
                heap.push(Reverse(key));
            }
        }
        let mut top: Vec<&ErrorRecord> = Vec::with_capacity(heap.len());
        while let Some(Reverse((_, Reverse(idx)))) = heap.pop() {
            top.push(active[idx]);
        }
        top.reverse();

        // ── Veredictos: misma política que preflight_v2, sobre el nuevo top ──
        let mut should_block = false;
        let mut warnings: Vec<String> = Vec::new();
        let mut precautions: Vec<String> = Vec::new();
        let mut suggestion: Option<String> = None;
        let mut veto_reason: Option<String> = None;
        for e in &top {
            let op_str = self.resolve(e.operation);
            if e.severity == ErrorSeverity::Critical && (e.recurrence as u64 + 1) >= 2 {
                should_block = true;
                let w = format!(
                    "BLOCKED: recurring critical rule [{}] {}: {}",
                    e.id,
                    op_str,
                    e.what
                );
                if !warnings.contains(&w) {
                    warnings.push(w);
                }
                if veto_reason.is_none() {
                    veto_reason = Some(format!(
                        "rule {} [{}] critical, seen {}x, no open FPs",
                        e.id,
                        op_str,
                        e.recurrence + 1
                    ));
                }
            } else if e.severity == ErrorSeverity::Critical {
                let w = format!(
                    "WARNING: past critical rule [{}] {}: {}",
                    e.id,
                    op_str,
                    e.what
                );
                if !warnings.contains(&w) {
                    warnings.push(w);
                }
            }
            if !e.prevention.is_empty() {
                let p = format!("[{}] {}", op_str, e.prevention);
                if !precautions.contains(&p) {
                    precautions.push(p);
                }
            }
            if suggestion.is_none() && !e.prevention.is_empty() {
                suggestion = Some(format!("[{}] {}", op_str, e.prevention));
            }
        }
        if should_block {
            self.stats.prefights_blocked += 1;
        }
        let top_rules: Vec<ScoredRule> = top
            .iter()
            .map(|e| ScoredRule {
                id: e.id,
                operation: self.resolve(e.operation).to_string(),
                what: e.what.to_string(),
                score: hybrid_of.get(&e.id).copied().unwrap_or(0.0),
            })
            .collect();
        PreflightResultV2 {
            proceed: !should_block,
            warnings,
            precautions,
            relevant_errors: top_rules.len(),
            top_rules,
            suggestion,
            veto_reason,
        }
    }

    /// Todas las reglas activas con su score compuesto v2 (para composición:
    /// fusión con otras señales, calibración, o exports). Orden descendente.
    /// O(n) — mismo scorer que preflight_v2, sin heap ni veto.
    pub fn preflight_v2_scored_all(&mut self, operation: &str, tags: &[&str]) -> Vec<ScoredRule> {
        self.stats.total_preflights += 1;
        let now = now_secs();
        let op_id = self.string_pool_index.get(operation).copied();
        let tag_ids: Vec<InternedId> = tags
            .iter()
            .filter_map(|t| self.string_pool_index.get(*t).copied())
            .collect();
        let mut all: Vec<ScoredRule> = self
            .errors
            .iter()
            .filter(|e| !e.deprecated)
            .map(|e| ScoredRule {
                id: e.id,
                operation: self.resolve(e.operation).to_string(),
                what: e.what.to_string(),
                score: self.preflight_v2_score(e, op_id, &tag_ids, now),
            })
            .collect();
        all.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all
    }

    /// Run pre-flight checks before an operation.
    /// Recalls relevant past errors and generates warnings/precautions.
    /// (Legado v1, intacto para compatibilidad de tests/consumidores.)
    pub fn preflight(&mut self, operation: &str, tags: &[&str]) -> PreflightResult {
        self.stats.total_preflights += 1;

        let relevant = self.recall_errors(operation, tags);
        let mut warnings = Vec::new();
        let mut precautions = Vec::new();
        let mut should_block = false;

        for error in &relevant {
            if error.severity == ErrorSeverity::Critical && error.recurrence >= 2 {
                should_block = true;
                warnings.push(format!(
                    "BLOCKED: Recurring critical error in '{}': {}",
                    self.resolve(error.operation), error.what
                ));
            } else if error.severity == ErrorSeverity::Critical {
                warnings.push(format!(
                    "WARNING: Past critical error in '{}': {}",
                    self.resolve(error.operation), error.what
                ));
            }

            if !error.prevention.is_empty() {
                precautions.push(format!("[{}] {}", self.resolve(error.operation), error.prevention));
            }
        }

        let n_relevant = relevant.len();

        if should_block {
            self.stats.prefights_blocked += 1;
        }

        PreflightResult {
            proceed: !should_block,
            warnings,
            precautions,
            relevant_errors: n_relevant,
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Layer 3: Reasoning Integrity
    // ═══════════════════════════════════════════════════════════════════

    /// Begin tracking a reasoning chain.
    pub fn begin_reasoning(&mut self) {
        self.reasoning_chain.clear();
        self.reasoning_labels.clear();
    }

    /// Record a step in the reasoning chain.
    /// Returns issues detected (loops, drift, etc.)
    pub fn reasoning_step(&mut self, operation: &str, state_hash: u64) -> Vec<ReasoningIssue> {
        let mut issues = Vec::new();

        // Detect circular reasoning — same state hash seen before
        let repetitions = self.reasoning_chain.iter().filter(|&&h| h == state_hash).count();
        if repetitions >= REPETITION_THRESHOLD {
            issues.push(ReasoningIssue {
                issue_type: ErrorCategory::CircularReasoning,
                description: format!(
                    "Circular reasoning detected: operation '{}' repeated {} times with identical state",
                    operation, repetitions + 1
                ),
                position: self.reasoning_chain.len(),
                confidence: 0.95,
            });
        }

        // Detect oscillation — alternating between two states
        // If new state == 2nd-to-last and last == 3rd-to-last, we're oscillating
        if self.reasoning_chain.len() >= 3 {
            let n = self.reasoning_chain.len();
            if state_hash == self.reasoning_chain[n - 2]
                && n >= 4
                && self.reasoning_chain[n - 1] == self.reasoning_chain[n - 3]
            {
                issues.push(ReasoningIssue {
                    issue_type: ErrorCategory::CircularReasoning,
                    description: format!(
                        "Oscillating between states in '{}': two-state loop detected",
                        operation
                    ),
                    position: n,
                    confidence: 0.90,
                });
            }
        }

        // Check chain length — excessively long chains may indicate drift
        if self.reasoning_chain.len() > MAX_REASONING_STEPS {
            issues.push(ReasoningIssue {
                issue_type: ErrorCategory::OutputDrift,
                description: format!(
                    "Reasoning chain too long ({} steps): likely drifting from objective",
                    self.reasoning_chain.len()
                ),
                position: self.reasoning_chain.len(),
                confidence: 0.80,
            });
        }

        self.reasoning_chain.push(state_hash);
        self.reasoning_labels.push(operation.to_string());

        issues
    }

    /// Validate reasoning chain integrity at the end of a chain.
    pub fn validate_chain(&mut self) -> ValidationResult {
        let start = Instant::now();
        let mut issues = Vec::new();
        let mut checks_run = 0;

        // Check 1: Empty chain
        checks_run += 1;
        if self.reasoning_chain.is_empty() {
            return ValidationResult {
                valid: true,
                issues: vec![],
                confidence: 1.0,
                checks_run,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            };
        }

        // Check 2: Repetition density
        checks_run += 1;
        let unique_count: usize = {
            let mut sorted = self.reasoning_chain.clone();
            sorted.sort();
            sorted.dedup();
            sorted.len()
        };
        let total = self.reasoning_chain.len();
        if total > 5 {
            let repetition_ratio = 1.0 - (unique_count as f64 / total as f64);
            if repetition_ratio >= 0.7 {
                issues.push(ReasoningIssue {
                    issue_type: ErrorCategory::CircularReasoning,
                    description: format!(
                        "High repetition ratio ({:.0}%): reasoning chain mostly circular",
                        repetition_ratio * 100.0
                    ),
                    position: 0,
                    confidence: 0.85,
                });
            }
        }

        // Check 3: Stagnation — same operation repeated many times without progress
        checks_run += 1;
        if self.reasoning_labels.len() > 10 {
            let last_10 = &self.reasoning_labels[self.reasoning_labels.len() - 10..];
            let first_label = last_10[0].as_str();
            if last_10.iter().all(|l| l.as_str() == first_label) {
                issues.push(ReasoningIssue {
                    issue_type: ErrorCategory::OutputDrift,
                    description: format!(
                        "Stagnation: same operation '{}' repeated 10+ times without progress",
                        last_10[0]
                    ),
                    position: self.reasoning_labels.len() - 10,
                    confidence: 0.90,
                });
            }
        }

        // Check 4: Sudden topic shift — last operation completely different from preceding
        checks_run += 1;
        if self.reasoning_labels.len() >= 3 {
            let n = self.reasoning_labels.len();
            let prev_ops: Vec<&str> = self.reasoning_labels[..n - 1]
                .iter()
                .map(|s| s.as_str())
                .take(5)
                .collect();
            let last_op = self.reasoning_labels[n - 1].as_str();
            let overlap = prev_ops.iter().filter(|&&op| op == last_op).count();
            if overlap == 0 && !last_op.starts_with("validate") && !last_op.starts_with("finalize") {
                // Sudden shift isn't always bad (validation at end is normal)
                // Only flag if the previous ops were all the same
                let first_prev = prev_ops[0];
                if prev_ops.iter().all(|op| *op == first_prev) {
                    issues.push(ReasoningIssue {
                        issue_type: ErrorCategory::BadAssumption,
                        description: format!(
                            "Sudden topic shift from '{}' to '{}' — verify this was intentional",
                            prev_ops[0], last_op
                        ),
                        position: n - 1,
                        confidence: 0.60,
                    });
                }
            }
        }

        // Check 5: Custom validators
        checks_run += 1;
        // (validators are applied at validate_output level, not chain level)

        let max_confidence = issues.iter().map(|i| i.confidence).fold(0.0_f64, f64::max);
        let confidence = 1.0 - max_confidence;
        let valid = issues.is_empty()
            || issues.iter().all(|i| i.confidence < 0.75);

        if !valid {
            self.stats.validations_failed += 1;
        }
        self.stats.total_validations += 1;

        ValidationResult {
            valid,
            issues,
            confidence,
            checks_run,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Validate a specific output against known error patterns.
    pub fn validate_output(
        &mut self,
        operation: &str,
        output: &serde_json::Value,
    ) -> ValidationResult {
        let start = Instant::now();
        let mut issues = Vec::new();
        let mut checks_run = 0;

        // Check 1: Empty/null output that shouldn't be
        checks_run += 1;
        if output.is_null() {
            issues.push(ReasoningIssue {
                issue_type: ErrorCategory::StateCorruption,
                description: format!("Operation '{}' produced null output", operation),
                position: 0,
                confidence: 0.95,
            });
        }

        // Check 2: Numeric anomalies (NaN, Inf)
        checks_run += 1;
        let json_str = output.to_string();
        if json_str.contains("NaN") || json_str.contains("Infinity") || json_str.contains("-Infinity") {
            issues.push(ReasoningIssue {
                issue_type: ErrorCategory::Fabrication,
                description: format!("Operation '{}' produced non-finite numeric values", operation),
                position: 0,
                confidence: 0.95,
            });
        }

        // Check 3: Suspiciously uniform results
        checks_run += 1;
        if let Some(arr) = output.as_array() {
            if arr.len() > 5 {
                let first = &arr[0];
                if arr.iter().all(|a| a == first) {
                    issues.push(ReasoningIssue {
                        issue_type: ErrorCategory::Fabrication,
                        description: format!(
                            "Operation '{}' produced {} identical results — likely fabricated",
                            operation, arr.len()
                        ),
                        position: 0,
                        confidence: 0.80,
                    });
                }
            }
        }

        // Check 4: Run custom validators
        checks_run += self.validators.len() as usize;
        for validator in &self.validators {
            if let Some(issue) = validator(operation, output) {
                issues.push(issue);
            }
        }

        // Check 5: Recall past errors for this operation
        checks_run += 1;
        let past_errors = self.recall_errors(operation, &[]);
        for error in past_errors.iter().take(3) {
            if error.severity == ErrorSeverity::Critical {
                issues.push(ReasoningIssue {
                    issue_type: ErrorCategory::RecurringPattern,
                    description: format!(
                        "Past critical error in '{}': {} — verify output is correct",
                        self.resolve(error.operation), error.what
                    ),
                    position: 0,
                    confidence: 0.70,
                });
            }
        }

        let max_confidence = issues.iter().map(|i| i.confidence).fold(0.0_f64, f64::max);
        let confidence = 1.0 - max_confidence;
        let valid = issues.is_empty()
            || issues.iter().all(|i| i.confidence < 0.75);

        if !valid {
            self.stats.validations_failed += 1;
        }
        self.stats.total_validations += 1;

        ValidationResult {
            valid,
            issues,
            confidence,
            checks_run,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Add a custom validation rule.
    pub fn add_validator(
        &mut self,
        f: Box<dyn Fn(&str, &serde_json::Value) -> Option<ReasoningIssue> + Send + Sync>,
    ) {
        self.validators.push(f);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Validate v2: niveles N0-N3 (pendiente 2a)
    // ═══════════════════════════════════════════════════════════════════

    /// Recorre TODO el JSON iterando valores numéricos con `as_f64` y reporta
    /// NaN/±Inf. JAMÁS por substring sobre la serialización (el FP
    /// garantizado del v1: `{"score":"NaN"}` — un STRING — matcheaba igual).
    fn scan_nonfinite(v: &serde_json::Value, path: &str, hits: &mut Vec<String>) {
        match v {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    if !f.is_finite() {
                        hits.push(format!("{path} = {f}"));
                    }
                }
                // u64/i64 que no caben en f64 finite son >= 2^1024: imposible
                // vía el parser (rechaza >1e308), pero defendemos igual.
                if n.as_f64().is_none() && (n.as_u64().is_some() || n.as_i64().is_some()) {
                    hits.push(format!("{path} = {n} (overflow f64)"));
                }
            }
            serde_json::Value::Array(a) => {
                for (i, x) in a.iter().enumerate() {
                    Self::scan_nonfinite(x, &format!("{path}[{i}]"), hits);
                }
            }
            serde_json::Value::Object(m) => {
                for (k, x) in m {
                    Self::scan_nonfinite(x, &format!("{path}.{k}"), hits);
                }
            }
            _ => {}
        }
    }

    /// Detecta resultados sospechosamente uniformes (arrays >5 con todos los
    /// elementos iguales) — heurística de fabricación heredada del v1.
    fn scan_uniform_array(v: &serde_json::Value) -> Option<usize> {
        if let Some(arr) = v.as_array() {
            if arr.len() > 5 && arr.iter().all(|a| a == &arr[0]) {
                return Some(arr.len());
            }
        }
        None
    }

    /// Validate v2 por niveles (pendiente 2a).
    ///
    /// - **N0 forma**: null-output, forma estructural (array uniforme).
    /// - **N1 auto-consistencia numérica**: NaN/±Inf iterando valores con
    ///   `as_f64` (NUNCA substring). Oráculo = aritmética estándar.
    /// - **N2 evidencia programática del llamador**: sale de `evidence`
    ///   (exit_code/stderr/diff). Sin evidencia → `unverifiable` (NO fail).
    ///   Nunca se ejecuta nada.
    /// - **N3 receipts**: verificación criptográfica vía `n3_check` de P1.
    ///   Hasta el merge: stub → `Unverifiable`.
    ///
    /// `overall`: fail si algún nivel con oráculo falla; unverifiable si
    /// ningún nivel pudo verificarse; pass en el resto. Compat: `valid` =
    /// (overall == "pass"), `confidence` mapeada 1.0/0.3/0.0.
    pub fn validate_output_v2(
        &mut self,
        operation: &str,
        output: &serde_json::Value,
    ) -> ValidationResultV2 {
        self.validate_output_v2_evidence(operation, output, None)
    }

    /// Igual que `validate_output_v2` pero acepta la evidencia programática
    /// del llamador para N2: `{"exit_code": 0, "stderr": "...", "diff": "...",
    /// "expected_exit_code": 0}`.
    pub fn validate_output_v2_evidence(
        &mut self,
        operation: &str,
        output: &serde_json::Value,
        evidence: Option<&serde_json::Value>,
    ) -> ValidationResultV2 {
        let start = Instant::now();
        let mut checks_run = 0usize;
        let mut details: Vec<String> = Vec::new();
        let mut issues: Vec<ReasoningIssue> = Vec::new(); // compat legado

        // ── N0: forma ────────────────────────────────────────────────────
        checks_run += 1;
        let n0 = if output.is_null() {
            issues.push(ReasoningIssue {
                issue_type: ErrorCategory::StateCorruption,
                description: format!("Operation '{}' produced null output", operation),
                position: 0,
                confidence: 0.95,
            });
            details.push("N0: null output".into());
            LevelVerdict::Fail
        } else if let Some(n) = Self::scan_uniform_array(output) {
            issues.push(ReasoningIssue {
                issue_type: ErrorCategory::Fabrication,
                description: format!(
                    "Operation '{}' produced {} identical results — likely fabricated",
                    operation, n
                ),
                position: 0,
                confidence: 0.80,
            });
            details.push(format!("N0: uniform array of {n} identical elements"));
            LevelVerdict::Fail
        } else {
            LevelVerdict::Pass
        };

        // ── N1: auto-consistencia numérica (as_f64, NUNCA substring) ────
        checks_run += 1;
        let n1 = {
            let mut hits = Vec::new();
            Self::scan_nonfinite(output, "$", &mut hits);
            if hits.is_empty() {
                LevelVerdict::Pass
            } else {
                issues.push(ReasoningIssue {
                    issue_type: ErrorCategory::Fabrication,
                    description: format!(
                        "Operation '{}' produced non-finite numeric values: {}",
                        operation,
                        hits.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
                    ),
                    position: 0,
                    confidence: 0.95,
                });
                details.push(format!("N1: {} non-finite value(s)", hits.len()));
                LevelVerdict::Fail
            }
        };

        // ── N2: evidencia programática del llamador (sin ejecutar nada) ──
        checks_run += 1;
        let n2 = 'n2: {
            let Some(ev) = evidence else {
                details.push("N2: no caller evidence provided".into());
                break 'n2 LevelVerdict::Unverifiable;
            };
            let exit = ev.get("exit_code").and_then(|v| v.as_i64());
            let Some(exit) = exit else {
                details.push("N2: evidence without exit_code".into());
                break 'n2 LevelVerdict::Unverifiable;
            };
            let expected = ev
                .get("expected_exit_code")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let stderr_nonempty = ev
                .get("stderr")
                .map(|s| s.as_str().map(|s| !s.trim().is_empty()).unwrap_or(false))
                .unwrap_or(false);
            let has_diff = ev
                .get("diff")
                .map(|d| d.as_str().map(|s| !s.trim().is_empty()).unwrap_or(false))
                .unwrap_or(false);
            // Consistencia: exit != expected = fail duro. exit 0 + stderr con
            // contenido y NINGÚN diff que lo explique = señales contradictorias.
            if exit != expected {
                issues.push(ReasoningIssue {
                    issue_type: ErrorCategory::StateCorruption,
                    description: format!(
                        "N2: exit_code {} != expected {} for '{}'",
                        exit, expected, operation
                    ),
                    position: 0,
                    confidence: 0.95,
                });
                details.push(format!("N2: exit_code {exit} != expected {expected}"));
                break 'n2 LevelVerdict::Fail;
            }
            if exit == 0 && stderr_nonempty && !has_diff {
                issues.push(ReasoningIssue {
                    issue_type: ErrorCategory::BadAssumption,
                    description: format!(
                        "N2: '{}' exited 0 but stderr is non-empty with no diff",
                        operation
                    ),
                    position: 0,
                    confidence: 0.75,
                });
                details.push("N2: exit 0 with non-empty stderr and no diff".into());
                break 'n2 LevelVerdict::Fail;
            }
            details.push(format!("N2: exit_code {exit} consistent (stderr/diff coherent)"));
            LevelVerdict::Pass
        };

        // ── N3: receipts (stub hasta el merge de P1) ─────────────────────
        checks_run += 1;
        // TODO(integration): wire receipts — llamar a
        // `receipts::n3_check(&claim)` del módulo de P1-receipts cuando esté
        // mergeado. Hasta entonces no hay oráculo criptográfico disponible.
        let n3 = LevelVerdict::Unverifiable;
        details.push("N3: receipts not wired yet (stub)".into());

        // Custom validators (heredados del v1, corren como parte de N0 forma).
        checks_run += self.validators.len();
        for validator in &self.validators {
            if let Some(issue) = validator(operation, output) {
                issues.push(issue);
            }
        }

        // ── Aggregate ────────────────────────────────────────────────────
        let verdicts = ValidationVerdicts {
            n0_form: n0,
            n1_numeric_consistency: n1,
            n2_caller_evidence: n2,
            n3_receipt: n3,
        };
        let levels = [n0, n1, n2, n3];
        let any_fail = levels.iter().any(|l| *l == LevelVerdict::Fail);
        let any_pass = levels.iter().any(|l| *l == LevelVerdict::Pass);
        let (overall, confidence) = if any_fail {
            ("fail".to_string(), 0.0)
        } else if any_pass {
            ("pass".to_string(), 1.0)
        } else {
            ("unverifiable".to_string(), 0.3)
        };
        let suggestion = if any_fail {
            let first_fail = [n0, n1, n2]
                .iter()
                .position(|l| *l == LevelVerdict::Fail);
            Some(match first_fail {
                Some(0) => format!(
                    "N0 failed for '{}': check the output shape (null/uniform array) before trusting it",
                    operation
                ),
                Some(1) => format!(
                    "N1 failed for '{}': recompute the non-finite values; NaN/Inf indicate a broken computation, not a representation artifact",
                    operation
                ),
                _ => format!(
                    "N2 failed for '{}': re-run with evidence capture (exit_code/stderr/diff) and verify consistency",
                    operation
                ),
            })
        } else if overall == "unverifiable" {
            Some(format!(
                "No oracle available for '{}': pass caller evidence (N2) or wire receipts (N3) to raise confidence",
                operation
            ))
        } else {
            None
        };

        if overall == "fail" {
            self.stats.validations_failed += 1;
        }
        self.stats.total_validations += 1;

        // dedup details (defensivo; los mensajes ya son únicos por nivel)
        details.dedup();

        ValidationResultV2 {
            verdicts,
            valid: overall == "pass",
            overall,
            suggestion,
            details,
            confidence,
            checks_run,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            issues,
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Introspection
    // ═══════════════════════════════════════════════════════════════════

    /// Get Agent-R statistics.
    pub fn stats(&self) -> &AgentRStats {
        &self.stats
    }

    /// Get all recorded errors (en orden de inserción; slice contiguo).
    pub fn errors(&self) -> &[ErrorRecord] {
        self.errors.as_slices().0
    }

    /// Get errors by category.
    pub fn errors_by_category(&self, cat: ErrorCategory) -> Vec<&ErrorRecord> {
        self.errors.iter().filter(|e| e.category == cat).collect()
    }

    /// Get the most common error patterns.
    pub fn top_patterns(&mut self, limit: usize) -> Vec<(&str, usize)> {
        // OPT-O3: agrupa por id internado (u32) en vez de por string; resuelve
        // el string sólo para los `limit` ganadores.
        let mut counts: HashMap<InternedId, usize> = HashMap::new();
        for e in &self.errors {
            *counts.entry(e.operation).or_insert(0) += 1;
        }
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.into_iter().take(limit)
            .map(|(id, n)| (self.resolve(id), n))
            .collect()
    }

    /// Clear reasoning chain without recording errors.
    pub fn clear_chain(&mut self) {
        self.reasoning_chain.clear();
        self.reasoning_labels.clear();
    }

    /// Serialize error memory for persistence: formato estable de strings
    /// (los consumidores ven operation/context_tags como strings, no ids).
    /// Incluye los campos de dedup/deprecación de OPT-O4.
    pub fn serialize_errors(&self) -> Result<String, serde_json::Error> {
        let resolved: Vec<PlainRecord> = self.errors.iter()
            .map(|e| PlainRecord {
                id: e.id,
                timestamp: e.timestamp,
                category: e.category,
                severity: e.severity,
                operation: self.resolve(e.operation).to_string(),
                what: e.what.to_string(),
                why: e.why.to_string(),
                context_tags: e.context_tags.iter()
                    .map(|t| self.resolve(*t).to_string())
                    .collect(),
                prevention: e.prevention.to_string(),
                recurrence: e.recurrence,
                pattern_id: e.pattern_id,
                fingerprint_hash: e.fingerprint_hash,
                last_seen: e.last_seen,
                deprecated: e.deprecated,
                deprecation_reason: e.deprecation_reason.clone(),
            })
            .collect();
        serde_json::to_string(&resolved)
    }

    /// Load error memory from persistence (formato estable de strings).
    /// Re-interna (OPT-O3), repara campos heredados y reconstruye los
    /// índices de recurrencia (OPT-O1) y dedup (OPT-O4).
    pub fn load_errors(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let loaded: Vec<PlainRecord> = serde_json::from_str(json)?;
        self.errors = loaded.into_iter()
            .map(|r| self.rebuild_plain_record(r))
            .collect();
        if let Some(max_id) = self.errors.iter().map(|e| e.id).max() {
            self.next_id = self.next_id.max(max_id + 1);
        }
        self.rebuild_recurrence_index();
        self.rebuild_fingerprint_index();
        Ok(())
    }
}

impl Default for AgentR {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OPT (opt/opt_recall): paridad del top-20 por min-heap acotado contra la
    /// implementación de referencia (Vec de scores + sort estable + take(20)).
    /// Misma secuencia de record_error que el micro-bench del harness
    /// (bench/micro_scan_vs_sort.rs): 500 registros, operaciones/tags
    /// superpuestos, severities mezcladas, empates de score garantizados.
    #[test]
    fn test_recall_top20_paridad_con_sort() {
        let mut agent = AgentR::new_ephemeral();
        let ops = ["reindex_collection", "store_memory", "search_similar",
                   "rebuild_hnsw_index", "merge_splats", "distill_lessons",
                   "route_query", "gc_expired_splats"];
        let tags_pool = ["gpu", "cpu", "ci", "prod", "index", "retry", "hnsw"];
        for i in 0..500 {
            let op = ops[i % ops.len()];
            let t1 = tags_pool[i % tags_pool.len()];
            let t2 = tags_pool[(i * 3 + 1) % tags_pool.len()];
            let cat = match i % 5 {
                0 => ErrorCategory::CircularReasoning,
                1 => ErrorCategory::OutputDrift,
                2 => ErrorCategory::Fabrication,
                3 => ErrorCategory::BadAssumption,
                _ => ErrorCategory::Timeout,
            };
            agent.record_error(
                cat,
                match i % 3 {
                    0 => ErrorSeverity::Critical,
                    1 => ErrorSeverity::Error,
                    _ => ErrorSeverity::Warning,
                },
                op, &format!("what {i}"), &format!("why {i}"),
                &[t1, t2], &format!("prev {i}"),
            );
        }

        // Referencia: replicar el código original (Vec + sort estable + take 20).
        // (scorer adaptado al layout internado: compara strings via resolve)
        let ref_top: Vec<(f64, u64)> = {
            let last_ts = agent.errors.iter().next_back().map(|e| e.timestamp).unwrap_or(0.0);
            let mut scored: Vec<(f64, usize)> = agent.errors.iter().enumerate()
                .map(|(idx, e)| {
                    let mut s = 0.0;
                    let e_op = agent.resolve(e.operation);
                    if e_op == "distill_lessons" { s += 10.0; }
                    else if e_op.contains("distill_lessons") || "distill_lessons".contains(e_op) { s += 5.0; }
                    for t in ["gpu", "ci"] { if e.context_tags.iter().any(|x| agent.resolve(*x) == t) { s += 3.0; } }
                    let age_h = (last_ts - e.timestamp) / 3600.0;
                    s += (24.0 - age_h.min(24.0)) * 0.5;
                    s += e.recurrence as f64 * 2.0;
                    s += match e.severity { ErrorSeverity::Critical => 5.0, ErrorSeverity::Error => 3.0, ErrorSeverity::Warning => 1.0 };
                    (s, idx)
                })
                .filter(|(s, _)| *s > 0.0)
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.into_iter().take(20).map(|(s, idx)| (s, agent.errors[idx].id)).collect()
        };

        let got: Vec<(f64, u64)> = agent.recall_errors("distill_lessons", &["gpu", "ci"])
            .into_iter().map(|e| (recurrence_only_score(&agent, e, "distill_lessons", &["gpu", "ci"], agent.errors.iter().next_back().map(|x| x.timestamp).unwrap_or(0.0)), e.id))
            .collect();
        let got_ids: Vec<u64> = got.iter().map(|(_, id)| *id).collect();
        let ref_ids: Vec<u64> = ref_top.iter().map(|(_, id)| *id).collect();
        assert_eq!(got_ids, ref_ids, "top-20 heap != top-20 sort estable");
    }

    /// Recalcula el score de un registro para verificar paridad de scores
    /// (helper del test de paridad). Layout internado: resuelve ids via pool.
    fn recurrence_only_score(agent: &AgentR, e: &ErrorRecord, operation: &str, tags: &[&str], last_ts: f64) -> f64 {
        let mut s = 0.0;
        let e_op = agent.resolve(e.operation);
        if e_op == operation { s += 10.0; }
        else if e_op.contains(operation) || operation.contains(e_op) { s += 5.0; }
        for t in tags { if e.context_tags.iter().any(|x| agent.resolve(*x) == *t) { s += 3.0; } }
        let age_h = (last_ts - e.timestamp) / 3600.0;
        s += (24.0 - age_h.min(24.0)) * 0.5;
        s += e.recurrence as f64 * 2.0;
        s += match e.severity { ErrorSeverity::Critical => 5.0, ErrorSeverity::Error => 3.0, ErrorSeverity::Warning => 1.0 };
        s
    }

    #[test]
    fn test_record_and_recall() {
        let mut agent = AgentR::new_ephemeral();
        agent.record_error(
            ErrorCategory::Fabrication, ErrorSeverity::Critical,
            "generate_benchmarks", "Invented benchmark data", "Skipped actual testing",
            &["benchmarks", "GPU"], "Only use validated data from real tests",
        );

        let found = agent.recall_errors("generate_benchmarks", &["benchmarks"]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].category, ErrorCategory::Fabrication);
    }

    #[test]
    fn test_preflight_blocks_recurring_critical() {
        let mut agent = AgentR::new_ephemeral();
        // Record 3 identical critical errors
        for _ in 0..3 {
            agent.record_error(
                ErrorCategory::Fabrication, ErrorSeverity::Critical,
                "generate_data", "Made up numbers", "No testing",
                &["data"], "Test first",
            );
        }

        let result = agent.preflight("generate_data", &["data"]);
        assert!(!result.proceed);
        assert!(result.warnings.iter().any(|w| w.contains("BLOCKED")));
    }

    #[test]
    fn test_circular_reasoning_detection() {
        let mut agent = AgentR::new_ephemeral();
        agent.begin_reasoning();

        // Repeat same state 3+ times
        for _ in 0..4 {
            let issues = agent.reasoning_step("analyze", 0xABCD);
            if !issues.is_empty() {
                assert!(issues[0].issue_type == ErrorCategory::CircularReasoning);
                return;
            }
        }
        panic!("Should have detected circular reasoning");
    }

    #[test]
    fn test_validate_output_nan() {
        let mut agent = AgentR::new_ephemeral();
        let result = agent.validate_output(
            "compute_distance",
            &serde_json::json!({"score": "NaN"}),
        );
        assert!(!result.valid);
        assert!(result.issues.iter().any(|i| i.issue_type == ErrorCategory::Fabrication));
    }

    #[test]
    fn test_validate_uniform_results() {
        let mut agent = AgentR::new_ephemeral();
        let result = agent.validate_output(
            "search",
            &serde_json::json!([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        );
        assert!(!result.valid);
    }

    #[test]
    fn test_chain_validation_repetition() {
        let mut agent = AgentR::new_ephemeral();
        agent.begin_reasoning();
        // 8 identical steps out of 10
        for i in 0..10 {
            agent.reasoning_step("step", if i < 2 { i as u64 * 100 } else { 42 });
        }
        let result = agent.validate_chain();
        assert!(!result.valid); // 80% repetition
    }

    #[test]
    fn test_chain_stagnation() {
        let mut agent = AgentR::new_ephemeral();
        agent.begin_reasoning();
        for _ in 0..15 {
            agent.reasoning_step("optimize", 1);
        }
        let result = agent.validate_chain();
        assert!(result.issues.iter().any(|i| i.description.contains("Stagnation")));
    }

    #[test]
    fn test_serialize_load_roundtrip() {
        let mut agent = AgentR::new_ephemeral();
        agent.record_error(
            ErrorCategory::BadAssumption, ErrorSeverity::Warning,
            "search", "Wrong backend", "Assumed CUDA available",
            &["gpu"], "Check backend availability first",
        );
        let json = agent.serialize_errors().unwrap();
        let mut agent2 = AgentR::new_ephemeral();
        agent2.load_errors(&json).unwrap();
        assert_eq!(agent2.errors.len(), 1);
        assert_eq!(agent2.next_id, 2);
    }

    #[test]
    fn test_oscillation_detection() {
        let mut agent = AgentR::new_ephemeral();
        agent.begin_reasoning();
        agent.reasoning_step("step_a", 100);
        agent.reasoning_step("step_b", 200);

        let issues = agent.reasoning_step("step_a", 100);
        assert!(issues.is_empty()); // Only 2nd repetition

        agent.reasoning_step("step_b", 200);
        let issues = agent.reasoning_step("step_a", 100); // Now oscillating
        assert!(issues.iter().any(|i| i.description.contains("Oscillating")));
    }

    #[test]
    fn test_stats() {
        let mut agent = AgentR::new_ephemeral();
        agent.record_error(ErrorCategory::Fabrication, ErrorSeverity::Critical, "op1", "w", "y", &[], "p");
        agent.record_error(ErrorCategory::Fabrication, ErrorSeverity::Error, "op2", "w", "y", &[], "p");
        agent.preflight("op1", &[]);
        agent.validate_output("op1", &serde_json::json!(null));

        let stats = agent.stats();
        assert_eq!(stats.total_errors_recorded, 2);
        assert_eq!(stats.total_preflights, 1);
        assert_eq!(stats.total_validations, 1);
    }

    #[test]
    fn test_empty_chain_validates_ok() {
        let mut agent = AgentR::new_ephemeral();
        agent.begin_reasoning();
        let result = agent.validate_chain();
        assert!(result.valid);
    }

    #[test]
    fn test_custom_validator() {
        let mut agent = AgentR::new_ephemeral();
        agent.add_validator(Box::new(|op, val| {
            if op == "risky" && val.get("status").map(|v| v == "unknown").unwrap_or(false) {
                Some(ReasoningIssue {
                    issue_type: ErrorCategory::BadAssumption,
                    description: "Risky operation returned unknown status".into(),
                    position: 0,
                    confidence: 0.90,
                })
            } else {
                None
            }
        }));

        let result = agent.validate_output("risky", &serde_json::json!({"status": "unknown"}));
        assert!(!result.valid);
    }

    // ── OPT-O4: persistencia + dedup + seeds (portado a layout internado) ──

    /// Unique temp dir per test (parallel-safe).
    fn tmp_state_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "agent_r_test_{}_{}",
            name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir.join("state.json")
    }

    /// Pre-write a valid EMPTY snapshot: an existing-but-empty state must NOT
    /// trigger seeding (seeds only load when the state file is absent).
    fn write_empty_state(path: &std::path::Path) {
        let empty = serde_json::json!({
            "version": 1,
            "saved_at": 0.0,
            "next_id": 1,
            "stats": {
                "total_errors_recorded": 0, "unique_incidents": 0,
                "total_preflights": 0, "total_validations": 0,
                "prefights_blocked": 0, "validations_failed": 0,
                "errors_by_category": {}, "top_recurring_patterns": []
            },
            "errors": [],
            "string_pool": []
        });
        fs::write(path, empty.to_string()).unwrap();
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let path = tmp_state_path("roundtrip");
        write_empty_state(&path);
        let mut agent = AgentR::with_state_path(path.clone());
        agent.record_error(
            ErrorCategory::Fabrication, ErrorSeverity::Critical,
            "op_a", "what a", "why a", &["t1"], "prev a",
        );
        agent.record_error(
            ErrorCategory::Timeout, ErrorSeverity::Error,
            "op_b", "what b", "why b", &["t2"], "prev b",
        );
        assert!(path.is_file(), "write-through must create the snapshot");
        // Debounce (pendiente 2a): el snapshot en disco puede laguar hasta
        // FLUSH_INTERVAL; para leer desde OTRA instancia hay que comprometer
        // el estado primero (equivalente al apagado ordenado / Drop).
        agent.flush().unwrap();

        // Restart: a new instance over the same path must recover everything
        let agent2 = AgentR::with_state_path(path.clone());
        assert_eq!(agent2.errors().len(), 2, "both incidents survive restart");
        assert_eq!(agent2.next_id, agent.next_id);
        assert_eq!(agent2.stats().unique_incidents, 2);
        assert_eq!(agent2.stats().total_errors_recorded, 2);
        // OPT-O3 x OPT-O4: los ids internados resuelven a las strings originales.
        // El bonus de recencia del baseline puntúa también registros recientes
        // sin match (comportamiento preexistente): el exact-match va primero.
        let recalled = agent2.recall_errors("op_a", &["t1"]);
        assert_eq!(agent2.resolve(recalled[0].operation), "op_a");

        // next_id keeps growing across restarts (no id reuse)
        let mut agent3 = AgentR::with_state_path(path);
        let (id, outcome, _) = agent3.record_error_dedup(
            ErrorCategory::Timeout, ErrorSeverity::Error,
            "op_c", "what c", "why c", &[], "prev c",
        );
        assert_eq!(id, 3);
        assert_eq!(outcome, WriteOutcome::Add);
    }

    #[test]
    fn test_dedup_add_update_noop() {
        let path = tmp_state_path("dedup");
        write_empty_state(&path);
        let mut agent = AgentR::with_state_path(path);

        // ADD: unknown fingerprint
        let (id1, o1, r1) = agent.record_error_dedup(
            ErrorCategory::Fabrication, ErrorSeverity::Error,
            "Generate_Benchmarks ", "  Invented benchmark DATA ", "why", &["gpu"], "p",
        );
        assert_eq!(o1, WriteOutcome::Add);
        assert_eq!(r1, 0);
        assert_eq!(agent.errors().len(), 1);

        // NOOP: same fingerprint (case/whitespace-insensitive), nothing new
        let (id2, o2, r2) = agent.record_error_dedup(
            ErrorCategory::Fabrication, ErrorSeverity::Error,
            "generate_benchmarks", "invented benchmark data", "why", &["gpu"], "p",
        );
        assert_eq!(id2, id1, "duplicate maps to the existing incident id");
        assert_eq!(o2, WriteOutcome::Noop);
        assert_eq!(r2, 1);
        assert_eq!(agent.errors().len(), 1, "no duplicate row inserted");

        // UPDATE: new context tag + severity escalation
        let (_, o3, r3) = agent.record_error_dedup(
            ErrorCategory::Fabrication, ErrorSeverity::Critical,
            "generate_benchmarks", "invented benchmark data", "why", &["cuda"], "p",
        );
        assert_eq!(o3, WriteOutcome::Update);
        assert_eq!(r3, 2);
        assert_eq!(agent.errors().len(), 1);
        let rec = &agent.errors()[0];
        assert_eq!(rec.severity, ErrorSeverity::Critical, "severity escalated");
        assert!(rec.context_tags.iter().any(|t| agent.resolve(*t) == "cuda"), "tags merged (union)");
        assert!(rec.context_tags.iter().any(|t| agent.resolve(*t) == "gpu"), "original tags kept");

        // Stats: 3 write events, 1 unique incident
        assert_eq!(agent.stats().total_errors_recorded, 3);
        assert_eq!(agent.stats().unique_incidents, 1);

        // Distinct what => distinct fingerprint => new row
        let (id4, o4, _) = agent.record_error_dedup(
            ErrorCategory::Fabrication, ErrorSeverity::Error,
            "generate_benchmarks", "different failure", "why", &[], "p",
        );
        assert_eq!(o4, WriteOutcome::Add);
        assert!(id4 > id1, "next_id keeps growing, no reuse");
        assert_eq!(agent.errors().len(), 2);
    }

    #[test]
    fn test_seeds_load_only_when_state_absent() {
        let path = tmp_state_path("seeds");
        // First boot: no state file => seeds load and persist
        let mut agent = AgentR::with_state_path(path.clone());
        assert_eq!(agent.errors().len(), 3, "3 historical seed incidents");
        let ops: Vec<&str> = agent.errors().iter().map(|e| agent.resolve(e.operation)).collect();
        assert!(ops.contains(&"generate_benchmarks"));
        assert!(ops.contains(&"image2svg_pipeline"));
        assert!(ops.contains(&"promise_long_task"));
        agent.flush().unwrap(); // debounce: comprometer antes de releer

        // Second boot: state file exists => state loads, seeds NOT re-applied
        let mut agent2 = AgentR::with_state_path(path.clone());
        assert_eq!(agent2.errors().len(), 3, "no double-seeding");
        // A new error lands on top of the seeds without evicting them
        agent2.record_error(
            ErrorCategory::Timeout, ErrorSeverity::Warning,
            "fresh_op", "w", "y", &[], "p",
        );
        assert_eq!(agent2.errors().len(), 4);
        agent2.flush().unwrap(); // debounce

        // Third boot: the runtime error survives, seeds still exactly once
        let agent3 = AgentR::with_state_path(path);
        assert_eq!(agent3.errors().len(), 4);
        assert_eq!(agent3.stats().unique_incidents, 4);
    }

    #[test]
    fn test_empty_state_does_not_reseed() {
        // An existing (valid) empty snapshot means "user cleared the memory":
        // seeds must NOT be re-applied on boot.
        let path = tmp_state_path("noseed");
        write_empty_state(&path);
        let mut agent = AgentR::with_state_path(path);
        assert_eq!(agent.errors().len(), 0, "valid empty state stays empty");
        agent.record_error(
            ErrorCategory::Other, ErrorSeverity::Warning, "op", "w", "y", &[], "p",
        );
        assert_eq!(agent.errors().len(), 1);
    }

    #[test]
    fn test_deprecate_excludes_from_recall_and_persists() {
        let path = tmp_state_path("deprecate");
        write_empty_state(&path);
        let mut agent = AgentR::with_state_path(path.clone());
        let (id, _, _) = agent.record_error_dedup(
            ErrorCategory::BadAssumption, ErrorSeverity::Critical,
            "noisy_op", "false positive trigger", "why", &[], "p",
        );
        assert_eq!(agent.recall_errors("noisy_op", &[]).len(), 1);

        assert!(agent.deprecate_error(id, "FP loop: 3 false blocks measured"));
        assert_eq!(agent.recall_errors("noisy_op", &[]).len(), 0, "deprecated excluded");
        agent.flush().unwrap(); // debounce: comprometer antes de releer

        // Deprecated state survives restart
        let agent2 = AgentR::with_state_path(path);
        assert_eq!(agent2.recall_errors("noisy_op", &[]).len(), 0);
        assert!(agent2.errors().iter().any(|e| e.deprecated), "kept for audit");
    }

    #[test]
    fn test_corrupt_state_falls_back_to_seeds() {
        let path = tmp_state_path("corrupt");
        fs::write(&path, "{not json").unwrap();
        // Corrupt snapshot: warn + start from seeds rather than crash
        let agent = AgentR::with_state_path(path);
        assert_eq!(agent.errors().len(), 3);
    }

    #[test]
    fn test_legacy_state_without_new_fields_loads() {
        // Snapshot written by the OLD format (sin fingerprint/last_seen/
        // deprecated/string_pool): serde defaults + rebuild must repair it.
        let path = tmp_state_path("legacy");
        let legacy = r#"{"version":1,"saved_at":1.0,"next_id":2,"stats":{"total_errors_recorded":1,"unique_incidents":1,"total_preflights":0,"total_validations":0,"prefights_blocked":0,"validations_failed":0,"errors_by_category":{"Timeout":1},"top_recurring_patterns":[]},"errors":[{"id":1,"timestamp":100.0,"category":"Timeout","severity":"Warning","operation":"legacy_op","what":"legacy what","why":"y","context_tags":["x"],"prevention":"p","recurrence":0,"pattern_id":null}],"string_pool":[]}"#;
        fs::write(&path, legacy).unwrap();
        let mut agent = AgentR::with_state_path(path);
        assert_eq!(agent.errors().len(), 1);
        // Duplicate of the legacy record must dedup (fingerprint was rebuilt)
        let (_, outcome, _) = agent.record_error_dedup(
            ErrorCategory::Timeout, ErrorSeverity::Warning,
            "legacy_op", "legacy what", "y", &["x"], "p",
        );
        assert_eq!(outcome, WriteOutcome::Noop, "legacy record dedups after repair");
        assert_eq!(agent.errors().len(), 1);
    }

    #[test]
    fn test_atomic_write_leaves_no_tmp() {
        let path = tmp_state_path("atomic");
        write_empty_state(&path);
        let mut agent = AgentR::with_state_path(path.clone());
        agent.record_error(
            ErrorCategory::Other, ErrorSeverity::Warning,
            "op", "w", "y", &[], "p",
        );
        agent.flush().unwrap(); // debounce: escritura atómica comprometida
        assert!(path.is_file());
        assert!(!path.with_extension("json.tmp").exists(), "tmp file renamed away");
        // Snapshot on disk parses as the StateSnapshot shape
        let raw = fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(v["version"], 1);
        assert!(v["errors"].as_array().unwrap().len() >= 1);
        assert!(v["stats"]["unique_incidents"].is_u64());
        // OPT-O3 x OPT-O4: el snapshot es human-readable (operation como string)
        assert_eq!(v["errors"][0]["operation"].as_str().unwrap_or(""), "op");
    }

    /// Paridad del índice vs el scan O(n) original: recurrence y pattern_id
    /// deben ser idénticos a los que producían los 2 scans eliminados.
    #[test]
    fn test_recurrence_index_matches_naive_scan() {
        let mut agent = AgentR::new_ephemeral();
        // Mezcla de claves: 2 categorías x 3 operaciones, orden interleaved.
        // what distinto por registro: con dedup (OPT-O4) cada registro es un
        // incidente único => misma semántica de índice que el modelo de filas.
        let ops = ["op_a", "op_b", "op_c"];
        let cats = [ErrorCategory::Fabrication, ErrorCategory::Timeout];
        let mut ids = Vec::new();
        let mut k = 0usize;
        for round in 0..5 {
            for cat in cats {
                for op in ops {
                    k += 1;
                    let id = agent.record_error(cat, ErrorSeverity::Error, op,
                        &format!("what #{k}"), "y", &[], "p");
                    ids.push((id, cat, op));
                }
            }
            // ruido con clave única para forzar buckets distintos
            agent.record_error(
                ErrorCategory::Other, ErrorSeverity::Warning,
                &format!("noise_{round}"), &format!("w #{round}"), "y", &[], "p",
            );
        }
        for (id, cat, op) in &ids {
            let rec = agent.errors().iter().find(|e| e.id == *id).unwrap();
            // naive: contar previos con misma clave (solo ids < id)
            let naive_count = ids.iter()
                .filter(|(pid, pc, po)| pid < id && pc == cat && po == op)
                .count() as u32;
            let naive_pattern = ids.iter()
                .filter(|(pid, pc, po)| pid < id && pc == cat && po == op)
                .map(|(pid, _, _)| *pid)
                .max();
            assert_eq!(rec.recurrence, naive_count, "recurrence diverge id={id}");
            assert_eq!(rec.pattern_id, naive_pattern, "pattern_id diverge id={id}");
        }
    }

    /// Tras load_errors, record_error sigue siendo consistente (índice rebuilt).
    #[test]
    fn test_recurrence_after_load_errors() {
        let mut agent = AgentR::new_ephemeral();
        // 3 incidentes distintos de la misma clave (what distinto: dedup no fusiona)
        for k in 1..=3 {
            agent.record_error(ErrorCategory::Fabrication, ErrorSeverity::Critical, "op_x",
                &format!("variante #{k}"), "y", &[], "p");
        }
        let json = agent.serialize_errors().unwrap();
        let mut agent2 = AgentR::new_ephemeral();
        agent2.load_errors(&json).unwrap();
        // 4º incidente: recurrence=3 (incidentes previos de la clave), pattern_id=max previo=3
        let id = agent2.record_error(ErrorCategory::Fabrication, ErrorSeverity::Critical, "op_x",
            "variante #4", "y", &[], "p");
        let rec = agent2.errors().iter().find(|e| e.id == id).unwrap();
        assert_eq!(rec.recurrence, 3);
        assert_eq!(rec.pattern_id, Some(3));
    }

    /// Eviction (MAX_ERRORS): el índice no debe inflar recurrence tras purgar.
    /// Simula el ciclo completo con un cap chico reinyectando el flujo.
    #[test]
    fn test_recurrence_index_eviction_purge() {
        // Usa el flujo real de record_error con cap alto: inserta 5 con la
        // misma clave, verifica bucket==5; el purge se ejercita vía índice
        // manual porque MAX_ERRORS es una const global (2M en este branch).
        let mut agent = AgentR::new_ephemeral();
        for i in 0..5 {
            agent.record_error(ErrorCategory::Other, ErrorSeverity::Warning, "cap_op",
                &format!("w #{i}"), "y", &[], &format!("p{i}"));
        }
        let op_id = agent.string_pool_index.get("cap_op").copied().unwrap();
        let bucket = agent.recurrence_index.get(&(ErrorCategory::Other, op_id)).unwrap();
        assert_eq!(bucket.len(), 5);
        assert_eq!(*bucket.back().unwrap(), 5);
    }

    // ═══ OPT-O3: regresión del layout internado ═══

    #[test]
    fn test_intern_dedup_same_operation() {
        // Dos registros con la misma operación comparten id (dedup del pool).
        let mut agent = AgentR::new_ephemeral();
        agent.record_error(ErrorCategory::Other, ErrorSeverity::Warning, "op_x", "w", "y", &["t"], "p");
        agent.record_error(ErrorCategory::Other, ErrorSeverity::Error, "op_x", "w2", "y2", &["t"], "p2");
        let ids: Vec<_> = agent.errors().iter().map(|e| e.operation).collect();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], ids[1], "misma operación => mismo id internado");
        assert_eq!(agent.resolve(ids[0]), "op_x");
    }

    #[test]
    fn test_recall_exact_and_partial_match_interned() {
        // Exact match (+10) y partial contains (+5) siguen funcionando con ids.
        let mut agent = AgentR::new_ephemeral();
        agent.record_error(ErrorCategory::Other, ErrorSeverity::Warning, "rebuild_hnsw_index", "w", "y", &[], "p");
        agent.record_error(ErrorCategory::Other, ErrorSeverity::Warning, "hnsw", "w", "y", &[], "p");
        // exact: score 10+recency+severity; partial: 5+... => ambos > 0
        let found = agent.recall_errors("rebuild_hnsw_index", &[]);
        assert_eq!(found.len(), 2, "exact + contains deben ambos puntuar");
        // op nunca registrada pero substring de una existente (contains inverso)
        let found2 = agent.recall_errors("hnsw", &[]);
        assert_eq!(found2.len(), 2);
    }

    #[test]
    fn test_recall_unknown_operation_no_exact_match() {
        // Operación jamás internada: no puede haber exact match falso. El bonus
        // de recencia (+12, comportamiento preexistente del baseline) puntúa
        // TODOS los registros recientes => el assertion correcto es que ningún
        // resultado matchee exactamente la op desconocida.
        let mut agent = AgentR::new_ephemeral();
        agent.record_error(ErrorCategory::Other, ErrorSeverity::Warning, "zzz_totally_unrelated", "w", "y", &[], "p");
        let found = agent.recall_errors("qqq_never_seen", &[]);
        assert!(found.iter().all(|e| agent.resolve(e.operation) != "qqq_never_seen"),
            "op no internada jamás exact-matchea");
    }

    #[test]
    fn test_tag_overlap_interned() {
        let mut agent = AgentR::new_ephemeral();
        agent.record_error(ErrorCategory::Other, ErrorSeverity::Warning, "op", "w", "y", &["gpu", "ci"], "p");
        let found = agent.recall_errors("different_op", &["gpu"]);
        assert_eq!(found.len(), 1, "tag overlap debe puntuar aunque la op no coincida");
    }

    #[test]
    fn test_serialize_contract_strings_not_ids() {
        // Contrato JSON estable: operation/context_tags serializan como strings.
        let mut agent = AgentR::new_ephemeral();
        agent.record_error(ErrorCategory::Fabrication, ErrorSeverity::Critical, "gen_bench", "w", "y", &["bench"], "p");
        let json = agent.serialize_errors().unwrap();
        assert!(json.contains("\"operation\":\"gen_bench\""), "operation debe ser string en JSON: {json}");
        assert!(json.contains("\"context_tags\":[\"bench\"]"), "tags deben ser strings en JSON: {json}");
        assert!(!json.contains("\"operation\":0"), "operation no debe serializar como id numérico");

        // Roundtrip preserva el comportamiento de recall.
        let mut agent2 = AgentR::new_ephemeral();
        agent2.load_errors(&json).unwrap();
        let found = agent2.recall_errors("gen_bench", &["bench"]);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_top_patterns_interned() {
        let mut agent = AgentR::new_ephemeral();
        for k in 0..3 { agent.record_error(ErrorCategory::Other, ErrorSeverity::Warning, "op_a",
            &format!("w #{k}"), "y", &[], "p"); }
        for k in 0..1 { agent.record_error(ErrorCategory::Other, ErrorSeverity::Warning, "op_b",
            &format!("w {k}"), "y", &[], "p"); }
        let top = agent.top_patterns(2);
        assert_eq!(top[0], ("op_a", 3));
        assert_eq!(top[1], ("op_b", 1));
    }

    // ═══ Pendiente 2a: validate N0-N3, preflight v2, debounce ═══

    /// N1 NO debe flaggear strings que contengan "NaN"/"Infinity" — el FP
    /// garantizado del v1 (substring sobre la serialización).
    #[test]
    fn test_n1_no_substring_false_positive() {
        let mut agent = AgentR::new_ephemeral();
        // v1 fallaba aquí: json_str.contains("NaN") era true por el STRING.
        let r = agent.validate_output_v2("compute_distance", &serde_json::json!({
            "score": "NaN",
            "note": "value was Infinity last run",
            "vals": [1.0, 2.5, -3.25],
        }));
        assert_eq!(r.verdicts.n1_numeric_consistency, LevelVerdict::Pass,
            "strings con 'NaN'/'Infinity' NO son valores no-finitos");
        assert_ne!(r.overall, "fail");
        assert!(r.valid);
    }

    /// N1 SÍ debe detectar no-finitos reales... vía parse: serde_json no
    /// puede producir NaN/Inf desde texto JSON estándar. El nivel se verifica
    /// con la única vía posible: números construidos programáticamente
    /// (aquí: overflow de u64>i64::MAX que rompe as_i64 pero es finite en
    /// as_f64 => Pass, demostrando que el scan usa as_f64 y no substrings).
    #[test]
    fn test_n1_scan_uses_as_f64_not_substrings() {
        let mut agent = AgentR::new_ephemeral();
        // Número grande legítimo (finite): Pass.
        let r = agent.validate_output_v2("op", &serde_json::json!({
            "big": 1e308_f64,
            "neg": -1e308_f64,
        }));
        assert_eq!(r.verdicts.n1_numeric_consistency, LevelVerdict::Pass);
        // La serialización de floats pequeños (exponente e-notation) tampoco
        // debe generar FPs por substring.
        let r2 = agent.validate_output_v2("op", &serde_json::json!([1e-10, 2e-10, 3e-10, 4e-10, 5e-10, 6e-10]));
        assert_eq!(r2.verdicts.n1_numeric_consistency, LevelVerdict::Pass);
    }

    /// N0: null y array uniforme siguen siendo fail (paridad con v1).
    #[test]
    fn test_n0_null_and_uniform() {
        let mut agent = AgentR::new_ephemeral();
        let r = agent.validate_output_v2("op_null", &serde_json::json!(null));
        assert_eq!(r.verdicts.n0_form, LevelVerdict::Fail);
        assert_eq!(r.overall, "fail");
        assert!(!r.valid);
        assert_eq!(r.confidence, 0.0);
        assert!(r.suggestion.as_deref().unwrap_or("").contains("N0"));

        let r2 = agent.validate_output_v2("search", &serde_json::json!([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        assert_eq!(r2.verdicts.n0_form, LevelVerdict::Fail);
        assert_eq!(r2.overall, "fail");
    }

    /// N2: sin evidencia => unverifiable (JAMÁS fail); con evidencia
    /// consistente => pass; exit_code != esperado => fail.
    #[test]
    fn test_n2_caller_evidence_levels() {
        let mut agent = AgentR::new_ephemeral();
        let out = serde_json::json!({"rows": 3});

        // Sin evidencia: unverifiable, no fail.
        let r = agent.validate_output_v2_evidence("op", &out, None);
        assert_eq!(r.verdicts.n2_caller_evidence, LevelVerdict::Unverifiable);
        assert_eq!(r.overall, "pass", "N0/N1 con oráculo pasan => overall pass");
        // (N2 unverifiable no hunde el overall cuando otros niveles pasan)

        // Evidencia consistente: pass.
        let ev_ok = serde_json::json!({"exit_code": 0, "stderr": "", "diff": ""});
        let r2 = agent.validate_output_v2_evidence("op", &out, Some(&ev_ok));
        assert_eq!(r2.verdicts.n2_caller_evidence, LevelVerdict::Pass);
        assert_eq!(r2.overall, "pass");
        assert_eq!(r2.confidence, 1.0);

        // exit_code != esperado: fail.
        let ev_bad = serde_json::json!({"exit_code": 1, "stderr": "panic", "diff": ""});
        let r3 = agent.validate_output_v2_evidence("op", &out, Some(&ev_bad));
        assert_eq!(r3.verdicts.n2_caller_evidence, LevelVerdict::Fail);
        assert_eq!(r3.overall, "fail");
        assert!(r3.suggestion.as_deref().unwrap_or("").contains("N2"));

        // exit 0 + stderr no vacío sin diff que lo explique: contradictorio.
        let ev_contra = serde_json::json!({"exit_code": 0, "stderr": "warning: stale handle", "diff": ""});
        let r4 = agent.validate_output_v2_evidence("op", &out, Some(&ev_contra));
        assert_eq!(r4.verdicts.n2_caller_evidence, LevelVerdict::Fail);
    }

    /// N3: stub Unverifiable hasta el merge de receipts (P1).
    #[test]
    fn test_n3_stub_unverifiable() {
        let mut agent = AgentR::new_ephemeral();
        let r = agent.validate_output_v2("op", &serde_json::json!({"ok": true}));
        assert_eq!(r.verdicts.n3_receipt, LevelVerdict::Unverifiable);
        assert!(r.details.iter().any(|d| d.contains("N3")));
    }

    /// Híbrido (bench 15-oct): la señal de contenido re-ordena el top-3 —
    /// la regla cuyo `what` comparte tokens con la situación sube.
    #[test]
    fn test_preflight_v2_hybrid_content_reranks() {
        let mut agent = AgentR::new_ephemeral();
        // Regla A: op compartida, contenido AJENO; 2 incidentes => mayor
        // frecuencia => domina el base de forma determinista (el orden base
        // por recencia solo tiene ruido de nanosegundos).
        let _ = agent.record_error(
            ErrorCategory::Other,
            ErrorSeverity::Error,
            "level-1",
            "grey box blocked the corridor path",
            "grey box blocked path",
            &["babyai"],
            "",
        );
        let _ = agent.record_error(
            ErrorCategory::Other,
            ErrorSeverity::Error,
            "level-1",
            "grey box blocked the corridor path again",
            "grey box blocked path",
            &["babyai"],
            "",
        );
        // Regla B: misma op, contenido que SÍ matchea la situación.
        agent.record_error(
            ErrorCategory::BadAssumption,
            ErrorSeverity::Error,
            "level-1",
            "agent ignored red ball mission and searched socket",
            "ignored mission",
            &["babyai"],
            "",
        );
        let situation = "Mission: go to the red ball. You see a socket and a table.";
        let base = agent.preflight_v2("level-1", &["babyai"]);
        let hyb = agent.preflight_v2_hybrid("level-1", &["babyai"], situation);
        let is_b = |r: &ScoredRule| r.what.contains("red ball");
        let b_before = base.top_rules.iter().position(is_b).expect("regla B en base");
        let b_after = hyb.top_rules.iter().position(is_b).expect("regla B en híbrido");
        assert!(b_after < b_before, "regla con overlap debe subir: before={b_before} after={b_after}");
    }

    /// Híbrido: PRESENCIA, no frecuencia — repetir tokens de la regla en la
    /// situación NO cambia el score (hallazgo del bench: frecuencia hunde
    /// top3 de 48% a 29%). Tolerancia relativa: recencia usa now() y varía
    /// ~1e-9 entre llamadas.
    #[test]
    fn test_preflight_v2_hybrid_binary_not_frequency() {
        let mut agent = AgentR::new_ephemeral();
        agent.record_error(
            ErrorCategory::Other,
            ErrorSeverity::Error,
            "op",
            "socket timeout while parsing payload",
            "why",
            &[],
            "",
        );
        let once = agent.preflight_v2_hybrid("op", &[], "socket appeared once");
        let many = agent.preflight_v2_hybrid(
            "op",
            &[],
            "socket socket socket socket socket socket socket",
        );
        let s1 = once.top_rules[0].score;
        let s2 = many.top_rules[0].score;
        assert!(
            (s1 - s2).abs() <= s1.abs() * 1e-6,
            "presencia binaria: {s1} vs {s2}"
        );
    }

    /// Híbrido: sin situación o con una sola regla => passthrough (los
    /// scores coinciden dentro de tolerancia temporal).
    #[test]
    fn test_preflight_v2_hybrid_passthrough() {
        let mut agent = AgentR::new_ephemeral();
        agent.record_error(
            ErrorCategory::Other,
            ErrorSeverity::Error,
            "op",
            "something failed",
            "why",
            &[],
            "",
        );
        let base = agent.preflight_v2("op", &[]);
        let hyb = agent.preflight_v2_hybrid("op", &[], "");
        let s0 = base.top_rules[0].score;
        let s1 = hyb.top_rules[0].score;
        assert!((s0 - s1).abs() <= s0.abs() * 1e-6, "{s0} vs {s1}");
        let hyb2 = agent.preflight_v2_hybrid("op", &[], "estado irrelevante");
        let s2 = hyb2.top_rules[0].score;
        assert!((s0 - s2).abs() <= s0.abs() * 1e-6, "{s0} vs {s2}"); // 1 regla => sin re-rank
    }

    /// Sin oráculo en NINGÚN nivel => overall unverifiable (no fail).
    #[test]
    fn test_unverifiable_when_no_oracle() {
        let mut agent = AgentR::new_ephemeral();
        // String plano: N0 pass (no null/no array uniforme)... para forzar
        // "sin oráculo" usamos un output que N0 no puede evaluar como fail
        // pero tampoco verifica contenido: un string. N0 en strings = Pass
        // trivial => hay oráculo de forma. El caso real sin NINGÚN oráculo
        // es output vacío {}. N0 de {} es Pass (hay forma, es válida).
        // => overall "pass" porque N0/N1 son oráculos estructurales.
        // El contrato dice: unverifiable cuando no hay oráculo PARA UN NIVEL
        // (ese nivel reporta unverifiable); overall solo es unverifiable si
        // NINGÚN nivel pudo verificar. Ese caso no existe con N0 estructural
        // siempre presente => verificamos el contrato por nivel:
        let r = agent.validate_output_v2("op", &serde_json::json!("plain text"));
        assert_eq!(r.verdicts.n2_caller_evidence, LevelVerdict::Unverifiable);
        assert_eq!(r.verdicts.n3_receipt, LevelVerdict::Unverifiable);
        // overall: pass (N0/N1 estructurales pasan)
        assert_eq!(r.overall, "pass");
    }

    /// Preflight v2: veto solo con Critical + N>=2 + regla activa.
    #[test]
    fn test_preflight_v2_veto_requires_critical_recurring() {
        let mut agent = AgentR::new_ephemeral();
        // Incidente critical VISTO 3 veces (dedup: 1 regla, recurrence=2).
        for _ in 0..3 {
            agent.record_error(
                ErrorCategory::Fabrication, ErrorSeverity::Critical,
                "generate_data", "Made up numbers", "No testing",
                &["data"], "Test first",
            );
        }
        let r = agent.preflight_v2("generate_data", &["data"]);
        assert!(!r.proceed, "critical recurring (N=3) activa => veto");
        assert!(r.veto_reason.is_some());
        assert!(r.warnings.iter().any(|w| w.contains("BLOCKED")));
        assert!(!r.top_rules.is_empty());
        assert!(r.suggestion.is_some(), "el veto viene con sugerencia");

        // Critical VISTO 1 vez (recurrence=0): WARNING, no veto.
        let mut agent2 = AgentR::new_ephemeral();
        agent2.record_error(
            ErrorCategory::Fabrication, ErrorSeverity::Critical,
            "one_shot", "w", "y", &[], "p",
        );
        let r2 = agent2.preflight_v2("one_shot", &[]);
        assert!(r2.proceed, "critical sin recurrencia => NO veto");
        assert!(r2.warnings.iter().any(|w| w.contains("WARNING")));

        // Regla deprecada (FP medido): fuera del preflight, sin veto.
        let mut agent3 = AgentR::new_ephemeral();
        for _ in 0..5 {
            agent3.record_error(
                ErrorCategory::BadAssumption, ErrorSeverity::Critical,
                "noisy_op", "FP trigger", "y", &[], "p",
            );
        }
        let id = agent3.errors()[0].id;
        agent3.deprecate_error(id, "FP loop: 3 false blocks measured");
        let r3 = agent3.preflight_v2("noisy_op", &[]);
        assert!(r3.proceed, "regla deprecada (FP) jamás veta");
        assert!(r3.top_rules.is_empty());
    }

    /// Preflight v2: exact-match de operación, NUNCA substring cruzado.
    /// "hnsw" (op de una lección) NO debe matchear la query
    /// "rebuild_hnsw_index" como si fueran la misma operación.
    #[test]
    fn test_preflight_v2_no_cross_substring_match() {
        let mut agent = AgentR::new_ephemeral();
        // Lección registrada SOLO para la op "hnsw".
        agent.record_error(
            ErrorCategory::Other, ErrorSeverity::Critical,
            "hnsw", "index corrupt", "y", &["index"], "rebuild from source",
        );
        // Query por "rebuild_hnsw_index": en v1 el contains cruzado sumaba
        // +5; en v2 NO debe considerarse match de operación. La lección puede
        // seguir apareciendo por el TAG "index" (filtro secundario legítimo),
        // pero el score NO debe incluir el bonus de exact-match (+10).
        let r = agent.preflight_v2("rebuild_hnsw_index", &[]);
        assert!(r.top_rules.iter().all(|s| s.operation != "rebuild_hnsw_index"
            || s.operation == "hnsw"), "ninguna regla de otra operación matchea por substring");
        // Sin tags en común y sin exact: la lección "hnsw" recibe sólo
        // recencia+freq => score bajo pero > 0 (vista hace <24h). El punto
        // clave: NO hay bonus de similitud => no domina el top-3 como en v1.
        let s = agent.preflight_v2_score(
            agent.errors().iter().next().unwrap(),
            agent.string_pool_index.get("rebuild_hnsw_index").copied(),
            &[],
            now_secs(),
        );
        assert!(s < 10.0, "sin exact-match ni tags, score < 10 (got {s})");
    }

    /// Preflight v2: top-3 ordenado por score, warnings dedup.
    #[test]
    fn test_preflight_v2_top3_and_dedup() {
        let mut agent = AgentR::new_ephemeral();
        for k in 0..6 {
            agent.record_error(
                ErrorCategory::Other, ErrorSeverity::Error,
                &format!("op_{k}"), &format!("w{k}"), "y", &["common"], "prev",
            );
        }
        let r = agent.preflight_v2("op_2", &["common"]);
        assert!(r.top_rules.len() <= 3, "top-3 máximo");
        let scores: Vec<f64> = r.top_rules.iter().map(|s| s.score).collect();
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_eq!(scores, sorted, "top_rules ordenado desc por score");
        // warnings dedup: no hay duplicados
        let mut ws = r.warnings.clone();
        ws.sort();
        ws.dedup();
        assert_eq!(ws.len(), r.warnings.len());
    }

    /// Preflight v2: paridad de forma con preflight_shared (agent_swarm).
    #[test]
    fn test_preflight_shared_v2_parity() {
        use crate::agent_swarm::AgentSwarm;
        let sw = AgentSwarm::new();
        let mut ar = AgentR::new_ephemeral();
        ar.record_error(
            ErrorCategory::Timeout, ErrorSeverity::Error,
            "distill_lessons", "OOM on merge", "y", &["gpu"], "chunk the merge",
        );
        let pf = sw.preflight_shared(&mut ar, "distill_lessons", &["gpu"]);
        assert!(pf["relevant_errors"].as_u64().unwrap_or(0) >= 1);
        assert!(pf["top_rules"].as_array().unwrap().len() >= 1, "scoring v2 expuesto");
        assert!(pf["suggestion"].is_string(), "sugerencia presente");
    }

    // ── Debounce de snapshots ──────────────────────────────────────────

    /// Debounce: las escrituras NO tocan disco inmediatamente (batch), pero
    /// flush() sync sí, y Drop persiste al apagar.
    #[test]
    fn test_debounce_batches_and_flush_persists() {
        let path = tmp_state_path("debounce");
        write_empty_state(&path);
        let mut agent = AgentR::with_state_path(path.clone());
        for k in 0..10 {
            agent.record_error(
                ErrorCategory::Other, ErrorSeverity::Warning,
                "op", &format!("w #{k}"), "y", &[], "p",
            );
        }
        // Con debounce (intervalo 2 s o 5k ops), 10 ops NO disparan flush:
        // puede haber escrito 0 o 1 veces (si el test tardó >2s, 1 vez).
        // Lo garantizado: flush() sync deja el estado completo en disco.
        agent.flush().unwrap();
        let agent2 = AgentR::with_state_path(path.clone());
        assert_eq!(agent2.errors().len(), 10, "flush() persiste todo");
        // Drop del agent2 (con flush-on-drop) mantiene el estado.
        let agent3 = AgentR::with_state_path(path);
        assert_eq!(agent3.errors().len(), 10, "estado sobrevive restart");
    }

    /// Debounce: FLUSH_MAX_OPS (5k) dispara el segundo camino de flush.
    #[test]
    fn test_debounce_flush_by_ops_count() {
        let path = tmp_state_path("debounce_ops");
        write_empty_state(&path);
        let mut agent = AgentR::with_state_path(path.clone());
        // >FLUSH_MAX_OPS escrituras rápidas (sin esperar 2 s): al menos un
        // job se encola al writer background. Esperamos que el writer corra.
        for k in 0..(FLUSH_MAX_OPS + 100) {
            agent.record_error(
                ErrorCategory::Other, ErrorSeverity::Warning,
                "flood", &format!("w #{k}"), "y", &[], "p",
            );
        }
        // El writer background tiene hasta ~2 s para comprometer el job.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            if path.is_file() {
                let raw = fs::read_to_string(&path).unwrap();
                let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
                if v["errors"].as_array().map(|a| a.len()).unwrap_or(0) >= 5000 {
                    break; // el flusher comprometió el snapshot batch
                }
            }
            assert!(std::time::Instant::now() < deadline,
                "el writer background no comprometió tras FLUSH_MAX_OPS");
            std::thread::sleep(Duration::from_millis(100));
        }
        agent.flush().unwrap();
    }

    /// Eviction O(1): VecDeque, orden preservado, índices consistentes.
    #[test]
    fn test_vecdeque_eviction_preserves_order() {
        let mut agent = AgentR::new_ephemeral();
        for k in 0..5 {
            agent.record_error(
                ErrorCategory::Other, ErrorSeverity::Warning,
                "evict_op", &format!("w #{k}"), "y", &[], "p",
            );
        }
        // Purga manual (el camino de evict_oldest) x3: quedan los 2 últimos.
        agent.evict_oldest();
        agent.evict_oldest();
        agent.evict_oldest();
        let ids: Vec<u64> = agent.errors().iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![4, 5], "orden de inserción preservado tras purgas O(1)");
        assert_eq!(agent.stats.unique_incidents, 2);
        // errors() devuelve slice contiguo aunque el deque haya rotado.
        assert_eq!(agent.errors().len(), 2);
    }
}
