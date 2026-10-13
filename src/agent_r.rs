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
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
}

const MAX_ERRORS: usize = 2_000_000; // rambench-nocap: relevo el cap para medir la firma real (solo en este branch)
const MAX_REASONING_STEPS: usize = 1000;
const REPETITION_THRESHOLD: usize = 3;
/// Snapshot format version; loaders reject newer versions. (OPT-O4)
const SNAPSHOT_VERSION: u32 = 1;

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
    errors: Vec<ErrorRecord>,
    next_id: u64,
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
    /// Dedup index: fingerprint_hash (FNV-1a estable) -> id del incidente.
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
}

impl AgentR {
    /// Default state snapshot path when AGENT_R_STATE_PATH is not set:
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
            errors: Vec::new(),
            next_id: 1,
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
                        agent.load_seeds();
                    }
                }
            } else {
                agent.load_seeds();
            }
        }
        agent
    }

    /// Create an Agent-R with persistence disabled (RAM only): modo por
    /// defecto de los benches/harness (no tocan disco). (OPT-O4)
    pub fn new_ephemeral() -> Self {
        Self {
            errors: Vec::new(),
            next_id: 1,
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

    // ── Persistence: write-through snapshot (OPT-O4) ────────────────────

    /// Atomically write the full state snapshot to `state_path` (tmp+rename).
    /// No-op when persistence is disabled.
    fn persist_snapshot(&self) {
        let Some(path) = &self.state_path else { return };
        if let Err(e) = self.write_state_to(path) {
            eprintln!(
                "[agent-r] WARN: persist to {} failed: {}",
                path.display(),
                e
            );
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
        })
    }

    /// Atomic write: serialize to `<path>.tmp` in the same directory, then
    /// rename over the target. Readers either see the old or the new file.
    fn write_state_to(&self, path: &std::path::Path) -> Result<(), String> {
        let json = self
            .snapshot_json()
            .map_err(|e| format!("serialize: {}", e))?;
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
        self.rebuild_recurrence_index();
        self.rebuild_fingerprint_index();
        Ok(self.errors.len())
    }

    /// Persist right now (public escape hatch for graceful shutdown / tests). (OPT-O4)
    pub fn flush(&self) -> Result<(), String> {
        match &self.state_path {
            Some(p) => self.write_state_to(p),
            None => Ok(()),
        }
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
        self.errors.push(record);
        if self.errors.len() > MAX_ERRORS {
            self.evict_oldest();
        }
        self.stats.unique_incidents = self.errors.len() as u64;
    }

    /// Purga el registro más viejo (eviction) manteniendo recurrence_index,
    /// fingerprint_index y unique_incidents consistentes. (OPT-O1 x OPT-O4)
    fn evict_oldest(&mut self) {
        let evicted = self.errors.remove(0);
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
            let idx = self
                .errors
                .iter()
                .position(|e| e.id == existing_id)
                .expect("fingerprint_index apunta a un id vivo");
            let existing = &mut self.errors[idx];
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

        self.errors.push(record);

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
        let last_ts = self.errors.last().map(|e| e.timestamp).unwrap_or(0.0);

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

    /// Run pre-flight checks before an operation.
    /// Recalls relevant past errors and generates warnings/precautions.
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
    // Introspection
    // ═══════════════════════════════════════════════════════════════════

    /// Get Agent-R statistics.
    pub fn stats(&self) -> &AgentRStats {
        &self.stats
    }

    /// Get all recorded errors.
    pub fn errors(&self) -> &[ErrorRecord] {
        &self.errors
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
            let last_ts = agent.errors.last().map(|e| e.timestamp).unwrap_or(0.0);
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
            .into_iter().map(|e| (recurrence_only_score(&agent, e, "distill_lessons", &["gpu", "ci"], agent.errors.last().map(|x| x.timestamp).unwrap_or(0.0)), e.id))
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
        let agent = AgentR::with_state_path(path.clone());
        assert_eq!(agent.errors().len(), 3, "3 historical seed incidents");
        let ops: Vec<&str> = agent.errors().iter().map(|e| agent.resolve(e.operation)).collect();
        assert!(ops.contains(&"generate_benchmarks"));
        assert!(ops.contains(&"image2svg_pipeline"));
        assert!(ops.contains(&"promise_long_task"));

        // Second boot: state file exists => state loads, seeds NOT re-applied
        let mut agent2 = AgentR::with_state_path(path.clone());
        assert_eq!(agent2.errors().len(), 3, "no double-seeding");
        // A new error lands on top of the seeds without evicting them
        agent2.record_error(
            ErrorCategory::Timeout, ErrorSeverity::Warning,
            "fresh_op", "w", "y", &[], "p",
        );
        assert_eq!(agent2.errors().len(), 4);

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
}
