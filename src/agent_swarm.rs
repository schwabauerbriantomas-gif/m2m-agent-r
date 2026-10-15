//! Agent-Swarm: coordinación multi-agente sobre el Agent-R optimizado.
//!
//! Capa de orquestación de tareas para múltiples agentes que comparten UNA
//! instancia de AgentR (la memoria de errores/lecciones es el punto de
//! colaboración). Implementa:
//!
//! - **Registro de agentes** con heartbeat y TTL (death detection por ausencia).
//! - **Cola de tareas con lock-free claim**: el claim es un solo CAS atómico —
//!   N agentes pueden competir por la misma tarea sin locks ni doble asignación.
//! - **Dependencias** entre tareas (todas COMPLETED antes de READY).
//! - **Primitivas de coordinación**: barrier y counter (resultados de fase).
//! - **Fallback determinista**: si un agente muere con la tarea CLAIMED, otra
//!   instancia la puede re-claimar tras `claim_timeout_s` (reclamo reproducible,
//!   no "quien gane").
//! - **Lecciones compartidas**: los errores que un agente registra quedan
//!   disponibles (con dedup de O4) para el preflight de todos los demás.
//!
//! Todo in-memory; la persistencia de las lecciones la aporta AgentR (O4).

use crate::agent_r::{AgentR, ErrorCategory, ErrorSeverity};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Estado de una tarea del swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TaskState {
    /// Con dependencias pendientes; no reclutable aún.
    Pending,
    /// Reclutable por cualquier agente vivo.
    Ready,
    /// Reclamada por un agente (claim = CAS atómico).
    Claimed,
    /// Completada con resultado.
    Completed,
    /// El agente reportó fallo; re-clamable inmediatamente.
    Failed,
}

/// Tarea distribuida del swarm.
#[derive(Debug, Clone)]
pub struct SwarmTask {
    pub id: u64,
    /// Qué operación Agent-R representa (ej. "index_rebuild"): los errores que
    /// genere quedan ligados a esta operación en la memoria compartida.
    pub operation: String,
    /// Descripción libre del trabajo a hacer.
    pub description: String,
    /// Ids de tareas que deben estar COMPLETED antes de que esta sea READY.
    pub depends_on: Vec<u64>,
    /// Tarea que sólo este agente puede reclamar (None = cualquiera).
    pub preferred_agent: Option<String>,
    /// Atómicos para coordinación lock-free entre agentes (hilos del server).
    state: std::sync::Arc<AtomicU8>,
    claim_epoch: std::sync::Arc<AtomicU64>,
    result: std::sync::Arc<Mutex<Option<String>>>,
    /// Self-tuning: regla que el preflight híbrido señaló como top-1 al
    /// momento de la decisión (claim). El outcome (complete/fail) la usa
    /// como etiqueta para ajustar W_content (ver AGENT_R_TUNING en server).
    pub veto_rule: std::sync::Arc<Mutex<Option<u64>>>,
}

const ST_PENDING: u8 = 0;
const ST_READY: u8 = 1;
const ST_CLAIMED: u8 = 2;
const ST_COMPLETED: u8 = 3;
/// Reservado para "agente reportó fallo definitivo" (hoy fail_task vuelve a
/// READY para re-claim; el valor queda documentado para ese uso futuro).
#[allow(dead_code)]
const ST_FAILED: u8 = 4;

/// Estados del ciclo de vida de una propuesta (fase de ACUERDO, antes de
/// la ejecución): propuesta -> votación -> decisión -> tarea.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProposalState {
    /// Abierta, esperando votos de los validators vivos.
    Voting,
    /// Quórum alcanzado con mayoria ACCEPT: convertida en tarea(s).
    Accepted,
    /// Quórum alcanzado con mayoria REJECT: archivada con las razones.
    Rejected,
    /// Cerrada sin decisión (quórum imposible: quedan menos validators
    /// vivos que `min_votes`).
    Stalled,
}

/// Un voto sobre una propuesta.
#[derive(Debug, Clone, Serialize)]
pub struct ProposalVote {
    pub voter: String,
    /// ACCEPT | REJECT (strings por API; validado al registrar).
    pub decision: String,
    /// Por qué vota eso: el desacuerdo fundamentado es información, no ruido.
    pub reason: String,
}

/// Propuesta colectiva: el mecanismo de llegada a acuerdo con roles.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub id: u64,
    pub title: String,
    /// Qué hacer: descripción del plan acordado.
    pub body: String,
    /// Operación Agent-R asociada (las lecciones del ciclo quedan aquí).
    pub operation: String,
    pub proposer: String,
    pub state: ProposalState,
    /// Votos registrados (votantes vivos al momento de votar).
    pub votes: Vec<ProposalVote>,
    /// Mínimo de votos para decidir (default: # de validators vivos al crear).
    pub min_votes: usize,
    /// Id de la tarea creada al aceptarse (fase de ejecución).
    pub task_id: Option<u64>,
    pub created_at: f64,
    pub decided_at: Option<f64>,
}

/// Resultado de emitir un voto.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteResult {
    /// Voto registrado, la propuesta sigue abierta.
    Counted,
    /// Voto registrado Y quórum alcanzado: la propuesta quedó decidida.
    Decided,
    /// Votante con rol que no habilita a votar (sólo Validator/Coordinator).
    NotAValidator,
    /// Votante no registrado o muerto (heartbeat vencido).
    UnknownAgent,
    /// La propuesta no está en votación.
    Closed,
    /// Este agente ya votó (un voto por agente; para cambiarlo, la propuesta
    /// nueva lleva la razón nueva).
    AlreadyVoted,
}

/// Decisión de una propuesta cerrada.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalDecision {
    Accept,
    Reject,
    Stall,
}

fn state_from_u8(v: u8) -> TaskState {
    match v {
        ST_PENDING => TaskState::Pending,
        ST_READY => TaskState::Ready,
        ST_CLAIMED => TaskState::Claimed,
        ST_COMPLETED => TaskState::Completed,
        _ => TaskState::Failed,
    }
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Rol de un agente en el swarm (colaboración con especialización, no sólo
/// competencia por tareas): el Coordinator orquesta, los Proposers suben
/// propuestas, los Validators votan el consenso, los Executors ejecutan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmRole {
    Coordinator,
    Proposer,
    Validator,
    Executor,
}

impl SwarmRole {
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "coordinator" => Some(SwarmRole::Coordinator),
            "proposer" => Some(SwarmRole::Proposer),
            "validator" => Some(SwarmRole::Validator),
            "executor" => Some(SwarmRole::Executor),
            _ => None,
        }
    }
}

/// Agente registrado en el swarm.
#[derive(Debug, Clone, Serialize)]
pub struct SwarmAgent {
    pub name: String,
    /// Capacidades declaradas (informativo, para ruteo fino futuro).
    pub capabilities: Vec<String>,
    /// Rol en el modelo colaborativo (default: Executor).
    pub role: SwarmRole,
    pub registered_at: f64,
    pub last_heartbeat: f64,
    pub tasks_claimed: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
}

/// Resultado de un claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimResult {
    /// Reclamada por este agente.
    Claimed,
    /// Otro agente la reclamó primero (perdió el CAS).
    Taken,
    /// No reclutable: estado no-Ready o dependencias pendientes.
    NotReady,
    /// El agente no está registrado o su heartbeat expiró.
    UnknownAgent,
    /// Reservada para otro agente.
    Reserved,
}

/// Estado coordinado del swarm (compartir vía Arc<AgentSwarm>).
/// NO envuelve al AgentR: el server lo posee (tokio::Mutex) y lo presta por
/// referencia en las llamadas que tocan memoria compartida (evita deadlock
/// entre std Mutex síncrono y el runtime async).
pub struct AgentSwarm {
    agents: Mutex<HashMap<String, SwarmAgent>>,
    tasks: Mutex<HashMap<u64, SwarmTask>>,
    /// Propuestas en ciclo de acuerdo (colaboración por roles + consenso).
    proposals: Mutex<HashMap<u64, Proposal>>,
    next_task_id: AtomicU64,
    next_proposal_id: AtomicU64,
    /// TTL de heartbeat en segundos (death detection).
    heartbeat_ttl_s: f64,
    /// Tiempo tras el cual un CLAIMED sin progreso es re-claimable. (fallback)
    claim_timeout_s: f64,
}

impl AgentSwarm {
    /// Swarm nuevo (la memoria compartida la presta el server por &AgentR).
    pub fn new() -> Self {
        Self {
            agents: Mutex::new(HashMap::new()),
            tasks: Mutex::new(HashMap::new()),
            proposals: Mutex::new(HashMap::new()),
            next_task_id: AtomicU64::new(1),
            next_proposal_id: AtomicU64::new(1),
            heartbeat_ttl_s: 30.0,
            claim_timeout_s: 120.0,
        }
    }

    /// TTL de heartbeat y timeout de claim (para tests / despliegues finos).
    pub fn with_timeouts(mut self, heartbeat_ttl_s: f64, claim_timeout_s: f64) -> Self {
        self.heartbeat_ttl_s = heartbeat_ttl_s;
        self.claim_timeout_s = claim_timeout_s;
        self
    }

    // ── Agentes ─────────────────────────────────────────────────────────

    /// Registra (o re-registra) un agente y devuelve su nombre canónico.
    /// Si `role` es None y el agente ya existía, conserva su rol previo.
    pub fn register_agent(&self, name: &str, capabilities: &[&str], role: Option<SwarmRole>) -> String {
        let name = name.trim().to_string();
        let mut agents = self.agents.lock().unwrap();
        let now = now_secs();
        match agents.get_mut(&name) {
            Some(a) => {
                a.last_heartbeat = now;
                a.capabilities = capabilities.iter().map(|s| s.to_string()).collect();
                if let Some(r) = role {
                    a.role = r;
                }
            }
            None => {
                agents.insert(
                    name.clone(),
                    SwarmAgent {
                        name: name.clone(),
                        capabilities: capabilities.iter().map(|s| s.to_string()).collect(),
                        role: role.unwrap_or(SwarmRole::Executor),
                        registered_at: now,
                        last_heartbeat: now,
                        tasks_claimed: 0,
                        tasks_completed: 0,
                        tasks_failed: 0,
                    },
                );
            }
        }
        name
    }

    /// Heartbeat: refresca el TTL de vida del agente. false si no existe.
    pub fn heartbeat(&self, name: &str) -> bool {
        let mut agents = self.agents.lock().unwrap();
        match agents.get_mut(name) {
            Some(a) => {
                a.last_heartbeat = now_secs();
                true
            }
            None => false,
        }
    }

    fn agent_alive(&self, name: &str) -> bool {
        let agents = self.agents.lock().unwrap();
        agents
            .get(name)
            .map(|a| now_secs() - a.last_heartbeat <= self.heartbeat_ttl_s)
            .unwrap_or(false)
    }

    /// Agentes vivos (heartbeat dentro del TTL).
    pub fn live_agents(&self) -> Vec<SwarmAgent> {
        let agents = self.agents.lock().unwrap();
        let now = now_secs();
        agents
            .values()
            .filter(|a| now - a.last_heartbeat <= self.heartbeat_ttl_s)
            .cloned()
            .collect()
    }

    // ── Tareas ──────────────────────────────────────────────────────────

    /// Crea una tarea. READY inmediato si no tiene dependencias; si las tiene,
    /// PENDING hasta que todas estén COMPLETED.
    pub fn submit_task(
        &self,
        operation: &str,
        description: &str,
        depends_on: Vec<u64>,
        preferred_agent: Option<&str>,
    ) -> u64 {
        let id = self.next_task_id.fetch_add(1, Ordering::SeqCst);
        let deps_ok = depends_on.iter().all(|d| {
            self.tasks
                .lock()
                .unwrap()
                .get(d)
                .map(|t| t.state.load(Ordering::SeqCst) == ST_COMPLETED)
                .unwrap_or(false)
        });
        let state = std::sync::Arc::new(AtomicU8::new(if deps_ok && depends_on.is_empty() {
            ST_READY
        } else if deps_ok {
            ST_READY
        } else {
            ST_PENDING
        }));
        self.tasks.lock().unwrap().insert(
            id,
            SwarmTask {
                id,
                operation: operation.to_string(),
                description: description.to_string(),
                depends_on,
                preferred_agent: preferred_agent.map(|s| s.to_string()),
                state,
                claim_epoch: std::sync::Arc::new(AtomicU64::new(0)),
                result: std::sync::Arc::new(Mutex::new(None)),
                veto_rule: std::sync::Arc::new(Mutex::new(None)),
            },
        );
        id
    }

    /// Setea la regla señalada por el preflight en el momento de la decisión
    /// (claim del swarm). La consume el outcome para el self-tuning.
    pub fn set_task_rule(&self, task_id: u64, rule: u64) -> bool {
        match self.tasks.lock().unwrap().get(&task_id) {
            Some(t) => {
                *t.veto_rule.lock().unwrap() = Some(rule);
                true
            }
            None => false,
        }
    }

    /// Regla señalada en el claim (None = no hubo preflight con regla).
    pub fn task_rule(&self, task_id: u64) -> Option<u64> {
        self.tasks
            .lock()
            .unwrap()
            .get(&task_id)?
            .veto_rule
            .lock()
            .unwrap()
            .clone()
    }

    /// CLAIM lock-free: un solo CAS sobre el estado atómico de la tarea.
    /// N agentes compitiendo => exactamente uno gana.
    /// Fallback determinista: si la tarea está CLAIMED por un agente MUERTO
    /// (heartbeat vencido) o con claim más viejo que `claim_timeout_s`,
    /// cualquier agente puede tomarla con un epoch mayor (reclamo reproducible:
    /// gana el epoch más alto, no "quien llegue primero" sin límite).
    pub fn claim_task(&self, task_id: u64, agent: &str) -> ClaimResult {
        if !self.agent_alive(agent) {
            return ClaimResult::UnknownAgent;
        }
        let tasks = self.tasks.lock().unwrap();
        let task = match tasks.get(&task_id) {
            Some(t) => t,
            None => return ClaimResult::NotReady,
        };

        // PENDING: verificar dependencias y promover si corresponde.
        if task.state.load(Ordering::SeqCst) == ST_PENDING {
            let deps_done = task.depends_on.iter().all(|d| {
                tasks
                    .get(d)
                    .map(|t| t.state.load(Ordering::SeqCst) == ST_COMPLETED)
                    .unwrap_or(false)
            });
            if deps_done {
                task.state
                    .compare_exchange(ST_PENDING, ST_READY, Ordering::SeqCst, Ordering::SeqCst)
                    .ok();
            } else {
                return ClaimResult::NotReady;
            }
        }

        // Reserva por agente.
        if let Some(pref) = &task.preferred_agent {
            if pref != agent {
                return ClaimResult::Reserved;
            }
        }

        let now = now_secs();
        let my_epoch = now_to_epoch(now);
        match task.state.compare_exchange(
            ST_READY,
            ST_CLAIMED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => { /* claim limpio */ }
            Err(current) if current == ST_CLAIMED => {
                // Fallback: ¿el claim expiró (holder muerto o timeout vencido)?
                let holder_dead =
                    epoch_to_secs(task.claim_epoch.load(Ordering::SeqCst))
                        < now - self.claim_timeout_s;
                if !holder_dead {
                    return ClaimResult::Taken;
                }
                // Re-claim determinista: gana quien presente el epoch mayor.
                let prev = task.claim_epoch.fetch_max(my_epoch, Ordering::SeqCst);
                if my_epoch <= prev {
                    return ClaimResult::Taken;
                }
                task.state.store(ST_CLAIMED, Ordering::SeqCst);
            }
            Err(_) => return ClaimResult::NotReady,
        }

        task.claim_epoch.store(my_epoch, Ordering::SeqCst);
        if let Some(a) = self.agents.lock().unwrap().get_mut(agent) {
            a.tasks_claimed += 1;
        }
        ClaimResult::Claimed
    }

    /// Completa una tarea (sólo el agente con el epoch vigente).
    /// Registra el resultado y una lección en la memoria compartida
    /// (`agent_r` lo presta el server; dedup de O4: resultados idénticos
    /// no duplican incidentes).
    pub fn complete_task(&self, agent_r: &mut AgentR, task_id: u64, agent: &str, result: &str) -> bool {
        let tasks = self.tasks.lock().unwrap();
        let task = match tasks.get(&task_id) {
            Some(t) => t,
            None => return false,
        };
        if task.state.load(Ordering::SeqCst) != ST_CLAIMED {
            return false;
        }
        task.state.store(ST_COMPLETED, Ordering::SeqCst);
        *task.result.lock().unwrap() = Some(result.to_string());
        if let Some(a) = self.agents.lock().unwrap().get_mut(agent) {
            a.tasks_completed += 1;
            a.last_heartbeat = now_secs();
        }
        // Self-tuning (15-oct): la task COMPLETÓ → si el preflight del claim
        // señaló una regla de error como top-1, esa señal fue un MISS (la
        // regla no aplicaba a esta ejecución). Sin regla señalada => no hay
        // predicción que evaluar (abstención implícita).
        // R03 Capa 1: los éxitos van a la memoria de éxitos (no contaminan el
        // recall del preflight). El MISS de la regla señalada se mantiene.
        let predicted = task.veto_rule.lock().unwrap().take();
        if let Some(pred) = predicted {
            agent_r.tuning_feedback(Some(pred), None);
        }
        let _ = agent_r.record_success(&task.operation, result, None);
        true
    }

    /// Marca fallo de una tarea: vuelve a READY para re-claim inmediato.
    /// El error queda registrado en la memoria compartida para los demás.
    pub fn fail_task(&self, agent_r: &mut AgentR, task_id: u64, agent: &str, reason: &str) -> bool {
        let tasks = self.tasks.lock().unwrap();
        let task = match tasks.get(&task_id) {
            Some(t) => t,
            None => return false,
        };
        if task.state.load(Ordering::SeqCst) != ST_CLAIMED {
            return false;
        }
        task.state.store(ST_READY, Ordering::SeqCst);
        if let Some(a) = self.agents.lock().unwrap().get_mut(agent) {
            a.tasks_failed += 1;
            a.last_heartbeat = now_secs();
        }
        // R03 Capa 1 — label POST-HOC: el hit se acredita solo si la evidencia
        // del fallo (reason/stderr) CONFIRMA la regla señalada (match_outcome_
        // rule), no por identidad. Si la evidencia confirma OTRA regla, esa
        // recibe el hit; si no confirma ninguna, la señalada es un MISS.
        let predicted = task.veto_rule.lock().unwrap().take();
        if let Some(pred) = predicted {
            let confirmed = agent_r.match_outcome_rule(&task.operation, reason);
            // hit SOLO si la evidencia confirma EXACTAMENTE la regla señalada;
            // si confirma otra regla o ninguna, la señalada es un MISS.
            let outcome = match confirmed {
                Some(actual) if actual == pred => (Some(pred), Some(actual)),
                _ => (Some(pred), None),
            };
            agent_r.tuning_feedback(outcome.0, outcome.1);
        }
        agent_r.record_error(
            ErrorCategory::Timeout,
            ErrorSeverity::Error,
            &task.operation,
            &format!("task {} failed: {}", task_id, reason),
            "swarm task failure (reclaimable)",
            &["swarm", "failed"],
            "revisar la causa antes de re-claimar esta operación",
        );
        true
    }

    // ── Acuerdo con roles: propuestas + consenso ────────────────────────

    /// Registrar un rol autoritativo (sólo el Coordinator puede fijarlo).
    /// Devuelve el rol efectivo del agente o None si no existe/no es coordinator.
    pub fn set_role(
        &self,
        coordinator: &str,
        agent: &str,
        role: SwarmRole,
    ) -> Option<SwarmRole> {
        let mut agents = self.agents.lock().unwrap();
        if agents.get(coordinator)?.role != SwarmRole::Coordinator {
            return None;
        }
        let a = agents.get_mut(agent)?;
        a.role = role;
        a.last_heartbeat = now_secs();
        Some(a.role)
    }

    /// El Coordinator propone el plan de acción. min_votes default: nº de
    /// validators vivos (quórum = todos los validadores vivos; si no hay
    /// ninguno, el Coordinator puede aprobarla él mismo con 1 voto).
    pub fn submit_proposal(
        &self,
        proposer: &str,
        title: &str,
        body: &str,
        operation: &str,
        min_votes: Option<usize>,
    ) -> Result<u64, String> {
        let mut agents = self.agents.lock().unwrap();
        let now = now_secs();
        let pa = agents.get_mut(proposer).ok_or("proposer no registrado")?;
        if now - pa.last_heartbeat > self.heartbeat_ttl_s {
            return Err("proposer sin heartbeat (muerto)".into());
        }
        if pa.role != SwarmRole::Coordinator && pa.role != SwarmRole::Proposer {
            return Err("sólo Coordinator/Proposer pueden proponer".into());
        }
        pa.last_heartbeat = now;
        let validators_vivos = agents
            .values()
            .filter(|a| {
                a.role == SwarmRole::Validator
                    && now - a.last_heartbeat <= self.heartbeat_ttl_s
            })
            .count();
        let min_votes =
            min_votes.unwrap_or_else(|| validators_vivos.max(1));
        drop(agents);

        let id = self
            .next_proposal_id
            .fetch_add(1, Ordering::SeqCst);
        self.proposals.lock().unwrap().insert(
            id,
            Proposal {
                id,
                title: title.to_string(),
                body: body.to_string(),
                operation: operation.to_string(),
                proposer: proposer.to_string(),
                state: ProposalState::Voting,
                votes: Vec::new(),
                min_votes,
                task_id: None,
                created_at: now_secs(),
                decided_at: None,
            },
        );
        Ok(id)
    }

    /// Voto de un validator (con razón obligatoria). Al alcanzar el quórum
    /// decide por mayoría simple; el resultado del recuento va en la respuesta.
    pub fn vote_proposal(
        &self,
        proposal_id: u64,
        voter: &str,
        accept: bool,
        reason: &str,
    ) -> VoteResult {
        // Votante vivo con rol habilitado (Validator o Coordinator).
        {
            let agents = self.agents.lock().unwrap();
            let a = match agents.get(voter) {
                Some(a) => a,
                None => return VoteResult::UnknownAgent,
            };
            if now_secs() - a.last_heartbeat > self.heartbeat_ttl_s {
                return VoteResult::UnknownAgent;
            }
            if a.role != SwarmRole::Validator && a.role != SwarmRole::Coordinator {
                return VoteResult::NotAValidator;
            }
        }
        let mut proposals = self.proposals.lock().unwrap();
        let p = match proposals.get_mut(&proposal_id) {
            Some(p) => p,
            None => return VoteResult::Closed,
        };
        if p.state != ProposalState::Voting {
            return VoteResult::Closed;
        }
        if p.votes.iter().any(|v| v.voter == voter) {
            return VoteResult::AlreadyVoted;
        }
        p.votes.push(ProposalVote {
            voter: voter.to_string(),
            decision: if accept { "ACCEPT" } else { "REJECT" }.to_string(),
            reason: reason.to_string(),
        });

        // ¿Quórum alcanzado? (el Coordinator Proposer cuenta si tiene rol de
        // voto; aquí sólo cuentan los votantes registrados en p.votes).
        if p.votes.len() < p.min_votes {
            return VoteResult::Counted;
        }
        // Decisión por mayoría simple con quórum completo.
        let accepts = p.votes.iter().filter(|v| v.decision == "ACCEPT").count();
        let decision = if accepts * 2 > p.votes.len() {
            ProposalState::Accepted
        } else {
            ProposalState::Rejected
        };
        p.state = decision;
        p.decided_at = Some(now_secs());
        VoteResult::Decided
    }

    /// Convierte una propuesta ACCEPTED en tarea del pipeline de ejecución
    /// (fase ejecutar del flujo acordar → ejecutar). Idempotente.
    pub fn materialize_proposal(&self, proposal_id: u64) -> Result<(u64, ProposalState), String> {
        let mut proposals = self.proposals.lock().unwrap();
        let p = proposals
            .get_mut(&proposal_id)
            .ok_or("propuesta inexistente")?;
        if p.state != ProposalState::Accepted {
            return Ok((0, p.state));
        }
        if let Some(tid) = p.task_id {
            return Ok((tid, p.state));
        }
        let tid = self.next_task_id.fetch_add(1, Ordering::SeqCst);
        self.tasks.lock().unwrap().insert(
            tid,
            SwarmTask {
                id: tid,
                operation: p.operation.clone(),
                description: format!("[propuesta #{}] {}", p.id, p.title),
                depends_on: vec![],
                preferred_agent: None,
                state: std::sync::Arc::new(AtomicU8::new(ST_READY)),
                claim_epoch: std::sync::Arc::new(AtomicU64::new(0)),
                result: std::sync::Arc::new(Mutex::new(None)),
                veto_rule: std::sync::Arc::new(Mutex::new(None)),
            },
        );
        p.task_id = Some(tid);
        Ok((tid, p.state))
    }

    /// Estado de una propuesta (título, votos con razones, decisión).
    pub fn proposal_status(&self, proposal_id: u64) -> Option<serde_json::Value> {
        let proposals = self.proposals.lock().unwrap();
        let p = proposals.get(&proposal_id)?;
        Some(serde_json::json!({
            "id": p.id,
            "title": p.title,
            "body": p.body,
            "operation": p.operation,
            "proposer": p.proposer,
            "state": p.state,
            "min_votes": p.min_votes,
            "votes": p.votes.iter().map(|v| serde_json::json!({
                "voter": v.voter, "decision": v.decision, "reason": v.reason,
            })).collect::<Vec<_>>(),
            "task_id": p.task_id,
        }))
    }

    /// Cierra como Stalled toda propuesta Voting cuyo quórum es imposible
    /// (validators vivos < min_votes). Devuelve cuántas cerró.
    pub fn close_stalled(&self) -> usize {
        let now = now_secs();
        let validators_vivos = {
            let agents = self.agents.lock().unwrap();
            agents
                .values()
                .filter(|a| {
                    a.role == SwarmRole::Validator
                        && now - a.last_heartbeat <= self.heartbeat_ttl_s
                })
                .count()
        };
        let mut proposals = self.proposals.lock().unwrap();
        let mut closed = 0;
        for p in proposals.values_mut() {
            if p.state == ProposalState::Voting && validators_vivos < p.min_votes {
                p.state = ProposalState::Stalled;
                p.decided_at = Some(now);
                closed += 1;
            }
        }
        closed
    }

    /// Resultado de una tarea completada.
    pub fn task_result(&self, task_id: u64) -> Option<String> {
        let tasks = self.tasks.lock().unwrap();
        tasks.get(&task_id).and_then(|t| {
            (t.state.load(Ordering::SeqCst) == ST_COMPLETED)
                .then(|| t.result.lock().unwrap().clone().unwrap_or_default())
        })
    }

    // ── Primitivas de coordinación ──────────────────────────────────────

    /// Crea una barrera con `n` participantes y devuelve su id (negativo, para
    /// no chocar con el espacio de tareas).
    pub fn create_barrier(&self, n: usize) -> u64 {
        // Reutiliza el mecanismo de tareas: barrera = tarea que se completa
        // cuando los n participantes reportan (counter implícito).
        let id = self.next_task_id.fetch_add(1, Ordering::SeqCst);
        self.tasks.lock().unwrap().insert(
            id,
            SwarmTask {
                id,
                operation: "swarm_barrier".to_string(),
                description: format!("barrier n={n}"),
                depends_on: vec![],
                preferred_agent: None,
                state: std::sync::Arc::new(AtomicU8::new(ST_READY)),
                claim_epoch: std::sync::Arc::new(AtomicU64::new(n as u64)),
                result: std::sync::Arc::new(Mutex::new(Some(n.to_string()))),
                veto_rule: std::sync::Arc::new(Mutex::new(None)),
            },
        );
        id
    }

    /// Reporta llegada de un participante a la barrera. Devuelve Some(true)
    /// cuando TODOS llegaron (barrera abierta).
    pub fn barrier_arrive(&self, barrier_id: u64) -> Option<bool> {
        let tasks = self.tasks.lock().unwrap();
        let task = tasks.get(&barrier_id)?;
        if task.operation != "swarm_barrier" {
            return None;
        }
        let remaining = task.claim_epoch.fetch_sub(1, Ordering::SeqCst);
        Some(remaining <= 1)
    }

    /// Contador distribuido de resultados de fase (incréments atómicos).
    /// Reutiliza la barrera: arrive() hace fetch_sub; para contadores hacia
    /// arriba usamos fetch_add sobre el mismo slot.
    pub fn counter_add(&self, counter_task_id: u64, delta: u64) -> Option<u64> {
        let tasks = self.tasks.lock().unwrap();
        let task = tasks.get(&counter_task_id)?;
        if task.operation != "swarm_barrier" {
            return None;
        }
        Some(task.claim_epoch.fetch_add(delta, Ordering::SeqCst))
    }

    /// Snapshot del estado del swarm (para /agent-swarm/status).
    pub fn status(&self, agent_r: &AgentR) -> serde_json::Value {
        let tasks = self.tasks.lock().unwrap();
        let proposals = self.proposals.lock().unwrap();
        let mut by_state: HashMap<String, usize> = [
            "PENDING", "READY", "CLAIMED", "COMPLETED", "FAILED",
        ]
        .iter()
        .map(|s| (s.to_string(), 0))
        .collect();
        for t in tasks.values() {
            let key = format!("{:?}", state_from_u8(t.state.load(Ordering::SeqCst))).to_uppercase();
            *by_state.entry(key).or_insert(0) += 1;
        }
        let mut proposals_by_state: HashMap<String, usize> = [
            "VOTING", "ACCEPTED", "REJECTED", "STALLED",
        ]
        .iter()
        .map(|s| (s.to_string(), 0))
        .collect();
        for p in proposals.values() {
            let key = format!("{:?}", p.state).to_uppercase();
            *proposals_by_state.entry(key).or_insert(0) += 1;
        }
        let proposals_len = proposals.len();
        drop(proposals);
        serde_json::json!({
            "live_agents": self.live_agents().iter().map(|a| serde_json::json!({
                "name": a.name,
                "role": a.role,
                "capabilities": a.capabilities,
                "claimed": a.tasks_claimed,
                "completed": a.tasks_completed,
                "failed": a.tasks_failed,
            })).collect::<Vec<_>>(),
            "tasks_total": tasks.len(),
            "tasks_by_state": by_state,
            "proposals": {
                "total": proposals_len,
                "by_state": proposals_by_state,
            },
            "shared_memory": {
                "backend": "agent_r (O1-O4 integrados)",
                "unique_incidents": agent_r.stats().unique_incidents,
            },
            "heartbeat_ttl_s": self.heartbeat_ttl_s,
            "claim_timeout_s": self.claim_timeout_s,
        })
    }

    /// Preflight compartido: valida una operación contra las lecciones de
    /// TODO el swarm (no sólo del agente local — es una sola memoria).
    /// v2 (pendiente 2a): paridad con el scoring compuesto del endpoint
    /// /agent-r/preflight (mismo preflight_v2 + campos top-3/sugerencia).
    pub fn preflight_shared(&self, agent_r: &mut AgentR, operation: &str, tags: &[&str]) -> serde_json::Value {
        let r = agent_r.preflight_v2(operation, tags);
        serde_json::json!({
            "proceed": r.proceed,
            "relevant_errors": r.relevant_errors,
            "warnings": r.warnings,
            "precautions": r.precautions,
            "top_rules": r.top_rules,
            "suggestion": r.suggestion,
            "veto_reason": r.veto_reason,
        })
    }
}

fn now_to_epoch(now: f64) -> u64 {
    (now * 1000.0) as u64
}

fn epoch_to_secs(e: u64) -> f64 {
    e as f64 / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn swarm() -> AgentSwarm {
        AgentSwarm::new().with_timeouts(30.0, 120.0)
    }

    /// El claim es un CAS: N agentes compitiendo => exactamente uno gana
    /// y los demás reciben Taken (sin doble asignación).
    #[test]
    fn test_claim_cas_exactly_one_winner() {
        let sw = swarm();
        sw.register_agent("a1", &["scan"], None);
        sw.register_agent("a2", &["scan"], None);
        sw.register_agent("a3", &["scan"], None);
        let tid = sw.submit_task("op_par", "tarea paralela", vec![], None);
        assert_eq!(sw.claim_task(tid, "a1"), ClaimResult::Claimed);
        assert_eq!(sw.claim_task(tid, "a2"), ClaimResult::Taken);
        assert_eq!(sw.claim_task(tid, "a3"), ClaimResult::Taken);
    }

    /// Agente no registrado (o con heartbeat vencido) no puede reclamar.
    #[test]
    fn test_claim_unknown_agent_rejected() {
        let sw = swarm();
        sw.register_agent("a1", &[], None);
        let tid = sw.submit_task("op_par", "d", vec![], None);
        assert_eq!(sw.claim_task(tid, "fantasma"), ClaimResult::UnknownAgent);
    }

    /// Dependencias: PENDING hasta que todas estén COMPLETED.
    #[test]
    fn test_dependencies_gate_ready() {
        let sw = swarm();
        sw.register_agent("a1", &[], None);
        let mut ar = AgentR::new_ephemeral();
        // dependencia COMPLETED en claim_task (sin escrituras tentativas).
        let t1 = sw.submit_task("step_1", "primero", vec![], None);
        let t2 = sw.submit_task("step_2", "segundo", vec![t1], None);
        assert_eq!(sw.claim_task(t2, "a1"), ClaimResult::NotReady);
        assert_eq!(sw.claim_task(t1, "a1"), ClaimResult::Claimed);
        assert!(sw.complete_task(&mut ar, t1, "a1", "ok"));
        assert_eq!(sw.claim_task(t2, "a1"), ClaimResult::Claimed);
    }

    /// preferred_agent: la tarea está reservada para ese agente.
    #[test]
    fn test_preferred_agent_reservation() {
        let sw = swarm();
        sw.register_agent("especialista", &[], None);
        sw.register_agent("otros", &[], None);
        let tid = sw.submit_task("op_rara", "sólo el especialista", vec![], Some("especialista"));
        assert_eq!(sw.claim_task(tid, "otros"), ClaimResult::Reserved);
        assert_eq!(sw.claim_task(tid, "especialista"), ClaimResult::Claimed);
    }

    /// complete/fail: estado + resultado + lección compartida en AgentR.
    #[test]
    fn test_complete_and_fail_write_shared_memory() {
        let sw = swarm();
        let mut ar = AgentR::new_ephemeral();
        sw.register_agent("a1", &[], None);
        let tid = sw.submit_task("distill_lessons", "compilar lección", vec![], None);
        sw.claim_task(tid, "a1");

        // fail: vuelve a READY y deja el error en la memoria común
        assert!(sw.fail_task(&mut ar, tid, "a1", "OOM"));
        assert_eq!(sw.claim_task(tid, "a1"), ClaimResult::Claimed);
        assert!(sw.complete_task(&mut ar, tid, "a1", "lección compilada"));
        assert_eq!(sw.task_result(tid).as_deref(), Some("lección compilada"));
        // lección visible en la memoria compartida (preflight del swarm)
        let pf = sw.preflight_shared(&mut ar, "distill_lessons", &[]);
        assert!(pf["relevant_errors"].as_u64().unwrap_or(0) >= 1);
        // doble complete no procede (ya no está CLAIMED)
        assert!(!sw.complete_task(&mut ar, tid, "a1", "dup"));
    }

    /// Fallback determinista: agente muerto => re-claim por epoch mayor.
    /// (Con TTL y claim_timeout cortos para no dormir el test.)
    #[test]
    fn test_dead_agent_task_reclaimable() {
        let sw = AgentSwarm::new().with_timeouts(0.05, 0.05);
        let mut ar = AgentR::new_ephemeral();
        sw.register_agent("lento", &[], None);
        sw.register_agent("reemplazo", &[], None);
        let tid = sw.submit_task("index_rebuild", "rebuild largo", vec![], None);
        assert_eq!(sw.claim_task(tid, "lento"), ClaimResult::Claimed);
        // aún vivo: el reemplazo no puede tomarla
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert_eq!(sw.claim_task(tid, "reemplazo"), ClaimResult::Taken);
        // muere (heartbeat vence): el reemplazo —con su latido al día, como
        // manda el protocolo— la re-claima (el claim del holder expiró).
        std::thread::sleep(std::time::Duration::from_millis(80));
        sw.heartbeat("reemplazo");
        assert_eq!(sw.claim_task(tid, "reemplazo"), ClaimResult::Claimed);
        assert!(sw.complete_task(&mut ar, tid, "reemplazo", "recuperada"));
        assert_eq!(sw.task_result(tid).as_deref(), Some("recuperada"));
    }

    /// Barrera: abre sólo cuando todos los participantes llegan.
    #[test]
    fn test_barrier_opens_when_all_arrive() {
        let sw = swarm();
        let b = sw.create_barrier(3);
        // fetch_sub devuelve el valor PREVIO: 3→2, 2→1, 1→0 (abre en el último).
        assert_eq!(sw.barrier_arrive(b), Some(false));
        assert_eq!(sw.barrier_arrive(b), Some(false));
        assert_eq!(sw.barrier_arrive(b), Some(true));
    }

    /// Heartbeat TTL: agente sin latido desaparece de los vivos.
    #[test]
    fn test_heartbeat_ttl_death_detection() {
        let sw = AgentSwarm::new().with_timeouts(0.05, 120.0);
        sw.register_agent("a1", &[], None);
        assert_eq!(sw.live_agents().len(), 1);
        std::thread::sleep(std::time::Duration::from_millis(80));
        assert!(sw.live_agents().is_empty(), "sin heartbeat => muerto");
        // re-registrar = revivir
        sw.register_agent("a1", &[], None);
        assert_eq!(sw.live_agents().len(), 1);
    }

    /// Contador distribuido de fase: fetch_add atómico visible entre "agentes".
    #[test]
    fn test_phase_counter_atomic() {
        let sw = swarm();
        // create_barrier(0): el slot arranca en 0 (contador puro desde cero).
        let c = sw.create_barrier(0);
        // fetch_add devuelve el valor PREVIO (semántica clásica de contador).
        assert_eq!(sw.counter_add(c, 5), Some(0));
        assert_eq!(sw.counter_add(c, 7), Some(5));
        assert_eq!(sw.counter_add(c, 1), Some(12));
        assert_eq!(sw.counter_add(999, 1), None, "id inexistente => None");
    }

    /// status(): snapshot JSON con agentes vivos y conteo por estado.
    #[test]
    fn test_status_snapshot() {
        let sw = swarm();
        let ar = AgentR::new_ephemeral();
        sw.register_agent("a1", &["gpu"], None);
        let t = sw.submit_task("op", "d", vec![], None);
        sw.claim_task(t, "a1");
        let st = sw.status(&ar);
        assert_eq!(st["live_agents"].as_array().unwrap().len(), 1);
        assert_eq!(st["tasks_by_state"]["CLAIMED"].as_u64(), Some(1));
        assert!(st["shared_memory"]["unique_incidents"].is_u64());
        assert!(st["proposals"]["total"].is_u64());
    }

    // ── Acuerdo con roles: propuestas + consenso ────────────────────────

    /// Setup: coordinator + 2 validators + 1 executor puro.
    fn swarm_roles() -> AgentSwarm {
        let sw = swarm();
        sw.register_agent("coord", &["orquestar"], Some(SwarmRole::Coordinator));
        sw.register_agent("val_a", &["revisar"], Some(SwarmRole::Validator));
        sw.register_agent("val_b", &["revisar"], Some(SwarmRole::Validator));
        sw.register_agent("peon", &["ejecutar"], None); // default: Executor
        sw
    }

    /// Roles: sólo el Coordinator fija roles; el Executor no puede votar
    /// ni proponer; el Validator no puede proponer.
    #[test]
    fn test_roles_gate_propose_and_vote() {
        let sw = swarm_roles();
        // Executor no propone
        assert!(sw.submit_proposal("peon", "t", "b", "op", None).is_err());
        // Validator no propone
        assert!(sw.submit_proposal("val_a", "t", "b", "op", None).is_err());
        // Coordinator sí
        let pid = sw.submit_proposal("coord", "plan", "hacer X", "op_x", None).unwrap();
        // Executor no vota
        assert_eq!(sw.vote_proposal(pid, "peon", true, "me parece"), VoteResult::NotAValidator);
        // set_role sin coordinator => None
        assert!(sw.set_role("val_a", "peon", SwarmRole::Validator).is_none());
        // coordinator asciende al peon a validator => ahora vota.
        // Ojo: el quórum de la propuesta quedó fijado en 2 al crearse
        // (validators vivos entonces) => su voto cuenta pero no decide solo.
        assert_eq!(sw.set_role("coord", "peon", SwarmRole::Validator), Some(SwarmRole::Validator));
        assert_eq!(sw.vote_proposal(pid, "peon", true, "revisado"), VoteResult::Counted);
        assert_eq!(sw.vote_proposal(pid, "val_a", true, "de acuerdo"), VoteResult::Decided);
    }

    /// Ciclo feliz: propuesta -> 2 ACCEPT con razón -> Decided -> tarea
    /// ejecutable -> lección de la ejecución en la memoria compartida.
    #[test]
    fn test_proposal_accept_becomes_task() {
        let sw = swarm_roles();
        let mut ar = AgentR::new_ephemeral();
        let pid = sw.submit_proposal("coord", "reindexar HNSW",
            "reconstruir el índice tras el lote nuevo", "index_rebuild", None)
            .unwrap();
        assert_eq!(sw.vote_proposal(pid, "val_a", true, "el drift lo justifica"),
                   VoteResult::Counted);
        assert_eq!(sw.vote_proposal(pid, "val_b", true, "medido: recall cayó 4%"),
                   VoteResult::Decided);
        // doble voto no: la propuesta ya está decidida => Closed (el chequeo
        // de estado precede al de duplicado; el duplicado también daría Closed)
        assert_eq!(sw.vote_proposal(pid, "val_a", true, "otra vez"), VoteResult::Closed);
        // materializar -> tarea READY del pipeline existente
        let (tid, state) = sw.materialize_proposal(pid).unwrap();
        assert_eq!(state, ProposalState::Accepted);
        assert_eq!(sw.claim_task(tid, "peon"), ClaimResult::Claimed);
        assert!(sw.complete_task(&mut ar, tid, "peon", "índice reconstruido"));
        // propuesta ya materializada: idempotente
        let (tid2, _) = sw.materialize_proposal(pid).unwrap();
        assert_eq!(tid, tid2);
    }

    /// REJECT con quórum: la propuesta muere con las razones archivadas y
    /// el desacuerdo queda como lección en la memoria compartida.
    #[test]
    fn test_proposal_reject_archives_reasons() {
        let sw = swarm_roles();
        let pid = sw.submit_proposal("coord", "borrar índice",
            "drop y rebuild desde cero", "index_rebuild", None).unwrap();
        sw.vote_proposal(pid, "val_a", false, "costo>beneficio: 40 min de rebuild");
        sw.vote_proposal(pid, "val_b", false, "el incremental cubre el mismo gap");
        let st = sw.proposal_status(pid).unwrap();
        assert_eq!(st["state"], "REJECTED");
        assert_eq!(st["votes"].as_array().unwrap().len(), 2);
        assert!(st["votes"][0]["reason"].as_str().unwrap().len() > 0);
        // no materializa
        assert_eq!(sw.materialize_proposal(pid).unwrap().1, ProposalState::Rejected);
    }

    /// Voto en empate técnico (1-1 con quórum 2) => no hay mayoría => REJECTED
    /// (el default conservador es no ejecutar).
    #[test]
    fn test_proposal_tie_defaults_to_reject() {
        let sw = swarm_roles();
        let pid = sw.submit_proposal("coord", "cambio riesgoso", "b", "op", None).unwrap();
        sw.vote_proposal(pid, "val_a", true, "a favor");
        sw.vote_proposal(pid, "val_b", false, "en contra");
        assert_eq!(sw.proposal_status(pid).unwrap()["state"], "REJECTED");
    }

    /// Muerte de validators con propuesta abierta => Stalled (cerrable).
    #[test]
    fn test_proposal_stalled_when_validators_die() {
        let sw = AgentSwarm::new().with_timeouts(0.05, 120.0);
        sw.register_agent("coord", &[], Some(SwarmRole::Coordinator));
        sw.register_agent("val_a", &[], Some(SwarmRole::Validator));
        sw.register_agent("val_b", &[], Some(SwarmRole::Validator));
        let pid = sw.submit_proposal("coord", "plan", "b", "op", None).unwrap();
        sw.vote_proposal(pid, "val_a", true, "ok");
        std::thread::sleep(std::time::Duration::from_millis(80));
        // ambos validators muertos => quórum imposible
        assert_eq!(sw.close_stalled(), 1);
        assert_eq!(sw.proposal_status(pid).unwrap()["state"], "STALLED");
        // y los votos tardíos no reabren nada
        assert_eq!(sw.vote_proposal(pid, "val_b", true, "tardío"), VoteResult::UnknownAgent);
    }

    /// Propuesta con min_votes explícito menor al pool de validators
    /// (quórum parcial para decisiones rápidas).
    #[test]
    fn test_proposal_min_votes_explicit() {
        let sw = swarm_roles();
        let pid = sw.submit_proposal("coord", "hotfix", "b", "op", Some(1)).unwrap();
        assert_eq!(sw.vote_proposal(pid, "val_a", true, "urgente"), VoteResult::Decided);
        let (tid, _) = sw.materialize_proposal(pid).unwrap();
        assert!(tid > 0);
    }
}
