//! agent_r::meta_curiosity — Curiosidad guiada (RSI, pedido de user 16-oct):
//! el sistema debe GENERARSE preguntas en puntos críticos en vez de esperar
//! que un humano las formule ("ya van 3 preguntas guías que podían ser
//! autónomas": valida numéricamente / ¿por qué baja el recall? / esta misma).
//!
//! Mecanismo (3 capas, todo en la lib):
//!   1. TRIGGERS programáticos (deterministas, baratos): el código llama
//!      observe_* en los puntos donde hoy un humano tendría que preguntar —
//!      delta de métrica fuera de rango, trade-off entre métricas, sorpresa
//!      (expectativa vs realidad), hallazgo de seguridad.
//!   2. Preguntas templadas + refinamiento opcional con el crítico glm-5.3-
//!      flash (que demostró recall 100% como detector; generar preguntas es
//!      el mismo músculo semántico).
//!   3. CICLO cerrado: la pregunta queda ABIERTA en el snapshot hasta que
//!      answer() la cierre con evidencia (lección en la memoria de errores) —
//!      la misma trazabilidad que el resto del RSI.
//!
//! Casos reales que motivan los triggers (ronda 2):
//!   - recall 100%→88.9% sin explicación  => observe_metric_delta
//!   - "valida numéricamente" (mejora afirmada sin A/B) => observe_claim_without_validation
//!   - receipt re-usable (T2)             => observe_security_finding

use crate::agent_r::AgentR;
use serde_json::{json, Value};

/// Punto crítico que dispara la pregunta.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CriticalPoint {
    /// Métrica que se movió fuera de lo esperado (ej: recall baja 11 pts).
    MetricDelta { metric: String, from: f64, to: f64 },
    /// Mejora afirmada sin validación numérica que la respalde.
    ClaimWithoutValidation { claim: String },
    /// Trade-off: una métrica sube mientras otra baja.
    TradeOff { up: String, down: String },
    /// Hallazgo de seguridad/fuzzing.
    SecurityFinding { finding: String },
    /// Sorpresa genérica: se esperaba X, se obtuvo Y.
    Surprise { expected: String, got: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Priority {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QStatus {
    Open,
    Answered,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Question {
    pub id: u64,
    pub timestamp: f64,
    pub point: CriticalPoint,
    pub text: String,
    pub priority: Priority,
    pub status: QStatus,
    #[serde(default)]
    pub answer: Option<String>,
    /// Id de la lección (ErrorRecord) que documenta la respuesta.
    #[serde(default)]
    pub lesson_id: Option<u64>,
    /// Receipt de la investigación que respondió la pregunta (N2).
    #[serde(default)]
    pub receipt_id: Option<String>,
}

/// Pregunta templada según el punto crítico (determinista, sin LLM).
pub fn template_question(point: &CriticalPoint) -> (String, Priority) {
    match point {
        CriticalPoint::MetricDelta { metric, from, to } => (
            format!(
                "¿Por qué {} cambió de {:.1} a {:.1}? Aislar la causa: ¿ruido del serving, cambio de protocolo, o artefacto del harness? Repetir la medición y comparar condiciones.",
                metric, from, to
            ),
            Priority::High,
        ),
        CriticalPoint::ClaimWithoutValidation { claim } => (
            format!(
                "¿\"{}\" se sostiene con números? Diseñar A/B con el mismo bench y split: solo cambia la variable afirmada.",
                claim
            ),
            Priority::High,
        ),
        CriticalPoint::TradeOff { up, down } => (
            format!(
                "¿Vale la pena el trade-off ({} sube, {} baja)? ¿En qué régimen de carga gana cada lado?",
                up, down
            ),
            Priority::Medium,
        ),
        CriticalPoint::SecurityFinding { finding } => (
            format!(
                "¿{} es explotable en producción? Fuzzear el vector y medir el daño máximo antes de fijar.",
                finding
            ),
            Priority::High,
        ),
        CriticalPoint::Surprise { expected, got } => (
            format!(
                "¿Por qué se obtuvo \"{}\" cuando se esperaba \"{}\"? Reproducir mínimo y aislar la variable.",
                got, expected
            ),
            Priority::Medium,
        ),
    }
}

/// Registro de preguntas (mismo patrón que insights/skills: vive en el
/// snapshot del Agent-R, con dedup por texto para no duplicar disparos).
pub fn ask(agent: &mut AgentR, point: CriticalPoint) -> u64 {
    let (text, priority) = template_question(&point);
    agent.add_question(point, text, priority)
}

/// Igual que `ask` pero con texto custom (cuando el templado no alcanza).
pub fn ask_custom(
    agent: &mut AgentR,
    point: CriticalPoint,
    text: String,
    priority: Priority,
) -> u64 {
    agent.add_question(point, text, priority)
}

/// Refinamiento con el crítico (opcional, feature "critic"): dada una pregunta
/// templada, el LLM la hace más investigable (sub-preguntas, método concreto).
/// Devuelve el texto enriquecido o el original si el crítico falla — NUNCA
/// bloquea el flujo por indisponibilidad del LLM.
#[cfg(feature = "critic")]
pub fn elaborate(agent: &AgentR, question_id: u64) -> Option<String> {
    let q = agent
        .questions_iter()
        .find(|q| q.id == question_id)?
        .clone();
    let critic = super::meta::Critic::from_env().ok()?;
    let sys = "Sos un asesor de investigación. Respondé SOLO JSON: {\"sharper\":\"pregunta refinada\",\"method\":\"cómo investigarla en 1-3 pasos concretos\"}.";
    let user = format!(
        "Pregunta del sistema: {}\n\nPunto crítico: {}\n\nRefiná la pregunta para que sea investigable y proponé el método.",
        q.text,
        serde_json::to_string(&q.point).unwrap_or_default()
    );
    let raw = critic.ask(sys, &user).ok()?;
    let v: Value = serde_json::from_str(
        raw.trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim(),
    )
    .ok()?;
    Some(format!(
        "{}\n[método sugerido] {}",
        v.get("sharper")?.as_str()?,
        v.get("method")?.as_str()?
    ))
}

#[cfg(not(feature = "critic"))]
pub fn elaborate(_agent: &AgentR, _question_id: u64) -> Option<String> {
    None
}

/// Helpers de observación para los triggers más comunes (azúcar sobre `ask`).
pub fn observe_metric_delta(
    agent: &mut AgentR,
    metric: &str,
    from: f64,
    to: f64,
    min_delta: f64,
) -> Option<u64> {
    if (to - from).abs() >= min_delta {
        Some(ask(
            agent,
            CriticalPoint::MetricDelta {
                metric: metric.into(),
                from,
                to,
            },
        ))
    } else {
        None
    }
}

pub fn observe_claim_without_validation(agent: &mut AgentR, claim: &str) -> u64 {
    ask(
        agent,
        CriticalPoint::ClaimWithoutValidation {
            claim: claim.into(),
        },
    )
}

pub fn observe_security_finding(agent: &mut AgentR, finding: &str) -> u64 {
    ask(
        agent,
        CriticalPoint::SecurityFinding {
            finding: finding.into(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_r::AgentR;

    #[test]
    fn test_ask_genera_pregunta_abierta_dedup() {
        let mut a = AgentR::new_ephemeral();
        let id1 = ask(
            &mut a,
            CriticalPoint::MetricDelta {
                metric: "recall".into(),
                from: 100.0,
                to: 88.9,
            },
        );
        assert_eq!(a.questions_iter().count(), 1);
        let q = a.questions_iter().find(|q| q.id == id1).unwrap();
        assert_eq!(q.status, QStatus::Open);
        assert_eq!(q.priority, Priority::High);
        assert!(q.text.contains("recall"));
        // mismo punto => misma pregunta (dedup por texto)
        let id2 = ask(
            &mut a,
            CriticalPoint::MetricDelta {
                metric: "recall".into(),
                from: 100.0,
                to: 88.9,
            },
        );
        assert_eq!(id1, id2, "pregunta duplicada = NOOP");
        assert_eq!(a.questions_iter().count(), 1);
    }

    #[test]
    fn test_answer_cierra_con_leccion() {
        let mut a = AgentR::new_ephemeral();
        let id = ask(
            &mut a,
            CriticalPoint::ClaimWithoutValidation {
                claim: "el label post-hoc mejora la precisión".into(),
            },
        );
        let lesson = a.record_error(
            crate::agent_r::ErrorCategory::SkippedValidation,
            crate::agent_r::ErrorSeverity::Warning,
            "rsi",
            "afirmación sin A/B: precisión 40->100 verificada recién al pedirlo",
            "no había trigger de validación numérica en el flujo",
            &["rsi", "curiosidad"],
            "observe_claim_without_validation en cada afirmación de mejora",
        );
        assert!(a.answer_question(
            id,
            "A/B: 40%->100% precisión, +1 top1 holdout",
            lesson,
            None
        ));
        let q = a.questions_iter().find(|q| q.id == id).unwrap();
        assert_eq!(q.status, QStatus::Answered);
        assert_eq!(q.lesson_id, Some(lesson));
    }

    #[test]
    fn test_observe_metric_delta_respeta_umbral() {
        let mut a = AgentR::new_ephemeral();
        // delta 2.5 < min_delta 5 => no pregunta
        assert!(observe_metric_delta(&mut a, "top1", 3.0, 5.5, 5.0).is_none());
        // delta 11 >= 5 => pregunta
        assert!(observe_metric_delta(&mut a, "recall", 100.0, 89.0, 5.0).is_some());
        assert_eq!(a.questions_iter().count(), 1);
    }

    #[test]
    fn test_preguntas_snapshot_roundtrip() {
        let dir = std::env::temp_dir().join(format!("rsi_curiosidad_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json");
        {
            let mut a = AgentR::with_state_path(path.clone());
            ask(
                &mut a,
                CriticalPoint::SecurityFinding {
                    finding: "receipt re-usable infla W".into(),
                },
            );
            a.flush().unwrap();
        }
        let b = AgentR::with_state_path(path);
        assert_eq!(b.questions_iter().count(), 1);
        assert!(b.questions_iter().next().unwrap().text.contains("receipt"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
