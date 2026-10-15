//! agent_r::meta — Crítico LLM externo envuelto en la lib (RSI Capa 2).
//!
//! REGLA (guardrail arxiv:2310.01798 + orden de user 15-oct): la crítica
//! LLM externa cuenta como señal SOLO si
//!   1. se emite un receipt HMAC del claim ANTES de consultar al crítico,
//!   2. el veredicto pasa por la validación programática N0–N3 (con store)
//!      DESPUÉS — el crítico nunca auto-aplica nada,
//!   3. todo queda en el snapshot del Agent-R (memoria, no stdout).
//!
//! Modelo: **glm-5.3-flash** (orden verbatim de user: "No uses glm 4.7 usa
//! glm 5.3 flash"). Endpoint z.ai coding PAAS, temperature 0.0,
//! max_tokens>=3000 (el reasoning_content consume tokens).
//!
//! Compila solo con feature `critic` (ureq). Sin la feature, todo el módulo
//! existe como stub que devuelve Unavailable — cero costo en binarios que no
//! lo usan.
//!
//! Uso típico (endpoint POST /agent-r/refine en api_server):
//! ```no_run
//! use m2m_vector_search::meta::Critic;
//! let critic = Critic::from_env().unwrap();
//! let out = critic.refine_operation("deploy_prod",
//!     "salida sospechosa del executor", "contexto de la operación");
//! // out.receipt_id: receipt del claim emitido ANTES de la crítica.
//! // out.validation: N0–N3 del output refinado (post-crítica).
//! ```

use crate::agent_r::AgentR;
use crate::receipts::ReceiptStore;
use serde_json::{json, Value};

/// Modelo del crítico — FIJO por orden de user (15-oct): glm-5.3-flash.
pub const CRITIC_MODEL: &str = "glm-5.3-flash";
/// Endpoint z.ai (plan coding de user; costo cubierto por el plan).
pub const CRITIC_ENDPOINT: &str = "https://api.z.ai/api/coding/paas/v4/chat/completions";
/// Piso de max_tokens: el modelo razona (reasoning_content consume tokens).
pub const CRITIC_MAX_TOKENS: u32 = 3000;

#[derive(Debug, Clone, serde::Serialize)]
pub struct RefineOutcome {
    /// Receipt HMAC emitido ANTES de consultar al crítico (claim sellado).
    pub receipt_claim: Option<String>,
    /// Receipt del resultado refinado (post-verificación).
    pub receipt_result: Option<String>,
    /// Texto propuesto por el crítico (crudo, para auditoría).
    pub critic_raw: Option<String>,
    /// Veredicto N0–N3 programático sobre el output refinado.
    pub validation: Value,
    /// "refined" | "kept_original" | "error"
    pub decision: &'static str,
    /// Motivo del decision (auditable).
    pub reason: String,
}

/// Stub cuando la feature `critic` está apagada: siempre Unavailable.
#[cfg(not(feature = "critic"))]
pub struct Critic;

#[cfg(not(feature = "critic"))]
impl Critic {
    pub fn from_env() -> Result<Self, String> {
        Err("feature 'critic' no compilada".into())
    }
}

#[cfg(feature = "critic")]
pub struct Critic {
    api_key: String,
    model: String,
    endpoint: String,
    http: ureq::Agent,
}

#[cfg(feature = "critic")]
impl Critic {
    /// Key de Z_AI_API_KEY (o ZAI_API_KEY). Sin key => error explícito.
    pub fn from_env() -> Result<Self, String> {
        let api_key = std::env::var("Z_AI_API_KEY")
            .or_else(|_| std::env::var("ZAI_API_KEY"))
            .map_err(|_| "Z_AI_API_KEY no seteada (crítico LLM deshabilitado)")?;
        Ok(Self {
            api_key,
            model: CRITIC_MODEL.into(),
            endpoint: CRITIC_ENDPOINT.into(),
            http: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(120))
                .build(),
        })
    }

    /// Modelo efectivo (siempre glm-5.3-flash hoy; testeable).
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Una llamada al crítico. Devuelve el texto del primer choice.
    pub fn ask(&self, system: &str, user: &str) -> Result<String, String> {
        let body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.0,
            "max_tokens": CRITIC_MAX_TOKENS,
        });
        let body_str = serde_json::to_string(&body).map_err(|e| e.to_string())?;
        let resp = self
            .http
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_string(&body_str)
            .map_err(|e| format!("crítico HTTP: {e}"))?;
        let text = resp
            .into_string()
            .map_err(|e| format!("crítico body: {e}"))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("crítico JSON: {e}"))?;
        v["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| format!("crítico sin content: {}", v))
    }

    /// Prompt de crítica: output del executor + contexto => JSON con
    /// {verdict: "ok"|"issues", issues: [...], refined: "..."}.
    /// El prompt EXIGE formato JSON para parseo programático (nada de prosa).
    pub fn critique_output(&self, operation: &str, output: &str, context: &str) -> Result<Value, String> {
        let system = "Sos un crítico estricto de outputs de agentes. Respondé SOLO JSON válido, sin markdown, con esta forma exacta: {\"verdict\":\"ok\"|\"issues\",\"issues\":[\"...\"],\"refined\":\"...\"}. En refined devolvé la versión corregida del output si hay issues; si verdict=ok, refined = output original.";
        let user = format!(
            "Operación: {operation}\n\nContexto:\n{context}\n\nOutput a criticar:\n{output}\n\nRespondé solo el JSON."
        );
        let raw = self.ask(system, &user)?;
        // El modelo puede envolver en ```json ... ``` — limpiar.
        let cleaned = raw
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        serde_json::from_str(cleaned).map_err(|e| format!("crítico no-JSON: {e}; raw={raw}"))
    }
}

/// Flujo completo Self-Refine integrado (disponible con y sin feature):
/// 1. receipt del claim (output original) — ANTES.
/// 2. crítica LLM (si hay feature + key) o saltear (sin feature).
/// 3. validación N0–N3 del output (refinado u original) con store — DESPUÉS.
/// 4. si la validación NO es `pass`, se mantiene el ORIGINAL (el crítico
///    jamás degrada el estado: decisión conservadora por guardrail).
///
/// `agent` provee validate_output_v2_full; `store` firma los receipts.
pub fn refine(
    agent: &mut AgentR,
    store: &ReceiptStore,
    operation: &str,
    output: &Value,
    context: &str,
) -> RefineOutcome {
    // 1. Receipt del claim ANTES de cualquier crítica.
    let claim = json!({
        "kind": "refine_claim",
        "operation": operation,
        "output": output,
        "context": context,
    });
    let receipt_claim = store.issue(&claim).map(|r| r.id).ok();

    // 2. Crítica (feature-gated).
    #[cfg(feature = "critic")]
    let refined = match Critic::from_env() {
        Ok(critic) => match critic.critique_output(
            operation,
            &serde_json::to_string_pretty(output).unwrap_or_default(),
            context,
        ) {
            Ok(v) if v["verdict"] == json!("issues") => Some(v),
            Ok(_) => None, // verdict ok => no hay nada que refinar
            Err(e) => {
                return RefineOutcome {
                    receipt_claim,
                    receipt_result: None,
                    critic_raw: None,
                    validation: json!({"error": e}),
                    decision: "kept_original",
                    reason: format!("crítico falló ({e}); se mantiene el original"),
                };
            }
        },
        Err(e) => {
            return RefineOutcome {
                receipt_claim,
                receipt_result: None,
                critic_raw: None,
                validation: json!({"critic": "unavailable", "why": e}),
                decision: "kept_original",
                reason: "sin crítico disponible; solo validación programática".into(),
            };
        }
    };
    #[cfg(not(feature = "critic"))]
    let refined: Option<Value> = None;

    let critic_raw = refined.as_ref().map(|v| v.to_string());
    let candidate: Value = refined
        .as_ref()
        .and_then(|v| serde_json::from_str(v["refined"].as_str().unwrap_or("")).ok())
        .unwrap_or_else(|| output.clone());

    // 3. Validación N0–N3 del candidato (con receipt del claim en evidencia).
    let evidence = json!({
        "receipt_id": receipt_claim,
        "refined_by": CRITIC_MODEL,
    });
    let validation = agent.validate_output_v2_full(operation, &candidate, Some(&evidence), Some(store));
    let overall = validation.overall.clone();

    // 4. Decisión conservadora: solo un candidato "pass" (N0/N1+N2/N3)
    // reemplaza al original. pass_unverified NO cuenta (R03 Capa 1).
    let (decision, reason, receipt_result) = if overall == "pass" {
        let rr = store
            .issue(&json!({
                "kind": "refine_result",
                "operation": operation,
                "output": candidate,
                "claim_receipt": receipt_claim,
                "validation": validation.overall,
            }))
            .map(|r| r.id)
            .ok();
        (
            "refined",
            "candidato pass verificado (N0–N3)".to_string(),
            rr,
        )
    } else if refined.is_some() {
        (
            "kept_original",
            format!("candidato refinado no pasó validación ({overall}); original conservado"),
            None,
        )
    } else {
        (
            "kept_original",
            "sin issues detectados por el crítico".to_string(),
            None,
        )
    };

    RefineOutcome {
        receipt_claim,
        receipt_result,
        critic_raw,
        validation: serde_json::to_value(&validation).unwrap_or_default(),
        decision,
        reason,
    }
}

#[cfg(all(test, feature = "critic"))]
mod tests {
    use super::*;

    #[test]
    fn test_model_fijo_glm53flash() {
        let c = Critic::from_env().or_else(|_| {
            std::env::set_var("Z_AI_API_KEY", "test-key");
            Critic::from_env()
        }).expect("critic");
        assert_eq!(c.model(), "glm-5.3-flash");
    }
}
