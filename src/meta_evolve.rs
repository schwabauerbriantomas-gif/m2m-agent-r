//! agent_r::meta::evolve — Evolución ACOTADA de parámetros del snapshot
//! (RSI Capa 3, fase 2: la variante viable de STOP/AlphaEvolve para Agent-R).
//!
//! QUÉ se evoluciona: SOLO parámetros acotados del snapshot (W_content),
//! NO código (Gödel-meta descartada con evidencia: 92% drops).
//!
//! CÓMO (promoción N0→N3 acotada):
//!   N0  candidato generado por mutación acotada (grid determinista).
//!   N1  replay del evaluator (bench holdout) con métrica programática.
//!   N2  comparación contra el incumbent con margen mínimo (+delta >= umbral).
//!   N3  receipt HMAC del experimento completo + rollback path firmado.
//!
//! El snapshot se persiste ANTES de aplicar (rollback por receipt: el path
//! del backup viaja en el receipt). Sin pass en N2 => incumbent se conserva.
//!
//! Uso (harness R02 como evaluator):
//! ```no_run
//! use agent_r::agent_r::AgentR;
//! use agent_r::meta_evolve::{EvolveSpace, ReplayScore};
//! use agent_r::receipts::ReceiptStore;
//!
//! let mut agent = AgentR::new_ephemeral();
//! let store = ReceiptStore::open("receipts").unwrap();
//! let space = EvolveSpace { w_values: vec![1.0, 1.5, 2.0, 2.5, 3.0] };
//! let report = agent_r::meta_evolve::evolve(&mut agent, &store, &space,
//!     |_a, w| ReplayScore { w, score: 1.0, tiebreak: 1.0, n: 10 }, 0.05);
//! // report.promoted: true si el mejor W superó al incumbent por >= margin.
//! ```

use crate::agent_r::AgentR;
use crate::receipts::ReceiptStore;
use serde_json::json;

/// Espacio de búsqueda acotado (determinista, SIN mutación estocástica:
/// reproducibilidad total — mismo grid, mismo resultado).
#[derive(Debug, Clone)]
pub struct EvolveSpace {
    /// Valores candidatos de W_content (grid fijo).
    pub w_values: Vec<f64>,
}

impl Default for EvolveSpace {
    fn default() -> Self {
        Self {
            w_values: vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        }
    }
}

/// Resultado del replay de UN candidato (el closure lo produce).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReplayScore {
    pub w: f64,
    /// Métrica principal (ej: top1 hits del holdout).
    pub score: f64,
    /// Métrica secundaria (ej: top3 hits) para desempate auditable.
    pub tiebreak: f64,
    /// N tareas evaluadas (0 => candidato inválido).
    pub n: u64,
}

/// Reporte completo de la evolución (receipt HMAC firmado).
#[derive(Debug, Clone, serde::Serialize)]
pub struct EvolveReport {
    pub incumbent_w: f64,
    pub incumbent_score: f64,
    pub best_w: f64,
    pub best_score: f64,
    /// true si el best reemplazó al incumbent (pass N2 con margen).
    pub promoted: bool,
    /// Delta mínimo exigido para promover (anti-ruido).
    pub margin: f64,
    pub receipts: Vec<String>,
    pub grid: Vec<ReplayScore>,
    pub backup_path: Option<String>,
}

/// Evolución acotada: para cada W del grid, re-encarnar el agente del
/// snapshot, correr el evaluator con ese W, y promover solo si supera al
/// incumbent por >= margin. Receipt por candidato + receipt final.
///
/// `evaluate` recibe (&mut AgentR, w) y devuelve ReplayScore. Debe ser
/// DETERMINISTA (mismo snapshot => mismo score) y programático (nada de
/// self-report): el harness R02 (bench C1/C2 re-encarnado) cumple ambos.
///
/// PERSISTENCIA: si el agente tiene state_path, se hace backup del snapshot
/// antes de aplicar el W ganador (rollback posible por el path del receipt).
pub fn evolve<F>(
    agent: &mut AgentR,
    store: &ReceiptStore,
    space: &EvolveSpace,
    mut evaluate: F,
    margin: f64,
) -> EvolveReport
where
    F: FnMut(&mut AgentR, f64) -> ReplayScore,
{
    // N0/N1: replay del grid (cada candidato se mide en agente limpio).
    // El caller provee el evaluator; acá solo orquestamos y sellamos.
    let mut grid: Vec<ReplayScore> = Vec::new();
    let mut receipts: Vec<String> = Vec::new();

    // El incumbent SIEMPRE entra al replay: si su W no está en el grid, se
    // mide aparte (si no, incumbent_score=-inf y cualquier candidato promueve
    // "gratis" — bug encontrado por la muestra del swarm con W=3.103).
    let incumbent_w = agent.tuning_stats().2;
    let incumbent_measured = space
        .w_values
        .iter()
        .any(|w| (w - incumbent_w).abs() < 1e-9);
    if !incumbent_measured {
        let s = evaluate(agent, incumbent_w);
        let receipt = store
            .issue(&json!({
                "kind": "evolve_incumbent",
                "w": s.w, "score": s.score, "tiebreak": s.tiebreak, "n": s.n,
            }))
            .map(|r| r.id)
            .ok();
        if let Some(r) = receipt {
            receipts.push(r);
        }
        grid.push(s);
    }

    for &w in &space.w_values {
        let score = evaluate(agent, w);
        let receipt = store
            .issue(&json!({
                "kind": "evolve_candidate",
                "w": score.w,
                "score": score.score,
                "tiebreak": score.tiebreak,
                "n": score.n,
            }))
            .map(|r| r.id)
            .ok();
        if let Some(r) = receipt {
            receipts.push(r);
        }
        grid.push(score);
    }

    // N2: incumbent = W actual del agente (ya medido arriba si está en el grid).
    let incumbent_w = agent.tuning_stats().2;
    let incumbent_score = grid
        .iter()
        .filter(|s| (s.w - incumbent_w).abs() < 1e-9)
        .map(|s| s.score)
        .next()
        .unwrap_or_else(|| {
            // Respaldo: con el incumbent medido arriba nunca debería pasar.
            grid.iter().map(|s| s.score).fold(f64::MIN, f64::min)
        });

    let best = grid
        .iter()
        .max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    a.tiebreak
                        .partial_cmp(&b.tiebreak)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        })
        .cloned()
        .expect("grid no vacío");

    // Promoción: best supera incumbent por >= margin (anti-ruido) y n>0.
    let promoted = best.n > 0
        && best.score >= incumbent_score + margin
        && (best.w - incumbent_w).abs() >= 1e-9;

    let mut backup_path = None;
    if promoted {
        // N3-rollback: backup del snapshot antes de tocar W.
        if let Some(path) = agent.state_path_opt() {
            let backup = path.with_extension("json.pre-evolve");
            if agent.backup_state_to(&backup).is_ok() {
                backup_path = Some(backup.display().to_string());
            }
            agent.set_w_content(best.w);
            agent.flush().ok();
        } else {
            // RAM-only: aplicar en memoria (el caller decide persistir).
            agent.set_w_content(best.w);
        }
    }

    let report = EvolveReport {
        incumbent_w,
        incumbent_score,
        best_w: best.w,
        best_score: best.score,
        promoted,
        margin,
        receipts,
        grid,
        backup_path,
    };

    // Receipt final del experimento completo (integridad auditable).
    if let Ok(r) = store.issue(&serde_json::to_value(&report).unwrap_or_default()) {
        // El id del receipt final no viaja en el propio payload (recursión);
        // queda en el store para el verify posterior.
        let _ = r;
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolve_promueve_solo_con_margen() {
        use crate::receipts::ReceiptStore;
        let dir = std::env::temp_dir().join(format!("rsi_evolve_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = ReceiptStore::open_no_env(&dir).unwrap();
        let mut a = AgentR::new_ephemeral(); // W inicial 3.0

        let space = EvolveSpace {
            w_values: vec![3.0, 1.0, 0.5],
        };
        // Evaluator sintético: score = función determinista que favorece w=1.0
        let report = evolve(
            &mut a,
            &store,
            &space,
            |_agent, w| ReplayScore {
                w,
                score: 100.0 - (w - 1.0).abs() * 10.0,
                tiebreak: 0.0,
                n: 10,
            },
            0.5,
        );
        assert_eq!(report.incumbent_w, 3.0);
        assert_eq!(report.best_w, 1.0);
        assert!(report.promoted, "1.0 supera 3.0 por 15 >= 0.5");
        assert_eq!(a.tuning_stats().2, 1.0, "W aplicado");
        assert_eq!(report.receipts.len(), 3, "un receipt por candidato");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_evolve_incumbent_fuera_del_grid_se_mide() {
        use crate::receipts::ReceiptStore;
        let dir = std::env::temp_dir().join(format!("rsi_evolve3_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = ReceiptStore::open_no_env(&dir).unwrap();
        let mut a = AgentR::new_ephemeral();
        a.set_w_content(3.103); // W del swarm real: NO está en el grid 0.5..4.0

        let space = EvolveSpace {
            w_values: vec![3.0, 3.5, 4.0],
        };
        // Evaluator: score = 2 hits para cualquier W (empate con el incumbent
        // que también daría 2). Con el bug viejo: incumbent_score=-inf =>
        // promovía "gratis". Con el fix: empate => NO promociona (sin margen).
        let report = evolve(
            &mut a,
            &store,
            &space,
            |_agent, w| ReplayScore {
                w,
                score: 2.0,
                tiebreak: 0.0,
                n: 3,
            },
            0.5,
        );
        assert!(!report.promoted, "empate total no promociona");
        assert_eq!(report.incumbent_score, 2.0, "incumbent medido (no -inf)");
        assert_eq!(a.tuning_stats().2, 3.103, "W del incumbent conservado");
        assert_eq!(report.receipts.len(), 4, "3 grid + 1 incumbent");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_evolve_no_promueve_bajo_margen() {
        use crate::receipts::ReceiptStore;
        let dir = std::env::temp_dir().join(format!("rsi_evolve2_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = ReceiptStore::open_no_env(&dir).unwrap();
        let mut a = AgentR::new_ephemeral();

        let space = EvolveSpace {
            w_values: vec![3.0, 3.4],
        };
        // Score diff 0.4 < margin 0.5 => NO promociona (anti-ruido)
        let report = evolve(
            &mut a,
            &store,
            &space,
            |_agent, w| ReplayScore {
                w,
                score: 30.0 + (w - 3.0) * 0.4 / 0.4,
                tiebreak: 0.0,
                n: 10,
            },
            0.5,
        );
        assert!(!report.promoted);
        assert_eq!(a.tuning_stats().2, 3.0, "incumbent conserva W");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
