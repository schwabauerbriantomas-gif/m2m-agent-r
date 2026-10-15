//! Demo curiosidad guiada: las 3 preguntas que user hizo manualmente en la
//! ronda 2, regeneradas por el sistema y respondidas con la evidencia real.
//! Uso: cargo run --release --example curiosidad_demo
use m2m_vector_search::agent_r::{AgentR, ErrorCategory, ErrorSeverity};
use m2m_vector_search::meta_curiosity::{self, CriticalPoint, QStatus};

fn main() {
    let mut a = AgentR::new_ephemeral();

    // P1 (user: "Valida numéricamente") — trigger: claim sin A/B
    let p1 = meta_curiosity::observe_claim_without_validation(
        &mut a,
        "el label post-hoc mejora la señal del tuning",
    );
    let l1 = a.record_error(
        ErrorCategory::SkippedValidation,
        ErrorSeverity::Warning,
        "rsi",
        "mejora afirmada sin A/B (label identity vs post-hoc)",
        "el flujo no tenía trigger de validación numérica",
        &["rsi", "curiosidad"],
        "observe_claim_without_validation en cada afirmación de mejora",
    );
    a.answer_question(
        p1,
        "A/B: precision 40%->100% (+60pts), +1 top1 holdout, receipt rcpt_1a14090c319",
        l1,
        None,
    );

    // P2 (user: "¿Por qué baja el recall?") — trigger: delta de métrica
    let p2 = meta_curiosity::observe_metric_delta(&mut a, "recall", 100.0, 88.9, 5.0)
        .expect("delta 11.1 >= 5 debe disparar la pregunta");
    let l2 = a.record_error(
        ErrorCategory::BadAssumption,
        ErrorSeverity::Warning,
        "rsi",
        "recall caía 100->88.9 por artefacto del harness: GT en el contexto anclaba al crítico",
        "el grader le pasaba la respuesta al evaluador en el contexto",
        &["rsi", "bench"],
        "el contexto del crítico NUNCA incluye el GT (aislado: 0/4 con GT vs 3/3 sin GT)",
    );
    a.answer_question(
        p2,
        "artefacto del harness: GT en contexto. Sin GT: detección 3/3-4/4 del mismo defecto",
        l2,
        None,
    );

    // P3 (user: la pregunta sobre preguntas) — trigger: surprise.
    // Queda ABIERTA: su respuesta es este módulo mismo (meta_curiosity).
    let _p3 = meta_curiosity::ask(
        &mut a,
        CriticalPoint::Surprise {
            expected: "el sistema formula sus preguntas de investigación".into(),
            got: "3 preguntas guía vinieron del humano en una sola ronda".into(),
        },
    );

    println!("preguntas generadas: {}", a.questions_iter().count());
    for q in a.questions_iter() {
        println!("  [{:?}] {} -> {:?}", q.priority, q.text, q.status);
    }
    let abiertas = a.questions_iter().filter(|q| q.status == QStatus::Open).count();
    println!("abiertas: {abiertas}");
}
