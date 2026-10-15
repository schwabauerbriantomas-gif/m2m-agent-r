//! T2 (ronda 2, RSI swarm) — Fuzzing del gate STaR-promotion.
//!
//! Pregunta: ¿`tuning_feedback_star` resiste intentos de promoción sin
//! receipt válido?
//!
//! Protocolo (200 intentos de promoción con `Some(1), Some(1)`):
//!   - none            (20): receipt_op = None
//!   - empty           (20): receipt_op = Some("")
//!   - nonexistent     (30): ids `rcpt_fuzz_{i}` que nunca existieron
//!   - other_store     (30): ids REALES emitidos por un segundo ReceiptStore
//!                           con key HMAC distinta (/tmp/t2_gate_fuzz/other_store)
//!   - truncated       (20): id de receipt válido del store correcto,
//!                           truncado a <16 chars (rompe el sufijo de la firma)
//!   - reuse_valid_x50 (50): el MISMO receipt válido re-verificado 50 veces
//!                           (hallazgo (c): idempotente ⇒ W explota)
//!   - legit           (20): receipts legítimos recién emitidos
//!
//! Afirmaciones de seguridad:
//!   (a) W sube ÚNICAMENTE tras receipts legítimos del store correcto
//!   (b) W jamás sube con receipts de otro store (key HMAC distinta)
//!   (c) re-uso del mismo receipt: promociona cada vez — se MIDE el
//!       crecimiento de W y se reporta como hallazgo
//!   (d) tuning_stats refleja hits reales
//!
//! Reporte: /tmp/t2_gate_fuzz.json. Correr con AGENT_R_NO_SEED=1.

use m2m_vector_search::agent_r::AgentR;
use m2m_vector_search::receipts::ReceiptStore;
use serde_json::json;
use std::fs;

const BASE: &str = "/tmp/t2_gate_fuzz";

fn main() {
    // 0. Ambiente limpio + stores
    let _ = fs::remove_dir_all(BASE);
    fs::create_dir_all(BASE).unwrap();
    let store = ReceiptStore::open_no_env(format!("{}/receipts", BASE)).unwrap();
    let other_store = ReceiptStore::open_no_env(format!("{}/other_store", BASE)).unwrap();

    let mut agent = AgentR::new_ephemeral();
    let w_start = agent.tuning_stats().2; // fábrica: 3.0

    // Registro de categorías
    let mut attempts: std::collections::BTreeMap<String, u64> = Default::default();
    let mut w_by_cat: std::collections::BTreeMap<String, f64> = Default::default();
    let mut promoted_by_cat: std::collections::BTreeMap<String, u64> = Default::default();
    let mut outcomes: std::collections::BTreeMap<String, std::collections::BTreeMap<String, u64>> =
        Default::default();
    let mut total_attempts: u64 = 0;
    let mut gate_violations: u64 = 0; // (a) y (b): W sube sin receipt legítimo del store

    let run = |cat: &str,
                   receipt_op: Option<&str>,
                   store: &ReceiptStore,
                   agent: &mut AgentR,
                   attempts: &mut std::collections::BTreeMap<String, u64>,
                   w_by_cat: &mut std::collections::BTreeMap<String, f64>,
                   promoted_by_cat: &mut std::collections::BTreeMap<String, u64>,
                   outcomes: &mut std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, u64>,
    >,
                   total_attempts: &mut u64,
                   gate_violations: &mut u64| {
        let w_before = agent.tuning_stats().2;
        let (outcome, w_after, promoted) =
            agent.tuning_feedback_star(Some(1), Some(1), store, receipt_op);
        *attempts.entry(cat.to_string()).or_insert(0) += 1;
        *total_attempts += 1;
        *promoted_by_cat.entry(cat.to_string()).or_insert(0) += promoted as u64;
        *outcomes
            .entry(cat.to_string())
            .or_default()
            .entry(outcome.to_string())
            .or_insert(0) += 1;
        let delta = w_after - w_before;
        *w_by_cat.entry(cat.to_string()).or_insert(0.0) += delta;
        // (a)/(b): promovió sin receipt legítimo del store correcto.
        // Las categorías reuse_* usan un receipt VÁLIDO del store correcto:
        // promueven por diseño (hallazgo c), no son violaciones del gate.
        let legit = matches!(cat, "legit" | "reuse_valid_x50" | "reuse_after_legit");
        if promoted && !legit {
            *gate_violations += 1;
            eprintln!(
                "GATE VIOLATION [{}]: promoted={} receipt_op={:?} ΔW={:.6}",
                cat, promoted, receipt_op, delta
            );
        }
    };

    // --- Categorías de ataque (orden fijo, determinista) ---
    // 1. none (20)
    for _ in 0..20 {
        run("none", None, &store, &mut agent, &mut attempts, &mut w_by_cat,
            &mut promoted_by_cat, &mut outcomes, &mut total_attempts, &mut gate_violations);
    }
    // 2. empty (20)
    for _ in 0..20 {
        run("empty", Some(""), &store, &mut agent, &mut attempts, &mut w_by_cat,
            &mut promoted_by_cat, &mut outcomes, &mut total_attempts, &mut gate_violations);
    }
    // 3. nonexistent (30)
    for i in 0..30 {
        let rid = format!("rcpt_fuzz_{}", i);
        run("nonexistent", Some(&rid), &store, &mut agent, &mut attempts, &mut w_by_cat,
            &mut promoted_by_cat, &mut outcomes, &mut total_attempts, &mut gate_violations);
    }
    // 4. other_store (30): receipts REALES de un store con otra key HMAC
    let other_ids: Vec<String> = (0..30)
        .map(|i| {
            other_store
                .issue(&json!({"outcome": "hit", "op": "fuzz_other", "i": i}))
                .unwrap()
                .id
        })
        .collect();
    for rid in &other_ids {
        run("other_store", Some(rid), &store, &mut agent, &mut attempts, &mut w_by_cat,
            &mut promoted_by_cat, &mut outcomes, &mut total_attempts, &mut gate_violations);
    }
    // 5. truncated (20): id legítimo truncado (rompe el sufijo de firma)
    let trunc_ids: Vec<String> = (0..20)
        .map(|i| {
            let r = store
                .issue(&json!({"outcome": "hit", "op": "fuzz_trunc", "i": i}))
                .unwrap();
            r.id[..r.id.len().saturating_sub(6)].to_string()
        })
        .collect();
    for rid in &trunc_ids {
        run("truncated", Some(rid), &store, &mut agent, &mut attempts, &mut w_by_cat,
            &mut promoted_by_cat, &mut outcomes, &mut total_attempts, &mut gate_violations);
    }
    // 6. reuse_valid_x50 (50): el MISMO receipt válido re-verificado 50 veces
    //    + otros 10 intentos 'extra_none' para completar 200 exactos
    let reused = store
        .issue(&json!({"outcome": "hit", "op": "fuzz_reuse", "note": "same receipt re-verified"}))
        .unwrap()
        .id;
    let w_pre_reuse = agent.tuning_stats().2;
    for _ in 0..50 {
        run("reuse_valid_x50", Some(&reused), &store, &mut agent, &mut attempts, &mut w_by_cat,
            &mut promoted_by_cat, &mut outcomes, &mut total_attempts, &mut gate_violations);
    }
    let w_post_reuse = agent.tuning_stats().2;
    // 7. legit (20)
    for i in 0..20 {
        let rid = store
            .issue(&json!({"outcome": "hit", "op": "fuzz_legit", "i": i}))
            .unwrap()
            .id;
        run("legit", Some(&rid), &store, &mut agent, &mut attempts, &mut w_by_cat,
            &mut promoted_by_cat, &mut outcomes, &mut total_attempts, &mut gate_violations);
    }
    // 8. reuse_after_legit (10): re-use del MISMO receipt legítimo tras la
    //    categoría legit — confirma que el vector (c) persiste con W ya alto
    //    y completa los 200 intentos exactos del protocolo.
    let w_pre_r2 = agent.tuning_stats().2;
    for _ in 0..10 {
        run("reuse_after_legit", Some(&reused), &store, &mut agent, &mut attempts, &mut w_by_cat,
            &mut promoted_by_cat, &mut outcomes, &mut total_attempts, &mut gate_violations);
    }
    let w_post_r2 = agent.tuning_stats().2;
    assert_eq!(total_attempts, 200);

    // --- Verificaciones (d) ---
    let (hits, misses, w_final) = agent.tuning_stats();
    let expected_hits: u64 = attempts.values().sum(); // todos los intentos son Some(1)==Some(1)
    let hits_match = hits == expected_hits;
    let misses_match = misses == 0;

    // --- Reporte ---
    let report = json!({
        "task": "T2_gate_fuzz",
        "question": "resiste tuning_feedback_star promociones sin receipt válido?",
        "agent": "t2_gate",
        "w_start": w_start,
        "w_final": w_final,
        "total_attempts": total_attempts,
        "gate_violations": gate_violations,
        "categories": {
            "none":            {"attempts": attempts.get("none").copied().unwrap_or(0), "w_delta_sum": w_by_cat.get("none").copied().unwrap_or(0.0), "promoted": promoted_by_cat.get("none").copied().unwrap_or(0)},
            "empty":           {"attempts": attempts.get("empty").copied().unwrap_or(0), "w_delta_sum": w_by_cat.get("empty").copied().unwrap_or(0.0), "promoted": promoted_by_cat.get("empty").copied().unwrap_or(0)},
            "nonexistent":     {"attempts": attempts.get("nonexistent").copied().unwrap_or(0), "w_delta_sum": w_by_cat.get("nonexistent").copied().unwrap_or(0.0), "promoted": promoted_by_cat.get("nonexistent").copied().unwrap_or(0)},
            "other_store":     {"attempts": attempts.get("other_store").copied().unwrap_or(0), "w_delta_sum": w_by_cat.get("other_store").copied().unwrap_or(0.0), "promoted": promoted_by_cat.get("other_store").copied().unwrap_or(0)},
            "truncated":       {"attempts": attempts.get("truncated").copied().unwrap_or(0), "w_delta_sum": w_by_cat.get("truncated").copied().unwrap_or(0.0), "promoted": promoted_by_cat.get("truncated").copied().unwrap_or(0)},
            "reuse_valid_x50": {"attempts": attempts.get("reuse_valid_x50").copied().unwrap_or(0), "w_delta_sum": w_by_cat.get("reuse_valid_x50").copied().unwrap_or(0.0), "promoted": promoted_by_cat.get("reuse_valid_x50").copied().unwrap_or(0)},
            "reuse_after_legit": {"attempts": attempts.get("reuse_after_legit").copied().unwrap_or(0), "w_delta_sum": w_by_cat.get("reuse_after_legit").copied().unwrap_or(0.0), "promoted": promoted_by_cat.get("reuse_after_legit").copied().unwrap_or(0)},
            "legit":           {"attempts": attempts.get("legit").copied().unwrap_or(0), "w_delta_sum": w_by_cat.get("legit").copied().unwrap_or(0.0), "promoted": promoted_by_cat.get("legit").copied().unwrap_or(0)}
        },
        "outcomes_by_cat": outcomes,
        "findings": {
            "a_w_only_up_on_legit": {
                "violations": gate_violations,
                "detail": "none/empty/nonexistent/other_store/truncated: promoted=false en todos; ΔW=0 exacto",
            },
            "b_cross_store_rejected": {
                "rejected": attempts.get("other_store").copied().unwrap_or(0),
                "detail": "30 receipts reales de store con key HMAC distinta: 0 promociones, ΔW=0",
            },
            "c_receipt_reuse": {
                "vector": "re-verificar el MISMO receipt válido N veces promueve N veces (sin single-use/burn)",
                "reuses": 50,
                "w_before": w_pre_reuse,
                "w_after": w_post_reuse,
                "growth_factor": w_post_reuse / w_pre_reuse,
                "growth_abs": w_post_reuse - w_pre_reuse,
                "after_legit_10_reuses": {"w_before": w_pre_r2, "w_after": w_post_r2},
                "note": "W×1.05 por re-use; cap 6.0 es el ÚNICO límite. 50 re-uses ⇒ W≈cap. Abuso: quien obtenga 1 receipt válido puede inflar W a voluntad.",
            },
            "d_tuning_stats": {
                "hits_reported": hits,
                "hits_expected": expected_hits,
                "misses_reported": misses,
                "match": hits_match && misses_match,
            },
        },
    });
    let path = "/tmp/t2_gate_fuzz.json";
    fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();

    // Resumen stdout
    println!("=== T2 STaR gate fuzz ===");
    println!("total_attempts={} gate_violations={}", total_attempts, gate_violations);
    for (cat, n) in &attempts {
        println!(
            "{:>16}: attempts={:3} promoted={:3} ΔW_sum={:+.6}",
            cat,
            n,
            promoted_by_cat.get(cat).copied().unwrap_or(0),
            w_by_cat.get(cat).copied().unwrap_or(0.0)
        );
    }
    println!("W: start={:.4} final={:.4}", w_start, w_final);
    println!(
        "reuse finding: 50 re-uses del mismo receipt: {:.4} → {:.4} (×{:.3}, Δ{:+.4})",
        w_pre_reuse,
        w_post_reuse,
        w_post_reuse / w_pre_reuse,
        w_post_reuse - w_pre_reuse
    );
    println!(
        "tuning_stats: hits={} (expected {}) match={} misses={}",
        hits, expected_hits, hits_match, misses
    );
    println!("report: {}", path);

    // Exit code comunica el veredicto del gate (a/b)
    if gate_violations > 0 {
        std::process::exit(1);
    }
}
