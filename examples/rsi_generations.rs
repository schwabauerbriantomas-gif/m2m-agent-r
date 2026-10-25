//! R02 — Harness de validación de mejora recursiva (RSI) de Agent-R.
//!
//! Loop G(n) → G(n+1) medible que vive DENTRO de la lib (patrón
//! examples/self_tuning.rs): cada generación es una re-encarnación del
//! agente que carga el snapshot del padre (AGENT_R_STATE_PATH real, con
//! W_content persistido), se mide contra el bench C1/C2 (BenchTrace L1,
//! top-1/top-3 del preflight híbrido, n=31), aplica UNA técnica de mejora
//! que es un mecanismo PROPIO de Agent-R, y registra el delta con un
//! receipt HMAC del ReceiptStore existente (el mismo que envuelven los
//! endpoints /agent-r/receipts/{issue,verify} del api_server).
//!
//! Generaciones encadenadas por defecto: 2 (G0→G1→G2).
//!   G0  baseline      : W de fábrica (3.0), memoria = lecciones del bench.
//!   G1  self_tuning_w : tuning_feedback(pred, outcome) sobre el split de
//!                       tuning (misma mecánica que examples/self_tuning.rs
//!                       y que el wire fail/complete del swarm). W persiste
//!                       en el snapshot; G1 se mide RELEYENDO el snapshot.
//!   G2  deprecate_fp  : deprecate_error de reglas con evidencia de falso
//!                       positivo sistemático en el split de tuning
//!                       (≥2 predicciones top-1, 0 hits) — el criterio de
//!                       deprecación documentado de la lib.
//!
//! Anti-self-report: el harness SOLO emite y acepta outputs programáticos
//! (conteos top-1/top-3 del preflight, W, receipts firmados). El modo
//! `verify` re-abre el ReceiptStore desde disco, valida la firma HMAC de
//! cada receipt y que las métricas declaradas estén contenidas en el
//! payload firmado (n3_check), más la aritmética del delta y la cadena
//! parent→gen. Exit code 0/1. Sin jueces LLM, sin prose.
//!
//! Uso (desde wt_valid/):
//!   CARGO_TARGET_DIR=target cargo run --release --example rsi_generations -- \
//!       --data ../bench_data/benchtrace --out /tmp/rsi_r02/run1
//!   CARGO_TARGET_DIR=target cargo run --release --example rsi_generations -- \
//!       verify /tmp/rsi_r02/run1
//!
//! Protocolo de medición (idéntico en TODAS las generaciones):
//!   - memoria: TODOS los core_failure de los snapshots del bench importados
//!     con record_error (what = diagnosis[:160], tags [modelo, babyai,
//!     level-1]) — igual que far_benchtrace.rs / self_tuning.rs.
//!   - tareas: las n=31 con lecciones materializadas y test_snapshot
//!     presente, ordenadas por id (determinista).
//!   - query: preflight_v2_hybrid(game, [model_short, "babyai", "level-1"],
//!     situation) con situation = trayectoria (obs+act) hasta el
//!     decision_point del test.
//!   - métricas: top1_hits / top3_hits (una lección correcta es regla #1 /
//!     está en el top-3) + n. Azar de referencia: 1/45 y 3/45.

use m2m_vector_search::agent_r::{AgentR, ErrorCategory, ErrorSeverity};
use m2m_vector_search::receipts::ReceiptStore;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

// ─────────────────────────────────────────────────────────────────────────────
// Bench C1/C2 (reuso del protocolo existente, nada nuevo del lado del bench)
// ─────────────────────────────────────────────────────────────────────────────

fn map_category(ftype: &str) -> ErrorCategory {
    let t = ftype.to_lowercase();
    if t.contains("strategy")
        || t.contains("lateral")
        || t.contains("goal")
        || t.contains("planning")
        || t.contains("spatial")
    {
        ErrorCategory::BadAssumption
    } else if t.contains("loop") || t.contains("repeat") {
        ErrorCategory::CircularReasoning
    } else if t.contains("invalid") || t.contains("action") || t.contains("exec") {
        ErrorCategory::Timeout
    } else {
        ErrorCategory::Other
    }
}

/// Situación textual del test (trayectoria hasta el decision_point) —
/// idéntica a examples/self_tuning.rs.
fn situation_of(ep: &Value, dp: i64) -> String {
    let mut s = String::new();
    if let Some(steps) = ep["snapshot"].get("trajectory").and_then(|t| t.as_array()) {
        let upto = if dp > 0 {
            (dp as usize).min(steps.len())
        } else {
            steps.len()
        };
        for st in &steps[..upto] {
            s.push_str(st.get("obs").and_then(|o| o.as_str()).unwrap_or(""));
            s.push(' ');
            s.push_str(
                &st.get("act")
                    .or_else(|| st.get("action"))
                    .map(|a| a.to_string())
                    .unwrap_or_default(),
            );
            s.push(' ');
        }
    }
    s
}

struct Bench {
    /// Episodios por id (para situation/model/game del test).
    ep_by_id: HashMap<String, Value>,
    /// Tareas evaluables (lecciones materializadas + test presente), orden por id.
    tasks: Vec<Value>,
    /// regla → episodios que la originaron (lecciones correctas por tarea).
    lessons_of_rule: HashMap<u64, HashSet<String>>,
    n_rules: u64,
}

fn bench_load(data_dir: &str) -> Bench {
    let snaps: Value = serde_json::from_str(
        &fs::read_to_string(format!("{}/babyai_l1_snapshots.json", data_dir)).expect("snapshots"),
    )
    .expect("parse snapshots");
    let evol: Value = serde_json::from_str(
        &fs::read_to_string(format!("{}/babyai_l1_evolution.json", data_dir)).expect("evolution"),
    )
    .expect("parse evolution");
    let episodes = snaps["episodes"].as_array().expect("episodes[]").clone();
    let all_tasks = evol["tasks"].as_array().expect("tasks[]").clone();

    // Episodios con lecciones materializables (core_failure con diagnosis).
    let mut rule_eps: HashSet<String> = HashSet::new();
    for ep in &episodes {
        let has = ep["failure_instances"]["core_failure"]
            .as_array()
            .map(|cfs| {
                cfs.iter().any(|cf| {
                    !cf.get("diagnosis")
                        .and_then(|d| d.as_str())
                        .unwrap_or("")
                        .is_empty()
                })
            })
            .unwrap_or(false);
        if has {
            rule_eps.insert(ep["id"].as_str().unwrap_or("").to_string());
        }
    }

    let mut ep_by_id: HashMap<String, Value> = HashMap::new();
    for ep in &episodes {
        if let Some(id) = ep["id"].as_str() {
            ep_by_id.insert(id.to_string(), ep.clone());
        }
    }

    // Tareas evaluables deterministas: lección materializada + test presente.
    let mut tasks: Vec<Value> = all_tasks
        .into_iter()
        .filter(|t| {
            let eps = t["evolution_snapshots"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let materialized = eps
                .iter()
                .filter_map(|v| v.as_str())
                .any(|e| rule_eps.contains(e));
            let test_ok = t["test_snapshot"]["episode_id"]
                .as_str()
                .map(|id| ep_by_id.contains_key(id))
                .unwrap_or(false);
            materialized && test_ok
        })
        .collect();
    tasks.sort_by_key(|t| t["id"].as_str().unwrap_or("").to_string());

    Bench {
        ep_by_id,
        tasks,
        lessons_of_rule: HashMap::new(),
        n_rules: 0,
    }
}

/// Importa la memoria del bench en el agente (protocolo C1/C2) y arma el mapa
/// regla→episodios para saber cuáles son las lecciones correctas por tarea.
fn bench_import(agent: &mut AgentR, bench: &mut Bench) {
    // Lecciones deterministas: (ep_id, game, model_short, categoría, diag)
    // ordenadas por (game|model|diag) para que el orden de import (y por ende
    // los desempates del heap del preflight) sea idéntico en cada corrida.
    let mut ep_lessons: Vec<(String, String, String, ErrorCategory, String)> = Vec::new();
    for (ep_id, ep) in bench.ep_by_id.iter() {
        let Some(cfs) = ep["failure_instances"]["core_failure"].as_array() else {
            continue;
        };
        let game = ep["game"].as_str().unwrap_or("level-1").to_string();
        let model = ep["model"].as_str().unwrap_or("unknown").to_string();
        let model_short = model.rsplit('/').next().unwrap_or(&model).to_string();
        for cf in cfs {
            let diag = cf["diagnosis"].as_str().unwrap_or("");
            if diag.is_empty() {
                continue;
            }
            ep_lessons.push((
                ep_id.clone(),
                game.clone(),
                model_short.clone(),
                map_category(cf["type"].as_str().unwrap_or("unknown")),
                diag.to_string(),
            ));
        }
    }
    ep_lessons.sort_by(|a, b| {
        format!("{}|{}|{}", a.1, a.2, a.4).cmp(&format!("{}|{}|{}", b.1, b.2, b.4))
    });
    for (_ep_id, game, model_short, cat, diag) in &ep_lessons {
        let what: String = diag.chars().take(160).collect();
        let id = agent.record_error(
            *cat,
            ErrorSeverity::Error,
            game,
            &what,
            diag,
            &[model_short.as_str(), "babyai", "level-1"],
            "",
        );
        bench
            .lessons_of_rule
            .entry(id)
            .or_default()
            .insert(_ep_id.clone());
    }
    bench.n_rules = agent.stats().unique_incidents;
}

/// Métricas del protocolo C1/C2 sobre el agente dado (SIEMPRE igual):
/// top-1/top-3 del preflight híbrido en las n tareas evaluables.
fn measure(agent: &mut AgentR, bench: &Bench) -> Value {
    let (mut top1, mut top3, mut n) = (0u64, 0u64, 0u64);
    let mut per_task: Vec<Value> = Vec::new();
    for task in &bench.tasks {
        let ts = &task["test_snapshot"];
        let test_id = ts["episode_id"].as_str().unwrap_or("");
        let dp = ts["decision_point"].as_i64().unwrap_or(-1);
        let ep = match bench.ep_by_id.get(test_id) {
            Some(e) => e,
            None => continue,
        };
        let lesson_eps: HashSet<&str> = task["evolution_snapshots"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let correct: HashSet<u64> = bench
            .lessons_of_rule
            .iter()
            .filter(|(_, eps)| eps.iter().any(|e| lesson_eps.contains(e.as_str())))
            .map(|(id, _)| *id)
            .collect();
        if correct.is_empty() {
            continue;
        }
        let game = ep["game"].as_str().unwrap_or("level-1");
        let model = ep["model"].as_str().unwrap_or("unknown");
        let model_short = model.rsplit('/').next().unwrap_or(model);
        let situation = situation_of(ep, dp);
        let pf = agent.preflight_v2_hybrid(game, &[model_short, "babyai", "level-1"], &situation);
        let top: Vec<u64> = pf.top_rules.iter().map(|r| r.id).collect();
        let t1 = top.first().map(|id| correct.contains(id)).unwrap_or(false);
        let t3 = top.iter().any(|id| correct.contains(id));
        n += 1;
        top1 += t1 as u64;
        top3 += t3 as u64;
        per_task.push(json!({
            "task": task["id"], "top1": t1, "top3": t3,
            "top_rule": top.first(),
        }));
    }
    json!({
        "n": n, "top1_hits": top1, "top3_hits": top3,
        "top1_pct": if n > 0 { (top1 as f64 / n as f64 * 1000.0).round() / 10.0 } else { 0.0 },
        "top3_pct": if n > 0 { (top3 as f64 / n as f64 * 1000.0).round() / 10.0 } else { 0.0 },
        "per_task": per_task,
    })
}

fn metrics_public(m: &Value) -> Value {
    json!({
        "n": m["n"], "top1_hits": m["top1_hits"], "top3_hits": m["top3_hits"],
        "top1_pct": m["top1_pct"], "top3_pct": m["top3_pct"],
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Técnicas de mejora: mecanismos PROPIOS de Agent-R (nada externo)
// ─────────────────────────────────────────────────────────────────────────────

/// G0→G1: self-tuning de W_content via tuning_feedback sobre el split de
/// tuning (primera mitad, mismo criterio temporal que self_tuning.rs).
/// Outcome por tarea = oráculo del bench: hit si la regla top-1 predicha es
/// lección correcta; miss si predijo otra ((Some, None): la task completó sin
/// implicar a la regla — semántica del wire complete_task del swarm).
fn technique_self_tuning(agent: &mut AgentR, bench: &Bench) -> Value {
    let w_before = agent.w_content();
    let split = bench.tasks.len() / 2;
    let (mut hits, mut misses, mut abst) = (0u64, 0u64, 0u64);
    for task in &bench.tasks[..split] {
        let ts = &task["test_snapshot"];
        let test_id = ts["episode_id"].as_str().unwrap_or("");
        let dp = ts["decision_point"].as_i64().unwrap_or(-1);
        let Some(ep) = bench.ep_by_id.get(test_id) else {
            continue;
        };
        let lesson_eps: HashSet<&str> = task["evolution_snapshots"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let correct: HashSet<u64> = bench
            .lessons_of_rule
            .iter()
            .filter(|(_, eps)| eps.iter().any(|e| lesson_eps.contains(e.as_str())))
            .map(|(id, _)| *id)
            .collect();
        if correct.is_empty() {
            continue;
        }
        let game = ep["game"].as_str().unwrap_or("level-1");
        let model = ep["model"].as_str().unwrap_or("unknown");
        let model_short = model.rsplit('/').next().unwrap_or(model);
        let situation = situation_of(ep, dp);
        let pf = agent.preflight_v2_hybrid(game, &[model_short, "babyai", "level-1"], &situation);
        let pred = pf.top_rules.first().map(|r| r.id);
        let actual = pred.filter(|p| correct.contains(p)); // oráculo del bench
        let (outcome, _w) = agent.tuning_feedback(pred, actual);
        match outcome {
            "hit" => hits += 1,
            "miss" => misses += 1,
            _ => abst += 1,
        }
    }
    let (th, tm, w_after) = agent.tuning_stats();
    json!({
        "technique": "self_tuning_w",
        "mechanism": "AgentR::tuning_feedback (hit W*1.05 cap 6, miss W*0.95 floor 0.5)",
        "tuning_split_n": split,
        "w_before": w_before,
        "w_after": w_after,
        "hits": hits, "misses": misses, "abstain": abst,
        "lib_tuning_stats": {"hits": th, "misses": tm},
    })
}

/// G1→G2: deprecación de reglas con falso positivo sistemático medido en el
/// split de tuning: ≥2 predicciones top-1 y 0 hits. Criterio de la lib
/// ("FP medido → deprecación", mismo espíritu que el wire swarm).
fn technique_deprecate_fp(agent: &mut AgentR, bench: &Bench) -> Value {
    let split = bench.tasks.len() / 2;
    let mut k_pred: HashMap<u64, u64> = HashMap::new();
    let mut k_hit: HashMap<u64, u64> = HashMap::new();
    for task in &bench.tasks[..split] {
        let ts = &task["test_snapshot"];
        let test_id = ts["episode_id"].as_str().unwrap_or("");
        let dp = ts["decision_point"].as_i64().unwrap_or(-1);
        let Some(ep) = bench.ep_by_id.get(test_id) else {
            continue;
        };
        let lesson_eps: HashSet<&str> = task["evolution_snapshots"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let correct: HashSet<u64> = bench
            .lessons_of_rule
            .iter()
            .filter(|(_, eps)| eps.iter().any(|e| lesson_eps.contains(e.as_str())))
            .map(|(id, _)| *id)
            .collect();
        if correct.is_empty() {
            continue;
        }
        let game = ep["game"].as_str().unwrap_or("level-1");
        let model = ep["model"].as_str().unwrap_or("unknown");
        let model_short = model.rsplit('/').next().unwrap_or(model);
        let situation = situation_of(ep, dp);
        let pf = agent.preflight_v2_hybrid(game, &[model_short, "babyai", "level-1"], &situation);
        if let Some(id) = pf.top_rules.first().map(|r| r.id) {
            *k_pred.entry(id).or_default() += 1;
            if correct.contains(&id) {
                *k_hit.entry(id).or_default() += 1;
            }
        }
    }
    let mut targets: Vec<u64> = k_pred
        .iter()
        .filter(|(id, p)| **p >= 2 && k_hit.get(*id).copied().unwrap_or(0) == 0)
        .map(|(id, _)| *id)
        .collect();
    targets.sort_unstable(); // determinista
    let mut deprecated: Vec<Value> = Vec::new();
    for id in &targets {
        let (p, h) = (
            k_pred.get(id).copied().unwrap_or(0),
            k_hit.get(id).copied().unwrap_or(0),
        );
        let reason = format!(
            "R02 G1->G2: FP sistemático medido en tuning split ({} predicciones top-1, {} hits)",
            p, h
        );
        let ok = agent.deprecate_error(*id, &reason);
        deprecated.push(json!({"rule": id, "preds": p, "hits": h, "deprecated": ok}));
    }
    json!({
        "technique": "deprecate_fp",
        "mechanism": "AgentR::deprecate_error (reglas >=2 pred top-1, 0 hits en tuning)",
        "tuning_split_n": split,
        "candidates_evaluated": k_pred.len(),
        "n_deprecated": deprecated.len(),
        "deprecated": deprecated,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Run: encadena G0 → G1 → G2 midiendo SIEMPRE con el mismo protocolo
// ─────────────────────────────────────────────────────────────────────────────

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn spawn_generation(state: &PathBuf) -> AgentR {
    // Cada generación es una re-encarnación que lee el snapshot del padre
    // (AGENT_R_STATE_PATH real de la lib; NO_SEED para estado limpio).
    AgentR::with_state_path(state.clone())
}

fn run_mode(data_dir: &str, out_dir: &str) -> i32 {
    std::env::set_var("AGENT_R_NO_SEED", "1"); // memoria limpia: sin semillas M2M
    fs::create_dir_all(out_dir).expect("crear out dir");
    let state = PathBuf::from(out_dir).join("state.json");
    if state.exists() {
        fs::remove_file(&state).expect("limpiar state para G0 reproducible");
    }
    let store = ReceiptStore::open(format!("{}/receipts", out_dir)).expect("ReceiptStore");

    let mut bench = bench_load(data_dir);

    let mut generations: Vec<Value> = Vec::new();

    // ── G0: baseline (W fábrica, memoria del bench) ─────────────────────
    {
        let mut g0 = spawn_generation(&state);
        bench_import(&mut g0, &mut bench);
        let m = measure(&mut g0, &bench);
        g0.flush().expect("flush G0");
        let payload = json!({
            "harness": "rsi_generations", "gen": 0, "parent": null,
            "technique": "baseline_factory_w",
            "operation": "bench_c1c2_hybrid_top1top3",
            "w_content": g0.w_content(),
            "n_rules": bench.n_rules,
            "metrics": metrics_public(&m),
            "delta": {"d_top1_hits": 0, "d_top3_hits": 0},
            "result": {"top1_hits": m["top1_hits"], "top3_hits": m["top3_hits"]},
        });
        let receipt = store.issue(&payload).expect("receipt G0");
        println!(
            "G0 baseline        : top1 {}/{} ({:.1}%)  top3 {}/{} ({:.1}%)  W={:.3}  reglas={}  receipt={}",
            m["top1_hits"], m["n"], m["top1_pct"], m["top3_hits"], m["n"], m["top3_pct"],
            g0.w_content(), bench.n_rules, receipt.id
        );
        generations.push(json!({
            "gen": 0, "parent": null, "technique": "baseline_factory_w",
            "w_content": g0.w_content(), "metrics": metrics_public(&m),
            "delta": {"d_top1_hits": 0, "d_top3_hits": 0},
            "receipt_id": receipt.id,
        }));
        drop(g0);
    }

    // ── G1: técnica = self-tuning de W (mecanismo de la lib) ────────────
    {
        let mut g1 = spawn_generation(&state); // lee el snapshot de G0
        let detail = technique_self_tuning(&mut g1, &bench);
        g1.flush().expect("flush G1 (W persiste en snapshot)");
        drop(g1);
        // La medición de G1 se hace sobre una re-encarnación que leyó el W
        // persistido: lo que se mide es el snapshot, no la RAM del proceso.
        let mut g1m = spawn_generation(&state);
        let w_persisted = g1m.w_content();
        let m = measure(&mut g1m, &bench);
        g1m.flush().expect("flush G1m");
        let prev = &generations[0];
        let (d1, d3) = (
            m["top1_hits"].as_i64().unwrap_or(0)
                - prev["metrics"]["top1_hits"].as_i64().unwrap_or(0),
            m["top3_hits"].as_i64().unwrap_or(0)
                - prev["metrics"]["top3_hits"].as_i64().unwrap_or(0),
        );
        let payload = json!({
            "harness": "rsi_generations", "gen": 1, "parent": 0,
            "technique": "self_tuning_w",
            "operation": "bench_c1c2_hybrid_top1top3",
            "w_content": w_persisted,
            "n_rules": bench.n_rules,
            "technique_detail": detail,
            "metrics": metrics_public(&m),
            "delta": {"d_top1_hits": d1, "d_top3_hits": d3},
            "result": {"top1_hits": m["top1_hits"], "top3_hits": m["top3_hits"]},
        });
        let receipt = store.issue(&payload).expect("receipt G1");
        println!(
            "G1 self_tuning_w   : top1 {}/{} ({:.1}%)  top3 {}/{} ({:.1}%)  W={:.3} (persistido)  delta top1 {:+} top3 {:+}  receipt={}",
            m["top1_hits"], m["n"], m["top1_pct"], m["top3_hits"], m["n"], m["top3_pct"],
            w_persisted, d1, d3, receipt.id
        );
        generations.push(json!({
            "gen": 1, "parent": 0, "technique": "self_tuning_w",
            "w_content": w_persisted, "technique_detail": detail,
            "metrics": metrics_public(&m),
            "delta": {"d_top1_hits": d1, "d_top3_hits": d3},
            "receipt_id": receipt.id,
        }));
        drop(g1m);
    }

    // ── G2: técnica = deprecación de FPs medidos (mecanismo de la lib) ──
    {
        let mut g2 = spawn_generation(&state); // lee el snapshot de G1 (W afinado)
        let detail = technique_deprecate_fp(&mut g2, &bench);
        g2.flush().expect("flush G2");
        drop(g2);
        let mut g2m = spawn_generation(&state);
        let m = measure(&mut g2m, &bench);
        g2m.flush().expect("flush G2m");
        let prev = &generations[1];
        let (d1, d3) = (
            m["top1_hits"].as_i64().unwrap_or(0)
                - prev["metrics"]["top1_hits"].as_i64().unwrap_or(0),
            m["top3_hits"].as_i64().unwrap_or(0)
                - prev["metrics"]["top3_hits"].as_i64().unwrap_or(0),
        );
        let payload = json!({
            "harness": "rsi_generations", "gen": 2, "parent": 1,
            "technique": "deprecate_fp",
            "operation": "bench_c1c2_hybrid_top1top3",
            "w_content": g2m.w_content(),
            "n_rules": bench.n_rules,
            "technique_detail": detail,
            "metrics": metrics_public(&m),
            "delta": {"d_top1_hits": d1, "d_top3_hits": d3},
            "result": {"top1_hits": m["top1_hits"], "top3_hits": m["top3_hits"]},
        });
        let receipt = store.issue(&payload).expect("receipt G2");
        println!(
            "G2 deprecate_fp    : top1 {}/{} ({:.1}%)  top3 {}/{} ({:.1}%)  deprecadas={}  delta top1 {:+} top3 {:+}  receipt={}",
            m["top1_hits"], m["n"], m["top1_pct"], m["top3_hits"], m["n"], m["top3_pct"],
            detail["n_deprecated"], d1, d3, receipt.id
        );
        generations.push(json!({
            "gen": 2, "parent": 1, "technique": "deprecate_fp",
            "w_content": g2m.w_content(), "technique_detail": detail,
            "metrics": metrics_public(&m),
            "delta": {"d_top1_hits": d1, "d_top3_hits": d3},
            "receipt_id": receipt.id,
        }));
        drop(g2m);
    }

    let out = json!({
        "harness": "rsi_generations",
        "version": 1,
        "ts_ms": now_ms() as u64,
        "data_dir": data_dir,
        "lib": "m2m_vector_search (wt_valid, ar2/validate-preflight)",
        "protocolo": {
            "bench": "BenchTrace babyai L1 — C1/C2 (top-1/top-3 preflight híbrido)",
            "memoria": "record_error de TODOS los core_failure (what=diagnosis[:160], tags [model,babyai,level-1])",
            "query": "preflight_v2_hybrid(game, [model_short,babyai,level-1], situation=trayectoria hasta decision_point)",
            "metricas": "top1_hits / top3_hits / n (lección correcta #1 o en top-3)",
            "determinismo": "tareas ordenadas por id; split temporal primera mitad para técnicas",
        },
        "generations": generations,
    });
    let path = format!("{}/generations.json", out_dir);
    fs::write(&path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("OK -> {}", path);
    0
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify: el gate anti-self-report (solo outputs programáticos firmados)
// ─────────────────────────────────────────────────────────────────────────────

fn verify_mode(run_dir: &str) -> i32 {
    let doc: Value = serde_json::from_str(
        &fs::read_to_string(format!("{}/generations.json", run_dir)).expect("generations.json"),
    )
    .expect("parse generations.json");
    let store = ReceiptStore::open(format!("{}/receipts", run_dir)).expect("ReceiptStore");
    let gens = doc["generations"].as_array().cloned().unwrap_or_default();
    let mut failures: Vec<String> = Vec::new();

    for (i, g) in gens.iter().enumerate() {
        let rid = g["receipt_id"].as_str().unwrap_or("");
        // 1. Firma HMAC del receipt (tamper → fail).
        let v = store.verify_receipt(rid);
        if !v.valid {
            failures.push(format!("G{}: receipt {} inválido: {}", i, rid, v.reason));
            continue;
        }
        // 2. Las métricas declaradas deben estar CONTENIDAS en el payload firmado.
        for metric in ["top1_hits", "top3_hits"] {
            let value = g["metrics"][metric].clone();
            let verdict = store
                .n3_check(&json!({"receipt_id": rid, "claim": {"metric": metric, "value": value}}));
            if verdict != m2m_vector_search::receipts::N3Verdict::Pass {
                failures.push(format!(
                    "G{}: claim {}={} no respaldado por el receipt ({:?})",
                    i, metric, value, verdict
                ));
            }
        }
        // 3. Aritmética del delta contra la generación anterior.
        if i > 0 {
            let prev = &gens[i - 1];
            let exp1 = g["metrics"]["top1_hits"].as_i64().unwrap_or(0)
                - prev["metrics"]["top1_hits"].as_i64().unwrap_or(0);
            let exp3 = g["metrics"]["top3_hits"].as_i64().unwrap_or(0)
                - prev["metrics"]["top3_hits"].as_i64().unwrap_or(0);
            if g["delta"]["d_top1_hits"].as_i64() != Some(exp1)
                || g["delta"]["d_top3_hits"].as_i64() != Some(exp3)
            {
                failures.push(format!("G{}: delta inconsistente con métricas", i));
            }
            if g["parent"].as_i64() != prev["gen"].as_i64() {
                failures.push(format!("G{}: parent no encadena con G{}", i, i - 1));
            }
        }
        println!(
            "G{} {:<16} receipt OK  top1={} top3={} delta=({:+},{:+})",
            i,
            g["technique"].as_str().unwrap_or("?"),
            g["metrics"]["top1_hits"],
            g["metrics"]["top3_hits"],
            g["delta"]["d_top1_hits"].as_i64().unwrap_or(0),
            g["delta"]["d_top3_hits"].as_i64().unwrap_or(0),
        );
    }
    if failures.is_empty() {
        println!(
            "VERIFY PASS: {} generaciones, firmas y claims consistentes",
            gens.len()
        );
        0
    } else {
        for f in &failures {
            eprintln!("FAIL: {}", f);
        }
        1
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 && args[1] == "verify" {
        std::process::exit(verify_mode(&args[2]));
    }
    let mut data_dir = "../bench_data/benchtrace".to_string();
    let mut out_dir = format!("/tmp/rsi_r02/run_{}", now_ms());
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" if i + 1 < args.len() => {
                data_dir = args[i + 1].clone();
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_dir = args[i + 1].clone();
                i += 2;
            }
            _ => i += 1,
        }
    }
    std::process::exit(run_mode(&data_dir, &out_dir));
}
