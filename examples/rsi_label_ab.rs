//! RSI-Capa 1 — Validación NUMÉRICA del label post-hoc (R03).
//!
//! A/B sobre el MISMO bench (31 tareas, protocolo C1/C2 del harness R02,
//! medición por re-encarnación del snapshot):
//!
//!   MODO A (viejo): fail_task ⇒ hit por identidad (Some(pred),Some(pred)).
//!   MODO B (nuevo): fail_task ⇒ hit solo si match_outcome_rule confirma
//!                   con la evidencia (texto del diagnóstico de la lección
//!                   correcta = emula el stderr real del fallo).
//!
//! Métricas por modo (todas programáticas, cero self-report):
//!   - label_precision : hits correctos / hits acreditados por el modo.
//!   - w_final         : W del híbrido tras el split de tuning.
//!   - holdout top1/top3: segunda mitad tras el tuning, snapshot re-encarnado.
//!
//! Uso (desde wt_valid/):
//!   CARGO_TARGET_DIR=target cargo run --release --example rsi_label_ab -- \
//!       --data ../bench_data/benchtrace --out /tmp/rsi_capa1

use m2m_vector_search::agent_r::{AgentR, ErrorCategory, ErrorSeverity};
use m2m_vector_search::receipts::ReceiptStore;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

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

fn situation_of(ep: &Value, dp: i64) -> String {
    let mut s = String::new();
    if let Some(steps) = ep["snapshot"].get("trajectory").and_then(|t| t.as_array()) {
        let upto = if dp > 0 { (dp as usize).min(steps.len()) } else { steps.len() };
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
    ep_by_id: HashMap<String, Value>,
    tasks: Vec<Value>,
    /// rule_id del agente → episode_ids que la originaron (GT por tarea).
    lessons_of_rule: HashMap<u64, HashSet<String>>,
}

fn bench_load(data_dir: &str) -> (HashMap<String, Value>, Vec<Value>) {
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

    let rule_eps: HashSet<String> = episodes
        .iter()
        .filter(|ep| {
            ep["failure_instances"]["core_failure"]
                .as_array()
                .map(|cfs| {
                    cfs.iter()
                        .any(|cf| !cf.get("diagnosis").and_then(|d| d.as_str()).unwrap_or("").is_empty())
                })
                .unwrap_or(false)
        })
        .map(|ep| ep["id"].as_str().unwrap_or("").to_string())
        .collect();

    let ep_by_id: HashMap<String, Value> = episodes
        .iter()
        .filter_map(|ep| ep["id"].as_str().map(|id| (id.to_string(), ep.clone())))
        .collect();

    let mut tasks: Vec<Value> = all_tasks
        .into_iter()
        .filter(|t| {
            let eps = t["evolution_snapshots"].as_array().cloned().unwrap_or_default();
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
    (ep_by_id, tasks)
}

fn bench_import(agent: &mut AgentR, bench: &mut Bench, ep_by_id: &HashMap<String, Value>) {
    let mut ep_lessons: Vec<(String, String, String, ErrorCategory, String)> = Vec::new();
    for (ep_id, ep) in ep_by_id.iter() {
        let Some(cfs) = ep["failure_instances"]["core_failure"].as_array() else { continue };
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
    ep_lessons.sort_by(|a, b| format!("{}|{}|{}", a.1, a.2, a.4).cmp(&format!("{}|{}|{}", b.1, b.2, b.4)));
    for (ep_id, game, model_short, cat, diag) in &ep_lessons {
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
        bench.lessons_of_rule.entry(id).or_default().insert(ep_id.clone());
    }
}

fn correct_rules_for(bench: &Bench, task: &Value) -> HashSet<u64> {
    let lesson_eps: HashSet<&str> = task["evolution_snapshots"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    bench
        .lessons_of_rule
        .iter()
        .filter(|(_, eps)| eps.iter().any(|e| lesson_eps.contains(e.as_str())))
        .map(|(id, _)| *id)
        .collect()
}

/// Diagnóstico completo de las reglas correctas (emula la evidencia/stderr
/// que un runner real capturaría del fallo).
fn evidence_of(agent: &AgentR, correct: &HashSet<u64>) -> String {
    let mut parts = Vec::new();
    for e in agent.errors() {
        if correct.contains(&e.id) {
            parts.push(format!("{} {} {}", e.what, e.why, e.prevention));
        }
    }
    parts.join(" ")
}

fn tuning_split(
    agent: &mut AgentR,
    bench: &Bench,
    tasks: &[Value],
    posthoc: bool,
) -> (u64, u64, u64) {
    let mut hits = 0u64;
    let mut misses = 0u64;
    let mut correct_hits = 0u64;
    for task in tasks {
        let ts = &task["test_snapshot"];
        let test_id = ts["episode_id"].as_str().unwrap_or("");
        let dp = ts["decision_point"].as_i64().unwrap_or(-1);
        let Some(ep) = bench.ep_by_id.get(test_id) else { continue };
        let correct = correct_rules_for(bench, task);
        if correct.is_empty() {
            continue;
        }
        let game = ep["game"].as_str().unwrap_or("level-1");
        let model = ep["model"].as_str().unwrap_or("unknown");
        let model_short = model.rsplit('/').next().unwrap_or(model);
        let situation = situation_of(ep, dp);
        let pf = agent.preflight_v2_hybrid(game, &[model_short, "babyai", "level-1"], &situation);
        let Some(pred_lib) = pf.top_rules.first().map(|r| r.id) else { continue };
        let pred_correct = correct.contains(&pred_lib);

        if posthoc {
            // MODO B: la evidencia del fallo decide. evidence_of usa los
            // diagnósticos de las reglas CORRECTAS del GT (= stderr real).
            let ev = evidence_of(agent, &correct);
            let confirmed = agent.match_outcome_rule("level-1", &ev);
            match confirmed {
                Some(actual) if actual == pred_lib => {
                    agent.tuning_feedback(Some(pred_lib), Some(pred_lib));
                    hits += 1;
                    if pred_correct {
                        correct_hits += 1;
                    }
                }
                _ => {
                    agent.tuning_feedback(Some(pred_lib), None);
                    misses += 1;
                }
            }
        } else {
            // MODO A: label por identidad (semántica vieja de fail_task)
            agent.tuning_feedback(Some(pred_lib), Some(pred_lib));
            hits += 1;
            if pred_correct {
                correct_hits += 1;
            }
        }
    }
    (hits, misses, correct_hits)
}

fn holdout_measure(agent: &mut AgentR, bench: &Bench, tasks: &[Value]) -> (u64, u64, u64) {
    let (mut top1, mut top3, mut n) = (0u64, 0u64, 0u64);
    for task in tasks {
        let ts = &task["test_snapshot"];
        let test_id = ts["episode_id"].as_str().unwrap_or("");
        let dp = ts["decision_point"].as_i64().unwrap_or(-1);
        let Some(ep) = bench.ep_by_id.get(test_id) else { continue };
        let correct = correct_rules_for(bench, task);
        if correct.is_empty() {
            continue;
        }
        n += 1;
        let game = ep["game"].as_str().unwrap_or("level-1");
        let model = ep["model"].as_str().unwrap_or("unknown");
        let model_short = model.rsplit('/').next().unwrap_or(model);
        let situation = situation_of(ep, dp);
        let pf = agent.preflight_v2_hybrid(game, &[model_short, "babyai", "level-1"], &situation);
        let top: Vec<u64> = pf.top_rules.iter().map(|r| r.id).collect();
        top1 += top.first().map(|id| correct.contains(id)).unwrap_or(false) as u64;
        top3 += top.iter().any(|id| correct.contains(id)) as u64;
    }
    (n, top1, top3)
}

fn run_mode(
    ep_by_id: &HashMap<String, Value>,
    tasks: &[Value],
    mode: &str,
    posthoc: bool,
    out_dir: &str,
) -> Value {
    let mut bench = Bench {
        ep_by_id: ep_by_id.clone(),
        tasks: Vec::new(),
        lessons_of_rule: HashMap::new(),
    };
    // Re-encarnación (protocolo R02): el agente NACE con state path (el ctor
    // carga lo que haya, vacío tras rm), se tunea, flush persiste, y la
    // medición corre sobre un agente NUEVO que carga ese snapshot de disco.
    std::env::set_var("AGENT_R_NO_SEED", "1");
    let state_path = PathBuf::from(format!("{out_dir}/state_{mode}.json"));
    let _ = fs::remove_file(&state_path);
    let mut agent = AgentR::with_state_path(state_path.clone());
    bench_import(&mut agent, &mut bench, ep_by_id);
    bench.tasks = tasks.to_vec();

    let split = tasks.len() / 2;
    let (hits, misses, correct_hits) = tuning_split(&mut agent, &bench, &tasks[..split], posthoc);
    agent.flush().expect("flush");
    drop(agent);
    let mut incarnate = AgentR::with_state_path(state_path);

    let (n_eval, top1, top3) = holdout_measure(&mut incarnate, &bench, &tasks[split..]);
    let (th, tm, w) = incarnate.tuning_stats();

    json!({
        "mode": mode, "posthoc": posthoc,
        "tuning_n": split, "hits": hits, "misses": misses,
        "correct_hits": correct_hits,
        "label_precision": if hits > 0 { (correct_hits as f64 / hits as f64 * 1000.0).round() / 10.0 } else { 0.0 },
        "w_final": w, "tuning_lib": {"hits": th, "misses": tm},
        "holdout_n": n_eval, "holdout_top1_hits": top1, "holdout_top3_hits": top3,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = args
        .iter()
        .position(|a| a == "--data")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "../bench_data/benchtrace".into());
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "/tmp/rsi_capa1".into());
    fs::create_dir_all(&out).unwrap();
    let store = ReceiptStore::open_no_env(std::path::Path::new(&format!("{out}/receipts")))
        .expect("receipt store");

    let (ep_by_id, tasks) = bench_load(&data);

    // A/B sobre el mismo bench y el mismo split: solo cambia el LABEL.
    let a = run_mode(&ep_by_id, &tasks, "A_identity", false, &out);
    let b = run_mode(&ep_by_id, &tasks, "B_posthoc", true, &out);

    let delta = json!({
        "label_precision_pts": b["label_precision"].as_f64().unwrap_or(0.0)
            - a["label_precision"].as_f64().unwrap_or(0.0),
        "holdout_top1_hits": b["holdout_top1_hits"].as_i64().unwrap_or(0)
            - a["holdout_top1_hits"].as_i64().unwrap_or(0),
        "holdout_top3_hits": b["holdout_top3_hits"].as_i64().unwrap_or(0)
            - a["holdout_top3_hits"].as_i64().unwrap_or(0),
    });

    let report = json!({
        "experimento": "capa1_label_posthoc_vs_identity",
        "fecha": "2026-10-15",
        "bench_n": tasks.len(),
        "protocolo": "mismo bench, mismo split, misma memoria semilla; solo cambia el label del tuning (identity vs post-hoc)",
        "A": a, "B": b, "delta": delta,
    });
    // Receipt HMAC del reporte completo (integridad auditable)
    let rid = store.issue(&report).map(|r| r.id).ok();
    let mut report = report;
    report["receipt_id"] = json!(rid);
    let path = format!("{out}/label_ab.json");
    fs::write(&path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    println!("guardado: {path}");
}
