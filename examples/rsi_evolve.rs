//! RSI Capa 3 REAL — evolve del W_content sobre el bench C1/C2.
//!
//! Grid determinista de W (0.5..4.0), replay como evaluator: por cada W se
//! RE-ENCARNA el snapshot incumbente y se mide el holdout (top1 score,
//! top3 tiebreak) con `preflight_v2_hybrid_w` (el probe LEE el snapshot,
//! nunca lo muta durante el grid). Promoción solo con margen >= 0.5 hits.
//!
//!   CARGO_TARGET_DIR=target cargo run --release --example rsi_evolve -- \
//!       --data ../bench_data/benchtrace --out /tmp/rsi_capa3

use m2m_vector_search::agent_r::{AgentR, ErrorCategory, ErrorSeverity};
use m2m_vector_search::meta_evolve::{evolve, EvolveSpace, ReplayScore};
use m2m_vector_search::receipts::ReceiptStore;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

// ── bench C1/C2 (protocolo idéntico al harness R02 / rsi_generations) ──────

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

/// Holdout con W FORZADO por parámetro (no muta el snapshot del probe).
fn holdout_measure_w(agent: &mut AgentR, bench: &Bench, tasks: &[Value], w: f64) -> (u64, u64, u64) {
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
        let pf = agent.preflight_v2_hybrid_w(game, &[model_short, "babyai", "level-1"], &situation, w);
        let top: Vec<u64> = pf.top_rules.iter().map(|r| r.id).collect();
        top1 += top.first().map(|id| correct.contains(id)).unwrap_or(false) as u64;
        top3 += top.iter().any(|id| correct.contains(id)) as u64;
    }
    (n, top1, top3)
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
        .unwrap_or_else(|| "/tmp/rsi_capa3".into());
    fs::create_dir_all(&out).unwrap();
    let store = ReceiptStore::open_no_env(std::path::Path::new(&format!("{out}/receipts")))
        .expect("receipt store");

    let (ep_by_id, tasks) = bench_load(&data);
    let state_path = PathBuf::from(format!("{out}/state_evolve.json"));
    let _ = fs::remove_file(&state_path);

    // Incumbent: memoria del bench importada UNA vez y flush al snapshot.
    let mut bench = Bench {
        ep_by_id: ep_by_id.clone(),
        tasks: Vec::new(),
        lessons_of_rule: HashMap::new(),
    };
    {
        let mut a = AgentR::with_state_path(state_path.clone());
        bench_import(&mut a, &mut bench, &ep_by_id);
        a.flush().expect("flush incumbent");
    }
    let lessons = bench.lessons_of_rule.clone();
    let split = tasks.len() / 2;

    // evolve: por cada W del grid re-encarna el snapshot (solo lectura) y
    // mide el holdout con ese W. Evaluator determinista y programático.
    // (bench_import del base es redundante con el snapshot, pero replica el
    // protocolo R02: lessons_of_rule del GT vive en el closure vía `lessons`.)
    let mut base = AgentR::with_state_path(state_path.clone());
    let space = EvolveSpace::default();
    let report = evolve(&mut base, &store, &space, |_agent, w| {
        let mut probe = AgentR::with_state_path(state_path.clone());
        let bench_probe = Bench {
            ep_by_id: ep_by_id.clone(),
            tasks: Vec::new(),
            lessons_of_rule: lessons.clone(),
        };
        let (n, top1, top3) =
            holdout_measure_w(&mut probe, &bench_probe, &tasks[split..], w);
        ReplayScore { w, score: top1 as f64, tiebreak: top3 as f64, n }
    }, 0.5);

    let mut report_json = serde_json::to_value(&report).unwrap_or_default();
    report_json["holdout_n"] = json!(tasks.len() - split);
    report_json["bench_n"] = json!(tasks.len());
    report_json["experimento"] = json!("capa3_evolve_w_content");
    report_json["nota"] = json!("grid determinista; evaluator = re-encarnación del snapshot incumbente con W forzado (preflight_v2_hybrid_w, solo lectura); score = top1 hits del holdout; margen 0.5");
    let path = format!("{out}/evolve.json");
    fs::write(&path, serde_json::to_string_pretty(&report_json).unwrap()).unwrap();
    println!("{}", serde_json::to_string_pretty(&report_json).unwrap());
    println!("guardado: {path}");
}
