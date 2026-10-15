//! Export de scores estructurales del preflight v2 para composición offline.
//!
//! Por cada tarea de evolution_evaluation.json emite: id de tarea, episodios
//! de evolución (lecciones correctas), y el score compuesto v2 (lib real,
//! cero replicación via preflight_v2_scored_all) de TODAS las reglas.
//! El compositor python (bench_data/benchtrace/compose_v2.py) fusiona esta
//! señal con la de contenido y mide complementariedad + ablations.
//!
//! Uso:
//!   CARGO_TARGET_DIR=target cargo run --release --example far_export_scores -- \
//!     ../bench_data/benchtrace

use m2m_vector_search::agent_r::{AgentR, ErrorCategory, ErrorSeverity};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;

fn map_category(ftype: &str) -> ErrorCategory {
    let t = ftype.to_lowercase();
    if t.contains("strategy") || t.contains("lateral") || t.contains("goal")
        || t.contains("planning") || t.contains("spatial")
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

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let snaps: Value = serde_json::from_str(
        &fs::read_to_string(format!("{}/babyai_l1_snapshots.json", data_dir))
            .expect("snapshots.json"),
    )
    .expect("parse snapshots");
    let evol: Value = serde_json::from_str(
        &fs::read_to_string(format!("{}/babyai_l1_evolution.json", data_dir))
            .expect("evolution.json"),
    )
    .expect("parse evolution");
    let episodes = snaps["episodes"].as_array().expect("episodes[]").clone();
    let tasks = evol["tasks"].as_array().expect("tasks[]").clone();

    // Memoria: todas las lecciones (idéntico a far_benchtrace.rs)
    let mut agent = AgentR::new_ephemeral();
    let mut ep_of_rule: HashMap<u64, Vec<String>> = HashMap::new();
    for ep in &episodes {
        let game = ep["game"].as_str().unwrap_or("level-1");
        let model = ep["model"].as_str().unwrap_or("unknown");
        let model_short = model.rsplit('/').next().unwrap_or(model).to_string();
        let ep_id = ep["id"].as_str().unwrap_or("").to_string();
        if let Some(cfs) = ep["failure_instances"]["core_failure"].as_array() {
            for cf in cfs {
                let ftype = cf["type"].as_str().unwrap_or("unknown");
                let diag = cf["diagnosis"].as_str().unwrap_or("");
                if diag.is_empty() {
                    continue;
                }
                let what: String = diag.chars().take(160).collect();
                let id = agent.record_error(
                    map_category(ftype),
                    ErrorSeverity::Error,
                    game,
                    &what,
                    diag,
                    &[&model_short, "babyai", "level-1"],
                    "",
                );
                ep_of_rule.entry(id).or_default().push(ep_id.clone());
            }
        }
    }

    // Modelo y game por episodio de test
    let mut model_of_ep: HashMap<String, String> = HashMap::new();
    let mut game_of_ep: HashMap<String, String> = HashMap::new();
    let mut ep_by_id_full: HashMap<String, Value> = HashMap::new();
    for ep in &episodes {
        let m = ep["model"].as_str().unwrap_or("unknown");
        model_of_ep.insert(
            ep["id"].as_str().unwrap_or("").to_string(),
            m.rsplit('/').next().unwrap_or(m).to_string(),
        );
        game_of_ep.insert(
            ep["id"].as_str().unwrap_or("").to_string(),
            ep["game"].as_str().unwrap_or("level-1").to_string(),
        );
        ep_by_id_full.insert(ep["id"].as_str().unwrap_or("").to_string(), ep.clone());
    }

    let mut out_tasks: Vec<Value> = Vec::new();
    for task in &tasks {
        let lesson_eps: Vec<String> = task["evolution_snapshots"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let ts = &task["test_snapshot"];
        let test_id = ts["episode_id"].as_str().unwrap_or("").to_string();
        let game = game_of_ep.get(&test_id).cloned().unwrap_or_else(|| "level-1".into());
        let model = model_of_ep.get(&test_id).cloned().unwrap_or_else(|| "unknown".into());

        let all = agent.preflight_v2_scored_all(&game, &[&model, "babyai", "level-1"]);
        // Híbrido: mismo re-ranking que producción (situación = trayectoria
        // hasta el decision_point, binario IDF, W=3) sobre el top-3 v2.
        let situation = {
            let ep = ep_by_id_full.get(&test_id);
            let dp = ts["decision_point"].as_i64().unwrap_or(-1);
            let mut s = String::new();
            if let Some(ep) = ep {
                let traj = ep["snapshot"].get("trajectory").and_then(|t| t.as_array());
                if let Some(steps) = traj {
                    let upto = if dp > 0 { (dp as usize).min(steps.len()) } else { steps.len() };
                    for st in &steps[..upto] {
                        s.push_str(st.get("obs").and_then(|o| o.as_str()).unwrap_or(""));
                        s.push(' ');
                        s.push_str(&st.get("act").or_else(|| st.get("action"))
                            .map(|a| a.to_string()).unwrap_or_default());
                        s.push(' ');
                    }
                }
            }
            s
        };
        let hyb = agent.preflight_v2_hybrid(&game, &[&model, "babyai", "level-1"], &situation);
        let hybrid_top: Vec<Value> = hyb
            .top_rules
            .iter()
            .map(|r| serde_json::json!({"id": r.id, "score": r.score}))
            .collect();
        let scored: Vec<Value> = all
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "eps": ep_of_rule.get(&r.id).cloned().unwrap_or_default(),
                    "score": r.score,
                    "what": r.what,
                })
            })
            .collect();
        out_tasks.push(serde_json::json!({
            "task_id": task["id"],
            "test_id": test_id,
            "decision_point": ts["decision_point"],
            "lesson_eps": lesson_eps,
            "scored_rules": scored,
            "hybrid_top": hybrid_top,
        }));
    }

    let out = serde_json::json!({ "n_rules": ep_of_rule.len(), "tasks": out_tasks });
    let path = format!("{}/structural_scores_l1.json", data_dir);
    fs::write(&path, serde_json::to_string(&out).unwrap()).unwrap();
    println!("OK -> {} ({} reglas, {} tareas)", path, ep_of_rule.len(), out_tasks.len());
}
