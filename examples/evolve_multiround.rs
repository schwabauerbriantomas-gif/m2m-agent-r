//! T3 — Convergencia de evolve multironda (5 rondas de tuning + evolve).
//!
//! Pregunta: ¿5 rondas de evolve+re-tuning convergen a un W estable, oscilan
//! o derivan? ¿El margen anti-ruido (0.5) aguanta?
//!
//! Protocolo por ronda r (r=1..5), sobre el bench C1/C2 real (31 tareas,
//! holdout = segunda mitad, n=16):
//!   (a) tuning_split con label post-hoc (match_outcome_rule + tuning_feedback)
//!       sobre la PRIMERA mitad — igual que rsi_label_ab.rs modo B;
//!   (b) evolve sobre el snapshot con grid 0.5..4.0 y margin 0.5
//!       (evaluator = re-encarnación, prefault_v2_hybrid_w, solo lectura);
//!   (c) registro {w_antes_tuning, w_post_tuning, w_post_evolve, promoted,
//!       incumbent_score, best_score}.
//! Cada ronda parte del snapshot resultante de la anterior (flush +
//! re-encarnación). Métrica final: varianza de W post-evolve en rondas 2-5,
//! nº de promociones, límites [0.5, 6.0].
//!
//!   CARGO_TARGET_DIR=target cargo run --release --example evolve_multiround -- \
//!       --data ../bench_data/benchtrace --out /tmp/t3_evolve_multiround

use m2m_vector_search::agent_r::{AgentR, ErrorCategory, ErrorSeverity};
use m2m_vector_search::meta_evolve::{evolve, EvolveSpace, ReplayScore};
use m2m_vector_search::receipts::ReceiptStore;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

// ── bench C1/C2: funciones idénticas a examples/rsi_evolve.rs ──────────────

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

/// Holdout con W forzado (idéntico a rsi_evolve.rs; no muta el snapshot).
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

// ── tuning con label post-hoc (idéntico a rsi_label_ab.rs modo B) ──────────

fn evidence_of(agent: &AgentR, correct: &HashSet<u64>) -> String {
    let mut parts = Vec::new();
    for e in agent.errors() {
        if correct.contains(&e.id) {
            parts.push(format!("{} {} {}", e.what, e.why, e.prevention));
        }
    }
    parts.join(" ")
}

fn tuning_split_posthoc(agent: &mut AgentR, bench: &Bench, tasks: &[Value]) -> (u64, u64, u64) {
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
    }
    (hits, misses, correct_hits)
}

// ── multironda ──────────────────────────────────────────────────────────────

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
        .unwrap_or_else(|| "/tmp/t3_evolve_multiround".into());
    fs::create_dir_all(&out).unwrap();
    let store = ReceiptStore::open_no_env(std::path::Path::new(&format!("{out}/receipts")))
        .expect("receipt store");

    let (ep_by_id, tasks) = bench_load(&data);
    let split = tasks.len() / 2;
    let state_path = PathBuf::from(format!("{out}/state_multiround.json"));
    let _ = fs::remove_file(&state_path);

    // Semilla: memoria del bench importada UNA vez y flush al snapshot
    // (protocolo idéntico a rsi_evolve.rs).
    let mut bench = Bench {
        ep_by_id: ep_by_id.clone(),
        tasks: Vec::new(),
        lessons_of_rule: HashMap::new(),
    };
    {
        let mut a = AgentR::with_state_path(state_path.clone());
        bench_import(&mut a, &mut bench, &ep_by_id);
        a.flush().expect("flush seed");
    }
    let lessons = bench.lessons_of_rule.clone();

    let space = EvolveSpace::default(); // grid 0.5..4.0
    let margin = 0.5;

    let mut rounds: Vec<Value> = Vec::new();
    let mut n_promotions = 0u64;
    let mut w_post_evolve_series: Vec<f64> = Vec::new();

    for r in 1..=5u32 {
        // La ronda parte del snapshot resultante de la anterior.
        let mut agent = AgentR::with_state_path(state_path.clone());
        let w_antes = agent.tuning_stats().2;

        // (a) tuning con label post-hoc sobre la PRIMERA mitad.
        let bench_r = Bench {
            ep_by_id: ep_by_id.clone(),
            tasks: Vec::new(),
            lessons_of_rule: lessons.clone(),
        };
        let (hits, misses, correct_hits) =
            tuning_split_posthoc(&mut agent, &bench_r, &tasks[..split]);
        agent.flush().expect("flush post-tuning");
        drop(agent);

        // Re-encarnación: el evolve corre sobre el snapshot post-tuning.
        let mut agent = AgentR::with_state_path(state_path.clone());
        let w_post_tuning = agent.tuning_stats().2;

        // (b) evolve: evaluator = re-encarnación del snapshot (solo lectura).
        let report = evolve(&mut agent, &store, &space, |_a, w| {
            let mut probe = AgentR::with_state_path(state_path.clone());
            let bench_probe = Bench {
                ep_by_id: ep_by_id.clone(),
                tasks: Vec::new(),
                lessons_of_rule: lessons.clone(),
            };
            let (n, top1, top3) =
                holdout_measure_w(&mut probe, &bench_probe, &tasks[split..], w);
            ReplayScore { w, score: top1 as f64, tiebreak: top3 as f64, n }
        }, margin);

        let w_post_evolve = agent.tuning_stats().2;
        if report.promoted {
            n_promotions += 1;
        }
        w_post_evolve_series.push(w_post_evolve);

        // Evolve ya hizo flush si promovió; si no, W no cambió pero
        // aseguramos la persistencia del snapshot de la ronda.
        agent.flush().expect("flush post-evolve");

        rounds.push(json!({
            "round": r,
            "w_antes_tuning": w_antes,
            "w_post_tuning": w_post_tuning,
            "w_post_evolve": w_post_evolve,
            "promoted": report.promoted,
            "incumbent_w": report.incumbent_w,
            "incumbent_score": report.incumbent_score,
            "best_w": report.best_w,
            "best_score": report.best_score,
            "margin": report.margin,
            "tuning": {"hits": hits, "misses": misses, "correct_hits": correct_hits},
            "backup_path": report.backup_path,
            "receipts": report.receipts,
            "grid": report.grid,
        }));
        println!(
            "ronda {r}: W {w_antes:.4} -tuning-> {w_post_tuning:.4} -evolve-> {w_post_evolve:.4} (promoted={}, best_w={}, inc_score={}, best_score={})",
            report.promoted, report.best_w, report.incumbent_score, report.best_score
        );
    }

    // ── métricas de convergencia ──────────────────────────────────────────
    // Varianza de W post-evolve entre rondas 2-5 (la 1 no cuenta).
    let tail = &w_post_evolve_series[1..];
    let mean = tail.iter().sum::<f64>() / tail.len() as f64;
    let variance = tail.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / tail.len() as f64;

    let out_of_bounds = w_post_evolve_series
        .iter()
        .filter(|w| **w < 0.5 - 1e-9 || **w > 6.0 + 1e-9)
        .count();

    // Clasificación: convergió / oscila / deriva monotónica.
    let monotonic_up = tail.windows(2).all(|p| p[1] > p[0] + 1e-9);
    let monotonic_down = tail.windows(2).all(|p| p[1] < p[0] - 1e-9);
    let monotonic_drift = monotonic_up || monotonic_down;
    let verdict = if variance < 0.01 {
        "convergio"
    } else if monotonic_drift {
        "deriva_monotonica"
    } else {
        "oscila"
    };

    let report = json!({
        "experimento": "t3_evolve_multiround_convergencia",
        "fecha": "2026-10-16",
        "bench_n": tasks.len(),
        "tuning_n": split,
        "holdout_n": tasks.len() - split,
        "margin": margin,
        "grid": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "w_bounds": [0.5, 6.0],
        "rounds": rounds,
        "convergencia": {
            "w_post_evolve_series": w_post_evolve_series,
            "variance_w_rounds_2_5": variance,
            "mean_w_rounds_2_5": mean,
            "n_promotions": n_promotions,
            "out_of_bounds": out_of_bounds,
            "verdict": verdict,
            "monotonic_drift": monotonic_drift,
            "monotonic_up": monotonic_up,
        },
    });
    let path = "/tmp/t3_evolve_multiround.json";
    fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    println!("variance W (rondas 2-5): {variance:.6}");
    println!("promociones: {n_promotions}");
    println!("fuera de límites [0.5, 6.0]: {out_of_bounds}");
    println!("veredicto: {verdict}");
    println!("guardado: {path}");
}
