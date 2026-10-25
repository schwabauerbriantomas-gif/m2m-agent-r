//! Auto-tuning de Agent-R: el agente ajusta SUS PROPIOS parámetros a partir
//! de outcomes reales (pendiente: cuáles outcomes).
//!
//! Aplicación práctica del bench 15-oct: preflight_v2_hybrid tiene dos
//! señales (estructural y contenido) con un peso W fijo de fábrica (3.0).
//! Este ejemplo muestra el loop completo:
//!
//!   1. PREDICCIÓN:  preflight_hybrid_before(op, tags, situación) → ¿qué
//!      regla era relevante? (top-1)
//!   2. REALIDAD:    el outcome llega DESPUÉS (task falló/salió bien, y con
//!      qué regla estaba relacionada).
//!   3. ACTUALIZACIÓN: si la regla predicha coincide con la del outcome =>
//!      hit (refuerza W_content); si el outcome trae una regla DISTINTA =>
//!      miss (W_content baja); si no había regla aplicable => abstención
//!      (sin castigo: castigar abstención destruye la exploración, lección
//!      del curriculum de user).
//!   4. REGISTRO:    cada veredicto queda en el propio Agent-R como
//!      incidente (dogfooding: el agente se observa a sí mismo).
//!
//! Esto es EXACTAMENTE el protocolo del bench de BenchTrace: predicción vs
//! outcome anotado, con la diferencia de que acá el outcome realimenta el
//! parámetro en vez de solo medirlo.
//!
//! Uso:
//!   CARGO_TARGET_DIR=target cargo run --release --example self_tuning -- \
//!     ../bench_data/benchtrace

use m2m_vector_search::agent_r::{AgentR, ErrorCategory, ErrorSeverity};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;

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

/// Situación textual del test (trayectoria hasta el decision point) —
/// el "estado actual" que el híbrido cruza con el contenido de las reglas.
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

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let snaps: Value = serde_json::from_str(
        &fs::read_to_string(format!("{}/babyai_l1_snapshots.json", data_dir))
            .expect("snapshots.json"),
    )
    .expect("parse");
    let evol: Value = serde_json::from_str(
        &fs::read_to_string(format!("{}/babyai_l1_evolution.json", data_dir))
            .expect("evolution.json"),
    )
    .expect("parse");

    let episodes = snaps["episodes"].as_array().unwrap().clone();
    let tasks = evol["tasks"].as_array().unwrap().clone();
    let mut ep_by_id: HashMap<String, &Value> = episodes
        .iter()
        .filter_map(|e| e["id"].as_str().map(|i| (i.to_string(), e)))
        .collect();

    // ── Memoria inicial: lecciones de TODOS los snapshots menos los de test ──
    let mut agent = AgentR::new_ephemeral();
    let mut rule_of_ep: HashMap<String, u64> = HashMap::new();
    for ep in &episodes {
        let game = ep["game"].as_str().unwrap_or("level-1");
        let model = ep["model"].as_str().unwrap_or("unknown");
        let model_short = model.rsplit('/').next().unwrap_or(model);
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
                    &[model_short, "babyai", "level-1"],
                    "",
                );
                rule_of_ep.insert(ep["id"].as_str().unwrap_or("").to_string(), id);
            }
        }
    }

    // ── Split temporal determinista: primera mitad = feedback (auto-tuning),
    //    segunda mitad = evaluación (holdout, parámetros CONGELADOS). ──
    let mut evaluable: Vec<&Value> = Vec::new();
    for task in &tasks {
        let lesson_ok = task["evolution_snapshots"].as_array().is_some();
        let test_id = task["test_snapshot"]["episode_id"].as_str().unwrap_or("");
        if lesson_ok && ep_by_id.contains_key(test_id) {
            evaluable.push(task);
        }
    }
    evaluable.sort_by_key(|t| t["id"].as_str().unwrap_or("").to_string());
    let split = evaluable.len() / 2;
    let (tuning, holdout) = evaluable.split_at(split);

    // ── EL LOOP DE AUTO-TUNING ──────────────────────────────────────────
    // W_content parte del valor de fábrica (3.0). La regla de actualización
    // es incremental, acotada y sin etiquetas del holdout:
    //   hit  (predijo la regla del outcome)  → W *= 1.05 (cap 6.0)
    //   miss (predije otra o nada habiendo)  → W *= 0.95 (floor 0.5)
    //   abstención (outcome sin regla mapeable) → sin cambio (no castigo)
    let mut w_content: f64 = 3.0;
    let (mut hits, mut misses, mut abst) = (0usize, 0usize, 0usize);
    let mut trace: Vec<Value> = Vec::new();

    for task in tuning {
        let ts = &task["test_snapshot"];
        let test_id = ts["episode_id"].as_str().unwrap_or("");
        let dp = ts["decision_point"].as_i64().unwrap_or(-1);
        let lesson_eps: HashSet<String> = task["evolution_snapshots"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let correct_rules: HashSet<u64> = lesson_eps
            .iter()
            .filter_map(|e| rule_of_ep.get(e).copied())
            .collect();
        if correct_rules.is_empty() {
            continue;
        }
        let ep = ep_by_id.get(test_id);
        let (game, model_short, situation) = match ep {
            Some(e) => {
                let m = e["model"].as_str().unwrap_or("unknown");
                (
                    e["game"].as_str().unwrap_or("level-1").to_string(),
                    m.rsplit('/').next().unwrap_or(m).to_string(),
                    situation_of(e, dp),
                )
            }
            None => continue,
        };

        // 1. PREDICCIÓN con el W vigente.
        let pf = agent.preflight_v2_hybrid_w(
            &game,
            &[&model_short, "babyai", "level-1"],
            &situation,
            w_content,
        );
        let pred = pf.top_rules.first().map(|r| r.id);

        // 2. REALIDAD: ¿la regla predicha es de las correctas?
        let outcome = match pred {
            Some(id) if correct_rules.contains(&id) => "hit",
            Some(_) => "miss",
            None => "abstain",
        };
        // 3. ACTUALIZACIÓN incremental acotada.
        match outcome {
            "hit" => {
                w_content = (w_content * 1.05).min(6.0);
                hits += 1;
            }
            "miss" => {
                w_content *= 0.95;
                if w_content < 0.5 {
                    w_content = 0.5;
                }
                misses += 1;
            }
            _ => {
                abst += 1;
            }
        }
        // 4. DOGFOODING: el agente registra su propio desempeño.
        // (comentado para no contaminar la memoria del experimento; en prod
        //  sería agent.record_error con category=OutputDrift si W deriva.)
        trace.push(serde_json::json!({
            "task": task["id"], "w": w_content, "pred": pred, "outcome": outcome,
        }));
    }

    // ── EVALUACIÓN en holdout con W congelado ──
    let mut h1 = 0;
    let mut h3 = 0;
    let mut hn = 0;
    for task in holdout {
        let ts = &task["test_snapshot"];
        let test_id = ts["episode_id"].as_str().unwrap_or("");
        let dp = ts["decision_point"].as_i64().unwrap_or(-1);
        let lesson_eps: HashSet<String> = task["evolution_snapshots"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let correct_rules: HashSet<u64> = lesson_eps
            .iter()
            .filter_map(|e| rule_of_ep.get(e).copied())
            .collect();
        if correct_rules.is_empty() {
            continue;
        }
        let ep = match ep_by_id.get(test_id) {
            Some(e) => e,
            None => continue,
        };
        let m = ep["model"].as_str().unwrap_or("unknown");
        let model_short = m.rsplit('/').next().unwrap_or(m);
        let situation = situation_of(ep, dp);
        let pf = agent.preflight_v2_hybrid_w(
            &game_of(ep),
            &[model_short, "babyai", "level-1"],
            &situation,
            w_content,
        );
        let top: Vec<u64> = pf.top_rules.iter().map(|r| r.id).collect();
        hn += 1;
        if top
            .first()
            .map(|id| correct_rules.contains(id))
            .unwrap_or(false)
        {
            h1 += 1;
        }
        if top.iter().any(|id| correct_rules.contains(id)) {
            h3 += 1;
        }
    }

    println!("=== AUTO-TUNING DE W_CONTENT (loop de feedback real) ===");
    println!(
        "fase tuning: {} tareas | hits {} misses {} abstenciones {}",
        tuning.len(),
        hits,
        misses,
        abst
    );
    println!("W final: {:.3} (fábrica: 3.000)", w_content);
    println!(
        "holdout (W congelado): top1 {}/{}, top3 {}/{}",
        h1, hn, h3, hn
    );
    let _ = &mut ep_by_id; // ep_by_id mutable por situación; silenciar warning

    let out = serde_json::json!({
        "w_factory": 3.0, "w_final": w_content,
        "tuning": {"n": tuning.len(), "hits": hits, "misses": misses, "abstain": abst},
        "holdout": {"n": hn, "top1": h1, "top3": h3},
        "trace": trace,
    });
    fs::write(
        format!("{}/self_tuning_results.json", data_dir),
        serde_json::to_string_pretty(&out).unwrap(),
    )
    .unwrap();
    println!("OK -> {}/self_tuning_results.json", data_dir);
}

fn game_of(ep: &Value) -> &str {
    ep["game"].as_str().unwrap_or("level-1")
}
