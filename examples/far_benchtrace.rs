//! FAR offline de Agent-R contra BenchTrace (Evolution Eval, babyai level-1).
//!
//! Protocolo (sin entorno vivo, sin API de pago — decisión de user):
//! 1. Importar TODOS los core_failure de los snapshots como memoria acumulada
//!    de Agent-R (~45 reglas; dedup por fingerprint).
//! 2. Para cada tarea de evolution_evaluation.json: sus `evolution_snapshots`
//!    definen las lecciones "correctas" (episodios que el agente debió
//!    aprender); el `test_snapshot` señala el episodio+decision_point donde
//!    la lección debía aplicarse. Con TODA la memoria cargada (distractores
//!    incluidos), medimos si una lección CORRECTA domina el preflight v2.
//! 3. Métricas (especificidad, no solo recall):
//!    - top1_precision: la regla #1 del preflight es lección correcta.
//!    - top3_recall: alguna lección correcta está en el top-3.
//!    - op_matters: control con operación NO vista — el score con la op
//!      correcta debe superar al control (valida exact-match anti-substring).
//! 4. Contexto del query = (game, [modelo, babyai, level-1]) — el dataset no
//!    expone la operación por paso: es el caso débil (todas las tareas
//!    comparten op). Documentado como límite del proxy.
//!
//! NOTA honesta: esto mide selección+recuperación de la lección correcta en
//! el punto de decisión, NO evitación real en entorno vivo (eso exigiría
//! correr BabyAI). Es el proxy offline del FAR del paper.
//!
//! Uso:
//!   CARGO_TARGET_DIR=target cargo run --release --example far_benchtrace -- \
//!     bench_data/benchtrace

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
        ErrorCategory::Timeout // acción no efectiva ~ execution
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
    println!(
        "snapshots: {}, tareas evolución: {}",
        episodes.len(),
        tasks.len()
    );

    // ── 1. Memoria acumulada: TODAS las lecciones + mapa regla→episodios ──
    let mut agent = AgentR::new_ephemeral();
    let mut lessons_of_rule: HashMap<u64, HashSet<String>> = HashMap::new();
    let mut type_of_ep: HashMap<String, String> = HashMap::new();
    let mut model_of_ep: HashMap<String, String> = HashMap::new();
    for ep in &episodes {
        let game = ep["game"].as_str().unwrap_or("level-1");
        let model = ep["model"].as_str().unwrap_or("unknown");
        let model_short = model.rsplit('/').next().unwrap_or(model).to_string();
        let ep_id = ep["id"].as_str().unwrap_or("").to_string();
        model_of_ep.insert(ep_id.clone(), model_short.clone());
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
                    "", // prevention: el dataset no la trae
                );
                lessons_of_rule.entry(id).or_default().insert(ep_id.clone());
                type_of_ep.insert(ep_id.clone(), ftype.to_string());
            }
        }
    }
    let n_rules = agent.stats().unique_incidents;
    println!("reglas en Agent-R (dedup por fingerprint): {}", n_rules);

    let ep_by_id: HashMap<&str, &Value> = episodes
        .iter()
        .filter_map(|e| e["id"].as_str().map(|i| (i, e)))
        .collect();

    // ── 2. Protocolo de evolución con distractores ─────────────────────
    let mut results: Vec<Value> = Vec::new();
    for task in &tasks {
        let task_id = task["id"].as_str().unwrap_or("?").to_string();
        let lesson_eps: Vec<String> = task["evolution_snapshots"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let in_file: HashSet<&String> = lesson_eps
            .iter()
            .filter(|s| type_of_ep.contains_key(*s))
            .collect();
        // Reglas correctas de esta tarea: toda regla originada en ALGUNO de
        // sus episodios de evolución (presentes en el archivo).
        let correct_rules: HashSet<u64> = lessons_of_rule
            .iter()
            .filter(|(_, eps)| eps.iter().any(|e| in_file.contains(e)))
            .map(|(id, _)| *id)
            .collect();

        let ts = &task["test_snapshot"];
        let test_id = ts["episode_id"].as_str().unwrap_or("");
        let dp = ts["decision_point"].as_i64().unwrap_or(-1);
        let (game, test_model) = match ep_by_id.get(test_id) {
            Some(e) => (
                e["game"].as_str().unwrap_or("level-1").to_string(),
                e["model"]
                    .as_str()
                    .unwrap_or("unknown")
                    .rsplit('/')
                    .next()
                    .unwrap_or("unknown")
                    .to_string(),
            ),
            None => ("level-1".to_string(), "unknown".to_string()),
        };
        // ¿El modelo del test coincide con el de las lecciones? (transfer)
        let lesson_models: HashSet<String> = in_file
            .iter()
            .filter_map(|e| model_of_ep.get(*e).cloned())
            .collect();
        let cross_model = !lesson_models.is_empty() && !lesson_models.contains(&test_model);

        // Query principal: op real del entorno.
        let pf = agent.preflight_v2(&game, &[&test_model, "babyai", "level-1"]);
        let top1_hit = pf
            .top_rules
            .first()
            .map(|r| correct_rules.contains(&r.id))
            .unwrap_or(false);
        let top3_hit = pf.top_rules.iter().any(|r| correct_rules.contains(&r.id));
        let top_score = pf.top_rules.first().map(|r| r.score).unwrap_or(0.0);

        // Control: op NUNCA vista (mismos tags). Si el exact-match de op
        // importa, el score correcto debe superar al control.
        let control_op = format!("unseen-op-{}", task_id);
        let pfc = agent.preflight_v2(&control_op, &[&test_model, "babyai", "level-1"]);
        let control_score = pfc.top_rules.first().map(|r| r.score).unwrap_or(0.0);

        results.push(serde_json::json!({
            "task_id": task_id,
            "n_lessons_in_file": in_file.len(),
            "n_correct_rules": correct_rules.len(),
            "decision_point": dp,
            "cross_model": cross_model,
            "top1_hit": top1_hit,
            "top3_hit": top3_hit,
            "top_score": top_score,
            "control_score": control_score,
            "op_matters": top_score > control_score,
            "lesson_types": in_file.iter().filter_map(|e| type_of_ep.get(*e)).cloned().collect::<Vec<_>>(),
            "top_rule_id": pf.top_rules.first().map(|r| r.id),
        }));
    }

    // ── 3. Métricas ────────────────────────────────────────────────────
    let n = results.len();
    let with_lessons: Vec<&Value> = results
        .iter()
        .filter(|r| r["n_correct_rules"].as_u64().unwrap_or(0) > 0)
        .collect();
    let nl = with_lessons.len();
    let top1 = with_lessons.iter().filter(|r| r["top1_hit"] == true).count();
    let top3 = with_lessons.iter().filter(|r| r["top3_hit"] == true).count();
    let opm = results.iter().filter(|r| r["op_matters"] == true).count();
    let mean = |vals: Vec<f64>| {
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    };
    let mean_top = mean(with_lessons.iter().filter_map(|r| r["top_score"].as_f64()).collect());
    let mean_ctl = mean(results.iter().filter_map(|r| r["control_score"].as_f64()).collect());

    println!(
        "\n=== Especificidad con {} reglas (distractores incluidos) en memoria ===",
        n_rules
    );
    println!("tareas con lecciones en archivo: {}/{}", nl, n);
    println!(
        "top1_precision (lección correcta es la regla #1): {}/{} = {:.1}%",
        top1,
        nl,
        if nl > 0 { top1 as f64 / nl as f64 * 100.0 } else { 0.0 }
    );
    println!(
        "top3_recall (alguna lección correcta en top-3):   {}/{} = {:.1}%",
        top3,
        nl,
        if nl > 0 { top3 as f64 / nl as f64 * 100.0 } else { 0.0 }
    );
    println!(
        "op_matters (score op-real > op-control):          {}/{} = {:.1}%",
        opm,
        n,
        if n > 0 { opm as f64 / n as f64 * 100.0 } else { 0.0 }
    );
    println!(
        "score medio top-1 (op real): {:.2} | control (op no vista): {:.2}",
        mean_top, mean_ctl
    );

    // Desglose cross-model (lecciones de un modelo aplicadas a tests de otro)
    let cross: Vec<&&Value> = with_lessons.iter().filter(|r| r["cross_model"] == true).collect();
    let same: Vec<&&Value> = with_lessons.iter().filter(|r| r["cross_model"] == false).collect();
    for (name, cohort) in [("mismo-modelo", &same), ("cross-modelo", &cross)] {
        if !cohort.is_empty() {
            let t1 = cohort.iter().filter(|r| r["top1_hit"] == true).count();
            println!(
                "  {:<13}: top1 {}/{} = {:.1}%",
                name,
                t1,
                cohort.len(),
                t1 as f64 / cohort.len() as f64 * 100.0
            );
        }
    }

    // Por tipo de falla de las lecciones
    let mut by_type: HashMap<String, (usize, usize)> = HashMap::new();
    for r in &with_lessons {
        let types = r["lesson_types"].as_array().cloned().unwrap_or_default();
        let t = types
            .first()
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let e = by_type.entry(t).or_insert((0, 0));
        e.0 += 1;
        if r["top3_hit"] == true {
            e.1 += 1;
        }
    }
    let mut rows: Vec<_> = by_type.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    println!("\nPor tipo de falla (top3):");
    for (t, (tot, ok)) in &rows {
        println!("  {:<45} {:>3}/{:<3}", t, ok, tot);
    }

    let out = serde_json::json!({
        "bench": "BenchTrace babyai/level-1 — Evolution Eval offline (especificidad FAR-proxy)",
        "n_rules_memoria": n_rules,
        "n_tasks": n,
        "n_tasks_with_lessons": nl,
        "top1_precision": if nl > 0 { top1 as f64 / nl as f64 } else { 0.0 },
        "top3_recall": if nl > 0 { top3 as f64 / nl as f64 } else { 0.0 },
        "op_matters_rate": if n > 0 { opm as f64 / n as f64 } else { 0.0 },
        "mean_top_score": mean_top,
        "mean_control_score": mean_ctl,
        "nota": "proxy offline de selección de lección correcta con distractores; NO es el FAR del paper (evitación real en entorno vivo). Contexto débil: todas las tareas comparten op 'level-1'.",
        "por_tipo": rows.iter().map(|(t,(tot,ok))| serde_json::json!({"type": t, "top3": ok, "total": tot})).collect::<Vec<_>>(),
        "results": results,
    });
    let out_path = format!("{}/far_results_l1.json", data_dir);
    fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("\nOK -> {}", out_path);
}
