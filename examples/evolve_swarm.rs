
use m2m_vector_search::agent_r::AgentR;
use m2m_vector_search::meta_evolve::{evolve, EvolveSpace, ReplayScore};
use m2m_vector_search::receipts::ReceiptStore;
use std::path::PathBuf;

fn main() {
    std::env::set_var("AGENT_R_NO_SEED", "1");
    let state = PathBuf::from("/tmp/rsi_muestra_state.json");
    let store = ReceiptStore::open_no_env(std::path::Path::new("/tmp/rsi_muestra_receipts")).unwrap();
    let mut base = AgentR::with_state_path(state.clone());
    let space = EvolveSpace::default();
    // Evaluator: para cada W, preflight de las 3 ops del pool sobre el
    // snapshot RE-ENCARNADO (solo lectura) — score = precautions relevantes
    // (la señal práctica del sistema: ¿el swarm avisa lo que duele?)
    let report = evolve(&mut base, &store, &space, |agent, w| {
        let mut probe = AgentR::with_state_path(state.clone());
        let (mut hits, mut n) = (0u64, 0u64);
        // Caso 1: index_rebuild con situación de shard lock (GT: regla 1)
        let pf = agent.preflight_v2_hybrid_w("index_rebuild", &["swarm"], "rebuild stalled shard lock held by stale writer", w);
        n += 1;
        if pf.suggestion.as_deref().map_or(false, |s| s.contains("shard lock")) { hits += 1; }
        // Caso 2: deploy_prod con situación de warmup (GT: regla 2)
        let pf = agent.preflight_v2_hybrid_w("deploy_prod", &["swarm"], "traffic switched before warmup health 503", w);
        n += 1;
        if pf.suggestion.as_deref().map_or(false, |s| s.contains("warm") || s.contains("health")) { hits += 1; }
        // Caso 3: op limpia sin situación patológica => NO debe vetar
        let pf = agent.preflight_v2_hybrid_w("index_rebuild", &["swarm"], "routine scheduled maintenance nothing wrong", w);
        n += 1;
        if pf.proceed { hits += 1; }
        ReplayScore { w, score: hits as f64, tiebreak: 0.0, n }
    }, 0.5);
    println!("{}", serde_json::to_string_pretty(&serde_json::to_value(&report).unwrap()).unwrap());
}
