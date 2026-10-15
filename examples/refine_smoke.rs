//! Smoke e2e de meta::refine: receipt claim → crítica glm-5.3-flash → N0-N3.
use m2m_vector_search::agent_r::AgentR;
use m2m_vector_search::meta;
use m2m_vector_search::receipts::ReceiptStore;

fn main() {
    std::env::set_var("AGENT_R_NO_SEED", "1");
    let store = ReceiptStore::open_no_env(std::path::Path::new("/tmp/refine_smoke/receipts")).unwrap();
    let mut agent = AgentR::new_ephemeral();
    let output = serde_json::json!({
        "cmd": "rm -rf /tmp/x",
        "force": true,
        "note": "borrar dir temporal de build"
    });
    let out = meta::refine(&mut agent, &store, "cleanup_build", &output, "Limpieza post-build en CI local; solo dirs de /tmp de esta sesión.");
    println!("decision: {}", out.decision);
    println!("reason: {}", out.reason);
    println!("receipt_claim: {:?}", out.receipt_claim.as_deref().map(|s| &s[..20]));
    println!("receipt_result: {:?}", out.receipt_result.as_deref().map(|s| &s[..20]));
    println!("validation overall: {}", out.validation["overall"].as_str().unwrap_or("?"));
}
