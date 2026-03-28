use m2m_vector_search::M2MConfig;

fn main() {
    let config = M2MConfig::default();
    println!("M2M Vector Search v{} — {} mode", env!("CARGO_PKG_VERSION"), config.device);
}
