//! Edge API — lightweight edge node API (stub).
//! Ported from m2m-vector-search Python.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct EdgeStatus {
    pub node_id: String,
    pub connected: bool,
    pub n_vectors: usize,
    pub memory_mb: f64,
}

/// Edge node API handler.
pub struct EdgeApi {
    node_id: String,
    n_vectors: usize,
}

impl EdgeApi {
    pub fn new(node_id: &str) -> Self { Self { node_id: node_id.to_string(), n_vectors: 0 } }

    pub fn status(&self) -> EdgeStatus {
        EdgeStatus {
            node_id: self.node_id.clone(),
            connected: true,
            n_vectors: self.n_vectors,
            memory_mb: (self.n_vectors * 768 * 4) as f64 / (1024.0 * 1024.0),
        }
    }
}
