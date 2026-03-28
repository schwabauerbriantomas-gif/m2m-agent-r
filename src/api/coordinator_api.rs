//! Coordinator API — cluster coordinator REST endpoints (stub).
//! Ported from m2m-vector-search Python.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    pub k: usize,
    pub filter: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub results: Vec<(usize, f64)>,
    pub latency_ms: f64,
    pub backend: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterInfo {
    pub n_nodes: usize,
    pub n_vectors: usize,
    pub healthy: bool,
}

/// Coordinator API handler.
pub struct CoordinatorApi {
    n_nodes: usize,
    n_vectors: usize,
}

impl CoordinatorApi {
    pub fn new() -> Self { Self { n_nodes: 1, n_vectors: 0 } }

    pub fn handle_search(&self, _req: &SearchRequest) -> SearchResponse {
        SearchResponse {
            results: vec![],
            latency_ms: 0.0,
            backend: "cpu".into(),
        }
    }

    pub fn cluster_info(&self) -> ClusterInfo {
        ClusterInfo {
            n_nodes: self.n_nodes,
            n_vectors: self.n_vectors,
            healthy: true,
        }
    }
}

impl Default for CoordinatorApi { fn default() -> Self { Self::new() } }
