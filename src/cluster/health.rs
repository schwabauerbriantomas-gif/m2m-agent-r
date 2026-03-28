//! Cluster health monitoring. Ported from m2m-vector-search Python.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct HealthStatus {
    pub node_id: String,
    pub healthy: bool,
    pub n_vectors: usize,
    pub last_check: f64,
}

/// Health checker for cluster nodes.
pub struct HealthChecker {
    statuses: Vec<HealthStatus>,
    timeout_secs: f64,
}

impl HealthChecker {
    pub fn new(timeout_secs: f64) -> Self {
        Self { statuses: Vec::new(), timeout_secs }
    }

    pub fn update(&mut self, node_id: &str, healthy: bool, n_vectors: usize) {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64();
        self.statuses.retain(|s| s.node_id != node_id);
        self.statuses.push(HealthStatus { node_id: node_id.to_string(), healthy, n_vectors, last_check: now });
    }

    pub fn healthy_nodes(&self) -> Vec<&HealthStatus> {
        self.statuses.iter().filter(|s| s.healthy).collect()
    }

    pub fn all_healthy(&self) -> bool { self.statuses.iter().all(|s| s.healthy) }
}
