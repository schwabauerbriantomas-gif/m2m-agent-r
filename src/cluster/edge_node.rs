//! Edge node for cluster. Ported from m2m-vector-search Python.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct EdgeNodeConfig {
    pub node_id: String,
    pub coordinator_address: String,
    pub max_vectors: usize,
}

/// Edge node that connects to a cluster coordinator.
pub struct EdgeNode {
    config: EdgeNodeConfig,
    connected: bool,
    n_vectors: usize,
}

impl EdgeNode {
    pub fn new(config: EdgeNodeConfig) -> Self {
        Self { config, connected: false, n_vectors: 0 }
    }

    pub fn connect(&mut self) -> Result<(), String> {
        self.connected = true;
        Ok(())
    }

    pub fn disconnect(&mut self) { self.connected = false; }
    pub fn is_connected(&self) -> bool { self.connected }
    pub fn n_vectors(&self) -> usize { self.n_vectors }
    pub fn config(&self) -> &EdgeNodeConfig { &self.config }
}
