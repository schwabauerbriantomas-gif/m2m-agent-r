//! Cluster load balancer. Ported from m2m-vector-search Python.

use std::collections::HashMap;

/// Simple load balancer for cluster nodes.
pub struct LoadBalancer {
    loads: HashMap<String, usize>,
}

impl LoadBalancer {
    pub fn new() -> Self { Self { loads: HashMap::new() } }

    pub fn register(&mut self, node_id: &str) {
        self.loads.entry(node_id.to_string()).or_insert(0);
    }

    pub fn select_least_loaded(&self) -> Option<String> {
        self.loads.iter().min_by_key(|(_, &load)| load).map(|(id, _)| id.clone())
    }

    pub fn increment_load(&mut self, node_id: &str) {
        *self.loads.entry(node_id.to_string()).or_insert(0) += 1;
    }

    pub fn decrement_load(&mut self, node_id: &str) {
        *self.loads.entry(node_id.to_string()).or_insert(0) = self.loads.get(node_id).unwrap_or(&0).saturating_sub(1);
    }
}

impl Default for LoadBalancer { fn default() -> Self { Self::new() } }
