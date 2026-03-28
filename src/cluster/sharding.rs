//! Cluster sharding. Ported from m2m-vector-search Python.

use std::collections::HashMap;

/// Shard assignment manager.
pub struct ShardManager {
    assignments: HashMap<String, String>, // shard_id -> node_id
}

impl ShardManager {
    pub fn new() -> Self { Self { assignments: HashMap::new() } }

    pub fn assign(&mut self, shard_id: &str, node_id: &str) {
        self.assignments.insert(shard_id.to_string(), node_id.to_string());
    }

    pub fn get_node(&self, shard_id: &str) -> Option<&str> {
        self.assignments.get(shard_id).map(|s| s.as_str())
    }

    pub fn get_shards_for_node(&self, node_id: &str) -> Vec<&str> {
        self.assignments.iter()
            .filter_map(|(s, n)| if n == node_id { Some(s.as_str()) } else { None })
            .collect()
    }

    pub fn n_shards(&self) -> usize { self.assignments.len() }
}

impl Default for ShardManager { fn default() -> Self { Self::new() } }
