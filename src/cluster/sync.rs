//! Cluster sync. Ported from m2m-vector-search Python.

use std::collections::HashMap;

/// Shard version tracker for cluster synchronization.
pub struct ClusterSync {
    versions: HashMap<String, u64>,
}

impl ClusterSync {
    pub fn new() -> Self { Self { versions: HashMap::new() } }

    pub fn update(&mut self, shard_id: &str, version: u64) {
        self.versions.insert(shard_id.to_string(), version);
    }

    pub fn get_version(&self, shard_id: &str) -> Option<u64> {
        self.versions.get(shard_id).copied()
    }

    pub fn needs_sync(&self, shard_id: &str, client_version: u64) -> bool {
        self.versions.get(shard_id).copied().map(|v| v > client_version).unwrap_or(false)
    }
}

impl Default for ClusterSync { fn default() -> Self { Self::new() } }
