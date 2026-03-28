//! Backend Communication — inter-node messaging protocol.
//! Ported from m2m-vector-search Python.

use serde::{Deserialize, Serialize};

/// Message types for inter-node communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendMessage {
    /// Search request
    Search { query_id: String, vector: Vec<f32>, k: usize },
    /// Search response
    SearchResponse { query_id: String, results: Vec<(usize, f32)> },
    /// Index request
    Index { shard_id: String, vectors: Vec<Vec<f32>> },
    /// Index response
    IndexResponse { shard_id: String, count: usize },
    /// Health check
    HealthCheck { node_id: String },
    /// Health response
    HealthResponse { node_id: String, healthy: bool, n_vectors: usize },
    /// Sync request
    Sync { shard_id: String, from_version: u64 },
    /// Sync response
    SyncResponse { shard_id: String, version: u64, data: Option<Vec<Vec<f32>>> },
}

/// Communication stats.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CommStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub errors: u64,
}

/// Backend communicator (local simulation).
pub struct BackendComm {
    node_id: String,
    stats: CommStats,
    message_log: Vec<BackendMessage>,
}

impl BackendComm {
    pub fn new(node_id: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            stats: CommStats::default(),
            message_log: Vec::new(),
        }
    }

    /// Send a message.
    pub fn send(&mut self, msg: BackendMessage) {
        self.stats.messages_sent += 1;
        self.stats.bytes_sent += estimate_size(&msg);
        self.message_log.push(msg);
    }

    /// Receive a message (from log).
    pub fn receive(&mut self) -> Option<BackendMessage> {
        if self.message_log.is_empty() {
            return None;
        }
        let msg = self.message_log.remove(0);
        self.stats.messages_received += 1;
        self.stats.bytes_received += estimate_size(&msg);
        Some(msg)
    }

    /// Get communicator stats.
    pub fn get_stats(&self) -> &CommStats {
        &self.stats
    }

    /// Get node ID.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }
}

fn estimate_size(msg: &BackendMessage) -> u64 {
    match msg {
        BackendMessage::Search { vector, .. } => vector.len() as u64 * 4 + 64,
        BackendMessage::SearchResponse { results, .. } => results.len() as u64 * 12 + 64,
        BackendMessage::Index { vectors, .. } => vectors.len() as u64 * 4 + 64,
        BackendMessage::IndexResponse { .. } => 64,
        BackendMessage::HealthCheck { .. } => 64,
        BackendMessage::HealthResponse { .. } => 64,
        BackendMessage::Sync { .. } => 64,
        BackendMessage::SyncResponse { data, .. } => data.as_ref().map(|d| d.len() as u64 * 4 + 64).unwrap_or(64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_receive() {
        let mut comm = BackendComm::new("node1");
        comm.send(BackendMessage::HealthCheck { node_id: "node1".into() });
        let msg = comm.receive().unwrap();
        assert!(matches!(msg, BackendMessage::HealthCheck { .. }));
        assert_eq!(comm.get_stats().messages_sent, 1);
    }
}
