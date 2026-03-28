//! Cluster client. Ported from m2m-vector-search Python.

use crate::backend_comm::{BackendComm, BackendMessage};

/// Client for communicating with cluster nodes.
pub struct ClusterClient {
    comm: BackendComm,
}

impl ClusterClient {
    pub fn new(node_id: &str) -> Self {
        Self { comm: BackendComm::new(node_id) }
    }

    pub fn send_heartbeat(&mut self, n_vectors: usize) {
        self.comm.send(BackendMessage::HealthCheck {
            node_id: self.comm.node_id().to_string(),
        });
    }

    pub fn send_search(&mut self, query: Vec<f32>, k: usize) {
        let query_id = format!("q-{}", rand::random::<u32>());
        self.comm.send(BackendMessage::Search {
            query_id,
            vector: query,
            k,
        });
    }
}
