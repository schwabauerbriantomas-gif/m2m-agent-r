//! Cluster router — routes queries to appropriate nodes.
//! Ported from m2m-vector-search Python.

use crate::cluster::protocol::NodeStatus;

/// Routes queries to cluster nodes based on shard assignment.
pub struct ClusterRouter {
    nodes: Vec<NodeStatus>,
}

impl ClusterRouter {
    pub fn new() -> Self { Self { nodes: Vec::new() } }

    pub fn register_node(&mut self, node: NodeStatus) {
        self.nodes.retain(|n| n.id != node.id);
        self.nodes.push(node);
    }

    pub fn route(&self, shard_id: &str) -> Option<String> {
        // Simple round-robin based on shard hash
        if self.nodes.is_empty() { return None; }
        let hash = shard_id.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let idx = (hash as usize) % self.nodes.len();
        self.nodes.get(idx).map(|n| n.id.clone())
    }

    pub fn healthy_nodes(&self) -> Vec<&NodeStatus> {
        self.nodes.iter().filter(|n| n.healthy).collect()
    }
}

impl Default for ClusterRouter { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::protocol::NodeRole;

    #[test]
    fn test_route() {
        let mut router = ClusterRouter::new();
        router.register_node(NodeStatus {
            id: "n1".into(), role: NodeRole::Worker, address: "localhost:8001".into(),
            healthy: true, n_vectors: 1000, last_heartbeat: 0.0,
        });
        let target = router.route("shard_42");
        assert_eq!(target.unwrap(), "n1");
    }
}
