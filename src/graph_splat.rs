//! Graph Splat — graph structures over Gaussian splats.
//! Ported from m2m-vector-search Python.

use std::collections::HashMap;

/// Edge in the graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub source_id: usize,
    pub target_id: usize,
    pub relation_type: String,
    pub weight: f64,
}

/// Entity node in the graph.
#[derive(Debug, Clone)]
pub struct GraphEntity {
    pub id: usize,
    pub name: String,
    pub entity_type: String,
    pub embedding: Option<Vec<f32>>,
}

/// Graph store for Gaussian splats.
pub struct GaussianGraphStore {
    entities: HashMap<usize, GraphEntity>,
    edges: Vec<GraphEdge>,
    next_id: usize,
}

impl GaussianGraphStore {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            edges: Vec::new(),
            next_id: 1,
        }
    }

    /// Add or get an entity node.
    pub fn add_entity(&mut self, name: &str, embedding: Option<&[f32]>, entity_type: &str) -> usize {
        // Check if entity already exists by name
        for (id, entity) in &self.entities {
            if entity.name == name {
                return *id;
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        self.entities.insert(id, GraphEntity {
            id,
            name: name.to_string(),
            entity_type: entity_type.to_string(),
            embedding: embedding.map(|e| e.to_vec()),
        });
        id
    }

    /// Add a relation edge.
    pub fn add_relation(&mut self, source_id: usize, target_id: usize, relation_type: &str, weight: f64) -> bool {
        self.edges.push(GraphEdge {
            source_id,
            target_id,
            relation_type: relation_type.to_string(),
            weight,
        });
        true
    }

    /// Get an entity by ID.
    pub fn get_entity(&self, id: usize) -> Option<&GraphEntity> {
        self.entities.get(&id)
    }

    /// Get edges for an entity.
    pub fn get_edges(&self, entity_id: usize) -> Vec<&GraphEdge> {
        self.edges.iter().filter(|e| e.source_id == entity_id || e.target_id == entity_id).collect()
    }

    /// Get all neighbors of an entity.
    pub fn get_neighbors(&self, entity_id: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        for edge in &self.edges {
            if edge.source_id == entity_id {
                neighbors.push(edge.target_id);
            } else if edge.target_id == entity_id {
                neighbors.push(edge.source_id);
            }
        }
        neighbors.sort();
        neighbors.dedup();
        neighbors
    }

    pub fn n_entities(&self) -> usize { self.entities.len() }
    pub fn n_edges(&self) -> usize { self.edges.len() }
}

impl Default for GaussianGraphStore {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_entity_and_edge() {
        let mut graph = GaussianGraphStore::new();
        let id1 = graph.add_entity("Entity1", Some(&[1.0, 2.0]), "test");
        let id2 = graph.add_entity("Entity2", Some(&[3.0, 4.0]), "test");
        assert_ne!(id1, id2);
        graph.add_relation(id1, id2, "RELATED", 0.8);
        assert_eq!(graph.n_entities(), 2);
        assert_eq!(graph.n_edges(), 1);
        let neighbors = graph.get_neighbors(id1);
        assert_eq!(neighbors, vec![id2]);
    }

    #[test]
    fn test_dedup_entity() {
        let mut graph = GaussianGraphStore::new();
        let id1 = graph.add_entity("Same", None, "test");
        let id2 = graph.add_entity("Same", None, "test");
        assert_eq!(id1, id2);
        assert_eq!(graph.n_entities(), 1);
    }
}
