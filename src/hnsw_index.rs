//! HNSW Index — Hierarchical Navigable Small World graph for ANN search.
//! Native Rust implementation optimized for speed.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use ndarray::{Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::interfaces::{IndexSearchResult, VectorIndex};

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Global counter of distance computations (rho_HNSW instrumentation).
pub static HNSW_DIST_COMPUTATIONS: AtomicU64 = AtomicU64::new(0);

/// Wrapper for (distance, index) that implements Ord for min-heap.
#[derive(Clone, Copy, Debug)]
struct DistIdx(f32, usize);

impl PartialEq for DistIdx {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for DistIdx {}
impl PartialOrd for DistIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DistIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap wrapper (negate distance).
#[derive(Clone, Copy, Debug)]
struct MaxDistIdx(f32, usize);
impl PartialEq for MaxDistIdx {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MaxDistIdx {}
impl PartialOrd for MaxDistIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxDistIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// A node in the HNSW graph.
#[allow(dead_code)]
struct HNSWNode {
    level: usize,
    neighbors: Vec<Vec<usize>>,
}

/// HNSW index for approximate nearest neighbor search.
#[allow(dead_code)]
pub struct HNSWIndex {
    dim: usize,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    level_mult: f64,
    metric: String,
    rng: ChaCha8Rng,

    vectors: Vec<Vec<f32>>,
    nodes: Vec<HNSWNode>,
    entry_point: usize,
    max_level: usize,
    removed: HashSet<usize>,
}

impl HNSWIndex {
    pub fn new(
        dim: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: &str,
        seed: u64,
    ) -> Self {
        Self {
            dim,
            m,
            m_max0: 2 * m,
            ef_construction,
            ef_search,
            level_mult: 1.0 / (m as f64).ln(),
            metric: metric.to_string(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            vectors: Vec::new(),
            nodes: Vec::new(),
            entry_point: 0,
            max_level: 0,
            removed: HashSet::new(),
        }
    }

    /// Set ef_search at runtime (rho_HNSW sweep).
    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef;
    }

    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        (-(r.ln()) * self.level_mult) as usize
    }

    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        HNSW_DIST_COMPUTATIONS.fetch_add(1, AtomicOrdering::Relaxed);
        if self.metric == "cosine" {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let denom = na * nb;
            if denom < 1e-10 {
                return 1.0;
            }
            1.0 - dot / denom
        } else {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt()
        }
    }

    fn greedy_closest(&self, query: &[f32], entry: usize, level: usize) -> usize {
        let mut current = entry;
        let mut current_dist = self.distance(query, &self.vectors[current]);

        loop {
            let mut changed = false;
            let node = &self.nodes[current];
            if level >= node.neighbors.len() {
                break;
            }
            for &neighbor in &node.neighbors[level] {
                if self.removed.contains(&neighbor) {
                    continue;
                }
                let d = self.distance(query, &self.vectors[neighbor]);
                if d < current_dist {
                    current_dist = d;
                    current = neighbor;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        level: usize,
    ) -> Vec<(f32, usize)> {
        let mut visited = HashSet::new();
        visited.insert(entry);
        let dist_ep = self.distance(query, &self.vectors[entry]);

        let mut candidates = BinaryHeap::new(); // min-heap via DistIdx
        let mut results = BinaryHeap::new(); // max-heap via MaxDistIdx

        candidates.push(DistIdx(dist_ep, entry));
        results.push(MaxDistIdx(dist_ep, entry));

        while let Some(DistIdx(dist_c, c)) = candidates.pop() {
            let farthest = match results.peek() {
                Some(MaxDistIdx(d, _)) => *d,
                None => break,
            };
            if dist_c > farthest {
                break;
            }

            let node = &self.nodes[c];
            if level >= node.neighbors.len() {
                continue;
            }

            for &neighbor in &node.neighbors[level] {
                if visited.contains(&neighbor) || self.removed.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let dist_n = self.distance(query, &self.vectors[neighbor]);
                let farthest = match results.peek() {
                    Some(MaxDistIdx(d, _)) => *d,
                    None => f32::MAX,
                };

                if dist_n < farthest || results.len() < ef {
                    candidates.push(DistIdx(dist_n, neighbor));
                    results.push(MaxDistIdx(dist_n, neighbor));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut r: Vec<(f32, usize)> = results.into_iter().map(|MaxDistIdx(d, i)| (d, i)).collect();
        r.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        r
    }

    fn insert(&mut self, vector: &[f32]) {
        let idx = self.vectors.len();
        let level = self.random_level();
        let mut neighbors = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            neighbors.push(Vec::new());
        }
        self.vectors.push(vector.to_vec());
        self.nodes.push(HNSWNode { level, neighbors });

        if idx == 0 {
            self.entry_point = 0;
            self.max_level = level;
            return;
        }

        let mut ep = self.entry_point;

        // Phase 1: greedy descent from top to above insertion level
        for lev in (level + 1..=self.max_level).rev() {
            ep = self.greedy_closest(vector, ep, lev);
        }

        // Phase 2: insert into each layer
        for lev in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(vector, ep, self.ef_construction, lev);
            let m_max = if lev == 0 { self.m_max0 } else { self.m };
            let m = self.m;

            // Select M closest neighbors
            let selected: Vec<usize> = candidates.iter().take(m).map(|(_, idx)| *idx).collect();

            // Set bidirectional connections
            self.nodes[idx].neighbors[lev] = selected.clone();
            for &neighbor_idx in &selected {
                let n_vec = self.vectors[neighbor_idx].clone();
                let neighbor_ids: Vec<(f32, usize)> = self.nodes[neighbor_idx].neighbors[lev]
                    .iter()
                    .map(|&ni| (self.distance(&n_vec, &self.vectors[ni]), ni))
                    .collect();

                let node = &mut self.nodes[neighbor_idx];
                if lev >= node.neighbors.len() {
                    continue;
                }
                if !node.neighbors[lev].contains(&idx) {
                    node.neighbors[lev].push(idx);
                }
                if node.neighbors[lev].len() > m_max {
                    let mut scored = neighbor_ids;
                    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    node.neighbors[lev] = scored.into_iter().take(m_max).map(|(_, i)| i).collect();
                }
            }

            if !candidates.is_empty() {
                ep = candidates[0].1;
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = idx;
        }
    }
}

impl VectorIndex for HNSWIndex {
    fn build(&mut self, vectors: Array2<f32>) {
        self.vectors.clear();
        self.nodes.clear();
        self.removed.clear();
        self.entry_point = 0;
        self.max_level = 0;

        for i in 0..vectors.nrows() {
            let row: Vec<f32> = vectors.row(i).to_vec();
            self.insert(&row);
        }
    }

    fn search(&self, query: ArrayView1<f32>, k: usize) -> IndexSearchResult {
        if self.vectors.is_empty() {
            return IndexSearchResult {
                indices: vec![],
                distances: vec![],
            };
        }

        let q: Vec<f32> = query.to_vec();
        let ef = self.ef_search.max(k);
        let mut ep = self.entry_point;

        // Phase 1: greedy descent from top layer to layer 1
        for level in (1..=self.max_level).rev() {
            ep = self.greedy_closest(&q, ep, level);
        }

        // Phase 2: search at layer 0
        let candidates = self.search_layer(&q, ep, ef, 0);

        let results: Vec<(f32, usize)> = candidates
            .into_iter()
            .filter(|(_, idx)| !self.removed.contains(idx))
            .take(k)
            .collect();

        IndexSearchResult {
            indices: results.iter().map(|(_, i)| *i).collect(),
            distances: results.iter().map(|(d, _)| *d).collect(),
        }
    }

    fn add(&mut self, vectors: Array2<f32>) {
        for i in 0..vectors.nrows() {
            let row: Vec<f32> = vectors.row(i).to_vec();
            self.insert(&row);
        }
    }

    fn remove(&mut self, indices: &[usize]) {
        for &idx in indices {
            self.removed.insert(idx);
        }
    }

    fn n_items(&self) -> usize {
        self.vectors.len() - self.removed.len()
    }

    fn supports_remove(&self) -> bool {
        true
    }
}
