//! SplatStore - Main API for M2M Vector Search.
//! Ported from Python splats.py. CPU-only.

use ndarray::{Array1, Array2, ArrayView1};

use crate::config::M2MConfig;
use crate::hrm2_engine::HRM2Engine;

/// Result of a neighbor search.
#[derive(Clone, Debug)]
pub struct NeighborResult {
    pub mu: Vec<f32>,
    pub alpha: f32,
    pub kappa: f32,
    pub index: usize,
    pub distance: f32,
}

/// Main SplatStore - wraps HRM2Engine for high-level vector operations.
pub struct SplatStore {
    config: M2MConfig,
    max_splats: usize,
    n_active: usize,
    mu: Array2<f32>,
    alpha: Array1<f32>,
    kappa: Array1<f32>,
    frequency: Array1<f32>,
    engine: HRM2Engine,
    next_id: u64,
}

impl SplatStore {
    pub fn new(config: M2MConfig) -> Self {
        let max_splats = config.max_splats;
        let dim = config.latent_dim;
        let n_coarse = (((max_splats as f64).sqrt() / 10.0) as usize).max(10);
        let n_fine = (max_splats / n_coarse).max(100);
        let engine = HRM2Engine::new(n_coarse, n_fine, dim, n_coarse.min(5), "cosine");

        Self {
            config,
            max_splats,
            n_active: 0,
            mu: Array2::zeros((max_splats, dim)),
            alpha: Array1::ones(max_splats),
            kappa: Array1::ones(max_splats),
            frequency: Array1::zeros(max_splats),
            engine,
            next_id: 0,
        }
    }

    /// Add a batch of splat vectors. Returns false if capacity exceeded.
    pub fn add_splat(&mut self, x: &Array2<f32>) -> bool {
        let n_new = x.nrows();
        if self.n_active + n_new > self.max_splats {
            return false;
        }
        for i in 0..n_new {
            self.mu.row_mut(self.n_active + i).assign(&x.row(i));
            self.alpha[self.n_active + i] = self.config.init_alpha as f32;
            self.kappa[self.n_active + i] = self.config.init_kappa as f32;
            self.frequency[self.n_active + i] = 1.0;
        }
        self.n_active += n_new;
        self.next_id += n_new as u64;
        true
    }

    /// Build the search index from active vectors.
    pub fn build_index(&mut self) {
        if self.n_active == 0 {
            return;
        }
        let embeddings = self.mu.slice(ndarray::s![..self.n_active, ..]).to_owned();
        self.engine.index(Some(embeddings));
    }

    /// Find k nearest neighbors for a single query.
    pub fn find_neighbors(&self, query: &ArrayView1<f32>, k: usize) -> Vec<NeighborResult> {
        let n = self.n_active;
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let k = k.min(n);
        let index_data = self.mu.slice(ndarray::s![..n, ..]);

        // Squared L2 distances
        let dists_sq: Vec<f32> = index_data
            .outer_iter()
            .map(|row| row.iter().zip(query.iter()).map(|(a, b)| { let d = a - b; d * d }).sum())
            .collect();

        // Partial sort top-k
        let mut indexed: Vec<(f32, usize)> = dists_sq.iter().enumerate().map(|(i, &d)| (d, i)).collect();
        if k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
        }
        indexed[..k].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        indexed[..k]
            .iter()
            .map(|&(dist_sq, idx)| NeighborResult {
                mu: index_data.row(idx).to_vec(),
                alpha: self.alpha[idx],
                kappa: self.kappa[idx],
                index: idx,
                distance: dist_sq.sqrt(),
            })
            .collect()
    }

    /// Batch find neighbors for multiple queries.
    pub fn find_neighbors_batch(&self, queries: &Array2<f32>, k: usize) -> Vec<Vec<NeighborResult>> {
        (0..queries.nrows()).map(|i| self.find_neighbors(&queries.row(i), k)).collect()
    }

    /// Shannon entropy of the kappa distribution.
    pub fn entropy(&self) -> f32 {
        if self.n_active == 0 {
            return 0.0;
        }
        let kappa = self.kappa.slice(ndarray::s![..self.n_active]);
        let total: f32 = kappa.sum();
        if total <= 0.0 {
            return 0.0;
        }
        let h: f32 = kappa.iter().filter(|&&k| k > 0.0).map(|&k| { let p = k / total; -p * p.ln() }).sum();
        let max_h = (self.n_active as f32).ln();
        if max_h <= 0.0 { 0.0 } else { (h / max_h).clamp(0.0, 1.0) }
    }

    /// Remove invalid splats (alpha ~0, NaN/Inf in mu).
    pub fn compact(&mut self) {
        if self.n_active == 0 { return; }
        let n = self.n_active;
        let mut write_idx = 0;
        for read_idx in 0..n {
            let alpha_ok = self.alpha[read_idx] >= 1e-6;
            let mu_ok = self.mu.row(read_idx).iter().all(|v| v.is_finite());
            if alpha_ok && mu_ok {
                if write_idx != read_idx {
                    let row_data = self.mu.row(read_idx).to_owned();
                    self.mu.row_mut(write_idx).assign(&row_data);
                    self.alpha[write_idx] = self.alpha[read_idx];
                    self.kappa[write_idx] = self.kappa[read_idx];
                    self.frequency[write_idx] = self.frequency[read_idx];
                }
                write_idx += 1;
            }
        }
        for i in write_idx..n {
            self.mu.row_mut(i).fill(0.0);
            self.alpha[i] = 0.0;
            self.kappa[i] = 0.0;
            self.frequency[i] = 0.0;
        }
        self.n_active = write_idx;
    }

    /// Get store statistics.
    pub fn get_statistics(&self) -> SplatStoreStats {
        SplatStoreStats {
            n_active: self.n_active,
            max_splats: self.max_splats,
            embedding_dim: self.config.latent_dim,
            entropy: self.entropy(),
        }
    }

    pub fn n_active(&self) -> usize { self.n_active }
    pub fn max_splats(&self) -> usize { self.max_splats }

    /// Get active mu vectors (n_active × dim slice).
    pub fn get_mu(&self) -> Option<Array2<f32>> {
        if self.n_active == 0 { return None; }
        Some(self.mu.slice(ndarray::s![..self.n_active, ..]).to_owned())
    }

    /// Get active alpha values.
    pub fn get_alpha(&self) -> Option<Array1<f32>> {
        if self.n_active == 0 { return None; }
        Some(self.alpha.slice(ndarray::s![..self.n_active]).to_owned())
    }

    /// Get active kappa values.
    pub fn get_kappa(&self) -> Option<Array1<f32>> {
        if self.n_active == 0 { return None; }
        Some(self.kappa.slice(ndarray::s![..self.n_active]).to_owned())
    }
}

#[derive(Debug, Clone)]
pub struct SplatStoreStats {
    pub n_active: usize,
    pub max_splats: usize,
    pub embedding_dim: usize,
    pub entropy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> M2MConfig {
        let mut c = M2MConfig::default();
        c.max_splats = 1000;
        c.latent_dim = 64;
        c
    }

    #[test]
    fn test_add_and_query() {
        let mut store = SplatStore::new(test_config());
        let data = Array2::from_shape_fn((100, 64), |(i, j)| ((i * 64 + j) as f32 * 0.01).sin());
        let mut normalized = data.clone();
        for mut row in normalized.rows_mut() {
            let norm = row.dot(&row).sqrt().max(1e-10);
            row.mapv_inplace(|v| v / norm);
        }
        assert!(store.add_splat(&normalized));
        assert_eq!(store.n_active(), 100);
        store.build_index();
        let query = normalized.row(0).to_owned();
        let results = store.find_neighbors(&query.view(), 5);
        assert_eq!(results.len(), 5);
        assert!(results[0].distance < 0.01, "First neighbor should be near-identical, got dist={}", results[0].distance);
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_capacity_limit() {
        let mut store = SplatStore::new(test_config());
        let data = Array2::zeros((1000, 64));
        assert!(store.add_splat(&data));
        let more = Array2::zeros((1, 64));
        assert!(!store.add_splat(&more));
    }

    #[test]
    fn test_entropy() {
        let mut store = SplatStore::new(test_config());
        assert_eq!(store.entropy(), 0.0);
        let data = Array2::ones((10, 64));
        store.add_splat(&data);
        let e = store.entropy();
        assert!(e > 0.0 && e <= 1.0, "Entropy should be in (0, 1], got {}", e);
    }

    #[test]
    fn test_compact() {
        let mut store = SplatStore::new(test_config());
        let data = Array2::from_shape_fn((10, 64), |(i, _)| (i as f32).cos());
        store.add_splat(&data);
        store.alpha[0] = 0.0;
        store.alpha[5] = 0.0;
        store.mu.row_mut(7).fill(f32::NAN);
        store.compact();
        assert_eq!(store.n_active(), 7);
    }

    #[test]
    fn test_empty_query() {
        let store = SplatStore::new(test_config());
        let q = Array1::zeros(64);
        let results = store.find_neighbors(&q.view(), 5);
        assert!(results.is_empty());
    }
}
