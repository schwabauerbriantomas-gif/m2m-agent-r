//! TurboQuant integration for M2M Vector Search.
//!
//! Compresses 640D splat embeddings to 4-8 bits per value with minimal accuracy loss.
//! Uses data-oblivious quantization — no training, no codebooks, instant indexing.

use ndarray::{Array2, ArrayView1};
use turbo_quant::{TurboQuantizer, PolarQuantizer, PolarCode, TurboCode};

/// Configuration for vector quantization.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Bits per value (3-8). 8 = near-lossless, 4 = aggressive compression.
    pub bits: u8,
    /// Number of random projections. Recommended: dim / 4 for search.
    pub projections: usize,
    /// Random seed for deterministic quantization.
    pub seed: u64,
    /// Which algorithm to use.
    pub algorithm: QuantAlgorithm,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantAlgorithm {
    /// Two-stage: PolarQuant + QJL residual. Best accuracy.
    TurboQuant,
    /// Single-stage polar coordinate encoding. Simpler, slightly less accurate.
    PolarQuant,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            projections: 16, // 64 / 4
            seed: 42,
            algorithm: QuantAlgorithm::TurboQuant,
        }
    }
}

impl QuantConfig {
    pub fn for_search(dim: usize) -> Self {
        Self {
            bits: 8,
            projections: dim / 4,
            seed: 42,
            algorithm: QuantAlgorithm::TurboQuant,
        }
    }

    pub fn for_compression(dim: usize) -> Self {
        Self {
            bits: 4,
            projections: dim / 8,
            seed: 42,
            algorithm: QuantAlgorithm::TurboQuant,
        }
    }

    pub fn for_max_compression(dim: usize) -> Self {
        Self {
            bits: 3,
            projections: dim / 16,
            seed: 42,
            algorithm: QuantAlgorithm::PolarQuant,
        }
    }
}

/// Compressed vector store using TurboQuant.
pub struct QuantizedStore {
    config: QuantConfig,
    dim: usize,
    turbo: Option<TurboQuantizer>,
    polar: Option<PolarQuantizer>,
    codes_turbo: Vec<TurboCode>,
    codes_polar: Vec<PolarCode>,
    ids: Vec<u64>,
}

impl QuantizedStore {
    pub fn new(dim: usize, config: QuantConfig) -> Result<Self, String> {
        let (turbo, polar) = match config.algorithm {
            QuantAlgorithm::TurboQuant => {
                let t = TurboQuantizer::new(dim, config.bits, config.projections, config.seed)
                    .map_err(|e| format!("TurboQuant init error: {}", e))?;
                (Some(t), None)
            }
            QuantAlgorithm::PolarQuant => {
                let p = PolarQuantizer::new(dim, config.bits, config.seed)
                    .map_err(|e| format!("PolarQuant init error: {}", e))?;
                (None, Some(p))
            }
        };

        Ok(Self {
            config,
            dim,
            turbo,
            polar,
            codes_turbo: Vec::new(),
            codes_polar: Vec::new(),
            ids: Vec::new(),
        })
    }

    /// Encode and store a batch of vectors.
    pub fn add_batch(&mut self, vectors: &Array2<f32>, start_id: u64) -> usize {
        let n = vectors.nrows();
        for i in 0..n {
            let v: Vec<f32> = vectors.row(i).to_vec();
            if let Some(ref turbo) = self.turbo {
                if let Ok(code) = turbo.encode(&v) {
                    self.codes_turbo.push(code);
                }
            } else if let Some(ref polar) = self.polar {
                if let Ok(code) = polar.encode(&v) {
                    self.codes_polar.push(code);
                }
            }
            self.ids.push(start_id + i as u64);
        }
        n
    }

    /// Encode a single vector.
    pub fn add_single(&mut self, vector: &ArrayView1<f32>, id: u64) -> bool {
        let v: Vec<f32> = vector.to_vec();
        if let Some(ref turbo) = self.turbo {
            match turbo.encode(&v) {
                Ok(code) => { self.codes_turbo.push(code); }
                Err(_) => return false,
            }
        } else if let Some(ref polar) = self.polar {
            match polar.encode(&v) {
                Ok(code) => { self.codes_polar.push(code); }
                Err(_) => return false,
            }
        }
        self.ids.push(id);
        true
    }

    /// Search for k nearest neighbors using inner product estimation.
    /// Returns (id, estimated_inner_product) pairs sorted by similarity (descending).
    pub fn search(&self, query: &ArrayView1<f32>, k: usize) -> Vec<(u64, f32)> {
        let q: Vec<f32> = query.to_vec();
        let n = self.ids.len();
        let k = k.min(n);
        if k == 0 { return Vec::new(); }

        let mut scores: Vec<(usize, f32)> = if let Some(ref turbo) = self.turbo {
            self.codes_turbo.iter().enumerate()
                .filter_map(|(i, code)| {
                    turbo.inner_product_estimate(code, &q).ok().map(|s| (i, s))
                })
                .collect()
        } else if let Some(ref polar) = self.polar {
            self.codes_polar.iter().enumerate()
                .filter_map(|(i, code)| {
                    polar.inner_product_estimate(code, &q).ok().map(|s| (i, s))
                })
                .collect()
        } else {
            Vec::new()
        };

        // Sort descending by inner product (higher = more similar)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        scores.into_iter()
            .map(|(i, score)| (self.ids[i], score))
            .collect()
    }

    /// Get number of stored codes.
    pub fn len(&self) -> usize { self.ids.len() }

    /// Check if store is empty.
    pub fn is_empty(&self) -> bool { self.ids.is_empty() }

    /// Estimate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let code_size = if self.turbo.is_some() {
            // TurboCode: roughly dim * bits / 8 bytes + overhead
            self.dim * self.config.bits as usize / 8 + 64
        } else {
            // PolarCode: roughly dim * bits / 8
            self.dim * self.config.bits as usize / 8 + 32
        };
        self.ids.len() * (code_size + 8) // 8 bytes for id
    }

    /// Get compression ratio vs raw f32 storage.
    pub fn compression_ratio(&self) -> f32 {
        let raw_bytes = self.dim * 4; // f32 per dim
        let quant_bytes = self.dim * self.config.bits as usize / 8;
        raw_bytes as f32 / quant_bytes.max(1) as f32
    }

    /// Get the config.
    pub fn config(&self) -> &QuantConfig { &self.config }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        Array2::from_shape_fn((n, dim), |(_, _)| rng.gen::<f32>())
    }

    #[test]
    fn test_turbo_quant_roundtrip() {
        let dim = 64;
        let config = QuantConfig::for_search(dim);
        let mut store = QuantizedStore::new(dim, config).unwrap();

        let vectors = random_vectors(100, dim, 42);
        store.add_batch(&vectors, 0);
        assert_eq!(store.len(), 100);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_turbo_quant_search() {
        let dim = 64;
        let config = QuantConfig::for_search(dim);
        let mut store = QuantizedStore::new(dim, config).unwrap();

        let vectors = random_vectors(100, dim, 42);
        store.add_batch(&vectors, 0);

        // Search with the first vector — should find itself near the top
        let results = store.search(&vectors.row(0), 5);
        assert_eq!(results.len(), 5);
        // With 8-bit quantization, the query vector should be in top-5
        let top_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(top_ids.contains(&0), "Query vector should be in top-5 results, got: {:?}", top_ids);
    }

    #[test]
    fn test_polar_quant_search() {
        let dim = 64;
        let config = QuantConfig {
            bits: 8,
            projections: 16,
            seed: 42,
            algorithm: QuantAlgorithm::PolarQuant,
        };
        let mut store = QuantizedStore::new(dim, config).unwrap();

        let vectors = random_vectors(50, dim, 123);
        store.add_batch(&vectors, 0);

        let results = store.search(&vectors.row(10), 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_compression_ratio() {
        let dim = 640;
        let config_8bit = QuantConfig::for_search(dim);
        let store_8 = QuantizedStore::new(dim, config_8bit).unwrap();
        assert!(store_8.compression_ratio() > 3.0);

        let config_4bit = QuantConfig::for_compression(dim);
        let store_4 = QuantizedStore::new(dim, config_4bit).unwrap();
        assert!(store_4.compression_ratio() > 6.0);

        let config_3bit = QuantConfig::for_max_compression(dim);
        let store_3 = QuantizedStore::new(dim, config_3bit).unwrap();
        assert!(store_3.compression_ratio() > 8.0);
    }

    #[test]
    fn test_add_single() {
        let dim = 64;
        let config = QuantConfig::default();
        let mut store = QuantizedStore::new(dim, config).unwrap();

        let v = Array1::from_elem(dim, 0.5f32);
        assert!(store.add_single(&v.view(), 42));
        assert_eq!(store.len(), 1);
        assert_eq!(store.ids[0], 42);
    }

    #[test]
    fn test_empty_search() {
        let dim = 64;
        let config = QuantConfig::default();
        let store = QuantizedStore::new(dim, config).unwrap();

        let v = Array1::from_elem(dim, 0.5f32);
        let results = store.search(&v.view(), 5);
        assert!(results.is_empty());
    }
}
