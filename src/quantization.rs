//! Vector quantization for M2M.
//! Scalar and polar quantization — no external dependencies.
//! Compresses embeddings to fewer bits with minimal accuracy loss.

use ndarray::{Array2, ArrayView1};

/// Quantization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantAlgorithm {
    Scalar,
    Polar,
}

/// Configuration for vector quantization.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub bits: u8,
    pub projections: usize,
    pub seed: u64,
    pub algorithm: QuantAlgorithm,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self { bits: 8, projections: 160, seed: 42, algorithm: QuantAlgorithm::Scalar }
    }
}

impl QuantConfig {
    pub fn for_search(dim: usize) -> Self {
        Self { bits: 8, projections: dim / 4, seed: 42, algorithm: QuantAlgorithm::Scalar }
    }

    pub fn for_max_compression(dim: usize) -> Self {
        Self { bits: 4, projections: dim / 8, seed: 42, algorithm: QuantAlgorithm::Polar }
    }
}

/// Scalar quantizer: maps f32 vectors to u8 with min/max scaling.
pub struct ScalarQuantizer {
    min_val: Vec<f32>,
    scale: Vec<f32>,
    dim: usize,
    bits: u8,
}

impl ScalarQuantizer {
    pub fn new(data: &Array2<f32>, bits: u8) -> Self {
        let dim = data.ncols();
        let mut min_val = vec![f32::MAX; dim];
        let mut max_val = vec![f32::MIN; dim];

        for row in data.outer_iter() {
            for j in 0..dim {
                if row[j] < min_val[j] { min_val[j] = row[j]; }
                if row[j] > max_val[j] { max_val[j] = row[j]; }
            }
        }

        let max_code = (1u32 << bits as usize) - 1;
        let scale: Vec<f32> = min_val.iter().zip(max_val.iter())
            .map(|(&mn, &mx)| {
                let range = mx - mn;
                if range > 1e-8 { max_code as f32 / range } else { 0.0 }
            }).collect();

        Self { min_val, scale, dim, bits }
    }

    pub fn quantize(&self, vector: &ArrayView1<f32>) -> Vec<u8> {
        let max_code = (1u32 << self.bits as usize) - 1;
        vector.iter().enumerate().map(|(j, &v)| {
            let q = ((v - self.min_val[j]) * self.scale[j]).round() as i32;
            q.clamp(0, max_code as i32) as u8
        }).collect()
    }

    pub fn dequantize(&self, codes: &[u8]) -> Vec<f32> {
        codes.iter().enumerate().map(|(j, &c)| {
            self.min_val[j] + c as f32 / self.scale[j].max(1e-8)
        }).collect()
    }

    pub fn dim(&self) -> usize { self.dim }
}

/// Polar quantizer: quantizes 2D projections in polar coordinates.
pub struct PolarQuantizer {
    projections: Vec<Vec<f32>>, // random projection vectors
    angle_bins: u16,
    magnitude_bits: u8,
    max_magnitude: f32,
    dim: usize,
}

impl PolarQuantizer {
    pub fn new(dim: usize, n_projections: usize, bits: u8, seed: u64) -> Self {
        let mut rng_state = seed;
        let projections: Vec<Vec<f32>> = (0..n_projections).map(|_| {
            let mut proj = Vec::with_capacity(dim);
            for _ in 0..dim {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                proj.push(((rng_state >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0) as f32);
            }
            // Normalize
            let norm: f32 = proj.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-8 { for v in proj.iter_mut() { *v /= norm; } }
            proj
        }).collect();

        Self { projections, angle_bins: (1u16 << bits) - 1, magnitude_bits: bits, max_magnitude: 1.0, dim }
    }

    pub fn quantize(&self, vector: &ArrayView1<f32>) -> Vec<u8> {
        self.projections.iter().map(|proj| {
            let dot: f32 = vector.iter().zip(proj.iter()).map(|(&a, &b)| a * b).sum();
            let code = ((dot + 1.0) * 0.5 * self.angle_bins as f32).round() as u8;
            code.min(self.angle_bins as u8)
        }).collect()
    }

    pub fn dequantize_approx(&self, codes: &[u8]) -> Vec<f32> {
        // Reconstruct approximate projection values
        codes.iter().map(|&c| {
            (c as f32 / self.angle_bins as f32) * 2.0 - 1.0
        }).collect()
    }

    pub fn dim(&self) -> usize { self.dim }
}

/// Compressed vector store using quantization.
pub struct QuantizedStore {
    config: QuantConfig,
    dim: usize,
    scalar: Option<ScalarQuantizer>,
    polar: Option<PolarQuantizer>,
    codes: Vec<Vec<u8>>,
    ids: Vec<u64>,
}

impl QuantizedStore {
    pub fn new(dim: usize, config: QuantConfig) -> Result<Self, String> {
        Ok(Self {
            config: config.clone(),
            dim,
            scalar: None,
            polar: None,
            codes: Vec::new(),
            ids: Vec::new(),
        })
    }

    /// Fit quantizer on data and store all vectors.
    pub fn fit_add(&mut self, vectors: &Array2<f32>, start_id: u64) -> usize {
        let n = vectors.nrows();

        match self.config.algorithm {
            QuantAlgorithm::Scalar => {
                let quantizer = ScalarQuantizer::new(vectors, self.config.bits);
                for i in 0..n {
                    self.codes.push(quantizer.quantize(&vectors.row(i)));
                    self.ids.push(start_id + i as u64);
                }
                self.scalar = Some(quantizer);
            }
            QuantAlgorithm::Polar => {
                let quantizer = PolarQuantizer::new(self.dim, self.config.projections, self.config.bits, self.config.seed);
                for i in 0..n {
                    self.codes.push(quantizer.quantize(&vectors.row(i)));
                    self.ids.push(start_id + i as u64);
                }
                self.polar = Some(quantizer);
            }
        }
        n
    }

    /// Encode and store a batch of vectors.
    pub fn add_batch(&mut self, vectors: &Array2<f32>, start_id: u64) -> usize {
        if self.scalar.is_none() && self.polar.is_none() {
            return self.fit_add(vectors, start_id);
        }

        let n = vectors.nrows();
        if let Some(ref q) = self.scalar {
            for i in 0..n {
                self.codes.push(q.quantize(&vectors.row(i)));
                self.ids.push(start_id + i as u64);
            }
        } else if let Some(ref q) = self.polar {
            for i in 0..n {
                self.codes.push(q.quantize(&vectors.row(i)));
                self.ids.push(start_id + i as u64);
            }
        }
        n
    }

    /// Add a single vector.
    pub fn add_single(&mut self, vector: &ArrayView1<f32>, id: u64) -> bool {
        if let Some(ref q) = self.scalar {
            self.codes.push(q.quantize(vector));
        } else if let Some(ref q) = self.polar {
            self.codes.push(q.quantize(vector));
        } else {
            return false;
        }
        self.ids.push(id);
        true
    }

    /// Search for k nearest neighbors using Hamming-like distance.
    pub fn search(&self, query: &ArrayView1<f32>, k: usize) -> Vec<(u64, f32)> {
        let query_code = if let Some(ref q) = self.scalar {
            q.quantize(query)
        } else if let Some(ref q) = self.polar {
            q.quantize(query)
        } else {
            return Vec::new();
        };

        let n = self.ids.len();
        let k = k.min(n);
        if k == 0 { return Vec::new(); }

        let mut scores: Vec<(usize, f32)> = self.codes.iter().enumerate().map(|(i, code)| {
            // L1 distance in quantized space (approximation)
            let dist: u32 = code.iter().zip(query_code.iter())
                .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u32)
                .sum();
            // Convert distance to similarity score (higher = more similar)
            let max_dist = code.len() as u32 * 255;
            let sim = 1.0 - dist as f32 / max_dist.max(1) as f32;
            (i, sim)
        }).collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        scores.into_iter().map(|(i, score)| (self.ids[i], score)).collect()
    }

    pub fn len(&self) -> usize { self.ids.len() }
    pub fn is_empty(&self) -> bool { self.ids.is_empty() }

    pub fn memory_usage(&self) -> usize {
        let code_size = self.dim * self.config.bits as usize / 8;
        self.ids.len() * (code_size + 8)
    }

    pub fn compression_ratio(&self) -> f32 {
        let raw_bytes = self.dim * 4;
        let quant_bytes = self.dim * self.config.bits as usize / 8;
        raw_bytes as f32 / quant_bytes.max(1) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        Array2::from_shape_fn((n, dim), |(_, _)| rng.gen::<f32>())
    }

    #[test]
    fn test_scalar_quant_roundtrip() {
        let data = random_vectors(50, 32, 42);
        let q = ScalarQuantizer::new(&data, 8);
        for i in 0..5 {
            let codes = q.quantize(&data.row(i));
            let recovered = q.dequantize(&codes);
            // 8-bit quantization should be close
            for j in 0..32 {
                assert!((data[[i, j]] - recovered[j]).abs() < 0.05,
                    "Roundtrip error too large at [{}, {}]", i, j);
            }
        }
    }

    #[test]
    fn test_add_single() {
        let data = random_vectors(10, 16, 42);
        let config = QuantConfig::for_search(16);
        let mut store = QuantizedStore::new(16, config).unwrap();
        store.fit_add(&data, 0);
        assert_eq!(store.len(), 10);
        let extra = Array2::from_shape_vec((1, 16), (0..16).map(|i| i as f32).collect()).unwrap();
        assert!(store.add_single(&extra.row(0), 10));
        assert_eq!(store.len(), 11);
    }

    #[test]
    fn test_search() {
        let data = random_vectors(100, 64, 42);
        let config = QuantConfig::for_search(64);
        let mut store = QuantizedStore::new(64, config).unwrap();
        store.fit_add(&data, 0);
        let results = store.search(&data.row(0), 5);
        assert_eq!(results.len(), 5);
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn test_compression_ratio() {
        let config = QuantConfig::for_search(64);
        let store = QuantizedStore::new(64, config).unwrap();
        assert!(store.compression_ratio() > 1.0);
    }

    #[test]
    fn test_polar_quant() {
        let data = random_vectors(50, 32, 42);
        let config = QuantConfig::for_max_compression(32);
        let mut store = QuantizedStore::new(32, config).unwrap();
        store.fit_add(&data, 0);
        assert_eq!(store.len(), 50);
        let results = store.search(&data.row(0), 5);
        assert_eq!(results.len(), 5);
    }
}
