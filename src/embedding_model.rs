//! Embedding model wrapper for M2M.
//!
//! Supports external encoders via trait. Concrete implementations
//! would use ONNX Runtime, candle, or HTTP API backends.
//! Ported from m2m-vector-search Python.

use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Encoder trait — implement for any embedding model.
pub trait Encoder: Send + Sync {
    /// Encode a single text, returning embedding vector.
    fn encode_one(&self, text: &str) -> Result<Array1<f32>, Box<dyn std::error::Error + Send + Sync>>;

    /// Encode multiple texts, returning embedding matrix [N, D].
    fn encode_batch(&self, texts: &[&str]) -> Result<Array2<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let mut rows = Vec::with_capacity(texts.len());
        for text in texts {
            rows.push(self.encode_one(text)?);
        }
        let dim = rows[0].len();
        let mut data = Vec::with_capacity(texts.len() * dim);
        for row in &rows {
            data.extend_from_slice(row.as_slice().unwrap());
        }
        Array2::from_shape_vec((texts.len(), dim), data)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    /// Get the embedding dimension.
    fn dim(&self) -> usize;

    /// Get the model name.
    fn model_name(&self) -> &str;
}

/// Simple hash-based encoder for testing (deterministic, no model needed).
pub struct HashEncoder {
    dim: usize,
}

impl HashEncoder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Encoder for HashEncoder {
    fn encode_one(&self, text: &str) -> Result<Array1<f32>, Box<dyn std::error::Error + Send + Sync>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut data = vec![0.0f32; self.dim];
        let bytes = text.as_bytes();

        for i in 0..self.dim {
            let mut hasher = DefaultHasher::new();
            (i, bytes).hash(&mut hasher);
            let h = hasher.finish();
            // Map hash to [-1, 1] range
            data[i] = ((h % 10000) as f32) / 5000.0 - 1.0;
        }

        // Normalize
        let norm: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in data.iter_mut() {
                *v /= norm;
            }
        }

        Ok(Array1::from_vec(data))
    }

    fn dim(&self) -> usize { self.dim }
    fn model_name(&self) -> &str { "hash-encoder" }
}

/// Encoder registry for managing multiple models.
pub struct EncoderRegistry {
    encoders: HashMap<String, Box<dyn Encoder>>,
    default: Option<String>,
}

impl EncoderRegistry {
    pub fn new() -> Self {
        Self {
            encoders: HashMap::new(),
            default: None,
        }
    }

    /// Register an encoder with a name.
    pub fn register(&mut self, name: &str, encoder: Box<dyn Encoder>, set_default: bool) {
        if set_default {
            self.default = Some(name.to_string());
        }
        self.encoders.insert(name.to_string(), encoder);
    }

    /// Get encoder by name.
    pub fn get(&self, name: &str) -> Option<&dyn Encoder> {
        self.encoders.get(name).map(|e| e.as_ref())
    }

    /// Get the default encoder.
    pub fn get_default(&self) -> Option<&dyn Encoder> {
        self.default.as_ref().and_then(|name| self.get(name))
    }

    /// List registered encoder names.
    pub fn list(&self) -> Vec<&str> {
        self.encoders.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for EncoderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_encoder() {
        let enc = HashEncoder::new(64);
        let v = enc.encode_one("hello world").unwrap();
        assert_eq!(v.len(), 64);
        // Verify normalized
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hash_encoder_deterministic() {
        let enc = HashEncoder::new(32);
        let v1 = enc.encode_one("test").unwrap();
        let v2 = enc.encode_one("test").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_hash_encoder_different() {
        let enc = HashEncoder::new(32);
        let v1 = enc.encode_one("hello").unwrap();
        let v2 = enc.encode_one("world").unwrap();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_encode_batch() {
        let enc = HashEncoder::new(16);
        let texts = vec!["hello", "world", "test"];
        let batch = enc.encode_batch(&texts).unwrap();
        assert_eq!(batch.nrows(), 3);
        assert_eq!(batch.ncols(), 16);
    }

    #[test]
    fn test_registry() {
        let mut reg = EncoderRegistry::new();
        reg.register("hash64", Box::new(HashEncoder::new(64)), true);
        reg.register("hash128", Box::new(HashEncoder::new(128)), false);

        assert_eq!(reg.list().len(), 2);
        let default = reg.get_default().unwrap();
        assert_eq!(default.dim(), 64);
    }
}
