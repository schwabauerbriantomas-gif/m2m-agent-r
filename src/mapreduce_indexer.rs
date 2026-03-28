//! MapReduce Indexer — distributed indexing framework.
//! Ported from m2m-vector-search Python.

use serde::Serialize;

/// Map phase output: (shard_id, vectors).
pub type MapOutput = Vec<(usize, Vec<f32>)>;

/// Reduce phase output: (index_id, stats).
#[derive(Debug, Clone, Serialize)]
pub struct ReduceOutput {
    pub index_id: String,
    pub total_vectors: usize,
    pub total_shards: usize,
    pub time_ms: f64,
}

/// MapReduce indexer configuration.
#[derive(Debug, Clone)]
pub struct MapReduceConfig {
    pub shard_size: usize,
    pub n_workers: usize,
    pub overlap: usize,
}

impl Default for MapReduceConfig {
    fn default() -> Self {
        Self { shard_size: 10_000, n_workers: 4, overlap: 0 }
    }
}

/// MapReduce indexer for M2M.
pub struct MapReduceIndexer {
    config: MapReduceConfig,
}

impl MapReduceIndexer {
    pub fn new(config: MapReduceConfig) -> Self { Self { config } }

    /// Map phase: split data into shards.
    pub fn map(&self, data: &[(String, Vec<f32>)]) -> Vec<MapOutput> {
        let n = data.len();
        let shard_size = self.config.shard_size;
        let n_shards = (n + shard_size - 1) / shard_size;

        (0..n_shards)
            .map(|shard_id| {
                let start = shard_id * shard_size;
                let end = (start + shard_size).min(n);
                (start..end)
                    .map(|i| (i - start, data[i].1.clone()))
                    .collect()
            })
            .collect()
    }

    /// Reduce phase: merge shard results.
    pub fn reduce(&self, shard_outputs: &[usize]) -> ReduceOutput {
        let total: usize = shard_outputs.iter().sum();
        ReduceOutput {
            index_id: format!("mr_{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()),
            total_vectors: total,
            total_shards: shard_outputs.len(),
            time_ms: 0.0,
        }
    }

    /// Execute full map-reduce pipeline.
    pub fn execute(&self, data: &[(String, Vec<f32>)]) -> ReduceOutput {
        let shards = self.map(data);
        let counts: Vec<usize> = shards.iter().map(|s| s.len()).collect();
        self.reduce(&counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_reduce() {
        let indexer = MapReduceIndexer::new(MapReduceConfig { shard_size: 3, n_workers: 2, overlap: 0 });
        let data: Vec<(String, Vec<f32>)> = (0..10)
            .map(|i| (format!("doc_{}", i), vec![i as f32; 4]))
            .collect();
        let result = indexer.execute(&data);
        assert_eq!(result.total_vectors, 10);
        assert_eq!(result.total_shards, 4); // ceil(10/3)
    }
}
