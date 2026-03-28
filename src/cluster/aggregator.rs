//! Result aggregator for distributed search. Ported from m2m-vector-search Python.

/// Aggregates search results from multiple nodes.
pub struct ResultAggregator;

impl ResultAggregator {
    /// Merge results from multiple nodes using score-based ranking.
    pub fn merge_results(results: &[Vec<(usize, f32)>], k: usize) -> Vec<(usize, f64)> {
        let mut all: Vec<(usize, f64)> = Vec::new();
        for node_results in results {
            for (idx, score) in node_results {
                all.push((*idx, *score as f64));
            }
        }
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.dedup_by(|a, b| a.0 == b.0);
        all.truncate(k);
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge() {
        let results = vec![
            vec![(1, 0.9), (2, 0.8)],
            vec![(3, 0.95), (1, 0.85)],
        ];
        let merged = ResultAggregator::merge_results(&results, 5);
        assert_eq!(merged[0].0, 3); // highest score
        assert_eq!(merged.len(), 3); // deduplicated
    }
}
