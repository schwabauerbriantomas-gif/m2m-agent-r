//! Embedding evaluation metrics.
//!
//! Computes precision, recall, MRR, NDCG, and embedding quality metrics.
//! Ported from m2m-vector-search Python.

use ndarray::Array2;
use serde::Serialize;

/// Embedding evaluation results.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingEvalReport {
    pub precision_at_1: f64,
    pub precision_at_5: f64,
    pub precision_at_10: f64,
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr: f64,
    pub ndcg_at_10: f64,
    pub avg_similarity: f64,
    pub embedding_norm_std: f64,
    pub n_queries: usize,
}

/// Evaluate embeddings quality.
pub struct EmbeddingEvaluator;

impl EmbeddingEvaluator {
    /// Evaluate retrieval quality given embeddings and labels.
    ///
    /// # Arguments
    /// * `embeddings` - Matrix [N, D] of embeddings
    /// * `labels` - Labels for each embedding
    /// * `n_queries` - Number of queries to evaluate (sampled from start)
    pub fn evaluate(embeddings: &Array2<f32>, labels: &[String], n_queries: usize) -> EmbeddingEvalReport {
        let n = embeddings.nrows();
        let actual_queries = n_queries.min(n);

        let mut p1_sum = 0.0;
        let mut p5_sum = 0.0;
        let mut p10_sum = 0.0;
        let mut r1_sum = 0.0;
        let mut r5_sum = 0.0;
        let mut r10_sum = 0.0;
        let mut mrr_sum = 0.0;
        let mut ndcg_sum = 0.0;
        let mut sim_sum = 0.0;
        let mut sim_count = 0;

        for i in 0..actual_queries {
            let query = embeddings.row(i);
            let query_label = &labels[i];

            // Compute cosine similarities
            let mut sims: Vec<(usize, f64)> = Vec::with_capacity(n);
            for j in 0..n {
                if j == i { continue; }
                let other = embeddings.row(j);
                let dot: f32 = query.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
                let norm_q: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_o: f32 = other.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sim = if norm_q > 0.0 && norm_o > 0.0 {
                    (dot / (norm_q * norm_o)) as f64
                } else {
                    0.0
                };
                sims.push((j, sim));
                sim_sum += sim;
                sim_count += 1;
            }

            // Sort by similarity descending
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Find how many have the same label
            let total_relevant: usize = labels.iter().filter(|l| **l == *query_label).count();
            if total_relevant == 0 { continue; }
            let relevant_minus_self = total_relevant.saturating_sub(1);

            // Compute metrics
            for (rank, (idx, _)) in sims.iter().enumerate() {
                let rank_0 = rank; // 0-indexed
                if labels[*idx] == *query_label {
                    if rank_0 == 0 { p1_sum += 1.0; }
                    if rank_0 < 5 { p5_sum += 1.0; }
                    if rank_0 < 10 { p10_sum += 1.0; }
                    if rank_0 == 0 { r1_sum += 1.0; }
                    if rank_0 < 5 { r5_sum += 1.0; }
                    if rank_0 < 10 { r10_sum += 1.0; }

                    // MRR
                    mrr_sum += 1.0 / (rank_0 + 1) as f64;

                    break; // Only count first relevant for MRR
                }
            }

            // NDCG@10
            let mut dcg = 0.0;
            let mut idcg = 0.0;
            let k10 = sims.iter().take(10);
            for (rank, (idx, _)) in k10.enumerate() {
                let rel = if labels[*idx].as_str() == *query_label { 1.0 } else { 0.0 };
                dcg += rel / (1.0 + rank as f64).ln();
            }
            // Ideal DCG
            let ideal_rels = std::cmp::min(10, relevant_minus_self);
            for i in 0..ideal_rels {
                idcg += 1.0 / (1.0 + i as f64).ln();
            }
            if idcg > 0.0 {
                ndcg_sum += dcg / idcg;
            } else {
                ndcg_sum = dcg; // if no relevant items, DCG is correct
            }
        }

        // Embedding norm statistics
        let mut norms = Vec::with_capacity(n);
        for i in 0..n {
            let row = embeddings.row(i);
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            norms.push(norm as f64);
        }
        let norm_mean = norms.iter().sum::<f64>() / norms.len() as f64;
        let norm_var: f64 = norms.iter().map(|n| (n - norm_mean).powi(2)).sum::<f64>() / norms.len() as f64;
        let norm_std = norm_var.sqrt();

        let avg_sim = if sim_count > 0 { sim_sum / sim_count as f64 } else { 0.0 };

        EmbeddingEvalReport {
            precision_at_1: if actual_queries > 0 { p1_sum / actual_queries as f64 } else { 0.0 },
            precision_at_5: if actual_queries > 0 { p5_sum / (actual_queries as f64 * 5.0).min(actual_queries as f64) } else { 0.0 },
            precision_at_10: if actual_queries > 0 { p10_sum / (actual_queries as f64 * 10.0).min(actual_queries as f64) } else { 0.0 },
            recall_at_1: if actual_queries > 0 { r1_sum / actual_queries as f64 } else { 0.0 },
            recall_at_5: if actual_queries > 0 { r5_sum / actual_queries as f64 } else { 0.0 },
            recall_at_10: if actual_queries > 0 { r10_sum / actual_queries as f64 } else { 0.0 },
            mrr: if actual_queries > 0 { mrr_sum / actual_queries as f64 } else { 0.0 },
            ndcg_at_10: if actual_queries > 0 { ndcg_sum / actual_queries as f64 } else { 0.0 },
            avg_similarity: avg_sim,
            embedding_norm_std: norm_std,
            n_queries: actual_queries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_clusters() {
        // 10 embeddings: 5 similar, 5 different
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for i in 0..5 {
            data.extend([1.0, 0.0, 0.0]); // cluster A
            labels.push("A".to_string());
        }
        for i in 0..5 {
            data.extend([0.0, 1.0, 0.0]); // cluster B
            labels.push("B".to_string());
        }
        let emb = Array2::from_shape_vec((10, 3), data).unwrap();

        let report = EmbeddingEvaluator::evaluate(&emb, &labels, 5);
        assert!(report.precision_at_1 > 0.0);
        assert!(report.avg_similarity >= 0.0);
        assert_eq!(report.n_queries, 5);
    }

    #[test]
    fn test_norm_std() {
        let data = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let emb = Array2::from_shape_vec((2, 3), data).unwrap();
        let labels = vec!["A".into(), "B".into()];
        let report = EmbeddingEvaluator::evaluate(&emb, &labels, 2);
        // All normalized vectors, std should be small
        assert!(report.embedding_norm_std < 0.1);
    }
}
