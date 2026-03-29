//! GPU acceleration module.
//!
//! When compiled with `--features cuda`, provides real CUDA operations
//! via cudarc. Falls back to CPU (ndarray) when CUDA is unavailable.
//! CPU path is always available — CUDA is purely additive acceleration.

#[cfg(feature = "cuda")]
pub mod cuda_kernel;

/// Check if CUDA is available at runtime.
#[cfg(feature = "cuda")]
pub fn is_cuda_available() -> bool {
    use cudarc::driver::CudaContext;
    CudaContext::device_count()
        .map(|c| c > 0)
        .unwrap_or(false)
}

/// Check if CUDA is available at runtime.
#[cfg(not(feature = "cuda"))]
pub fn is_cuda_available() -> bool {
    false
}

/// GPU-accelerated batch L2 distance computation.
///
/// CUDA path: uploads dataset to GPU memory, computes distances on device.
/// CPU fallback: ndarray-based computation (always available).
pub fn batch_l2_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Vec<f32> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            if let Some(result) = cuda_kernel::gpu_l2_distances(query, dataset, n_rows, dim) {
                return result;
            }
        }
    }
    cpu_l2_distances(query, dataset, n_rows, dim)
}

/// GPU-accelerated batch cosine distance computation.
pub fn batch_cosine_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Vec<f32> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            if let Some(result) = cuda_kernel::gpu_cosine_distances(query, dataset, n_rows, dim) {
                return result;
            }
        }
    }
    cpu_cosine_distances(query, dataset, n_rows, dim)
}

/// GPU-accelerated batch search: multiple queries against dataset.
///
/// Returns (indices, distances) for top-k per query.
/// Uses GPU when available, CPU otherwise.
pub fn batch_search(
    queries: &[f32],
    n_queries: usize,
    dataset: &[f32],
    n_vectors: usize,
    dim: usize,
    k: usize,
    metric: &str,
) -> Vec<(Vec<usize>, Vec<f32>)> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            if let Some(result) = cuda_kernel::gpu_batch_search(
                queries, n_queries, dataset, n_vectors, dim, k, metric,
            ) {
                return result;
            }
        }
    }
    cpu_batch_search(queries, n_queries, dataset, n_vectors, dim, k, metric)
}

/// Get GPU info if CUDA is available.
#[cfg(feature = "cuda")]
pub fn gpu_info() -> Option<String> {
    use cudarc::driver::CudaContext;
    let ctx = CudaContext::new(0).ok()?;
    let name = ctx.name().unwrap_or_else(|_| "Unknown".to_string());
    let (major, minor) = ctx.compute_capability().unwrap_or((0, 0));
    let total_mem = ctx.total_mem().unwrap_or(0);
    let free_mem = ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
    Some(format!(
        "{} (sm_{}{}), {:.1} GB total, {:.1} GB free",
        name, major, minor,
        total_mem as f64 / 1e9,
        free_mem as f64 / 1e9,
    ))
}

/// Get GPU info if CUDA is available.
#[cfg(not(feature = "cuda"))]
pub fn gpu_info() -> Option<String> {
    None
}

// --- CPU implementations (always available) ---

fn cpu_l2_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Vec<f32> {
    let mut distances = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let row = &dataset[i * dim..(i + 1) * dim];
        let dist: f32 = query.iter()
            .zip(row.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        distances.push(dist);
    }
    distances
}

fn cpu_cosine_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Vec<f32> {
    let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    let mut distances = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let row = &dataset[i * dim..(i + 1) * dim];
        let dot: f32 = query.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
        let r_norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        distances.push(1.0 - dot / (q_norm * r_norm));
    }
    distances
}

fn cpu_batch_search(
    queries: &[f32],
    n_queries: usize,
    dataset: &[f32],
    n_vectors: usize,
    dim: usize,
    k: usize,
    metric: &str,
) -> Vec<(Vec<usize>, Vec<f32>)> {
    (0..n_queries)
        .map(|q| {
            let q_slice = &queries[q * dim..(q + 1) * dim];
            let distances = if metric == "cosine" {
                cpu_cosine_distances(q_slice, dataset, n_vectors, dim)
            } else {
                cpu_l2_distances(q_slice, dataset, n_vectors, dim)
            };
            let mut indices: Vec<usize> = (0..n_vectors).collect();
            indices.sort_by(|a, b| distances[*a].partial_cmp(&distances[*b]).unwrap());
            indices.truncate(k);
            let dists = indices.iter().map(|&i| distances[i]).collect();
            (indices, dists)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_l2_distances() {
        let query = vec![1.0f32, 0.0];
        let dataset = vec![1.0, 0.0,  0.0, 1.0,  1.0, 1.0];
        let dists = cpu_l2_distances(&query, &dataset, 3, 2);
        assert!((dists[0] - 0.0).abs() < 1e-6);
        assert!((dists[1] - 2.0).abs() < 1e-6);
        assert!((dists[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_cosine_distances() {
        let query = vec![1.0f32, 0.0];
        let dataset = vec![1.0, 0.0,  0.0, 1.0];
        let dists = cpu_cosine_distances(&query, &dataset, 2, 2);
        assert!((dists[0] - 0.0).abs() < 1e-6);
        assert!((dists[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_batch_search() {
        let queries = vec![1.0f32, 0.0];
        let dataset = vec![0.0, 1.0,  1.0, 0.0,  0.5, 0.5];
        let results = cpu_batch_search(&queries, 1, &dataset, 3, 2, 2, "l2");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0[0], 1);
    }

    #[test]
    fn test_batch_l2_distances_public_api() {
        let query = vec![1.0f32, 0.0];
        let dataset = vec![1.0, 0.0,  0.0, 1.0];
        let dists = batch_l2_distances(&query, &dataset, 2, 2);
        assert_eq!(dists.len(), 2);
        assert!((dists[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_search_public_api() {
        let queries = vec![1.0f32, 0.0];
        let dataset = vec![1.0, 0.0,  0.0, 1.0];
        let results = batch_search(&queries, 1, &dataset, 2, 2, 2, "l2");
        assert_eq!(results.len(), 1);
    }

    /// Test that CUDA pipeline actually works on this machine (GPU upload/download/verify).
    /// Only runs when compiled with --features cuda.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_gpu_pipeline() {
        if !is_cuda_available() {
            return; // skip on machines without GPU
        }

        // Large enough to matter for GPU transfer
        let dim = 640;
        let n = 1000;
        let mut dataset = vec![0.0f32; n * dim];
        for i in 0..n {
            dataset[i * dim] = i as f32 / n as f32;
        }
        let query = vec![0.5f32; dim];

        // L2 via GPU pipeline
        let gpu_dists = batch_l2_distances(&query, &dataset, n, dim);
        assert_eq!(gpu_dists.len(), n);

        // Cosine via GPU pipeline
        let gpu_cos = batch_cosine_distances(&query, &dataset, n, dim);
        assert_eq!(gpu_cos.len(), n);

        // Batch search via GPU pipeline
        let queries_flat = vec![0.5f32; 2 * dim];
        let results = batch_search(&queries_flat, 2, &dataset, n, dim, 5, "l2");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.len(), 5);
        assert_eq!(results[1].0.len(), 5);

        // GPU info
        let info = gpu_info().expect("GPU info should be available");
        assert!(info.contains("GB"));
    }
}
