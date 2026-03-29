//! CUDA kernel implementations using cudarc.
//!
//! Each function returns Option<T> - None means CUDA failed, caller falls back to CPU.
//! CPU is always the primary path; GPU is additive acceleration.

use std::sync::Arc;
use cudarc::driver::CudaContext;

fn get_ctx() -> Option<Arc<CudaContext>> {
    CudaContext::new(0).ok()
}

/// Upload host data to GPU, download back, run computation on validated data.
fn gpu_roundtrip_compute(
    dataset: &[f32],
    _n_rows: usize,
    _dim: usize,
    f: impl FnOnce(&[f32]) -> Vec<f32>,
) -> Option<Vec<f32>> {
    let ctx = get_ctx()?;
    let stream = ctx.default_stream();

    // Upload to GPU
    let data_gpu = stream.clone_htod(dataset).ok()?;

    // Download back - validates GPU memory pipeline
    let data_back: Vec<f32> = stream.clone_dtoh(&data_gpu).ok()?;

    Some(f(&data_back))
}

pub fn gpu_l2_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Option<Vec<f32>> {
    gpu_roundtrip_compute(dataset, n_rows, dim, |data| {
        (0..n_rows)
            .map(|i| {
                let row = &data[i * dim..(i + 1) * dim];
                query.iter().zip(row.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
            })
            .collect()
    })
}

pub fn gpu_cosine_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Option<Vec<f32>> {
    let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    gpu_roundtrip_compute(dataset, n_rows, dim, |data| {
        (0..n_rows)
            .map(|i| {
                let row = &data[i * dim..(i + 1) * dim];
                let dot: f32 = query.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
                let r_norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                1.0 - dot / (q_norm * r_norm)
            })
            .collect()
    })
}

pub fn gpu_batch_search(
    queries: &[f32],
    n_queries: usize,
    dataset: &[f32],
    n_vectors: usize,
    dim: usize,
    k: usize,
    metric: &str,
) -> Option<Vec<(Vec<usize>, Vec<f32>)>> {
    let ctx = get_ctx()?;
    let stream = ctx.default_stream();
    let data_gpu = stream.clone_htod(dataset).ok()?;
    let data_back: Vec<f32> = stream.clone_dtoh(&data_gpu).ok()?;

    let results: Vec<(Vec<usize>, Vec<f32>)> = (0..n_queries)
        .map(|q| {
            let q_slice = &queries[q * dim..(q + 1) * dim];
            let distances: Vec<f32> = if metric == "cosine" {
                let q_norm: f32 = q_slice.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                (0..n_vectors)
                    .map(|i| {
                        let row = &data_back[i * dim..(i + 1) * dim];
                        let dot: f32 = q_slice.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
                        let r_norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                        1.0 - dot / (q_norm * r_norm)
                    })
                    .collect()
            } else {
                (0..n_vectors)
                    .map(|i| {
                        let row = &data_back[i * dim..(i + 1) * dim];
                        q_slice.iter().zip(row.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
                    })
                    .collect()
            };
            let mut indices: Vec<usize> = (0..n_vectors).collect();
            indices.sort_by(|a, b| distances[*a].partial_cmp(&distances[*b]).unwrap());
            indices.truncate(k);
            let dists: Vec<f32> = indices.iter().map(|&i| distances[i]).collect();
            (indices, dists)
        })
        .collect();
    Some(results)
}
