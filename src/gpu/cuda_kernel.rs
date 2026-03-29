//! CUDA kernel implementations using cudarc + custom PTX kernels.
//!
//! Architecture:
//! - Custom PTX kernels (distance.cu) compiled at build time by build.rs/nvcc
//! - cudarc loads PTX module and launches kernels
//! - Dataset persists in GPU memory between calls (no re-upload)
//! - CPU fallback always available when CUDA fails
//!
//! Performance optimizations:
//! - float4 vectorized loads for D divisible by 4
//! - Shared memory caching for query vectors
//! - __launch_bounds__(256) for optimal occupancy on sm_86
//! - --use_fast_math and -O3 during PTX compilation

use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaSlice, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

/// Persistent GPU state — keeps dataset in VRAM between calls.
pub struct GpuIndex {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    data_gpu: Option<CudaSlice<f32>>,
    n_vectors: usize,
    dim: usize,
    kernel_loaded: bool,
}

impl GpuIndex {
    /// Create a new GPU index. Returns None if CUDA is unavailable.
    pub fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        Some(GpuIndex {
            ctx,
            stream,
            data_gpu: None,
            n_vectors: 0,
            dim: 0,
            kernel_loaded: false,
        })
    }

    /// Upload dataset to GPU memory (persists until next upload or drop).
    pub fn upload_dataset(&mut self, dataset: &[f32], n_vectors: usize, dim: usize) -> bool {
        self.data_gpu = self.stream.clone_htod(dataset).ok();
        if self.data_gpu.is_some() {
            self.n_vectors = n_vectors;
            self.dim = dim;
            self.kernel_loaded = self.try_load_kernels();
            true
        } else {
            false
        }
    }

    /// Try to load PTX kernels compiled at build time.
    fn try_load_kernels(&self) -> bool {
        let ptx_path = option_env!("M2M_PTX_PATH");
        match ptx_path {
            Some(path) if !path.is_empty() => {
                // Verify PTX file actually exists
                let exists = std::path::Path::new(path).exists();
                if exists {
                    eprintln!("[m2m] PTX kernels available: {}", path);
                } else {
                    eprintln!("[m2m] PTX path set but file not found: {}", path);
                }
                exists
            }
            _ => {
                eprintln!("[m2m] No PTX kernels (built without nvcc or CUDA toolkit)");
                false
            }
        }
    }

    /// L2 distances: query vs all vectors. Returns distances or None.
    pub fn l2_distances(&self, query: &[f32]) -> Option<Vec<f32>> {
        let n = self.n_vectors;
        let dim = self.dim;
        let data_gpu = self.data_gpu.as_ref()?;

        // Try PTX kernel path first
        if let Some(result) = self.launch_l2_kernel(query, data_gpu, n, dim) {
            return Some(result);
        }

        // Fallback: upload query, download dataset, compute on CPU
        let data_back: Vec<f32> = self.stream.clone_dtoh(data_gpu).ok()?;
        let mut distances = Vec::with_capacity(n);
        for i in 0..n {
            let row = &data_back[i * dim..(i + 1) * dim];
            let dist: f32 = query.iter().zip(row.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            distances.push(dist);
        }
        Some(distances)
    }

    /// Launch custom L2 distance PTX kernel.
    fn launch_l2_kernel(&self, query: &[f32], data_gpu: &CudaSlice<f32>, n: usize, dim: usize) -> Option<Vec<f32>> {
        let ptx_path = option_env!("M2M_PTX_PATH")?;
        if ptx_path.is_empty() {
            return None;
        }

        // Load PTX module
        let module = self.ctx.load_module(Ptx::from_file(ptx_path)).ok()?;
        let func: CudaFunction = module.load_function("l2_distance_kernel").ok()?;

        // Allocate GPU buffers
        let query_gpu = self.stream.clone_htod(query).ok()?;
        let mut output_gpu = self.stream.alloc_zeros::<f32>(n).ok()?;

        // Shared memory = dim * sizeof(float) for query cache
        let shared_mem_bytes = dim * 4;

        // Launch configuration: 256 threads per block
        let block_dim = 256;
        let grid_dim = (n + block_dim - 1) / block_dim;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim as u32, 1, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        // Launch kernel
        // cudarc launch: params are (func, cfg, &mut output_gpu, ..., n, dim)
        unsafe {
            self.stream.launch_builder(&func)
                .arg(&query_gpu)
                .arg(data_gpu)
                .arg(&mut output_gpu)
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .launch(cfg)
                .ok()?;
        }

        // Download results
        self.stream.clone_dtoh(&output_gpu).ok()
    }

    /// Batch L2 distances: multiple queries vs all vectors.
    pub fn batch_l2_distances(&self, queries: &[f32], n_queries: usize) -> Option<Vec<Vec<f32>>> {
        let n = self.n_vectors;
        let dim = self.dim;
        let data_gpu = self.data_gpu.as_ref()?;

        // Try batch kernel
        if let Some(result) = self.launch_batch_l2_kernel(queries, data_gpu, n_queries, n, dim) {
            return Some(result);
        }

        // Fallback: per-query
        let data_back: Vec<f32> = self.stream.clone_dtoh(data_gpu).ok()?;
        let mut all_dists = Vec::with_capacity(n_queries);
        for q in 0..n_queries {
            let q_slice = &queries[q * dim..(q + 1) * dim];
            let mut distances = Vec::with_capacity(n);
            for i in 0..n {
                let row = &data_back[i * dim..(i + 1) * dim];
                let dist: f32 = q_slice.iter().zip(row.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                distances.push(dist);
            }
            all_dists.push(distances);
        }
        Some(all_dists)
    }

    fn launch_batch_l2_kernel(&self, queries: &[f32], data_gpu: &CudaSlice<f32>, n_queries: usize, n: usize, dim: usize) -> Option<Vec<Vec<f32>>> {
        let ptx_path = option_env!("M2M_PTX_PATH")?;
        if ptx_path.is_empty() {
            return None;
        }

        let module = self.ctx.load_module(Ptx::from_file(ptx_path)).ok()?;
        let func: CudaFunction = module.load_function("batch_l2_distance_kernel").ok()?;

        let queries_gpu = self.stream.clone_htod(queries).ok()?;
        let mut output_gpu = self.stream.alloc_zeros::<f32>(n_queries * n).ok()?;

        let block_dim = 256;
        let grid_x = ((n + block_dim - 1) / block_dim) as u32;
        let grid_y = n_queries as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream.launch_builder(&func)
                .arg(&queries_gpu)
                .arg(data_gpu)
                .arg(&mut output_gpu)
                .arg(&(n_queries as i32))
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .launch(cfg)
                .ok()?;
        }

        let flat: Vec<f32> = self.stream.clone_dtoh(&output_gpu).ok()?;
        // De-interleave: output is [Q, N] row-major
        let result: Vec<Vec<f32>> = (0..n_queries)
            .map(|q| flat[q * n..(q + 1) * n].to_vec())
            .collect();
        Some(result)
    }

    /// Cosine distances: query vs all vectors.
    pub fn cosine_distances(&self, query: &[f32]) -> Option<Vec<f32>> {
        let n = self.n_vectors;
        let dim = self.dim;
        let data_gpu = self.data_gpu.as_ref()?;

        // Try PTX kernel
        if let Some(result) = self.launch_cosine_kernel(query, data_gpu, n, dim) {
            return Some(result);
        }

        // Fallback
        let data_back: Vec<f32> = self.stream.clone_dtoh(data_gpu).ok()?;
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        let distances: Vec<f32> = (0..n)
            .map(|i| {
                let row = &data_back[i * dim..(i + 1) * dim];
                let dot: f32 = query.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
                let r_norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                1.0 - dot / (q_norm * r_norm)
            })
            .collect();
        Some(distances)
    }

    fn launch_cosine_kernel(&self, query: &[f32], data_gpu: &CudaSlice<f32>, n: usize, dim: usize) -> Option<Vec<f32>> {
        let ptx_path = option_env!("M2M_PTX_PATH")?;
        if ptx_path.is_empty() {
            return None;
        }

        let module = self.ctx.load_module(Ptx::from_file(ptx_path)).ok()?;
        let func: CudaFunction = module.load_function("cosine_distance_kernel").ok()?;

        let query_gpu = self.stream.clone_htod(query).ok()?;
        let mut output_gpu = self.stream.alloc_zeros::<f32>(n).ok()?;

        let shared_mem_bytes = dim * 4 + 256 * 4; // query + reduction workspace
        let block_dim = 256;
        let grid_dim = (n + block_dim - 1) / block_dim;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim as u32, 1, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        unsafe {
            self.stream.launch_builder(&func)
                .arg(&query_gpu)
                .arg(data_gpu)
                .arg(&mut output_gpu)
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .launch(cfg)
                .ok()?;
        }

        self.stream.clone_dtoh(&output_gpu).ok()
    }

    pub fn n_vectors(&self) -> usize { self.n_vectors }
    pub fn dim(&self) -> usize { self.dim }
    pub fn is_loaded(&self) -> bool { self.data_gpu.is_some() }
}

// ============================================================================
// Public API — wraps GpuIndex for the gpu module
// ============================================================================

/// GPU L2 distance computation with persistent dataset.
pub fn gpu_l2_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Option<Vec<f32>> {
    let mut idx = GpuIndex::new()?;
    idx.upload_dataset(dataset, n_rows, dim);
    idx.l2_distances(query)
}

/// GPU cosine distance computation.
pub fn gpu_cosine_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Option<Vec<f32>> {
    let mut idx = GpuIndex::new()?;
    idx.upload_dataset(dataset, n_rows, dim);
    idx.cosine_distances(query)
}

/// GPU batch search — uses batch kernel when available.
pub fn gpu_batch_search(
    queries: &[f32],
    n_queries: usize,
    dataset: &[f32],
    n_vectors: usize,
    dim: usize,
    k: usize,
    metric: &str,
) -> Option<Vec<(Vec<usize>, Vec<f32>)>> {
    let mut idx = GpuIndex::new()?;
    idx.upload_dataset(dataset, n_vectors, dim);

    // Batch distance computation
    let all_dists = if metric == "cosine" {
        // Use per-query cosine (no batch cosine kernel for simplicity)
        (0..n_queries)
            .map(|q| idx.cosine_distances(&queries[q * dim..(q + 1) * dim]))
            .collect::<Option<Vec<Vec<f32>>>>()?
    } else {
        idx.batch_l2_distances(queries, n_queries)?
    };

    // Top-k selection on CPU (fast for small k)
    let results: Vec<(Vec<usize>, Vec<f32>)> = all_dists.iter().map(|distances| {
        let mut indices: Vec<usize> = (0..distances.len()).collect();
        indices.sort_by(|a, b| distances[*a].partial_cmp(&distances[*b]).unwrap());
        indices.truncate(k);
        let dists: Vec<f32> = indices.iter().map(|&i| distances[i]).collect();
        (indices, dists)
    }).collect();

    Some(results)
}
