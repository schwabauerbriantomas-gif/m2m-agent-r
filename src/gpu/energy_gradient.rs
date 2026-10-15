//! Fused energy+gradient GPU wrapper (cudarc 0.19) — integrado del swarm de
//! kernels (A02, 2026-10-15: 360x e2e vs CPU, max_abs_diff 1.2e-7, B=256/N=50k/D=64).
//!
//! NVRTC runtime-compile con fallback a PTX precompilado (kernels/energy_gradient.ptx,
//! committeado, generado por nvcc --ptx -arch=sm_86 --use_fast_math). cudarc 0.19.10
//! PANICA si falta libnvrtc => catch_unwind (lección del swarm, ver INFORME_FINAL.md).
//!
//! Mirrors the pattern of wt_valid/src/gpu/cuda_kernel.rs but compiles the
//! kernel source at runtime with NVRTC (compile_ptx -> Ptx), as the BRIEF
//! prescribes for new kernels. Splats (mu/alpha/kappa) persist in VRAM;
//! only x[B,D] is uploaded per call, only E[B] + G[B,D] come back.
//!
//! Correctness contract vs src/ebm/energy_api.rs::energy_gradient:
//!   w_i = alpha_i * exp(-kappa_i * ||x-mu_i||^2),  Z = sum w_i
//!   G_d = sum_i 2*kappa_i*w_i*(x_d - mu_i_d) / Z   (0 if Z <= 1e-10)
//!   E   = -log(Z)                                  (10.0 if Z <= 1e-10)

use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};

pub const KERNEL_SRC: &str = include_str!("../../kernels/energy_gradient.cu");

const WARPS_PER_BLOCK: usize = 8;
const BLOCK_SIZE: usize = WARPS_PER_BLOCK * 32;

/// Kernel entry point by D: template COMPS = ceil(D/32), so g[]/diff[] are
/// fixed-size register arrays. c1/c2/c4/c8/c16 cover D = 1..512.
fn kernel_name_for_dim(d: usize) -> Option<&'static str> {
    let comps = d.div_ceil(32);
    match comps {
        1 => Some("energy_gradient_fused_c1"),
        2 => Some("energy_gradient_fused_c2"),
        3 | 4 => Some("energy_gradient_fused_c4"),
        5..=8 => Some("energy_gradient_fused_c8"),
        9..=16 => Some("energy_gradient_fused_c16"),
        _ => None, // D > 512: fall back to CPU
    }
}

/// Persistent GPU state: splats stay in VRAM between calls.
pub struct EnergyGpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    func: CudaFunction,
    mu_gpu: CudaSlice<f32>,
    alpha_gpu: CudaSlice<f32>,
    kappa_gpu: CudaSlice<f32>,
    n: usize,
    d: usize,
}

impl EnergyGpu {
    /// Compile the kernel with NVRTC and upload splats. Returns None if
    /// CUDA/NVRTC is unavailable (caller falls back to CPU like GpuIndex does).
    pub fn new(mu: &[f32], alpha: &[f32], kappa: &[f32], n: usize, d: usize) -> Option<Self> {
        assert_eq!(mu.len(), n * d, "mu must be [N,D] row-major");
        assert_eq!(alpha.len(), n);
        assert_eq!(kappa.len(), n);

        // NOTE: cudarc 0.19's `use_fast_math` field only emits --fmad=true, so
        // --use_fast_math is passed explicitly via raw options (verified in
        // cudarc-0.19.10/src/nvrtc/safe.rs CompileOptions::build). sm_86 target.
        //
        // Compilation strategy (mirrors repo's M2M_PTX_PATH pattern):
        //   1. NVRTC runtime compile (preferred — no offline step). cudarc
        //      PANICS (not Err) when libnvrtc.so is absent — this WSL has no
        //      libnvrtc, only the Windows nvcc.exe — hence catch_unwind.
        //   2. Precompiled PTX from nvcc.exe --ptx (M2M_EGRAD_PTX_PATH env or
        //      kernels/energy_gradient.ptx), which is what build.rs produces
        //      for distance.cu on this machine.
        let opts = CompileOptions {
            arch: Some("compute_86"),
            options: vec!["--use_fast_math".into()],
            ..Default::default()
        };
        let ptx: Ptx = {
            // Silence the panic message cudarc prints when libnvrtc.so is
            // missing (it panics rather than returning Err); the fallback
            // PTX path below is the expected route on this WSL setup.
            let prev_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let r = std::panic::catch_unwind(|| compile_ptx_with_opts(KERNEL_SRC, opts).ok())
                .ok()
                .flatten();
            std::panic::set_hook(prev_hook);
            r
        }
        .or_else(Self::precompiled_ptx)?;
        let ctx = CudaContext::new(0).ok()?;
        let module = ctx.load_module(ptx).ok()?;
        let kname = kernel_name_for_dim(d)?;
        let func = module.load_function(kname).ok()?;

        let stream = ctx.default_stream();
        let mu_gpu = stream.clone_htod(mu).ok()?;
        let alpha_gpu = stream.clone_htod(alpha).ok()?;
        let kappa_gpu = stream.clone_htod(kappa).ok()?;

        Some(EnergyGpu { ctx, stream, func, mu_gpu, alpha_gpu, kappa_gpu, n, d })
    }

    /// Fused E+G for a batch of queries. Returns (E[B], G[B*D]).
    ///
    /// `with_transfers`: time H2D+D2H inside this call (bench flag). When
    /// false, x must already be resident — use [upload_x] + [launch].
    pub fn energy_gradient_batch(
        &self,
        x: &[f32],
        b: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        assert_eq!(x.len(), b * self.d);
        let x_gpu = self.stream.clone_htod(x).ok()?;
        let mut e_gpu = self.stream.alloc_zeros::<f32>(b).ok()?;
        let mut g_gpu = self.stream.alloc_zeros::<f32>(b * self.d).ok()?;
        self.launch(&x_gpu, &mut e_gpu, &mut g_gpu, b)?;
        let e = self.stream.clone_dtoh(&e_gpu).ok()?;
        let g = self.stream.clone_dtoh(&g_gpu).ok()?;
        Some((e, g))
    }

    /// Launch the kernel on already-resident buffers (no H2D/D2H).
    pub fn launch(
        &self,
        x_gpu: &CudaSlice<f32>,
        e_gpu: &mut CudaSlice<f32>,
        g_gpu: &mut CudaSlice<f32>,
        b: usize,
    ) -> Option<()> {
        let d = self.d as i32;
        // shared mem: s_x[D] + s_g[WARPS*D]
        let shared = (self.d + WARPS_PER_BLOCK * self.d) * 4;
        let cfg = LaunchConfig {
            grid_dim: (b as u32, 1, 1), // one block per query
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.func)
                .arg(x_gpu)
                .arg(&self.mu_gpu)
                .arg(&self.alpha_gpu)
                .arg(&self.kappa_gpu)
                .arg(e_gpu)
                .arg(g_gpu)
                .arg(&(b as i32))
                .arg(&(self.n as i32))
                .arg(&d)
                .launch(cfg)
                .ok()?;
        }
        self.stream.synchronize().ok()?;
        Some(())
    }

    /// Fallback: load the PTX precompiled by nvcc.exe (build.rs-style).
    fn precompiled_ptx() -> Option<Ptx> {
        let path = std::env::var("M2M_EGRAD_PTX_PATH")
            .ok()
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| "kernels/energy_gradient.ptx".to_string());
        let src = std::fs::read_to_string(&path).ok()?;
        eprintln!("[energy_gpu] NVRTC unavailable, using precompiled PTX: {path}");
        Some(Ptx::from_src(src))
    }

    pub fn upload_x(&self, x: &[f32]) -> Option<CudaSlice<f32>> {
        self.stream.clone_htod(x).ok()
    }

    pub fn alloc_e(&self, b: usize) -> Option<CudaSlice<f32>> {
        self.stream.alloc_zeros::<f32>(b).ok()
    }

    pub fn alloc_g(&self, b: usize) -> Option<CudaSlice<f32>> {
        self.stream.alloc_zeros::<f32>(b * self.d).ok()
    }

    pub fn download(
        &self,
        e_gpu: &CudaSlice<f32>,
        g_gpu: &CudaSlice<f32>,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let e = self.stream.clone_dtoh(e_gpu).ok()?;
        let g = self.stream.clone_dtoh(g_gpu).ok()?;
        Some((e, g))
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.n, self.d)
    }
}
