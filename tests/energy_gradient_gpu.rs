//! Test de integración del kernel energy_gradient fusionado (swarm A02).
//!
//! Valida el wrapper GPU contra la implementación CPU de referencia
//! (`crate::ebm::energy_api::EnergyApi`) con:
//!   1. Caso reducido N=1k D=64: max_abs_diff < 1e-3 (gradiente) y < 1e-5 (energía)
//!   2. Queries CERCA de centros de splats (lección del bench A02: con x
//!      uniforme el exp underflowea, ambos lados caen en el guard y el test
//!      es verde-vacío).
//!
//! Corre SOLO con --features cuda Y GPU disponible (si no, no-op como el
//! resto del módulo gpu: la GPU es aceleración aditiva, nunca requisito).

#![cfg(feature = "cuda")]

use m2m_vector_search::ebm::energy_api::EBMEnergy;
use m2m_vector_search::gpu::energy_gradient::EnergyGpu;
use ndarray::Array2;

/// Genera splats sintéticos reproducibles (xorshift, sin dependencias).
fn gen_splats(n: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Array2<f32>) {
    let mut s = seed | 1;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u32 << 21) as f32
    };
    // mu en [-2, 2]^d, alpha en [0.5, 1.5], kappa en [0.5, 2.5] (paisaje NO difuso)
    let mut mu = Vec::with_capacity(n * d);
    for _ in 0..n * d {
        mu.push(rnd() * 4.0 - 2.0);
    }
    let alpha: Vec<f32> = (0..n).map(|_| 0.5 + rnd()).collect();
    let kappa: Vec<f32> = (0..n).map(|_| 0.5 + rnd() * 2.0).collect();
    let mu_arr = Array2::from_shape_vec((n, d), mu.clone()).unwrap();
    (mu, alpha, kappa, mu_arr)
}

/// Queries cerca de centros de splats reales (E~1, Z sano, gradiente acumula).
fn queries_near_centers(mu: &[f32], n: usize, d: usize, b: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u32 << 21) as f32
    };
    let mut x = Vec::with_capacity(b * d);
    for q in 0..b {
        let center = (q * 7919) % n; // splat elegido determinista
        for j in 0..d {
            let m = mu[center * d + j];
            x.push(m + (rnd() - 0.5) * 0.5); // jitter ±0.25
        }
    }
    x
}

#[test]
fn energy_gradient_gpu_matches_cpu() {
    let d = 64usize;
    let n = 1_000usize;
    let b = 8usize;

    let (mu, alpha, kappa, mu_arr) = gen_splats(n, d, 20261015);
    let x = queries_near_centers(&mu, n, d, b, 20261016);

    // CPU de referencia: la EBMEnergy REAL del repo (no un puerto).
    let api = EBMEnergy::with_splats(
        mu_arr,
        ndarray::Array1::from(alpha.clone()),
        ndarray::Array1::from(kappa.clone()),
    );
    let x_arr = Array2::from_shape_vec((b, d), x.clone()).unwrap();
    let e_cpu = api.energy_batch(&x_arr);
    let g_cpu: Vec<Vec<f32>> = (0..b)
        .map(|q| api.energy_gradient(&x[q * d..(q + 1) * d]))
        .collect();

    // GPU
    let Some(gpu) = EnergyGpu::new(&mu, &alpha, &kappa, n, d) else {
        eprintln!("[test] GPU no disponible — no-op (aceleración aditiva)");
        return;
    };
    let (e_gpu, g_gpu) = gpu.energy_gradient_batch(&x, b).expect("launch GPU");

    // Energía: tolerancia 1e-5 (fast math, |E| ~ O(1))
    for q in 0..b {
        let diff = (e_cpu[q] - e_gpu[q]).abs();
        assert!(
            diff < 1e-5,
            "E[{q}]: cpu={} gpu={} diff={diff:.2e}",
            e_cpu[q],
            e_gpu[q]
        );
    }
    // Gradiente: tolerancia 1e-3 (consenso del brief; medido 1.2e-7)
    let mut max_diff: f32 = 0.0;
    for q in 0..b {
        for j in 0..d {
            let diff = (g_cpu[q][j] - g_gpu[q * d + j]).abs();
            max_diff = max_diff.max(diff);
        }
    }
    assert!(
        max_diff < 1e-3,
        "gradiente max_abs_diff={max_diff:.2e} >= 1e-3"
    );
    eprintln!("[test] OK: max_abs_diff gradiente = {max_diff:.2e} (n={n}, d={d}, b={b})");
}
