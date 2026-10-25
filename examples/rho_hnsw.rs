//! rho_HNSW: exponente de distancias-computadas vs N a recall fijo.
//! Espejo del protocolo Python (rho.py): mismos corpus (real MiniLM anidado
//! 2K/4K/full + sintetico 64 gaussianas + 20% fondo), M=32, efC=400, metric L2,
//! seed fija, GT = top-10 brute-force desde la query (incluye self-hit, igual
//! criterio para brute y HNSW; el punto full-recall sigue siendo 1.0).
//!
//! Uso: rho_hnsw <embeddings.npy> <n_docs>
//! Output: una linea JSON por caso, con sufijo _OK.
use std::sync::atomic::Ordering as AtomicOrdering;

use m2m_vector_search::hnsw_index::{HNSWIndex, HNSW_DIST_COMPUTATIONS};
use m2m_vector_search::interfaces::VectorIndex;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const D: usize = 384;
const SEED: u64 = 20260922;

struct F32x {
    data: Vec<f32>,
    n: usize,
}

impl F32x {
    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * D..(i + 1) * D]
    }
}

fn normalize(x: &mut [f32]) {
    let n: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n > 1e-12 {
        for v in x.iter_mut() {
            *v /= n;
        }
    }
}

/// sintetico: 64 gaussianas en S^383 + 20% fondo uniforme (seed + N)
fn synth_corpus(n: usize) -> F32x {
    let mut rng = StdRng::seed_from_u64(SEED + n as u64);
    let m = 64usize;
    let bg = (n as f64 * 0.20) as usize;
    let ncl = n - bg;
    let mut means = vec![0f32; m * D];
    for i in 0..m {
        let mut v: Vec<f32> = (0..D).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
        normalize(&mut v);
        means[i * D..(i + 1) * D].copy_from_slice(&v);
    }
    let mut data = vec![0f32; n * D];
    for i in 0..ncl {
        let c = rng.gen_range(0..m);
        for j in 0..D {
            // Irwin-Hall(4) centrado, std ~0.35 tras escalar (aprox del gaussiano Python)
            let g: f32 =
                (rng.gen::<f32>() + rng.gen::<f32>() + rng.gen::<f32>() + rng.gen::<f32>() - 2.0)
                    * 0.605;
            data[i * D + j] = means[c * D + j] + g;
        }
        normalize(&mut data[i * D..(i + 1) * D]);
    }
    for i in ncl..n {
        for j in 0..D {
            data[i * D + j] = rng.gen::<f32>() * 2.0 - 1.0;
        }
        normalize(&mut data[i * D..(i + 1) * D]);
    }
    F32x { data, n }
}

fn brute_gt(cx: &F32x, q: &[f32], k: usize) -> Vec<usize> {
    let mut ds: Vec<(f32, usize)> = (0..cx.n)
        .map(|i| {
            let d: f32 = q
                .iter()
                .zip(cx.row(i))
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>();
            (d, i)
        })
        .collect();
    ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    ds[..k].iter().map(|(_, i)| *i).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let npy_path = args.get(1).cloned();
    let npy_n: Option<usize> = args.get(2).and_then(|s| s.parse().ok());

    let mut cases: Vec<(String, F32x)> = Vec::new();

    // ---------- corpus real: prefijos anidados 2000 ⊂ 4000 ⊂ full ----------
    if let (Some(path), Some(n)) = (&npy_path, npy_n) {
        let bytes = std::fs::read(path).expect("leer npy");
        assert!(
            bytes.len() > 128 && &bytes[0..6] == b"\x93NUMPY",
            "no es npy"
        );
        let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let off = 10 + hlen;
        let n_av = (bytes.len() - off) / (D * 4);
        assert!(n_av >= n, "npy tiene {} filas, pido {}", n_av, n);
        for &ni in &[2000usize, 4000, n] {
            let lo = off;
            let hi = off + ni * D * 4;
            let data: Vec<f32> = bytes[lo..hi]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            cases.push((format!("real-{}", ni), F32x { data, n: ni }));
        }
    }

    // ---------- sintetico ----------
    for &n in &[2000usize, 8000, 32000, 64000] {
        cases.push((format!("synth-{}", n), synth_corpus(n)));
    }

    println!("{{\"start\": true, \"cases\": {}}}", cases.len());

    for (tag, cx) in cases {
        let mode = std::env::args().nth(3).unwrap_or_else(|| "all".to_string());
        if mode != "all" && !tag.starts_with(&mode) {
            continue;
        }
        let n = cx.n;
        let arr = Array2::from_shape_vec((n, D), cx.data.clone()).unwrap();

        let t0 = std::time::Instant::now();
        let mut idx = HNSWIndex::new(D, 32, 400, 100, "L2", SEED);
        idx.build(arr);
        let build_s = t0.elapsed().as_secs_f32();

        // queries deterministas
        let (queries, gt): (Vec<Vec<f32>>, Vec<Vec<usize>>) = if tag.starts_with("real") {
            // filas del bloque comun 0..2000: mismas queries en los 3 tamanos
            let mut qs = Vec::new();
            for qi in 0..100usize {
                qs.push(cx.row(5 * qi + 3).to_vec());
            }
            let gts = qs.iter().map(|q| brute_gt(&cx, q, 10)).collect();
            (qs, gts)
        } else {
            let mut rng = StdRng::seed_from_u64(SEED + 1);
            let mut qs = Vec::new();
            for _ in 0..200usize {
                let mut v: Vec<f32> = (0..D).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                normalize(&mut v);
                qs.push(v);
            }
            let gts = qs.iter().map(|q| brute_gt(&cx, q, 10)).collect();
            (qs, gts)
        };

        let mut efs: Vec<usize> = vec![10, 20, 40, 80, 160, 320, 640, 1280];
        if tag.starts_with("real") {
            efs = vec![2, 3, 4, 6, 8, 10, 20, 40, 80];
        }
        let mut sweep = Vec::new();
        for &ef in &efs {
            idx.set_ef_search(ef);
            HNSW_DIST_COMPUTATIONS.store(0, AtomicOrdering::Relaxed);
            let mut rec_sum = 0f32;
            for (qi, q) in queries.iter().enumerate() {
                let av = ndarray::ArrayView1::from(q.as_slice());
                let r = idx.search(av, 10);
                let hit = r.indices.iter().filter(|i| gt[qi].contains(i)).count();
                rec_sum += hit as f32 / 10.0;
            }
            let dists = HNSW_DIST_COMPUTATIONS.load(AtomicOrdering::Relaxed);
            let mean_rec = rec_sum / queries.len() as f32;
            sweep.push(serde_json::json!({
                "ef": ef,
                "dist_per_query": dists as f64 / queries.len() as f64,
                "frac_N": (dists as f64 / (queries.len() as f64 * n as f64) * 100000.0).round() / 100000.0,
                "recall@10": (mean_rec * 10000.0).round() / 10000.0,
            }));
        }

        println!(
            "{}",
            serde_json::json!({
                "tag": tag, "n": n,
                "build_s": (build_s * 100.0).round() / 100.0,
                "curve": sweep, "status": format!("{}_OK", tag)
            })
        );
    }
}
